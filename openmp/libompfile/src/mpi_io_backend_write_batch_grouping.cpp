#include "debug_log.h"
#include "mpi_io_backend.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

uint64_t
MPIIOBackend::getWriteBatchGroupKey(const WriteBatchRequest &request) const {
  return request.GroupKey;
}

void MPIIOBackend::completeWriteBatchRequest(WriteBatchRequest &request,
                                             int status, int errnum) {
  const std::lock_guard<std::mutex> lock(write_batch_mutex);
  request.Status = status;
  request.Errno = errnum;
  request.Done = true;
  write_batch_queue_cv.notify_all();
}

void MPIIOBackend::processWriteBatch(std::vector<WriteBatchRequest *> &batch) {
  if (batch.empty())
    return;

  write_batch_count.fetch_add(1, std::memory_order_relaxed);
  std::unordered_map<uint64_t, std::vector<WriteBatchRequest *>> grouped;
  grouped.reserve(batch.size());
  for (WriteBatchRequest *request : batch)
    grouped[getWriteBatchGroupKey(*request)].push_back(request);

  for (auto &entry : grouped)
    processWriteGroup(entry.second);
}

void MPIIOBackend::processWriteGroup(std::vector<WriteBatchRequest *> &group) {
  if (group.empty())
    return;

  // Debug: log all write request offsets before sorting
  for (size_t dbg_i = 0; dbg_i < group.size(); ++dbg_i) {
    io_log("[write-batch] req[%zu] offset=%ld size=%zu\n", dbg_i,
           group[dbg_i]->Offset, group[dbg_i]->Data.size());
  }

  std::sort(group.begin(), group.end(),
            [](const WriteBatchRequest *lhs, const WriteBatchRequest *rhs) {
              if (lhs->Offset != rhs->Offset)
                return lhs->Offset < rhs->Offset;
              return lhs->Data.size() < rhs->Data.size();
            });

  struct CoalescedWrite {
    long Start = 0;
    long End = 0;
    std::vector<char> Data;
    std::vector<WriteBatchRequest *> Requests;
  };

  // Pass 1: decide coalescing groups and their [Start, End) extent by
  // adjacency in offset order, exactly as before. Byte content is NOT
  // materialized here.
  std::vector<CoalescedWrite> coalesced;
  coalesced.reserve(group.size());
  for (WriteBatchRequest *request : group) {
    if (request->Offset < 0) {
      completeWriteBatchRequest(*request, -1, EINVAL);
      continue;
    }
    if (request->Data.size() >
            static_cast<size_t>(std::numeric_limits<long>::max()) ||
        request->Offset > std::numeric_limits<long>::max() -
                              static_cast<long>(request->Data.size())) {
      completeWriteBatchRequest(*request, -1, EOVERFLOW);
      continue;
    }

    const long start = request->Offset;
    const long end = start + static_cast<long>(request->Data.size());
    if (coalesced.empty()) {
      CoalescedWrite item{};
      item.Start = start;
      item.End = end;
      item.Requests.push_back(request);
      coalesced.push_back(std::move(item));
      continue;
    }

    CoalescedWrite &tail = coalesced.back();
    const bool mergeable = start <= tail.End;
    const long merged_end = std::max(tail.End, end);
    const uint64_t merged_size = static_cast<uint64_t>(merged_end - tail.Start);
    const bool exceeds_cap = write_batch_max_batch_bytes > 0 &&
                             merged_size > write_batch_max_batch_bytes;
    if (!mergeable || exceeds_cap) {
      CoalescedWrite item{};
      item.Start = start;
      item.End = end;
      item.Requests.push_back(request);
      coalesced.push_back(std::move(item));
      continue;
    }

    tail.End = merged_end;
    tail.Requests.push_back(request);
  }

  // Pass 2: materialize each group's bytes by applying member requests in
  // real submission order (ascending DebugRequestId, assigned when the
  // request was enqueued), not offset order. Offset order only decided which
  // requests coalesce; on a genuine byte-range overlap within a group, the
  // request issued last must be the one whose bytes survive, independent of
  // which one happens to sit at the lower offset.
  for (CoalescedWrite &item : coalesced) {
    item.Data.resize(static_cast<size_t>(item.End - item.Start), 0);
    std::sort(item.Requests.begin(), item.Requests.end(),
              [](const WriteBatchRequest *lhs, const WriteBatchRequest *rhs) {
                return lhs->DebugRequestId < rhs->DebugRequestId;
              });
    for (WriteBatchRequest *request : item.Requests) {
      if (request->Data.empty())
        continue;
      const size_t copy_offset =
          static_cast<size_t>(request->Offset - item.Start);
      std::memcpy(item.Data.data() + copy_offset, request->Data.data(),
                  request->Data.size());
    }
  }

  // Debug: log coalesced write ranges after grouping
  for (size_t dbg_i = 0; dbg_i < coalesced.size(); ++dbg_i) {
    io_log("[write-coalesced] range[%zu] start=%ld end=%ld size=%zu "
           "requests=%zu\n",
           dbg_i, coalesced[dbg_i].Start, coalesced[dbg_i].End,
           coalesced[dbg_i].Data.size(), coalesced[dbg_i].Requests.size());
  }

  int remote_handle = -1;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = remote_file_handle_map.find(group.front()->FileHandle);
    if (it != remote_file_handle_map.end())
      remote_handle = it->second;
  }
  if (remote_handle < 0) {
    write_batch_failure_count.fetch_add(1, std::memory_order_relaxed);
    for (WriteBatchRequest *request : group)
      completeWriteBatchRequest(*request, -1, EBADF);
    return;
  }

  for (CoalescedWrite &item : coalesced) {
    write_batch_coalesced_write_count.fetch_add(1, std::memory_order_relaxed);
    write_batch_coalesced_bytes_total.fetch_add(item.Data.size(),
                                                std::memory_order_relaxed);
    if (item.Requests.size() > 1) {
      write_batch_saved_events.fetch_add(item.Requests.size() - 1,
                                         std::memory_order_relaxed);
    }

    size_t bytes_written = 0;
    const int rc =
        writeAtRemoteHandle(remote_handle, item.Start, item.Data.data(),
                            item.Data.size(), bytes_written);
    if (rc != 0 || bytes_written != item.Data.size()) {
      const int errnum = errno != 0 ? errno : EIO;
      write_batch_failure_count.fetch_add(1, std::memory_order_relaxed);
      for (WriteBatchRequest *request : item.Requests)
        completeWriteBatchRequest(*request, -1, errnum);
      continue;
    }

    for (WriteBatchRequest *request : item.Requests)
      completeWriteBatchRequest(*request, 0, 0);
  }
}
