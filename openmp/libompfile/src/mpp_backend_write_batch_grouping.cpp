#include "debug_log.h"
#include "mpp_backend.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

uint64_t
MppBackend::getWriteBatchGroupKey(const WriteBatchRequest &request) const {
  return request.GroupKey;
}

void MppBackend::completeWriteBatchRequest(WriteBatchRequest &request,
                                           int status, int errnum) {
  write_batch_batcher.complete(request, status, errnum);
}

void MppBackend::processWriteBatch(std::vector<WriteBatchRequest *> &batch) {
  if (batch.empty())
    return;

  counters.write_batch_count.fetch_add(1, std::memory_order_relaxed);
  std::unordered_map<uint64_t, std::vector<WriteBatchRequest *>> grouped;
  grouped.reserve(batch.size());
  for (WriteBatchRequest *request : batch)
    grouped[getWriteBatchGroupKey(*request)].push_back(request);

  for (auto &entry : grouped)
    processWriteGroup(entry.second);
}

void MppBackend::processWriteGroup(std::vector<WriteBatchRequest *> &group) {
  if (group.empty())
    return;

  // Debug: log all write request offsets before sorting
  for (size_t dbg_i = 0; dbg_i < group.size(); ++dbg_i) {
    io_log("[write-batch] req[%zu] offset=%ld size=%zu\n", dbg_i,
           group[dbg_i]->Offset, group[dbg_i]->Data.size());
  }

  struct CoalescedWrite {
    long Start = 0;
    long End = 0;
    std::vector<char> Data;
    std::vector<WriteBatchRequest *> Requests;
  };

  // Pass 1: sort by (offset, size) and decide coalescing groups and their
  // [Start, End) extent by adjacency in offset order, exactly as before
  // (the coalescer in ompfile_batcher.h, shared with the two-phase read).
  // Byte content is NOT materialized here. A member rejected here (EINVAL,
  // EOVERFLOW) is completed at once and joins no run.
  //
  // The remote handle is the lowest-offset member's, as before: group.front()
  // once sorted. A completed member's owner may return and destroy it, so the
  // handle is read before the first completion (the first reject runs after
  // the sort and before anything is completed) or, with no rejects, after
  // the pass, when every member is still alive.
  std::vector<WriteBatchRequest *> rejected;
  std::vector<CoalescedWrite> coalesced;
  int file_id = -1;
  bool file_id_taken = false;
  ompfile::batch::coalesceSortedRuns(
      group, settings.write_batch_max_batch_bytes, coalesced,
      [](const WriteBatchRequest &r) { return r.Offset; },
      [](const WriteBatchRequest &r) { return r.Data.size(); },
      [this, &rejected, &group, &file_id,
       &file_id_taken](WriteBatchRequest &r, int errnum) {
        if (!file_id_taken) {
          file_id = group.front()->FileHandle;
          file_id_taken = true;
        }
        completeWriteBatchRequest(r, -1, errnum);
        rejected.push_back(&r);
      });
  if (!file_id_taken)
    file_id = group.front()->FileHandle;

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

  int remote_handle = lookupRemoteHandle(file_id);
  if (remote_handle < 0) {
    counters.write_batch_failure_count.fetch_add(1, std::memory_order_relaxed);
    // A member pass 1 rejected is already complete and may be gone; only
    // its address is compared.
    for (WriteBatchRequest *request : group)
      if (std::find(rejected.begin(), rejected.end(), request) ==
          rejected.end())
        completeWriteBatchRequest(*request, -1, EBADF);
    return;
  }

  for (CoalescedWrite &item : coalesced) {
    counters.write_batch_coalesced_write_count.fetch_add(
        1, std::memory_order_relaxed);
    counters.write_batch_coalesced_bytes_total.fetch_add(
        item.Data.size(), std::memory_order_relaxed);
    if (item.Requests.size() > 1) {
      counters.write_batch_saved_events.fetch_add(item.Requests.size() - 1,
                                                  std::memory_order_relaxed);
    }

    size_t bytes_written = 0;
    const int rc =
        writeAtRemoteHandle(remote_handle, item.Start, item.Data.data(),
                            item.Data.size(), bytes_written);
    if (rc != 0 || bytes_written != item.Data.size()) {
      const int errnum = errno != 0 ? errno : EIO;
      counters.write_batch_failure_count.fetch_add(1,
                                                   std::memory_order_relaxed);
      for (WriteBatchRequest *request : item.Requests)
        completeWriteBatchRequest(*request, -1, errnum);
      continue;
    }

    for (WriteBatchRequest *request : item.Requests)
      completeWriteBatchRequest(*request, 0, 0);
  }
}
