#include "mpi_io_backend.h"
#include "debug_log.h"
#include "mpp_shim.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mpi.h>
#include <unordered_map>
#include <vector>

MPIIOBackend::MPIIOBackend() {
  int provided = 0;
  int initialized = 0;

  // Check if MPI is already initialized
  MPI_Initialized(&initialized);
  if (!initialized) {
    // Initialize MPI with thread support
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    io_log("MPI_Init_thread called, provided = %d\n", provided);
    externally_initialized = 0;
  } else {
    io_log("MPI already initialized by user program.\n");
    externally_initialized = 1;
  }

  const char *mpp_open_env = std::getenv("LIBOMPFILE_MPP_OPEN");
  if (mpp_open_env && mpp_open_env[0] == '1' && mpp_open_env[1] == '\0') {
    mpp_open_enabled = true;
    io_log("LIBOMPFILE_MPP_OPEN=1: enabling MPP open/close path.\n");
  }

  const char *mpp_io_env = std::getenv("LIBOMPFILE_MPP_IO");
  if (mpp_io_env && mpp_io_env[0] == '1' && mpp_io_env[1] == '\0') {
    if (mpp_open_enabled) {
      mpp_io_enabled = true;
      io_log("LIBOMPFILE_MPP_IO=1: enabling MPP I/O path.\n");
    } else {
      io_log("LIBOMPFILE_MPP_IO set without MPP open enabled.\n");
    }
  }

  const bool remote_only = mpp_open_enabled && mpp_io_enabled;
  if (!remote_only) {
    MPI_Comm source_comm = MPI_COMM_WORLD;
    const char *comm_self_env = std::getenv("LIBOMPFILE_MPI_COMM_SELF");
    if (comm_self_env && comm_self_env[0] == '1' && comm_self_env[1] == '\0') {
      source_comm = MPI_COMM_SELF;
      io_log("LIBOMPFILE_MPI_COMM_SELF=1: using MPI_COMM_SELF for file I/O.\n");
    }

    // Duplicate the chosen communicator for file I/O.
    MPI_Comm_dup(source_comm, &file_comm);
    io_log("MPI_Comm_dup completed for file I/O.\n");
  } else {
    io_log("Remote-only MPP mode: skipping MPI communicator duplication.\n");
  }

  if (ompfile::mpp::init()) {
    const char *env = std::getenv("LIBOMPFILE_MPP_PING");
    if (env && env[0] == '1' && env[1] == '\0') {
      if (!ompfile::mpp::ping())
        io_log("MPP shim ping failed.\n");
    }
  }

  two_phase_enabled = parseBoolEnv("LIBOMPFILE_OPT_TWO_PHASE", false);
  two_phase_window_us =
      parseUint64Env("LIBOMPFILE_OPT_TWO_PHASE_WINDOW_US", 0);
  two_phase_max_batch_bytes =
      parseUint64Env("LIBOMPFILE_OPT_TWO_PHASE_MAX_BATCH_BYTES", 0);

  io_log("Two-phase guard config: enabled=%d window_us=%llu "
         "max_batch_bytes=%llu\n",
         static_cast<int>(two_phase_enabled),
         static_cast<unsigned long long>(two_phase_window_us),
         static_cast<unsigned long long>(two_phase_max_batch_bytes));

  if (isTwoPhaseActive()) {
    io_log("Two-phase batching active (leader/follower mode).\n");
  } else if (two_phase_enabled) {
    io_log("Two-phase requested but inactive (requires remote-only MPP mode).\n");
  }

}

MPIIOBackend::~MPIIOBackend() {
  reportPhase0Stats();
  ompfile::mpp::finalize();
  if (file_comm != MPI_COMM_NULL) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
      MPI_Comm_free(&file_comm);
  }
  // if(!externally_initialized) {
  //   // Free the duplicated communicator
  //   io_log("MPI_Comm_free completed.\n");
  // }
}

int MPIIOBackend::open(const char *filename) {
  int file_id = getNextFileHandle();
  const bool remote_only = mpp_open_enabled && mpp_io_enabled;

  io_log("Opening file %s with file_id %d\n", filename, file_id);

  if (remote_only) {
    int remote_handle = -1;
    bool open_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      open_ok = ompfile::mpp::open(filename, O_RDWR, 0666, remote_handle);
    }
    if (!open_ok) {
      io_log("MPP open failed for %s\n", filename);
      errno = EIO;
      return -1;
    }
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      remote_file_handle_map[file_id] = remote_handle;
      logical_handle_set.insert(file_id);
    }
    io_log("Remote-only open completed for file_id %d\n", file_id);
    return file_id;
  }

  MPI_File file_handle;
  int ret = MPI_File_open(file_comm, filename, MPI_MODE_RDWR, MPI_INFO_NULL,
                          &file_handle);
  if (ret != MPI_SUCCESS) {
    char err_str[MPI_MAX_ERROR_STRING];
    int err_len = 0;
    MPI_Error_string(ret, err_str, &err_len);
    io_log("MPI_File_open failed for %s: %.*s (code %d)\n", filename,
           err_len, err_str, ret);
    errno = EIO;
    return -1;
  }
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    file_handle_map[file_id] = file_handle;
    logical_handle_set.insert(file_id);
  }

  if (mpp_open_enabled) {
    int remote_handle = -1;
    bool open_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      open_ok = ompfile::mpp::open(filename, O_RDWR, 0666, remote_handle);
    }
    if (!open_ok) {
      io_log("MPP open failed for %s\n", filename);
      MPI_File_close(&file_handle);
      {
        const std::lock_guard<std::mutex> lock(handle_mutex);
        file_handle_map.erase(file_id);
        logical_handle_set.erase(file_id);
      }
      errno = EIO;
      return -1;
    }
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      remote_file_handle_map[file_id] = remote_handle;
    }
  }
  return file_id;
}

int MPIIOBackend::write(int file_id, const void *data, size_t size) {
  if (mpp_open_enabled && mpp_io_enabled) {
    io_log("Error: write() without explicit offset is unsupported in "
           "remote-only MPP mode.\n");
    errno = ENOTSUP;
    return -1;
  }

  io_log("Writing %zu bytes to file %d\n", size, file_id);

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }
    file = it->second;
  }

  int ret = MPI_File_write(file, data, size, MPI_BYTE, MPI_STATUS_IGNORE);

  if (ret != MPI_SUCCESS) {
    io_log("Error: Write failed\n");
    return -1;
  }

  io_log("Write completed\n");

  return 0;
}

int MPIIOBackend::close(int file_id) {
  io_log("Closing file %d\n", file_id);
  const bool remote_only = mpp_open_enabled && mpp_io_enabled;
  int remote_handle = -1;
  bool has_remote_handle = false;
  MPI_File mpi_file = MPI_FILE_NULL;

  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    if (logical_handle_set.find(file_id) == logical_handle_set.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }

    if (mpp_open_enabled) {
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end()) {
        remote_handle = it->second;
        has_remote_handle = true;
        remote_file_handle_map.erase(it);
      }
    }

    if (!remote_only) {
      auto it = file_handle_map.find(file_id);
      if (it != file_handle_map.end()) {
        mpi_file = it->second;
        file_handle_map.erase(it);
      }
    }

    logical_handle_set.erase(file_id);
  }

  int mpp_ret = 0;
  if (has_remote_handle) {
    bool close_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      close_ok = ompfile::mpp::close(remote_handle);
    }
    if (!close_ok) {
      io_log("MPP close failed for file %d\n", file_id);
      mpp_ret = -1;
    }
  }

  if (remote_only) {
    if (mpp_ret != 0)
      return -1;
    io_log("Remote-only close completed\n");
    return 0;
  }

  if (mpi_file == MPI_FILE_NULL) {
    io_log("Error: Missing MPI file handle for file %d\n", file_id);
    return -1;
  }

  int ret = MPI_File_close(&mpi_file);

  if (ret != MPI_SUCCESS) {
    io_log("Error: Close failed\n");
    return -1;
  }

  if (mpp_ret != 0)
    return -1;

  io_log("Close completed\n");

  return 0;
}

int MPIIOBackend::read(int file_id, void *data, size_t size) {
  if (mpp_open_enabled && mpp_io_enabled) {
    io_log("Error: read() without explicit offset is unsupported in "
           "remote-only MPP mode.\n");
    errno = ENOTSUP;
    return -1;
  }

  io_log("Reading %zu bytes from file %d\n", size, file_id);

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }
    file = it->second;
  }

  int ret = MPI_File_read(file, data, size, MPI_BYTE, MPI_STATUS_IGNORE);

  if (ret != MPI_SUCCESS) {
    io_log("Error: Read failed\n");
    return -1;
  }

  io_log("Read completed\n");

  return 0;
}

int MPIIOBackend::seek(int file_id, long offset) {
  if (mpp_open_enabled && mpp_io_enabled) {
    io_log("Error: seek() is unsupported in remote-only MPP mode.\n");
    errno = ENOTSUP;
    return -1;
  }

  io_log("Setting file pointer to offset %ld\n", offset);

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }
    file = it->second;
  }

  int ret = MPI_File_seek(file, offset, MPI_SEEK_SET);

  if (ret != MPI_SUCCESS) {
    io_log("Error: Seek failed\n");
    return -1;
  }

  io_log("Seek completed\n");

  return 0;
}

int MPIIOBackend::readAt(int file_id, long offset, void *data, size_t size) {
  ompfile::OmpFileReadRequestContext context{};
  context.FileHandle = file_id;
  context.Offset = static_cast<int64_t>(offset);
  context.Size = static_cast<uint64_t>(size);
  return readAtWithContext(context, data, size);
}

bool MPIIOBackend::hasUsablePlannedRead(
    const ompfile::OmpFileReadRequestContext &context) const {
  if ((context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PLAN) == 0)
    return false;
  if (context.Plan.Status != 0)
    return false;
  if (!mpp_open_enabled)
    return false;
  return true;
}

int MPIIOBackend::readAtWithContext(
    const ompfile::OmpFileReadRequestContext &context, void *data,
    size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  pread_request_count.fetch_add(1, std::memory_order_relaxed);

  io_log("Reading %zu bytes from file %d at offset %lld\n", size, file_id,
         static_cast<long long>(offset));

  if (isTwoPhaseActive())
    return readAtTwoPhase(context, data, size);

  if (hasUsablePlannedRead(context)) {
    int remote_handle = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end())
        remote_handle = it->second;
    }
    if (remote_handle >= 0) {
      remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
      remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);
      io_log("Phase 1 planned read route: request_id=%llu file=%d "
             "path_key=%llu aggregator_rank=%d\n",
             static_cast<unsigned long long>(context.RequestId), file_id,
             static_cast<unsigned long long>(context.PathKey),
             context.Plan.AggregatorRank);
      bool pread_ok = false;
      {
        const std::lock_guard<std::mutex> lock(mpp_call_mutex);
        pread_ok = ompfile::mpp::pread(remote_handle, offset, data, size);
      }
      if (!pread_ok) {
        io_log("MPP pread failed for file %d\n", file_id);
        return -1;
      }

      io_log("MPP read at offset completed.\n");
      return 0;
    }
    io_log("Phase 1 planner produced a route but remote handle is missing for "
           "file %d; falling back to baseline pread path.\n",
           file_id);
  }

  if (two_phase_enabled)
    two_phase_fallback_count.fetch_add(1, std::memory_order_relaxed);

  return readAtFallback(file_id, offset, data, size);
}

int MPIIOBackend::readAtFallback(int file_id, long offset, void *data,
                                 size_t size) {
  const bool remote_only = mpp_open_enabled && mpp_io_enabled;

  if (remote_only) {
    int remote_handle = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end())
        remote_handle = it->second;
    }
    if (remote_handle < 0) {
      io_log("Error: Missing remote handle for file %d\n", file_id);
      return -1;
    }
    remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
    remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);
    bool pread_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      pread_ok = ompfile::mpp::pread(remote_handle, offset, data, size);
    }
    if (!pread_ok) {
      io_log("MPP pread failed for file %d\n", file_id);
      return -1;
    }

    io_log("MPP read at offset completed.\n");
    return 0;
  }

  // Check if the file handle is valid
  if (mpp_io_enabled) {
    int remote_handle = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end())
        remote_handle = it->second;
    }
    if (remote_handle < 0) {
      io_log("Error: Missing remote handle for file %d\n", file_id);
      return -1;
    }
    remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
    remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);
    bool pread_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      pread_ok = ompfile::mpp::pread(remote_handle, offset, data, size);
    }
    if (!pread_ok) {
      io_log("MPP pread failed for file %d\n", file_id);
      return -1;
    }

    io_log("MPP read at offset completed.\n");
    return 0;
  }

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }
    file = it->second;
  }

  // Perform the actual read at the specified offset
  int ret =
      MPI_File_read_at(file, offset, data, size, MPI_BYTE, MPI_STATUS_IGNORE);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Read at offset failed\n");
    return -1;
  }

  io_log("Read at offset completed\n");
  return 0;
}

int MPIIOBackend::readAtTwoPhase(
    const ompfile::OmpFileReadRequestContext &context, void *data,
    size_t size) {
  TwoPhaseReadRequest request{};
  request.FileHandle = context.FileHandle;
  request.ClientRank = context.ClientRank;
  request.Offset = static_cast<long>(context.Offset);
  request.Size = size;
  request.Buffer = data;
  request.PathKey = context.PathKey;
  request.HasPathKey =
      (context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY) != 0;

  std::vector<TwoPhaseReadRequest *> batch;
  std::unique_lock<std::mutex> lock(two_phase_mutex);
  two_phase_queue.push_back(&request);
  two_phase_queue_cv.notify_all();

  while (!request.Done) {
    if (!two_phase_batch_in_progress) {
      two_phase_batch_in_progress = true;

      if (two_phase_window_us > 0) {
        two_phase_queue_cv.wait_for(lock,
                                    std::chrono::microseconds(two_phase_window_us));
      }

      while (!two_phase_queue.empty()) {
        batch.push_back(two_phase_queue.front());
        two_phase_queue.pop_front();
      }

      lock.unlock();
      processTwoPhaseBatch(batch);
      batch.clear();
      lock.lock();

      two_phase_batch_in_progress = false;
      two_phase_queue_cv.notify_all();
    } else {
      two_phase_queue_cv.wait(
          lock, [this, &request] { return request.Done || !two_phase_batch_in_progress; });
    }
  }

  if (request.Status != 0) {
    errno = request.Errno;
    return -1;
  }

  return 0;
}

void MPIIOBackend::processTwoPhaseBatch(std::vector<TwoPhaseReadRequest *> &batch) {
  if (batch.empty())
    return;

  two_phase_batch_count.fetch_add(1, std::memory_order_relaxed);

  std::unordered_map<uint64_t, std::vector<TwoPhaseReadRequest *>> grouped;
  grouped.reserve(batch.size());
  for (TwoPhaseReadRequest *request : batch)
    grouped[getTwoPhaseGroupKey(*request)].push_back(request);

  for (auto &entry : grouped)
    processTwoPhaseGroup(entry.second);
}

void MPIIOBackend::processTwoPhaseGroup(std::vector<TwoPhaseReadRequest *> &group) {
  if (group.empty())
    return;

  if (group.size() == 1) {
    TwoPhaseReadRequest *request = group.front();
    const int rc = readAtFallback(request->FileHandle, request->Offset,
                                  request->Buffer, request->Size);
    completeTwoPhaseRequest(*request, rc, rc == 0 ? 0 : errno);
    return;
  }

  std::sort(group.begin(), group.end(),
            [](const TwoPhaseReadRequest *lhs, const TwoPhaseReadRequest *rhs) {
              if (lhs->Offset != rhs->Offset)
                return lhs->Offset < rhs->Offset;
              return lhs->Size < rhs->Size;
            });

  struct CoalescedRead {
    long Start = 0;
    long End = 0;
    std::vector<TwoPhaseReadRequest *> Requests;
    bool Skip = false;
    int ErrorStatus = 0;
    int ErrorErrno = 0;
  };

  std::vector<CoalescedRead> coalesced;
  coalesced.reserve(group.size());
  for (TwoPhaseReadRequest *request : group) {
    const long start = request->Offset;
    const long end = start + static_cast<long>(request->Size);
    if (coalesced.empty()) {
      CoalescedRead item{};
      item.Start = start;
      item.End = end;
      item.Requests.push_back(request);
      coalesced.push_back(std::move(item));
      continue;
    }

    CoalescedRead &tail = coalesced.back();
    const bool contiguous = start <= tail.End;
    const long merged_end = std::max(tail.End, end);
    const uint64_t merged_size =
        static_cast<uint64_t>(merged_end - tail.Start);
    const bool exceeds_max_batch =
        two_phase_max_batch_bytes > 0 && merged_size > two_phase_max_batch_bytes;

    if (contiguous && !exceeds_max_batch) {
      tail.End = merged_end;
      tail.Requests.push_back(request);
      continue;
    }

    CoalescedRead item{};
    item.Start = start;
    item.End = end;
    item.Requests.push_back(request);
    coalesced.push_back(std::move(item));
  }

  ompfile::OmpFileIOBatchRequest batch_request{};
  batch_request.AbiVersion = ompfile::OMPFILE_SCHED_BATCH_ABI_VERSION;
  batch_request.SegmentCount = static_cast<uint32_t>(coalesced.size());
  batch_request.RequestFlags =
      ompfile::OMPFILE_BATCH_REQ_FAIL_ON_ANY_ERROR |
      ompfile::OMPFILE_BATCH_REQ_DISABLE_SCALAR_FALLBACK;
  batch_request.BatchId =
      two_phase_planner_batch_id.fetch_add(1, std::memory_order_relaxed);
  std::vector<ompfile::OmpFileIOBatchSegment> batch_segments;
  batch_segments.reserve(coalesced.size());
  for (size_t i = 0; i < coalesced.size(); ++i) {
    const CoalescedRead &item = coalesced[i];
    ompfile::OmpFileIOBatchSegment segment{};
    segment.SegmentId = static_cast<uint64_t>(i);
    segment.FileHandle = item.Requests.front()->FileHandle;
    segment.ClientRank = item.Requests.front()->ClientRank;
    segment.Offset = item.Start;
    segment.Size = static_cast<uint64_t>(item.End - item.Start);
    segment.PathKey = item.Requests.front()->HasPathKey
                          ? item.Requests.front()->PathKey
                          : static_cast<uint64_t>(0);
    batch_segments.push_back(segment);
  }

  ompfile::OmpFileIOBatchPlan batch_plan{};
  std::vector<ompfile::OmpFileIOBatchPlanEntry> batch_entries;
  if (ompfile::mpp::schedBatchRequest(batch_request, batch_segments, batch_plan,
                                      batch_entries)) {
    two_phase_planner_batch_count.fetch_add(1, std::memory_order_relaxed);
    two_phase_planner_segment_count.fetch_add(batch_entries.size(),
                                              std::memory_order_relaxed);
    if ((batch_plan.PlanFlags & ompfile::OMPFILE_BATCH_PLAN_SCALAR_FALLBACK) !=
        0) {
      two_phase_planner_scalar_fallback_count.fetch_add(
          1, std::memory_order_relaxed);
    }

    if (batch_entries.size() != coalesced.size()) {
      two_phase_planner_error_count.fetch_add(1, std::memory_order_relaxed);
      io_log("Two-phase planner returned mismatched entry count: expected=%zu "
             "actual=%zu\n",
             coalesced.size(), batch_entries.size());
    } else {
      for (size_t i = 0; i < batch_entries.size(); ++i) {
        const ompfile::OmpFileIOBatchPlanEntry &entry = batch_entries[i];
        CoalescedRead &item = coalesced[i];
        if (entry.Status != 0) {
          item.Skip = true;
          item.ErrorStatus = -1;
          item.ErrorErrno = entry.Errno != 0 ? entry.Errno : EIO;
        }
      }
    }
  } else {
    two_phase_planner_error_count.fetch_add(1, std::memory_order_relaxed);
  }

  two_phase_coalesced_read_count.fetch_add(coalesced.size(),
                                           std::memory_order_relaxed);

  for (CoalescedRead &item : coalesced) {
    if (item.Skip) {
      for (TwoPhaseReadRequest *request : item.Requests)
        completeTwoPhaseRequest(*request, item.ErrorStatus, item.ErrorErrno);
      continue;
    }

    const size_t read_size = static_cast<size_t>(item.End - item.Start);
    two_phase_coalesced_bytes_total.fetch_add(read_size,
                                              std::memory_order_relaxed);
    std::vector<char> buffer(read_size);

    const int rc = readAtFallback(item.Requests.front()->FileHandle, item.Start,
                                  buffer.data(), read_size);
    if (rc != 0) {
      const int errnum = errno;
      for (TwoPhaseReadRequest *request : item.Requests)
        completeTwoPhaseRequest(*request, rc, errnum);
      continue;
    }

    for (TwoPhaseReadRequest *request : item.Requests) {
      const size_t scatter_offset =
          static_cast<size_t>(request->Offset - item.Start);
      std::memcpy(request->Buffer, buffer.data() + scatter_offset,
                  request->Size);
      completeTwoPhaseRequest(*request, 0, 0);
    }
  }
}

uint64_t MPIIOBackend::getTwoPhaseGroupKey(
    const TwoPhaseReadRequest &request) const {
  if (request.HasPathKey)
    return request.PathKey;
  return static_cast<uint64_t>(static_cast<uint32_t>(request.FileHandle));
}

void MPIIOBackend::completeTwoPhaseRequest(TwoPhaseReadRequest &request,
                                           int status, int errnum) {
  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  request.Status = status;
  request.Errno = errnum;
  request.Done = true;
}

bool MPIIOBackend::isTwoPhaseActive() const {
  return two_phase_enabled && mpp_open_enabled && mpp_io_enabled;
}

int MPIIOBackend::writeAt(int file_id, long offset, const void *data,
                          size_t size) {
  const bool remote_only = mpp_open_enabled && mpp_io_enabled;

  io_log("Writing %zu bytes to file %d at offset %lld\n", size, file_id,
         static_cast<long long>(offset));

  if (remote_only) {
    int remote_handle = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end())
        remote_handle = it->second;
    }
    if (remote_handle < 0) {
      io_log("Error: Missing remote handle for file %d\n", file_id);
      return -1;
    }
    bool pwrite_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      pwrite_ok = ompfile::mpp::pwrite(remote_handle, offset, data, size);
    }
    if (!pwrite_ok) {
      io_log("MPP pwrite failed for file %d\n", file_id);
      return -1;
    }

    io_log("MPP write at offset completed.\n");
    return 0;
  }

  // Check if the file handle is valid
  if (mpp_io_enabled) {
    int remote_handle = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end())
        remote_handle = it->second;
    }
    if (remote_handle < 0) {
      io_log("Error: Missing remote handle for file %d\n", file_id);
      return -1;
    }
    bool pwrite_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      pwrite_ok = ompfile::mpp::pwrite(remote_handle, offset, data, size);
    }
    if (!pwrite_ok) {
      io_log("MPP pwrite failed for file %d\n", file_id);
      return -1;
    }

    io_log("MPP write at offset completed.\n");
    return 0;
  }

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }
    file = it->second;
  }

  // Perform the actual write at the specified offset
  int ret =
      MPI_File_write_at(file, offset, data, size, MPI_BYTE, MPI_STATUS_IGNORE);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Write at offset failed\n");
    return -1;
  }

  io_log("Write at offset completed\n");
  return 0;
}

int MPIIOBackend::getNextFileHandle() {
  return next_file_handle.fetch_add(1, std::memory_order_relaxed);
}

bool MPIIOBackend::parseBoolEnv(const char *name, bool default_value) {
  const char *env = std::getenv(name);
  if (!env)
    return default_value;

  if (env[0] == '1' && env[1] == '\0')
    return true;
  if (env[0] == '0' && env[1] == '\0')
    return false;

  io_log("Invalid boolean value for %s='%s'; using default=%d\n", name, env,
         static_cast<int>(default_value));
  return default_value;
}

uint64_t MPIIOBackend::parseUint64Env(const char *name, uint64_t default_value) {
  const char *env = std::getenv(name);
  if (!env)
    return default_value;

  errno = 0;
  char *end_ptr = nullptr;
  const unsigned long long value = std::strtoull(env, &end_ptr, 10);
  if (errno != 0 || end_ptr == env || (end_ptr && *end_ptr != '\0')) {
    io_log("Invalid integer value for %s='%s'; using default=%llu\n", name, env,
           static_cast<unsigned long long>(default_value));
    return default_value;
  }

  return static_cast<uint64_t>(value);
}

bool MPIIOBackend::shouldReportStats() {
  return parseBoolEnv("LIBOMPFILE_OPT_STATS", false);
}

void MPIIOBackend::reportPhase0Stats() const {
  if (!two_phase_enabled && !shouldReportStats())
    return;

  const uint64_t pread_reqs =
      pread_request_count.load(std::memory_order_relaxed);
  const uint64_t remote_events =
      remote_pread_event_count.load(std::memory_order_relaxed);
  const uint64_t remote_bytes =
      remote_pread_bytes_total.load(std::memory_order_relaxed);
  const uint64_t fallback_count =
      two_phase_fallback_count.load(std::memory_order_relaxed);
  const uint64_t batch_count =
      two_phase_batch_count.load(std::memory_order_relaxed);
  const uint64_t coalesced_reads =
      two_phase_coalesced_read_count.load(std::memory_order_relaxed);
  const uint64_t coalesced_bytes =
      two_phase_coalesced_bytes_total.load(std::memory_order_relaxed);
  const uint64_t planner_batches =
      two_phase_planner_batch_count.load(std::memory_order_relaxed);
  const uint64_t planner_segments =
      two_phase_planner_segment_count.load(std::memory_order_relaxed);
  const uint64_t planner_scalar_fallbacks =
      two_phase_planner_scalar_fallback_count.load(std::memory_order_relaxed);
  const uint64_t planner_errors =
      two_phase_planner_error_count.load(std::memory_order_relaxed);

  const double avg_remote_bytes =
      remote_events == 0 ? 0.0
                         : static_cast<double>(remote_bytes) /
                               static_cast<double>(remote_events);
  const double avg_coalesced_bytes =
      coalesced_reads == 0 ? 0.0
                           : static_cast<double>(coalesced_bytes) /
                                 static_cast<double>(coalesced_reads);

  io_log("Two-phase stats: pread_requests=%llu remote_pread_events=%llu "
         "remote_pread_bytes_total=%llu remote_pread_avg_bytes=%.2f "
         "batch_count=%llu coalesced_reads=%llu coalesced_bytes=%llu "
         "coalesced_avg_bytes=%.2f planner_batches=%llu "
         "planner_segments=%llu planner_scalar_fallbacks=%llu "
         "planner_errors=%llu fallbacks=%llu two_phase_enabled=%d "
         "two_phase_active=%d window_us=%llu max_batch_bytes=%llu\n",
         static_cast<unsigned long long>(pread_reqs),
         static_cast<unsigned long long>(remote_events),
         static_cast<unsigned long long>(remote_bytes), avg_remote_bytes,
         static_cast<unsigned long long>(batch_count),
         static_cast<unsigned long long>(coalesced_reads),
         static_cast<unsigned long long>(coalesced_bytes), avg_coalesced_bytes,
         static_cast<unsigned long long>(planner_batches),
         static_cast<unsigned long long>(planner_segments),
         static_cast<unsigned long long>(planner_scalar_fallbacks),
         static_cast<unsigned long long>(planner_errors),
         static_cast<unsigned long long>(fallback_count),
         static_cast<int>(two_phase_enabled),
         static_cast<int>(isTwoPhaseActive()),
         static_cast<unsigned long long>(two_phase_window_us),
         static_cast<unsigned long long>(two_phase_max_batch_bytes));
}
