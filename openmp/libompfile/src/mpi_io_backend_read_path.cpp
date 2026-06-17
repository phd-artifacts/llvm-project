#include "debug_log.h"
#include "mpi_io_backend.h"
#include "mpp_shim.h"

#include <cassert>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>

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
  if (strict_mpp_init_failed)
    return failStrictMpp("pread");
  if (context.Offset <
          static_cast<int64_t>(std::numeric_limits<long>::min()) ||
      context.Offset >
          static_cast<int64_t>(std::numeric_limits<long>::max())) {
    errno = EOVERFLOW;
    return -1;
  }
  if (context.Size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    errno = EOVERFLOW;
    return -1;
  }

  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  assert(size == static_cast<size_t>(context.Size) || context.Size == 0);
  if (offset < 0) {
    errno = EINVAL;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }
  pread_request_count.fetch_add(1, std::memory_order_relaxed);
  recordForecastRead(context.Hint, size);
  io_trace("MPIIOBackend::readAtWithContext enter this=%p req_id=%llu file=%d "
           "offset=%ld size=%zu ctx_flags=0x%x has_plan=%d has_path_key=%d "
           "plan_status=%d aggregator=%d remote_handle=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(context.RequestId), file_id, offset,
           size, context.ContextFlags,
           (context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PLAN) != 0,
           (context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY) != 0,
           context.Plan.Status, context.Plan.AggregatorRank,
           context.Plan.RemoteHandle);

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
      size_t bytes_read = 0;
      {
        const std::lock_guard<std::mutex> lock(mpp_call_mutex);
        pread_ok =
            ompfile::mpp::preadEx(remote_handle, offset, data, size, bytes_read);
      }
      if (!pread_ok) {
        io_log("MPP pread failed for file %d\n", file_id);
        return -1;
      }
      if (bytes_read > size) {
        errno = EPROTO;
        return -1;
      }
      if (bytes_read < size && data) {
        const size_t short_bytes = size - bytes_read;
        short_read_count.fetch_add(1, std::memory_order_relaxed);
        short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
        std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
        io_trace("MPIIOBackend::readAtWithContext short-read req_id=%llu "
                 "file=%d requested=%zu read=%zu short=%zu\n",
                 static_cast<unsigned long long>(context.RequestId), file_id,
                 size, bytes_read, short_bytes);
      }

      io_log("MPP read at offset completed.\n");
      io_trace("MPIIOBackend::readAtWithContext planned read success req_id=%llu "
               "file=%d remote_handle=%d\n",
               static_cast<unsigned long long>(context.RequestId), file_id,
               remote_handle);
      return 0;
    }
    io_log("Phase 1 planner produced a route but remote handle is missing for "
           "file %d; falling back to baseline pread path.\n",
           file_id);
    traceHandleState("readAtWithContext.plan_missing_remote", file_id, -1);
  }

  if (two_phase_enabled)
    two_phase_fallback_count.fetch_add(1, std::memory_order_relaxed);

  io_trace("MPIIOBackend::readAtWithContext fallback req_id=%llu file=%d "
           "offset=%ld size=%zu\n",
           static_cast<unsigned long long>(context.RequestId), file_id, offset,
           size);
  size_t bytes_read = 0;
  return readAtFallbackWithBytes(file_id, offset, data, size, bytes_read,
                                 &context.Hint);
}

int MPIIOBackend::readAtFallback(int file_id, long offset, void *data,
                                 size_t size) {
  size_t bytes_read = 0;
  return readAtFallbackWithBytes(file_id, offset, data, size, bytes_read);
}

int MPIIOBackend::readAtFallbackWithBytes(int file_id, long offset, void *data,
                                           size_t size, size_t &bytes_read,
                                           const ompfile::OmpFileIOHint *hint,
                                           bool force_no_stage) {
  bytes_read = 0;
  if (offset < 0) {
    errno = EINVAL;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }

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
      traceHandleState("readAtFallback.mpp_io.missing_remote", file_id,
                       remote_handle);
      errno = EBADF;
      return -1;
    }
    remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
    remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);
    bool pread_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      if (hint && shouldUseNoStageFallback(*hint, force_no_stage)) {
        pread_ok = ompfile::mpp::preadNoStageEx(remote_handle, offset, data,
                                                size, bytes_read);
      } else {
        pread_ok =
            ompfile::mpp::preadEx(remote_handle, offset, data, size, bytes_read);
      }
    }
    if (!pread_ok) {
      io_log("MPP pread failed for file %d\n", file_id);
      traceHandleState("readAtFallback.mpp_io.pread_fail", file_id,
                       remote_handle);
      return -1;
    }
    if (bytes_read > size) {
      errno = EPROTO;
      return -1;
    }
    if (bytes_read < size && data) {
      const size_t short_bytes = size - bytes_read;
      short_read_count.fetch_add(1, std::memory_order_relaxed);
      short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
      std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
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
      errno = EBADF;
      return -1;
    }
    file = it->second;
  }

  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    errno = EOVERFLOW;
    return -1;
  }

  MPI_Status status{};
  const int ret = MPI_File_read_at(file, offset, data, static_cast<int>(size),
                                   MPI_BYTE, &status);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Read at offset failed\n");
    return -1;
  }
  int count = 0;
  if (MPI_Get_count(&status, MPI_BYTE, &count) != MPI_SUCCESS || count < 0) {
    errno = EIO;
    return -1;
  }

  bytes_read = static_cast<size_t>(count);
  if (bytes_read > size) {
    errno = EPROTO;
    return -1;
  }
  if (bytes_read < size && data) {
    const size_t short_bytes = size - bytes_read;
    short_read_count.fetch_add(1, std::memory_order_relaxed);
    short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
    std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
  }

  io_log("Read at offset completed\n");
  return 0;
}
