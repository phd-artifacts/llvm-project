#include "debug_log.h"
#include "mpi_io_backend.h"
#include "mpp_shim.h"

#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace {

bool freshnessWriteThroughModeEnabled() {
  const char *mode = std::getenv("LIBOMPFILE_TILE_FRESHNESS_WRITE_MODE");
  if (mode && std::strcmp(mode, "write-back") == 0)
    return false;
  return true;
}

} // namespace

bool MPIIOBackend::isWriteBatchActive() const {
  return write_batch_enabled && mpp_remote_only;
}

int MPIIOBackend::writeAtRemoteHandle(int remote_handle, long offset,
                                      const void *data, size_t size,
                                      size_t &bytes_written) {
  bytes_written = 0;
  if (size == 0)
    return 0;

  const char *cursor = static_cast<const char *>(data);
  long current_offset = offset;
  size_t remaining = size;

  while (remaining > 0) {
    remote_pwrite_event_count.fetch_add(1, std::memory_order_relaxed);
    remote_pwrite_bytes_total.fetch_add(remaining, std::memory_order_relaxed);

    bool pwrite_ok = false;
    size_t call_bytes_written = 0;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      pwrite_ok = ompfile::mpp::pwriteEx(remote_handle, current_offset, cursor,
                                         remaining, call_bytes_written);
    }
    if (!pwrite_ok) {
      const int saved_errno = errno;
      io_trace("MPIIOBackend::writeAtRemoteHandle pwriteEx fail "
               "remote_handle=%d offset=%ld remaining=%zu cursor=%p errno=%d\n",
               remote_handle, current_offset, remaining,
               static_cast<const void *>(cursor), saved_errno);
      if (errno == EIO && remaining > 0) {
        short_write_count.fetch_add(1, std::memory_order_relaxed);
        short_write_bytes_total.fetch_add(remaining, std::memory_order_relaxed);
      }
      errno = saved_errno;
      return -1;
    }
    if (call_bytes_written > remaining) {
      errno = EPROTO;
      return -1;
    }
    if (call_bytes_written == 0) {
      short_write_count.fetch_add(1, std::memory_order_relaxed);
      short_write_bytes_total.fetch_add(remaining, std::memory_order_relaxed);
      errno = EIO;
      return -1;
    }

    bytes_written += call_bytes_written;
    if (call_bytes_written < remaining) {
      short_write_count.fetch_add(1, std::memory_order_relaxed);
      short_write_bytes_total.fetch_add(remaining - call_bytes_written,
                                        std::memory_order_relaxed);
    }

    if (call_bytes_written >
            static_cast<size_t>(std::numeric_limits<long>::max()) ||
        current_offset > std::numeric_limits<long>::max() -
                             static_cast<long>(call_bytes_written)) {
      errno = EOVERFLOW;
      return -1;
    }
    current_offset += static_cast<long>(call_bytes_written);
    cursor += call_bytes_written;
    remaining -= call_bytes_written;
  }

  assert(bytes_written == size &&
         "writeAtRemoteHandle must account for all requested bytes.");
  return 0;
}

int MPIIOBackend::writeAtWithContext(
    const ompfile::OmpFileWriteRequestContext &context, const void *data,
    size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  uint64_t group_key = static_cast<uint64_t>(static_cast<uint32_t>(file_id));
  if ((context.ContextFlags & ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY) != 0)
    group_key = context.PathKey;
  else {
    uint64_t path_key = 0;
    if (getFilePathKey(file_id, path_key))
      group_key = path_key;
  }
  group_key = mixWriteHintIntoKey(group_key, context.Hint);

  if (strict_mpp_init_failed)
    return failStrictMpp("pwrite");
  const bool remote_only = mpp_remote_only;
  pwrite_request_count.fetch_add(1, std::memory_order_relaxed);
  if (offset < 0) {
    errno = EINVAL;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }
  recordForecastWrite(context.Hint, size);
  invalidateTwoPhaseReadCacheForFile(file_id);

  const auto report_freshness_write_commit = [&]() {
    if ((context.Hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) == 0)
      return true;

    uint64_t path_key = 0;
    if ((context.ContextFlags & ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY) != 0)
      path_key = context.PathKey;
    else if (!getFilePathKey(file_id, path_key))
      return true;

    if (path_key == 0)
      return true;

    int writer_rank = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto owner_it = remote_file_owner_rank_map.find(file_id);
      if (owner_it != remote_file_owner_rank_map.end())
        writer_rank = owner_it->second;
    }
    if (writer_rank < 0) {
      io_log("Freshness write-commit failed: missing remote owner rank for "
             "file %d.\n",
             file_id);
      errno = ENOKEY;
      return false;
    }

    const bool write_through_mode = freshnessWriteThroughModeEnabled();
    uint64_t committed_version = 0;
    if (!ompfile::mpp::freshnessWriteCommit(
            path_key, writer_rank, context.Hint.TileId, write_through_mode,
            committed_version)) {
      io_log("Freshness write-commit failed path_key=%llu writer=%d tile=%llu "
             "mode=%s errno=%d\n",
             static_cast<unsigned long long>(path_key), writer_rank,
             static_cast<unsigned long long>(context.Hint.TileId),
             write_through_mode ? "write-through" : "write-back", errno);
      return false;
    }

    io_trace("MPIIOBackend::writeAtWithContext freshness write-commit "
             "path_key=%llu writer=%d tile=%llu mode=%s version=%llu\n",
             static_cast<unsigned long long>(path_key), writer_rank,
             static_cast<unsigned long long>(context.Hint.TileId),
             write_through_mode ? "write-through" : "write-back",
             static_cast<unsigned long long>(committed_version));
    return true;
  };

  io_log("Writing %zu bytes to file %d at offset %lld\n", size, file_id,
         static_cast<long long>(offset));

  if (isWriteBatchActive()) {
    const int batched_rc =
        writeAtBatched(file_id, offset, data, size, group_key);
    if (batched_rc == 0) {
      recordWriteEpochForContext(file_id, offset, size, context.Hint);
      triggerStageInvalidationAfterSuccessfulRemoteWrite(file_id);
      if (!report_freshness_write_commit()) {
        if (errno == 0)
          errno = EIO;
        return -1;
      }
    }
    return batched_rc;
  }

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
      errno = EBADF;
      return -1;
    }
    size_t bytes_written = 0;
    if (writeAtRemoteHandle(remote_handle, offset, data, size, bytes_written) !=
        0) {
      io_log("MPP pwrite failed for file %d (remote_handle=%d offset=%lld "
             "size=%zu errno=%d)\n",
             file_id, remote_handle, static_cast<long long>(offset), size,
             errno);
      return -1;
    }
    assert(bytes_written == size && "Remote pwrite must complete full request");

    recordWriteEpochForContext(file_id, offset, size, context.Hint);
    triggerStageInvalidationAfterSuccessfulRemoteWrite(file_id);
    if (!report_freshness_write_commit()) {
      if (errno == 0)
        errno = EIO;
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
      errno = EBADF;
      return -1;
    }
    size_t bytes_written = 0;
    if (writeAtRemoteHandle(remote_handle, offset, data, size, bytes_written) !=
        0) {
      io_log("MPP pwrite failed for file %d (remote_handle=%d offset=%lld "
             "size=%zu errno=%d)\n",
             file_id, remote_handle, static_cast<long long>(offset), size,
             errno);
      return -1;
    }
    assert(bytes_written == size && "Remote pwrite must complete full request");

    recordWriteEpochForContext(file_id, offset, size, context.Hint);
    if (!report_freshness_write_commit()) {
      if (errno == 0)
        errno = EIO;
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
      errno = EBADF;
      return -1;
    }
    file = it->second;
  }

  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    errno = EOVERFLOW;
    return -1;
  }

  // Perform the actual write at the specified offset
  MPI_Status status{};
  const int ret = MPI_File_write_at(file, offset, data, static_cast<int>(size),
                                    MPI_BYTE, &status);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Write at offset failed\n");
    return -1;
  }

  int count = 0;
  if (MPI_Get_count(&status, MPI_BYTE, &count) != MPI_SUCCESS || count < 0) {
    errno = EIO;
    return -1;
  }
  const size_t bytes_written = static_cast<size_t>(count);
  if (bytes_written > size) {
    errno = EPROTO;
    return -1;
  }
  if (bytes_written < size) {
    const size_t short_bytes = size - bytes_written;
    short_write_count.fetch_add(1, std::memory_order_relaxed);
    short_write_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
    errno = EIO;
    return -1;
  }

  recordWriteEpochForContext(file_id, offset, size, context.Hint);
  io_log("Write at offset completed\n");
  return 0;
}

int MPIIOBackend::writeAt(int file_id, long offset, const void *data,
                          size_t size) {
  ompfile::OmpFileWriteRequestContext context{};
  context.FileHandle = file_id;
  context.Offset = static_cast<int64_t>(offset);
  context.Size = static_cast<uint64_t>(size);
  uint64_t path_key = 0;
  if (getFilePathKey(file_id, path_key)) {
    context.PathKey = path_key;
    context.ContextFlags |= ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY;
  }
  return writeAtWithContext(context, data, size);
}

int MPIIOBackend::writeAtBatched(int file_id, long offset, const void *data,
                                 size_t size, uint64_t group_key) {
  WriteBatchRequest request{};
  request.DebugRequestId =
      write_batch_request_id.fetch_add(1, std::memory_order_relaxed);
  request.FileHandle = file_id;
  request.Offset = offset;
  request.GroupKey = group_key;
  request.Data.resize(size);
  if (size > 0)
    std::memcpy(request.Data.data(), data, size);

  std::vector<WriteBatchRequest *> batch;
  const auto enqueue_ts = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(write_batch_mutex);
  const size_t queue_size_before = write_batch_queue.size();
  updateAtomicMax(write_queue_max_depth,
                  static_cast<uint64_t>(queue_size_before + 1));
  write_batch_queue.push_back(&request);
  write_batch_queue_cv.notify_all();

  while (!request.Done) {
    if (!write_batch_in_progress) {
      write_batch_in_progress = true;
      write_leader_turn_count.fetch_add(1, std::memory_order_relaxed);
      if (write_batch_window_us > 0) {
        const auto window_wait_begin = std::chrono::steady_clock::now();
        write_batch_queue_cv.wait_for(
            lock, std::chrono::microseconds(write_batch_window_us));
        const auto window_wait_end = std::chrono::steady_clock::now();
        const auto window_wait_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                window_wait_end - window_wait_begin)
                .count();
        if (window_wait_us > 0) {
          write_window_wait_us_total.fetch_add(
              static_cast<uint64_t>(window_wait_us), std::memory_order_relaxed);
        }
      }

      while (!write_batch_queue.empty()) {
        batch.push_back(write_batch_queue.front());
        write_batch_queue.pop_front();
      }

      lock.unlock();
      const auto batch_exec_begin = std::chrono::steady_clock::now();
      processWriteBatch(batch);
      const auto batch_exec_end = std::chrono::steady_clock::now();
      const auto batch_exec_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              batch_exec_end - batch_exec_begin)
              .count();
      if (batch_exec_us > 0) {
        write_batch_exec_us_total.fetch_add(
            static_cast<uint64_t>(batch_exec_us), std::memory_order_relaxed);
      }
      batch.clear();
      lock.lock();

      write_batch_in_progress = false;
      write_batch_queue_cv.notify_all();
    } else {
      write_follower_wait_count.fetch_add(1, std::memory_order_relaxed);
      const auto follower_wait_begin = std::chrono::steady_clock::now();
      write_batch_queue_cv.wait(lock, [this, &request] {
        return request.Done || !write_batch_in_progress;
      });
      const auto follower_wait_end = std::chrono::steady_clock::now();
      const auto follower_wait_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              follower_wait_end - follower_wait_begin)
              .count();
      if (follower_wait_us > 0) {
        write_follower_wait_us_total.fetch_add(
            static_cast<uint64_t>(follower_wait_us), std::memory_order_relaxed);
      }
    }
  }

  const auto done_ts = std::chrono::steady_clock::now();
  const auto request_wait_us =
      std::chrono::duration_cast<std::chrono::microseconds>(done_ts -
                                                            enqueue_ts)
          .count();
  if (request_wait_us > 0) {
    write_request_wait_us_total.fetch_add(
        static_cast<uint64_t>(request_wait_us), std::memory_order_relaxed);
  }

  if (request.Status != 0) {
    errno = request.Errno;
    return -1;
  }
  return 0;
}
