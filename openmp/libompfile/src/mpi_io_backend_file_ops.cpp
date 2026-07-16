#include "debug_log.h"
#include "mpi_io_backend.h"
#include "mpp_shim.h"

#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

int MPIIOBackend::open(const char *filename) {
  return openWithPlan(filename, nullptr);
}

int MPIIOBackend::openWithPlan(const char *filename,
                               const ompfile::OmpFileIOPlan *plan) {
  if (strict_mpp_init_failed)
    return failStrictMpp("open");
  int file_id = getNextFileHandle();
  const bool remote_only = mpp_remote_only;
  const bool have_open_plan = plan && plan->Status == 0 &&
                              plan->AggregatorRank >= 0;

  io_log("Opening file %s with file_id %d\n", filename, file_id);
  io_trace("MPIIOBackend::open enter this=%p file_id=%d remote_only=%d "
           "filename=%s have_plan=%d aggregator=%d remote_handle=%d\n",
           static_cast<void *>(this), file_id, static_cast<int>(remote_only),
           filename ? filename : "(null)", static_cast<int>(have_open_plan),
           have_open_plan ? plan->AggregatorRank : -1,
           have_open_plan ? plan->RemoteHandle : -1);

  auto openRemoteHandle = [&](int &remote_handle_out, int &owner_rank_out) {
    bool open_ok = false;
    owner_rank_out = -1;
    const auto lock = instrumentedMppCallLock();
    if (have_open_plan) {
      open_ok = ompfile::mpp::openOnRank(filename, open_flags, 0666,
                                         plan->AggregatorRank,
                                         remote_handle_out);
      if (open_ok)
        owner_rank_out = plan->AggregatorRank;
      if (!open_ok) {
        io_log("Planned MPP open failed for %s on rank %d; falling back to "
               "default open path.\n",
               filename, plan->AggregatorRank);
      }
    }
    if (!open_ok) {
      open_ok = ompfile::mpp::open(filename, open_flags, 0666,
                                   remote_handle_out);
      if (open_ok &&
          !ompfile::mpp::handleOwnerRank(remote_handle_out, owner_rank_out)) {
        io_log("MPP open owner-rank query failed for %s (handle=%d errno=%d)\n",
               filename ? filename : "(null)", remote_handle_out, errno);
        open_ok = false;
        if (errno == 0)
          errno = ENOKEY;
      }
    }
    return open_ok;
  };

  if (remote_only) {
    int remote_handle = -1;
    int owner_rank = -1;
    const bool open_ok = openRemoteHandle(remote_handle, owner_rank);
    if (!open_ok) {
      io_log("MPP open failed for %s\n", filename);
      if (errno == 0)
        errno = EIO;
      return -1;
    }
    {
      const auto lock = instrumentedHandleLock();
      remote_file_handle_map[file_id] = remote_handle;
      remote_file_owner_rank_map[file_id] = owner_rank;
      remote_read_handle_cache[file_id].clear();
      logical_handle_set.insert(file_id);
      traceHandleStateLocked("open.remote_only.insert", file_id, remote_handle);
    }
    io_log("Remote-only open completed for file_id %d\n", file_id);
    rememberFilePath(file_id, filename);
    rememberFilePathKey(file_id, filename);
    traceHandleState("open.remote_only.done", file_id, remote_handle);
    return file_id;
  }

  MPI_File file_handle;
  int ret = MPI_File_open(file_comm, filename, mpi_open_mode, MPI_INFO_NULL,
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
    const auto lock = instrumentedHandleLock();
    file_handle_map[file_id] = file_handle;
    remote_read_handle_cache[file_id].clear();
    logical_handle_set.insert(file_id);
    traceHandleStateLocked("open.mpi.insert", file_id, -1);
  }

  if (mpp_open_enabled) {
    int remote_handle = -1;
    int owner_rank = -1;
    const bool open_ok = openRemoteHandle(remote_handle, owner_rank);
    if (!open_ok) {
      io_log("MPP open failed for %s\n", filename);
      MPI_File_close(&file_handle);
      {
        const auto lock = instrumentedHandleLock();
        file_handle_map.erase(file_id);
        logical_handle_set.erase(file_id);
        traceHandleStateLocked("open.mpp.fail.cleanup", file_id, -1);
      }
      errno = EIO;
      return -1;
    }
    {
      const auto lock = instrumentedHandleLock();
      remote_file_handle_map[file_id] = remote_handle;
      remote_file_owner_rank_map[file_id] = owner_rank;
      traceHandleStateLocked("open.mpp.remote.insert", file_id, remote_handle);
    }
  }
  rememberFilePath(file_id, filename);
  rememberFilePathKey(file_id, filename);
  traceHandleState("open.done", file_id, -1);
  return file_id;
}

int MPIIOBackend::write(int file_id, const void *data, size_t size) {
  if (strict_mpp_init_failed)
    return failStrictMpp("write");
  if (mpp_open_enabled && mpp_io_enabled) {
    io_log("Error: write() without explicit offset is unsupported in "
           "remote-only MPP mode.\n");
    errno = ENOTSUP;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    errno = EOVERFLOW;
    return -1;
  }
  invalidateTwoPhaseReadCacheForFile(file_id);

  io_log("Writing %zu bytes to file %d\n", size, file_id);

  MPI_File file = MPI_FILE_NULL;
  {
    const auto lock = instrumentedHandleLock();
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    file = it->second;
  }

  MPI_Status status{};
  const int ret = MPI_File_write(file, data, static_cast<int>(size), MPI_BYTE,
                                 &status);

  if (ret != MPI_SUCCESS) {
    io_log("Error: Write failed\n");
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

  io_log("Write completed\n");

  return 0;
}

int MPIIOBackend::close(int file_id) {
  if (strict_mpp_init_failed)
    return failStrictMpp("close");
  io_log("Closing file %d\n", file_id);
  std::vector<int> rebalanced_handles;
  uint64_t close_path_key = 0;
  const bool remote_only = mpp_remote_only;
  int remote_handle = -1;
  bool has_remote_handle = false;
  MPI_File mpi_file = MPI_FILE_NULL;
  io_trace("MPIIOBackend::close enter this=%p file_id=%d remote_only=%d\n",
           static_cast<void *>(this), file_id, static_cast<int>(remote_only));

  {
    const auto lock = instrumentedHandleLock();
    if (logical_handle_set.find(file_id) == logical_handle_set.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      traceHandleStateLocked("close.invalid_handle", file_id, -1);
      errno = EBADF;
      return -1;
    }

    auto path_key_it = file_path_key_map.find(file_id);
    if (path_key_it != file_path_key_map.end())
      close_path_key = path_key_it->second;

    auto read_cache_it = remote_read_handle_cache.find(file_id);
    if (read_cache_it != remote_read_handle_cache.end()) {
      for (const auto &entry : read_cache_it->second)
        rebalanced_handles.push_back(entry.second);
      remote_read_handle_cache.erase(read_cache_it);
    }

    file_path_map.erase(file_id);
    file_path_key_map.erase(file_id);
    file_write_epoch_history.erase(file_id);

    if (mpp_open_enabled) {
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end()) {
        remote_handle = it->second;
        has_remote_handle = true;
      } else {
        remote_file_owner_rank_map.erase(file_id);
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
    traceHandleStateLocked("close.after_local_erase", file_id, remote_handle);
  }

  invalidateTwoPhaseReadCacheForFile(file_id);

  int mpp_ret = 0;
  if (has_remote_handle) {
    bool close_ok = false;
    {
      const auto lock = instrumentedMppCallLock();
      close_ok = ompfile::mpp::close(remote_handle);
    }
    if (!close_ok) {
      io_log("MPP close failed for file %d\n", file_id);
      mpp_ret = -1;
    } else {
      const auto lock = instrumentedHandleLock();
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end() && it->second == remote_handle)
        remote_file_handle_map.erase(it);
      remote_file_owner_rank_map.erase(file_id);
    }
  }

  for (int cached_handle : rebalanced_handles) {
    if (cached_handle < 0)
      continue;
    if (has_remote_handle && cached_handle == remote_handle)
      continue;
    bool close_ok = false;
    {
      const auto lock = instrumentedMppCallLock();
      close_ok = ompfile::mpp::close(cached_handle);
    }
    if (!close_ok) {
      io_log("MPP close failed for cached rebalanced handle (file %d, handle=%d)\n",
             file_id, cached_handle);
      mpp_ret = -1;
    }
  }

  if (remote_only) {
    if (mpp_ret != 0)
      return -1;
    if (close_path_key != 0) {
      int source_rank = -1;
      uint64_t flushed_version = 0;
      (void)ompfile::mpp::flushDirtyTile(close_path_key, source_rank,
                                         flushed_version);
    }
    io_log("Remote-only close completed\n");
    traceHandleState("close.remote_only.done", file_id, remote_handle);
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

  if (close_path_key != 0) {
    int source_rank = -1;
    uint64_t flushed_version = 0;
    (void)ompfile::mpp::flushDirtyTile(close_path_key, source_rank,
                                       flushed_version);
  }

  if (mpp_ret != 0)
    return -1;

  io_log("Close completed\n");
  traceHandleState("close.done", file_id, remote_handle);

  return 0;
}

int MPIIOBackend::read(int file_id, void *data, size_t size) {
  if (strict_mpp_init_failed)
    return failStrictMpp("read");
  if (mpp_open_enabled && mpp_io_enabled) {
    io_log("Error: read() without explicit offset is unsupported in "
           "remote-only MPP mode.\n");
    errno = ENOTSUP;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    errno = EOVERFLOW;
    return -1;
  }

  io_log("Reading %zu bytes from file %d\n", size, file_id);

  MPI_File file = MPI_FILE_NULL;
  {
    const auto lock = instrumentedHandleLock();
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    file = it->second;
  }

  MPI_Status status{};
  const int ret =
      MPI_File_read(file, data, static_cast<int>(size), MPI_BYTE, &status);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Read failed\n");
    errno = EIO;
    return -1;
  }

  int count = 0;
  if (MPI_Get_count(&status, MPI_BYTE, &count) != MPI_SUCCESS || count < 0) {
    errno = EIO;
    return -1;
  }
  const size_t bytes_read = static_cast<size_t>(count);
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

  io_log("Read completed\n");

  return 0;
}

int MPIIOBackend::seek(int file_id, long offset) {
  if (strict_mpp_init_failed)
    return failStrictMpp("seek");
  if (mpp_open_enabled && mpp_io_enabled) {
    io_log("Error: seek() is unsupported in remote-only MPP mode.\n");
    errno = ENOTSUP;
    return -1;
  }

  io_log("Setting file pointer to offset %ld\n", offset);

  MPI_File file = MPI_FILE_NULL;
  {
    const auto lock = instrumentedHandleLock();
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      errno = EBADF;
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
