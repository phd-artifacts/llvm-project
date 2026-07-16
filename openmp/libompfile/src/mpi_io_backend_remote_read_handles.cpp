#include "mpi_io_backend.h"
#include "mpp_shim.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>

bool MPIIOBackend::getOrCreateRemoteReadHandleForRank(int file_id,
                                                      int target_rank,
                                                      int &remote_handle_out) {
  remote_handle_out = -1;
  std::string path;

  {
    const auto lock = instrumentedHandleLock();
    if (logical_handle_set.find(file_id) == logical_handle_set.end()) {
      errno = EBADF;
      return false;
    }
    auto cache_it = remote_read_handle_cache.find(file_id);
    if (cache_it != remote_read_handle_cache.end()) {
      auto rank_it = cache_it->second.find(target_rank);
      if (rank_it != cache_it->second.end()) {
        remote_handle_out = rank_it->second;
        return true;
      }
    }

    auto path_it = file_path_map.find(file_id);
    if (path_it == file_path_map.end()) {
      errno = ENOENT;
      return false;
    }
    path = path_it->second;
  }

  if (path.empty()) {
    errno = ENOENT;
    return false;
  }

  int opened_handle = -1;
  {
    const auto lock = instrumentedMppCallLock();
    const int read_flags = open_flags == O_RDONLY ? O_RDONLY : O_RDWR;
    if (!ompfile::mpp::openOnRank(path.c_str(), read_flags, 0666, target_rank,
                                  opened_handle)) {
      return false;
    }
  }

  bool must_close_opened_handle = false;
  bool closed_during_open = false;
  {
    const auto lock = instrumentedHandleLock();
    if (logical_handle_set.find(file_id) == logical_handle_set.end()) {
      must_close_opened_handle = true;
      closed_during_open = true;
      errno = EBADF;
    } else {
      auto &rank_cache = remote_read_handle_cache[file_id];
      auto [it, inserted] = rank_cache.try_emplace(target_rank, opened_handle);
      if (!inserted) {
        must_close_opened_handle = true;
        opened_handle = it->second;
      }
    }
  }

  if (must_close_opened_handle) {
    const auto lock = instrumentedMppCallLock();
    (void)ompfile::mpp::close(opened_handle);
    if (closed_during_open)
      return false;
  }

  remote_handle_out = opened_handle;
  return true;
}

void MPIIOBackend::evictAndCloseRemoteReadHandle(int file_id,
                                                 int target_rank) {
  int stale_handle = -1;
  {
    const auto lock = instrumentedHandleLock();
    auto cache_it = remote_read_handle_cache.find(file_id);
    if (cache_it != remote_read_handle_cache.end()) {
      auto rank_it = cache_it->second.find(target_rank);
      if (rank_it != cache_it->second.end()) {
        stale_handle = rank_it->second;
        cache_it->second.erase(rank_it);
      }
    }
  }
  if (stale_handle >= 0) {
    const auto lock = instrumentedMppCallLock();
    (void)ompfile::mpp::close(stale_handle);
  }
}

bool MPIIOBackend::writeAtRemoteRankWithBytes(
    int file_id, int target_rank, long offset, const void *data, size_t size,
    size_t &bytes_written) {
  bytes_written = 0;
  if (!mpp_remote_only) {
    errno = ENOTSUP;
    return false;
  }
  if (target_rank < 0 || offset < 0 || (!data && size > 0)) {
    errno = EINVAL;
    return false;
  }
  int last_errno = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt > 0)
      evictAndCloseRemoteReadHandle(file_id, target_rank);

    int remote_handle = -1;
    if (!getOrCreateRemoteReadHandleForRank(file_id, target_rank,
                                            remote_handle)) {
      last_errno = errno;
      continue;
    }

    bytes_written = 0;
    if (writeAtRemoteHandle(remote_handle, offset, data, size,
                            bytes_written) == 0)
      return true;

    last_errno = errno;
    if (last_errno != 0 && last_errno != EBADF)
      break;
  }

  errno = last_errno != 0 ? last_errno : EIO;
  return false;
}

bool MPIIOBackend::readAtRemoteRankNoStageWithBytes(
    int file_id, int target_rank, long offset, void *data, size_t size,
    size_t &bytes_read) {
  bytes_read = 0;
  if (!mpp_remote_only) {
    errno = ENOTSUP;
    return false;
  }
  if (target_rank < 0) {
    errno = EINVAL;
    return false;
  }
  int last_errno = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt > 0)
      evictAndCloseRemoteReadHandle(file_id, target_rank);

    int remote_handle = -1;
    if (!getOrCreateRemoteReadHandleForRank(file_id, target_rank,
                                            remote_handle)) {
      last_errno = errno;
      continue;
    }

    remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
    remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);

    bytes_read = 0;
    bool ok = false;
    {
      const auto lock = instrumentedMppCallLock();
      ok = ompfile::mpp::preadNoStageEx(remote_handle, offset, data, size,
                                        bytes_read);
    }
    if (ok)
      goto pread_ok;

    last_errno = errno;
    if (last_errno != 0 && last_errno != EBADF)
      break;
  }

  errno = last_errno != 0 ? last_errno : EIO;
  return false;

pread_ok:
  if (bytes_read > size) {
    errno = EPROTO;
    return false;
  }
  if (bytes_read < size && data) {
    const size_t short_bytes = size - bytes_read;
    short_read_count.fetch_add(1, std::memory_order_relaxed);
    short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
    std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
  }
  return true;
}

bool MPIIOBackend::readAtRemoteRankWithBytes(int file_id, int target_rank,
                                             long offset, void *data,
                                             size_t size, size_t &bytes_read) {
  bytes_read = 0;
  if (!mpp_remote_only) {
    errno = ENOTSUP;
    return false;
  }
  if (target_rank < 0) {
    errno = EINVAL;
    return false;
  }
  int last_errno = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt > 0)
      evictAndCloseRemoteReadHandle(file_id, target_rank);

    int remote_handle = -1;
    if (!getOrCreateRemoteReadHandleForRank(file_id, target_rank,
                                            remote_handle)) {
      last_errno = errno;
      continue;
    }

    remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
    remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);

    bytes_read = 0;
    bool ok = false;
    {
      const auto lock = instrumentedMppCallLock();
      ok = ompfile::mpp::preadEx(remote_handle, offset, data, size, bytes_read);
    }
    if (ok)
      goto pread_ok;

    last_errno = errno;
    if (last_errno != 0 && last_errno != EBADF)
      break;
  }

  errno = last_errno != 0 ? last_errno : EIO;
  return false;

pread_ok:
  if (bytes_read > size) {
    errno = EPROTO;
    return false;
  }
  if (bytes_read < size && data) {
    const size_t short_bytes = size - bytes_read;
    short_read_count.fetch_add(1, std::memory_order_relaxed);
    short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
    std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
  }
  return true;
}
