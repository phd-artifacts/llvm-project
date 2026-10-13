#include "debug_log.h"
#include "mpi_io_backend.h"
#include "mpp_shim.h"

#include <cerrno>
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
  const bool have_open_plan = plan && plan->Status == 0 &&
                              plan->AggregatorRank >= 0;

  io_log("Opening file %s with file_id %d\n", filename, file_id);
  io_trace("MPIIOBackend::open enter this=%p file_id=%d "
           "filename=%s have_plan=%d aggregator=%d remote_handle=%d\n",
           static_cast<void *>(this), file_id,
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

int MPIIOBackend::write(int file_id, const void *data, size_t size) {
  if (strict_mpp_init_failed)
    return failStrictMpp("write");
  (void)file_id;
  (void)data;
  (void)size;
  io_log("Error: write() without explicit offset is unsupported in "
         "remote-only MPP mode.\n");
  errno = ENOTSUP;
  return -1;
}

int MPIIOBackend::commit(int file_id) {
  uint64_t commit_path_key = 0;
  {
    const auto lock = instrumentedHandleLock();
    if (logical_handle_set.find(file_id) == logical_handle_set.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    auto path_key_it = file_path_key_map.find(file_id);
    if (path_key_it != file_path_key_map.end())
      commit_path_key = path_key_it->second;
  }

  // No tracked path key means this handle has no staged identity, so no proxy
  // stage can be holding its bytes and there is nothing to drain.
  if (commit_path_key == 0)
    return 0;

  // Deliberately NOT the flush_dirty_tile protocol that close runs. That one
  // asks the headnode for the rank holding the freshest tile and carries a
  // freshness version, which only exists once a write has been
  // freshness-committed - so it rejects a commit issued mid-wave with EINVAL
  // before it ever looks at the stage. (Close never noticed because it
  // discards the result; the proxy's own close handler does the real flush.)
  // This broadcasts a version-free stage flush instead.
  if (!ompfile::mpp::commitStagePathKey(commit_path_key)) {
    io_log("Error: stage commit failed for file %d errno=%d path_key=%llu\n",
           file_id, errno,
           static_cast<unsigned long long>(commit_path_key));
    return -1;
  }

  io_trace("MPIIOBackend::commit done file_id=%d path_key=%llu\n", file_id,
           static_cast<unsigned long long>(commit_path_key));
  return 0;
}

int MPIIOBackend::close(int file_id) {
  if (strict_mpp_init_failed)
    return failStrictMpp("close");
  io_log("Closing file %d\n", file_id);
  std::vector<int> rebalanced_handles;
  uint64_t close_path_key = 0;
  int remote_handle = -1;
  bool has_remote_handle = false;
  io_trace("MPIIOBackend::close enter this=%p file_id=%d\n",
           static_cast<void *>(this), file_id);

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

    auto it = remote_file_handle_map.find(file_id);
    if (it != remote_file_handle_map.end()) {
      remote_handle = it->second;
      has_remote_handle = true;
    } else {
      remote_file_owner_rank_map.erase(file_id);
    }

    logical_handle_set.erase(file_id);
    traceHandleStateLocked("close.after_local_erase", file_id, remote_handle);
  }

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

int MPIIOBackend::read(int file_id, void *data, size_t size) {
  if (strict_mpp_init_failed)
    return failStrictMpp("read");
  (void)file_id;
  (void)data;
  (void)size;
  io_log("Error: read() without explicit offset is unsupported in "
         "remote-only MPP mode.\n");
  errno = ENOTSUP;
  return -1;
}

int MPIIOBackend::seek(int file_id, long offset) {
  if (strict_mpp_init_failed)
    return failStrictMpp("seek");
  (void)file_id;
  (void)offset;
  io_log("Error: seek() is unsupported in remote-only MPP mode.\n");
  errno = ENOTSUP;
  return -1;
}
