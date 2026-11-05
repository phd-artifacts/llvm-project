#include "mpp_backend.h"
#include "debug_log.h"

#include <atomic>
#include <mutex>
#include <sstream>

int MppBackend::getNextFileHandle() {
  return next_file_handle.fetch_add(1, std::memory_order_relaxed);
}

bool MppBackend::getFilePathKey(int file_id, uint64_t &path_key_out) {
  const auto lock = instrumentedHandleLock();
  const auto it = handles.file_path_key_map.find(file_id);
  if (it == handles.file_path_key_map.end())
    return false;
  path_key_out = it->second;
  return true;
}

int MppBackend::lookupRemoteHandle(int file_id) const {
  const auto lock = instrumentedHandleLock();
  auto it = remote.remote_file_handle_map.find(file_id);
  return it != remote.remote_file_handle_map.end() ? it->second : -1;
}

int MppBackend::lookupRemoteOwnerRank(int file_id) const {
  const auto lock = instrumentedHandleLock();
  auto it = remote.remote_file_owner_rank_map.find(file_id);
  return it != remote.remote_file_owner_rank_map.end() ? it->second : -1;
}

void MppBackend::rememberFilePathKey(int file_id, const char *path) {
  const uint64_t path_key = ompfile::keys::pathKey(path);
  {
    const auto lock = instrumentedHandleLock();
    handles.file_path_key_map[file_id] = path_key;
  }
}

void MppBackend::rememberFilePath(int file_id, const char *path) {
  if (!path)
    return;
  const auto lock = instrumentedHandleLock();
  handles.file_path_map[file_id] = path;
  write_epochs.file_write_epoch_history.erase(file_id);
}

void MppBackend::traceHandleStateLocked(const char *where, int file_id,
                                        int remote_handle) const {
  if (!io_trace_enabled())
    return;

  const bool logical_present =
      handles.logical_handle_set.find(file_id) !=
      handles.logical_handle_set.end();
  const auto remote_it = remote.remote_file_handle_map.find(file_id);
  const bool remote_present = remote_it != remote.remote_file_handle_map.end();
  const int remote_mapped_handle = remote_present ? remote_it->second : -1;

  std::ostringstream logical_ids;
  int logical_emitted = 0;
  for (int id : handles.logical_handle_set) {
    if (logical_emitted++ >= 8) {
      logical_ids << "...";
      break;
    }
    if (logical_emitted > 1)
      logical_ids << ",";
    logical_ids << id;
  }
  if (logical_emitted == 0)
    logical_ids << "(empty)";

  std::ostringstream remote_ids;
  int remote_emitted = 0;
  for (const auto &entry : remote.remote_file_handle_map) {
    if (remote_emitted++ >= 8) {
      remote_ids << "...";
      break;
    }
    if (remote_emitted > 1)
      remote_ids << ",";
    remote_ids << entry.first << "->" << entry.second;
  }
  if (remote_emitted == 0)
    remote_ids << "(empty)";

  io_trace("MPIIOBackend::%s this=%p file_id=%d remote_handle_arg=%d "
           "logical_present=%d remote_present=%d remote_mapped=%d "
           "logical_count=%zu remote_count=%zu "
           "logical_ids=[%s] remote_ids=[%s]\n",
           where ? where : "(unknown)", static_cast<const void *>(this), file_id,
           remote_handle, static_cast<int>(logical_present),
           static_cast<int>(remote_present), remote_mapped_handle,
           handles.logical_handle_set.size(),
           remote.remote_file_handle_map.size(), logical_ids.str().c_str(),
           remote_ids.str().c_str());
}

void MppBackend::traceHandleState(const char *where, int file_id,
                                  int remote_handle) {
  if (!io_trace_enabled())
    return;
  const auto lock = instrumentedHandleLock();
  traceHandleStateLocked(where, file_id, remote_handle);
}
