#include "mpp_backend.h"
#include "ompfile_config.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

void MppBackend::recordWriteEpochForContext(
    int file_id, long offset, size_t size, const ompfile::OmpFileIOHint &hint) {
  if (size == 0)
    return;
  if (offset < 0 ||
      size > static_cast<size_t>(std::numeric_limits<long>::max()) ||
      offset > std::numeric_limits<long>::max() - static_cast<long>(size)) {
    return;
  }

  WriteEpochEntry entry{};
  entry.Start = offset;
  entry.End = offset + static_cast<long>(size);
  entry.HasEpoch = (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_EPOCH) != 0;
  entry.EpochId = entry.HasEpoch ? hint.EpochId : 0;
  entry.HasTile = (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0;
  entry.TileId = entry.HasTile ? hint.TileId : 0;

  const auto lock = instrumentedHandleLock();
  entry.Sequence = write_epochs.next_write_sequence++;
  auto &history = write_epochs.file_write_epoch_history[file_id];
  history.push_back(entry);
  constexpr size_t kMaxWriteEpochHistory = 8192;
  if (history.size() > kMaxWriteEpochHistory) {
    history.erase(history.begin(),
                  history.begin() + (history.size() - kMaxWriteEpochHistory));
  }
}

void MppBackend::noteAppliedRebalancedReadForFile(int file_id) {
  bool count_writable_read_applied = false;
  {
    const auto lock = instrumentedHandleLock();
    auto it = write_epochs.file_write_epoch_history.find(file_id);
    const bool has_local_write_history =
        it != write_epochs.file_write_epoch_history.end() &&
        !it->second.empty();
    const bool has_file_metadata =
        handles.logical_handle_set.find(file_id) !=
            handles.logical_handle_set.end() ||
        handles.file_path_map.find(file_id) != handles.file_path_map.end() ||
        handles.file_path_key_map.find(file_id) !=
            handles.file_path_key_map.end();
    count_writable_read_applied = has_file_metadata || has_local_write_history;
  }
  if (count_writable_read_applied) {
    counters.writable_read_rebalance_applied_count.fetch_add(
        1, std::memory_order_relaxed);
  }
}

void MppBackend::noteWritableReadRebalanceBlocked(const char *reason) {
  counters.writable_read_rebalance_blocked_count.fetch_add(
      1, std::memory_order_relaxed);
  if (!reason || std::strcmp(reason, "remote-only-writable-file") == 0) {
    counters.writable_read_rebalance_blocked_disabled_count.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  if (std::strcmp(reason, "missing-file-metadata") == 0 ||
      std::strcmp(reason, "missing-read-epoch") == 0 ||
      std::strcmp(reason, "missing-write-epoch") == 0) {
    counters.writable_read_rebalance_blocked_missing_metadata_count.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  if (std::strcmp(reason, "newer-write-epoch") == 0) {
    counters.writable_read_rebalance_blocked_stale_metadata_count.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  if (std::strcmp(reason, "invalid-read-range") == 0) {
    counters.writable_read_rebalance_blocked_invalid_destination_count
        .fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (std::strcmp(reason, "unsafe-write-history") == 0) {
    counters.writable_read_rebalance_blocked_pending_invalidation.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  counters.writable_read_rebalance_blocked_unsafe_overlap_count.fetch_add(
      1, std::memory_order_relaxed);
}

// Whether a planned read may be served by the aggregator instead of the
// file's owner handle. Since LIBOMPFILE_OPT_WRITABLE_READ_REBALANCE went
// (Oct 2026) a file this rank has open or has written (the remote-only
// writable context, i.e. every file libompfile opened) is never rebalanced:
// it is blocked with "remote-only-writable-file" and takes the owner path, or
// the dirty-owner routes under LIBOMPFILE_OPT_DIRTY_OWNER_FORWARDING. The
// write-epoch overlap scan that decided the opt-in case went with the knob;
// Phase E rebuilds the decision from the task graph. `hint` is kept for that
// and for the two-phase planner's call site.
bool MppBackend::canApplyRebalancedRead(const ompfile::OmpFileIOHint &hint,
                                        int file_id, long start, size_t size,
                                        const char *&reason_out,
                                        int &reason_errno_out) {
  (void)hint;
  reason_out = "none";
  reason_errno_out = 0;
  bool has_remote_only_writable_context = false;
  {
    const auto lock = instrumentedHandleLock();
    auto it = write_epochs.file_write_epoch_history.find(file_id);
    const bool has_local_write_history =
        (it != write_epochs.file_write_epoch_history.end()) &&
        !it->second.empty();
    const bool has_file_metadata =
        handles.logical_handle_set.find(file_id) !=
            handles.logical_handle_set.end() ||
        handles.file_path_map.find(file_id) != handles.file_path_map.end() ||
        handles.file_path_key_map.find(file_id) !=
            handles.file_path_key_map.end();
    has_remote_only_writable_context =
        has_file_metadata || has_local_write_history;
  }
  if (has_remote_only_writable_context) {
    counters.writable_read_rebalance_candidate_count.fetch_add(
        1, std::memory_order_relaxed);
  }

  if (start < 0 ||
      size > static_cast<size_t>(std::numeric_limits<long>::max()) ||
      start > std::numeric_limits<long>::max() - static_cast<long>(size)) {
    reason_out = "invalid-read-range";
    reason_errno_out = EINVAL;
    if (has_remote_only_writable_context)
      noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  if (has_remote_only_writable_context) {
    reason_out = "remote-only-writable-file";
    reason_errno_out = EAGAIN;
    noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  // No metadata and no write history: an empty read is trivially safe, any
  // other read names a file this backend does not know.
  if (size == 0)
    return true;
  reason_out = "missing-file-metadata";
  reason_errno_out = ENOKEY;
  return false;
}
