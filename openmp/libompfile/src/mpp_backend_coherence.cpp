#include "mpp_backend.h"
#include "ompfile_config.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

bool MppBackend::rangesOverlap(long a_start, long a_end, long b_start,
                               long b_end) const {
  return ompfile::route::rangesOverlap(a_start, a_end, b_start, b_end);
}

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

bool MppBackend::canApplyRebalancedRead(const ompfile::OmpFileIOHint &hint,
                                        int file_id, long start, size_t size,
                                        const char *&reason_out,
                                        int &reason_errno_out) {
  reason_out = "none";
  reason_errno_out = 0;
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
  const bool has_remote_only_writable_context =
      has_file_metadata || has_local_write_history;
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

  if (has_remote_only_writable_context &&
      !settings.writable_read_rebalance_enabled) {
    reason_out = "remote-only-writable-file";
    reason_errno_out = EAGAIN;
    noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  if (size == 0) {
    if (has_remote_only_writable_context) {
      counters.writable_read_rebalance_eligible_count.fetch_add(
          1, std::memory_order_relaxed);
    }
    return true;
  }

  if (has_remote_only_writable_context && has_local_write_history &&
      (config_.Shared.stageReadthrough() && config_.Shared.stageWriteBack())) {
    reason_out = "unsafe-write-history";
    reason_errno_out = EAGAIN;
    noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  const long end = start + static_cast<long>(size);
  const bool read_has_epoch =
      (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_EPOCH) != 0;
  const bool read_has_tile =
      (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0;
  const uint64_t read_epoch = hint.EpochId;

  if (it == write_epochs.file_write_epoch_history.end() || it->second.empty()) {
    if (!has_file_metadata) {
      reason_out = "missing-file-metadata";
      reason_errno_out = ENOKEY;
      if (has_remote_only_writable_context)
        noteWritableReadRebalanceBlocked(reason_out);
      return false;
    }
    if (has_remote_only_writable_context) {
      counters.writable_read_rebalance_eligible_count.fetch_add(
          1, std::memory_order_relaxed);
    }
    return true;
  }

  bool saw_overlap = false;
  bool saw_missing_write_epoch = false;
  bool saw_newer_write_epoch = false;
  // The blocking reason is precedence-ordered (missing-read-epoch >
  // missing-write-epoch > newer-write-epoch > unsafe-write-overlap), so this
  // O(n) scan under handle_mutex can stop as soon as an overlapping entry
  // fixes the highest-precedence reason no later entry could change. This
  // shortens the critical section (the epoch history is capped at 8192) for
  // the common case where the outcome is decided early.
  for (const WriteEpochEntry &entry : it->second) {
    if (!rangesOverlap(start, end, entry.Start, entry.End))
      continue;
    // Byte-range overlap is authoritative for writable remote-read rebalance.
    // Tile IDs describe the solver's logical matrix block, but the packed file
    // can still expose stale bytes through the source fd when another proxy
    // owns dirty write-back data. Do not let mismatched tile metadata make an
    // overlapping byte range eligible for rebalanced reads.
    saw_overlap = true;
    if (!read_has_epoch)
      break; // outcome fixed: "missing-read-epoch" dominates everything below.
    if (!entry.HasEpoch) {
      // "missing-write-epoch" dominates "newer-write-epoch"/"unsafe-overlap".
      saw_missing_write_epoch = true;
      break;
    }
    if (entry.EpochId > read_epoch)
      saw_newer_write_epoch = true;
  }

  if (saw_overlap) {
    if (!read_has_epoch) {
      reason_out = "missing-read-epoch";
      reason_errno_out = ENOKEY;
    } else if (saw_missing_write_epoch) {
      reason_out = "missing-write-epoch";
      reason_errno_out = ENOKEY;
    } else if (saw_newer_write_epoch) {
      reason_out = "newer-write-epoch";
      reason_errno_out = ESTALE;
    } else {
      reason_out = "unsafe-write-overlap";
      reason_errno_out = EAGAIN;
    }
    if (has_remote_only_writable_context)
      noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  if (has_remote_only_writable_context && has_local_write_history) {
    reason_out = "unsafe-write-history";
    reason_errno_out = EAGAIN;
    noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  if (has_remote_only_writable_context) {
    counters.writable_read_rebalance_eligible_count.fetch_add(
        1, std::memory_order_relaxed);
  }
  return true;
}
