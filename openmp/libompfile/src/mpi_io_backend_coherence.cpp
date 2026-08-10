#include "mpi_io_backend.h"
#include "mpp_shim.h"

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

namespace {

bool isWritebackStageModeEnv() {
  const char *StageMode = std::getenv("LIBOMPFILE_STAGE_MODE");
  const char *WriteMode = std::getenv("LIBOMPFILE_STAGE_WRITE_MODE");
  return StageMode && std::strcmp(StageMode, "readthrough") == 0 &&
         WriteMode && std::strcmp(WriteMode, "write-back") == 0;
}

} // namespace

bool MPIIOBackend::rangesOverlap(long a_start, long a_end, long b_start,
                                 long b_end) const {
  if (a_end <= a_start || b_end <= b_start)
    return false;
  return a_start < b_end && b_start < a_end;
}

void MPIIOBackend::recordWriteEpochForContext(
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
  entry.Sequence = next_write_sequence++;
  auto &history = file_write_epoch_history[file_id];
  history.push_back(entry);
  auto &coherence = writable_stage_coherence[file_id];
  ++coherence.WriteGeneration;
  coherence.InvalidationFailed = false;
  constexpr size_t kMaxWriteEpochHistory = 8192;
  if (history.size() > kMaxWriteEpochHistory) {
    history.erase(history.begin(),
                  history.begin() + (history.size() - kMaxWriteEpochHistory));
  }
}

#if defined(OMPFILE_ENABLE_TEST_ACCESS)
bool MPIIOBackend::hasCompletedStageInvalidationForTesting(int file_id) const {
  const auto lock = instrumentedHandleLock();
  auto it = writable_stage_coherence.find(file_id);
  if (it == writable_stage_coherence.end())
    return false;
  const WritableStageCoherenceState &coherence = it->second;
  return coherence.WriteGeneration != 0 && !coherence.InvalidationFailed &&
         coherence.InvalidatedGeneration >= coherence.WriteGeneration;
}
#endif // OMPFILE_ENABLE_TEST_ACCESS

bool MPIIOBackend::globallyInvalidateStageForFile(int file_id) {
  uint64_t path_key = 0;
  uint64_t generation = 0;
  std::string path;
  {
    const auto lock = instrumentedHandleLock();
    auto path_key_it = file_path_key_map.find(file_id);
    auto path_it = file_path_map.find(file_id);
    auto coherence_it = writable_stage_coherence.find(file_id);
    if (path_key_it == file_path_key_map.end() || path_key_it->second == 0 ||
        path_it == file_path_map.end() || path_it->second.empty() ||
        coherence_it == writable_stage_coherence.end() ||
        coherence_it->second.WriteGeneration == 0) {
      stage_global_invalidations_failed.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    path_key = path_key_it->second;
    path = path_it->second;
    generation = coherence_it->second.WriteGeneration;
  }

  stage_global_invalidations_requested.fetch_add(1, std::memory_order_relaxed);
  bool ok = false;
  {
    const auto lock = instrumentedMppCallLock();
    ok = ompfile::mpp::stageInvalidatePathKey(path_key, generation,
                                              path.c_str());
  }

  completeStageInvalidationForFile(file_id, generation, ok);
  return ok;
}

bool MPIIOBackend::sourceStageAffinityPreReadInvalidateForFile(int file_id) {
  uint64_t path_key = 0;
  std::string path;
  {
    const auto lock = instrumentedHandleLock();
    auto path_key_it = file_path_key_map.find(file_id);
    auto path_it = file_path_map.find(file_id);
    if (path_key_it == file_path_key_map.end() || path_key_it->second == 0 ||
        path_it == file_path_map.end() || path_it->second.empty()) {
      errno = ENOKEY;
      return false;
    }
    path_key = path_key_it->second;
    path = path_it->second;
  }

  const auto lock = instrumentedMppCallLock();
  return ompfile::mpp::stageInvalidatePathKey(path_key, /*generation=*/0,
                                              path.c_str());
}

void MPIIOBackend::completeStageInvalidationForFile(int file_id,
                                                    uint64_t generation,
                                                    bool ok) {
  const auto lock = instrumentedHandleLock();
  auto &coherence = writable_stage_coherence[file_id];
  if (!ok) {
    if (generation == coherence.WriteGeneration)
      coherence.InvalidationFailed = true;
    stage_global_invalidations_failed.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  if (generation == coherence.WriteGeneration) {
    coherence.InvalidatedGeneration = generation;
    coherence.InvalidationFailed = false;
  }
  stage_global_invalidations_completed.fetch_add(1, std::memory_order_relaxed);
}

void MPIIOBackend::triggerStageInvalidationAfterSuccessfulRemoteWrite(
    int file_id) {
  if (stage_global_invalidation_enabled && mpp_remote_only)
    (void)globallyInvalidateStageForFile(file_id);
}

void MPIIOBackend::noteAppliedRebalancedReadForFile(int file_id) {
  bool count_writable_read_applied = false;
  bool count_after_invalidation_applied = false;
  if (mpp_remote_only) {
    const auto lock = instrumentedHandleLock();
    auto it = file_write_epoch_history.find(file_id);
    const bool has_local_write_history =
        it != file_write_epoch_history.end() && !it->second.empty();
    const bool has_file_metadata =
        logical_handle_set.find(file_id) != logical_handle_set.end() ||
        file_path_map.find(file_id) != file_path_map.end() ||
        file_path_key_map.find(file_id) != file_path_key_map.end();
    count_writable_read_applied = has_file_metadata || has_local_write_history;
    auto coherence_it = writable_stage_coherence.find(file_id);
    count_after_invalidation_applied =
        has_local_write_history &&
        coherence_it != writable_stage_coherence.end() &&
        coherence_it->second.WriteGeneration != 0 &&
        !coherence_it->second.InvalidationFailed &&
        coherence_it->second.InvalidatedGeneration >=
            coherence_it->second.WriteGeneration;
  }
  if (count_writable_read_applied) {
    writable_read_rebalance_applied_count.fetch_add(1,
                                                    std::memory_order_relaxed);
  }
  if (count_after_invalidation_applied) {
    writable_read_rebalance_after_invalidation_applied.fetch_add(
        1, std::memory_order_relaxed);
  }
}

void MPIIOBackend::noteWritableReadRebalanceBlocked(const char *reason) {
  writable_read_rebalance_blocked_count.fetch_add(1, std::memory_order_relaxed);
  if (!reason || std::strcmp(reason, "remote-only-writable-file") == 0) {
    writable_read_rebalance_blocked_disabled_count.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  if (std::strcmp(reason, "missing-file-metadata") == 0 ||
      std::strcmp(reason, "missing-read-epoch") == 0 ||
      std::strcmp(reason, "missing-write-epoch") == 0) {
    writable_read_rebalance_blocked_missing_metadata_count.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  if (std::strcmp(reason, "newer-write-epoch") == 0) {
    writable_read_rebalance_blocked_stale_metadata_count.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  if (std::strcmp(reason, "invalid-read-range") == 0) {
    writable_read_rebalance_blocked_invalid_destination_count.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  if (std::strcmp(reason, "unsafe-write-history") == 0) {
    writable_read_rebalance_blocked_pending_invalidation.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }
  writable_read_rebalance_blocked_unsafe_overlap_count.fetch_add(
      1, std::memory_order_relaxed);
}

bool MPIIOBackend::canApplyRebalancedRead(const ompfile::OmpFileIOHint &hint,
                                          int file_id, long start, size_t size,
                                          const char *&reason_out,
                                          int &reason_errno_out,
                                          bool record_stats) {
  reason_out = "none";
  reason_errno_out = 0;
  const auto lock = instrumentedHandleLock();
  auto it = file_write_epoch_history.find(file_id);
  const bool has_local_write_history =
      (it != file_write_epoch_history.end()) && !it->second.empty();
  const bool has_file_metadata =
      logical_handle_set.find(file_id) != logical_handle_set.end() ||
      file_path_map.find(file_id) != file_path_map.end() ||
      file_path_key_map.find(file_id) != file_path_key_map.end();
  const bool has_remote_only_writable_context =
      mpp_remote_only && (has_file_metadata || has_local_write_history);
  if (record_stats && has_remote_only_writable_context) {
    writable_read_rebalance_candidate_count.fetch_add(
        1, std::memory_order_relaxed);
  }

  if (start < 0 ||
      size > static_cast<size_t>(std::numeric_limits<long>::max()) ||
      start > std::numeric_limits<long>::max() - static_cast<long>(size)) {
    reason_out = "invalid-read-range";
    reason_errno_out = EINVAL;
    if (record_stats && has_remote_only_writable_context)
      noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  if (has_remote_only_writable_context && !writable_read_rebalance_enabled) {
    reason_out = "remote-only-writable-file";
    reason_errno_out = EAGAIN;
    if (record_stats)
      noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  if (size == 0) {
    if (record_stats && has_remote_only_writable_context) {
      writable_read_rebalance_eligible_count.fetch_add(
          1, std::memory_order_relaxed);
    }
    return true;
  }

  if (has_remote_only_writable_context && has_local_write_history &&
      isWritebackStageModeEnv()) {
    reason_out = "unsafe-write-history";
    reason_errno_out = EAGAIN;
    if (record_stats)
      noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  const long end = start + static_cast<long>(size);
  const bool read_has_epoch =
      (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_EPOCH) != 0;
  const bool read_has_tile =
      (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0;
  const uint64_t read_epoch = hint.EpochId;

  if (it == file_write_epoch_history.end() || it->second.empty()) {
    if (!has_file_metadata) {
      reason_out = "missing-file-metadata";
      reason_errno_out = ENOKEY;
      if (record_stats && has_remote_only_writable_context)
        noteWritableReadRebalanceBlocked(reason_out);
      return false;
    }
    if (record_stats && has_remote_only_writable_context) {
      writable_read_rebalance_eligible_count.fetch_add(
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
    if (record_stats && has_remote_only_writable_context)
      noteWritableReadRebalanceBlocked(reason_out);
    return false;
  }

  if (has_remote_only_writable_context && has_local_write_history) {
    auto coherence_it = writable_stage_coherence.find(file_id);
    const bool invalidation_complete =
        coherence_it != writable_stage_coherence.end() &&
        coherence_it->second.WriteGeneration != 0 &&
        !coherence_it->second.InvalidationFailed &&
        coherence_it->second.InvalidatedGeneration >=
            coherence_it->second.WriteGeneration;
    if (!invalidation_complete) {
      reason_out = "unsafe-write-history";
      reason_errno_out = EAGAIN;
      if (record_stats)
        noteWritableReadRebalanceBlocked(reason_out);
      return false;
    }
    if (record_stats) {
      writable_read_rebalance_after_invalidation_eligible.fetch_add(
          1, std::memory_order_relaxed);
    }
  }

  if (record_stats && has_remote_only_writable_context) {
    writable_read_rebalance_eligible_count.fetch_add(1,
                                                     std::memory_order_relaxed);
  }
  return true;
}

bool MPIIOBackend::commitTileFreshnessWrite(uint64_t path_key, int writer_rank,
                                            bool write_through_mode,
                                            uint64_t &committed_version_out) {
  committed_version_out = 0;
  if (path_key == 0 || writer_rank < 0) {
    errno = EINVAL;
    return false;
  }

  const auto lock = instrumentedHandleLock();
  TileFreshnessEntry &entry = tile_freshness_table[path_key];
  assert(entry.latest_version < std::numeric_limits<uint64_t>::max());
  ++entry.latest_version;
  entry.fresh_ranks.clear();
  entry.fresh_ranks.insert(writer_rank);
  if (write_through_mode) {
    entry.pfs_version = entry.latest_version;
    entry.dirty = false;
  } else {
    entry.dirty = true;
  }
  committed_version_out = entry.latest_version;
  return true;
}

bool MPIIOBackend::markTileFresh(uint64_t path_key, int rank,
                                 uint64_t version) {
  if (path_key == 0 || rank < 0 || version == 0) {
    errno = EINVAL;
    return false;
  }

  const auto lock = instrumentedHandleLock();
  auto it = tile_freshness_table.find(path_key);
  if (it == tile_freshness_table.end()) {
    errno = ENOKEY;
    return false;
  }

  TileFreshnessEntry &entry = it->second;
  if (entry.latest_version == 0 || version != entry.latest_version) {
    errno = ESTALE;
    return false;
  }
  entry.fresh_ranks.insert(rank);
  return true;
}

bool MPIIOBackend::isTileStale(uint64_t path_key, int rank,
                               uint64_t local_version) const {
  if (path_key == 0 || rank < 0 || local_version == 0)
    return true;

  const auto lock = instrumentedHandleLock();
  auto it = tile_freshness_table.find(path_key);
  if (it == tile_freshness_table.end())
    return true;

  const TileFreshnessEntry &entry = it->second;
  if (local_version != entry.latest_version)
    return true;
  return entry.fresh_ranks.find(rank) == entry.fresh_ranks.end();
}

MPIIOBackend::TileFreshnessEntry
MPIIOBackend::tileFreshnessEntry(uint64_t path_key) const {
  if (path_key == 0)
    return {};
  const auto lock = instrumentedHandleLock();
  auto it = tile_freshness_table.find(path_key);
  if (it == tile_freshness_table.end())
    return {};
  return it->second;
}
