#ifndef MPI_IO_BACKEND_H
#define MPI_IO_BACKEND_H

#include "abstract_backend.h"
#include "ompfile_sched.h"
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include <cstddef> // for size_t
#include <cstdint>
#include <deque>
#include <fcntl.h>
#include <mpi.h>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class MPIIOBackend : public IOBackend {
private:
  enum class TwoPhasePolicy {
    Disabled,
    Enabled,
    Auto,
  };

  enum class ForecastMode {
    Off,
    CholeskyOracle,
  };

  enum class StageAffinityMode {
    Off,
    CholeskyOracle,
  };

  MPI_Comm file_comm = MPI_COMM_NULL;
  int externally_initialized;
  int mpi_provided_thread_level = -1;
  // When false (default) the InstrumentedLock guards are plain mutex locks
  // with no timing/counting overhead. Set from LIBOMPFILE_OPT_STATS at
  // construction so production runs pay nothing.
  bool lock_stats_enabled = false;
  std::unordered_map<int, MPI_File> file_handle_map;
  std::unordered_map<int, int> remote_file_handle_map;
  std::unordered_map<int, int> remote_file_owner_rank_map;
  std::unordered_map<int, std::unordered_map<int, int>> remote_read_handle_cache;
  std::unordered_map<int, std::string> file_path_map;
  std::unordered_map<int, uint64_t> file_path_key_map;
  struct TileFreshnessEntry {
    uint64_t latest_version = 0;
    std::set<int> fresh_ranks;
    bool dirty = false;
    uint64_t pfs_version = 0;
  };
  std::unordered_map<uint64_t, TileFreshnessEntry> tile_freshness_table;
  struct WritableStageCoherenceState {
    uint64_t WriteGeneration = 0;
    uint64_t InvalidatedGeneration = 0;
    bool InvalidationFailed = false;
  };
  struct WriteEpochEntry {
    long Start = 0;
    long End = 0;
    uint64_t EpochId = 0;
    uint64_t TileId = 0;
    uint64_t Sequence = 0;
    bool HasEpoch = false;
    bool HasTile = false;
  };
  std::unordered_map<int, std::vector<WriteEpochEntry>> file_write_epoch_history;
  struct DirtyOwnerReadCacheEntry {
    int FileId = -1;
    int SourceRank = -1;
    uint64_t Version = 0;
    long Start = 0;
    long End = 0;
    std::vector<char> Data;
  };
  std::deque<DirtyOwnerReadCacheEntry> dirty_owner_read_cache;
  std::unordered_map<int, WritableStageCoherenceState>
      writable_stage_coherence;
  uint64_t next_write_sequence = 1;
  std::unordered_set<int> logical_handle_set;
  std::atomic<int> next_file_handle{0};
  mutable std::mutex handle_mutex;
  mutable std::mutex mpp_call_mutex;
  bool mpp_open_enabled = false;
  bool mpp_io_enabled = false;
  bool mpp_requested = false;
  bool mpp_remote_only = false;
  bool mpp_init_succeeded = false;
  bool allow_fallback = false;
  bool strict_mpp_required = false;
  bool strict_mpp_init_failed = false;
  int open_flags = O_RDWR;
  int mpi_open_mode = MPI_MODE_RDWR;
  mutable std::atomic<bool> strict_failure_logged{false};
  bool writable_read_rebalance_enabled = false;
  bool dirty_owner_forwarding_enabled = false;
  bool dirty_clean_pfs_rebalance_enabled = false;
  bool dirty_clean_pfs_allow_clean_pfs_enabled = false;
  bool stage_global_invalidation_enabled = false;
  TwoPhasePolicy two_phase_policy = TwoPhasePolicy::Disabled;
  bool two_phase_enabled = false;
  TwoPhasePolicy write_batch_policy = TwoPhasePolicy::Disabled;
  bool write_batch_enabled = false;
  ForecastMode forecast_mode = ForecastMode::Off;
  StageAffinityMode stage_affinity_mode = StageAffinityMode::Off;
  static constexpr uint64_t kDefaultTwoPhaseWindowUs = 2000;
  static constexpr uint64_t kDefaultTwoPhaseMaxBatchBytes = 8 * 1024 * 1024;
  static constexpr uint64_t kDefaultTwoPhaseSieveBytes = 256 * 1024;
  static constexpr uint64_t kDefaultWriteBatchWindowUs = 2000;
  static constexpr uint64_t kDefaultWriteBatchMaxBatchBytes =
      16 * 1024 * 1024;
  uint64_t two_phase_window_us = 0;
  uint64_t two_phase_max_batch_bytes = 0;
  uint64_t two_phase_sieve_bytes = 0;
  uint64_t write_batch_window_us = 0;
  uint64_t write_batch_max_batch_bytes = 0;
  std::atomic<uint64_t> pread_request_count{0};
  std::atomic<uint64_t> remote_pread_event_count{0};
  std::atomic<uint64_t> remote_pread_bytes_total{0};
  std::atomic<uint64_t> pwrite_request_count{0};
  std::atomic<uint64_t> remote_pwrite_event_count{0};
  std::atomic<uint64_t> remote_pwrite_bytes_total{0};
  std::atomic<uint64_t> short_read_count{0};
  std::atomic<uint64_t> short_read_bytes_total{0};
  std::atomic<uint64_t> short_write_count{0};
  std::atomic<uint64_t> short_write_bytes_total{0};
  std::atomic<uint64_t> two_phase_fallback_count{0};
  std::atomic<uint64_t> two_phase_batch_count{0};
  std::atomic<uint64_t> two_phase_coalesced_read_count{0};
  std::atomic<uint64_t> two_phase_coalesced_bytes_total{0};
  std::atomic<uint64_t> two_phase_planner_batch_count{0};
  std::atomic<uint64_t> two_phase_planner_segment_count{0};
  std::atomic<uint64_t> two_phase_planner_affinity_count{0};
  std::atomic<uint64_t> two_phase_planner_rebalanced_count{0};
  std::atomic<uint64_t> two_phase_planner_rebalanced_applied_count{0};
  std::atomic<uint64_t> two_phase_planner_rebalanced_skipped_count{0};
  std::atomic<uint64_t> two_phase_planner_rebalanced_conflict_count{0};
  std::atomic<uint64_t> two_phase_planner_scalar_fallback_count{0};
  std::atomic<uint64_t> two_phase_planner_error_count{0};
  std::atomic<uint64_t> writable_read_rebalance_candidate_count{0};
  std::atomic<uint64_t> writable_read_rebalance_eligible_count{0};
  std::atomic<uint64_t> writable_read_rebalance_applied_count{0};
  std::atomic<uint64_t> writable_read_rebalance_blocked_count{0};
  std::atomic<uint64_t> writable_read_rebalance_blocked_disabled_count{0};
  std::atomic<uint64_t> writable_read_rebalance_blocked_missing_metadata_count{0};
  std::atomic<uint64_t> writable_read_rebalance_blocked_stale_metadata_count{0};
  std::atomic<uint64_t> writable_read_rebalance_blocked_unsafe_overlap_count{0};
  std::atomic<uint64_t>
      writable_read_rebalance_blocked_invalid_destination_count{0};
  std::atomic<uint64_t> dirty_owner_forward_attempt_count{0};
  std::atomic<uint64_t> dirty_owner_forward_success_count{0};
  std::atomic<uint64_t> dirty_owner_forward_failure_count{0};
  std::atomic<uint64_t> dirty_owner_forward_bytes_total{0};
  std::atomic<uint64_t> stage_global_invalidations_requested{0};
  std::atomic<uint64_t> stage_global_invalidations_completed{0};
  std::atomic<uint64_t> stage_global_invalidations_failed{0};
  std::atomic<uint64_t>
      writable_read_rebalance_blocked_pending_invalidation{0};
  std::atomic<uint64_t>
      writable_read_rebalance_after_invalidation_eligible{0};
  std::atomic<uint64_t>
      writable_read_rebalance_after_invalidation_applied{0};
  std::atomic<uint64_t> two_phase_cache_hit_count{0};
  std::atomic<uint64_t> two_phase_planner_batch_id{1};
  std::atomic<uint64_t> two_phase_request_id{1};
  std::atomic<uint64_t> write_batch_count{0};
  std::atomic<uint64_t> write_batch_coalesced_write_count{0};
  std::atomic<uint64_t> write_batch_coalesced_bytes_total{0};
  std::atomic<uint64_t> write_batch_saved_events{0};
  std::atomic<uint64_t> write_batch_failure_count{0};
  std::atomic<uint64_t> two_phase_queue_max_depth{0};
  std::atomic<uint64_t> two_phase_leader_turn_count{0};
  std::atomic<uint64_t> two_phase_follower_wait_count{0};
  std::atomic<uint64_t> two_phase_follower_wait_us_total{0};
  std::atomic<uint64_t> two_phase_window_wait_us_total{0};
  std::atomic<uint64_t> two_phase_batch_exec_us_total{0};
  std::atomic<uint64_t> two_phase_request_wait_us_total{0};
  std::atomic<uint64_t> write_queue_max_depth{0};
  std::atomic<uint64_t> write_leader_turn_count{0};
  std::atomic<uint64_t> write_follower_wait_count{0};
  std::atomic<uint64_t> write_follower_wait_us_total{0};
  std::atomic<uint64_t> write_window_wait_us_total{0};
  std::atomic<uint64_t> write_batch_exec_us_total{0};
  std::atomic<uint64_t> write_request_wait_us_total{0};
  std::atomic<uint64_t> write_batch_request_id{1};
  std::atomic<uint64_t> forecast_hint_read_count{0};
  std::atomic<uint64_t> forecast_hint_write_count{0};
  std::atomic<uint64_t> forecast_hint_read_bytes_total{0};
  std::atomic<uint64_t> forecast_hint_write_bytes_total{0};
  std::atomic<uint64_t> forecast_source_read_stageable_count{0};
  std::atomic<uint64_t> forecast_source_read_stageable_bytes_total{0};
  std::atomic<uint64_t> forecast_target_read_bypass_count{0};
  std::atomic<uint64_t> forecast_target_read_bypass_bytes_total{0};
  std::atomic<uint64_t> forecast_role_unknown_count{0};
  std::atomic<uint64_t> forecast_role_unknown_bytes_total{0};
  std::atomic<uint64_t> stage_affinity_source_candidate_count{0};
  std::atomic<uint64_t> stage_affinity_source_candidate_bytes{0};
  std::atomic<uint64_t> stage_affinity_source_applied_count{0};
  std::atomic<uint64_t> stage_affinity_source_applied_bytes{0};
  std::atomic<uint64_t> stage_affinity_source_fallback_count{0};
  std::atomic<uint64_t> dirty_owner_batch_candidate_count{0};
  std::atomic<uint64_t> dirty_owner_batch_count{0};
  std::atomic<uint64_t> dirty_owner_single_read_count{0};
  std::atomic<uint64_t> dirty_owner_remote_read_event_count{0};
  std::atomic<uint64_t> dirty_owner_remote_read_bytes_total{0};
  std::atomic<uint64_t> dirty_owner_batch_segment_count{0};
  std::atomic<uint64_t> dirty_owner_batch_saved_events{0};
  std::atomic<uint64_t> dirty_owner_batch_bytes{0};
  std::atomic<uint64_t> dirty_owner_batch_failure_count{0};
  std::atomic<uint64_t> dirty_owner_queue_max_depth{0};
  std::atomic<uint64_t> dirty_owner_follower_wait_count{0};
  std::atomic<uint64_t> dirty_owner_follower_wait_us_total{0};
  std::atomic<uint64_t> dirty_owner_batch_window_wait_us_total{0};
  // Lock-contention instrumentation for the code-review Phase 3 (C2/C3/P1)
  // decision. Sampled at the hottest per-request acquisition sites via
  // instrumentedLock() so the "contended" ratio faithfully reflects whether
  // a lock is usually free or usually held under the Cholesky workload;
  // hold-time is a lower bound (only instrumented sites contribute). See
  // reportPhase0Stats for how these surface.
  // mutable so the instrumented lock helpers can be const (some const
  // methods, e.g. tileFreshnessEntry, acquire handle_mutex).
  mutable std::atomic<uint64_t> handle_mutex_acquire_count{0};
  mutable std::atomic<uint64_t> handle_mutex_contended_count{0};
  mutable std::atomic<uint64_t> handle_mutex_wait_us_total{0};
  mutable std::atomic<uint64_t> handle_mutex_hold_us_total{0};
  mutable std::atomic<uint64_t> mpp_call_mutex_acquire_count{0};
  mutable std::atomic<uint64_t> mpp_call_mutex_contended_count{0};
  mutable std::atomic<uint64_t> mpp_call_mutex_wait_us_total{0};
  mutable std::atomic<uint64_t> mpp_call_mutex_hold_us_total{0};

  // RAII lock guard that optionally records contention/wait/hold-time into
  // the counters above. When instrumentation is disabled (the default, unless
  // LIBOMPFILE_OPT_STATS=1) it is a plain lock_guard-equivalent with zero
  // extra overhead on the hot path. When enabled it try_lock()s first: if
  // that succeeds the lock was uncontended (no waiting); if it fails another
  // thread held it, so we count a contended acquisition and time the blocking
  // lock(). All ~68 backend acquisition sites route through this guard, so the
  // contended ratio and hold time are complete (not sampled) when enabled.
  class InstrumentedLock {
  public:
    InstrumentedLock(std::mutex &m, bool enabled,
                     std::atomic<uint64_t> &acquires,
                     std::atomic<uint64_t> &contended,
                     std::atomic<uint64_t> &wait_us,
                     std::atomic<uint64_t> &hold_us)
        : mutex_(m), enabled_(enabled), hold_us_(hold_us) {
      if (!enabled_) {
        mutex_.lock();
        return;
      }
      acquires.fetch_add(1, std::memory_order_relaxed);
      if (!mutex_.try_lock()) {
        contended.fetch_add(1, std::memory_order_relaxed);
        const auto wait_start = std::chrono::steady_clock::now();
        mutex_.lock();
        wait_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wait_start)
                .count(),
            std::memory_order_relaxed);
      }
      hold_start_ = std::chrono::steady_clock::now();
    }
    ~InstrumentedLock() {
      if (enabled_)
        hold_us_.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - hold_start_)
                .count(),
            std::memory_order_relaxed);
      mutex_.unlock();
    }
    InstrumentedLock(const InstrumentedLock &) = delete;
    InstrumentedLock &operator=(const InstrumentedLock &) = delete;

  private:
    std::mutex &mutex_;
    bool enabled_;
    std::atomic<uint64_t> &hold_us_;
    std::chrono::steady_clock::time_point hold_start_;
  };

  InstrumentedLock instrumentedHandleLock() const {
    return InstrumentedLock(handle_mutex, lock_stats_enabled,
                            handle_mutex_acquire_count,
                            handle_mutex_contended_count,
                            handle_mutex_wait_us_total,
                            handle_mutex_hold_us_total);
  }
  InstrumentedLock instrumentedMppCallLock() const {
    return InstrumentedLock(mpp_call_mutex, lock_stats_enabled,
                            mpp_call_mutex_acquire_count,
                            mpp_call_mutex_contended_count,
                            mpp_call_mutex_wait_us_total,
                            mpp_call_mutex_hold_us_total);
  }

  bool two_phase_batch_in_progress = false;
  bool write_batch_in_progress = false;
  bool dirty_owner_batch_in_progress = false;

  struct TwoPhaseReadRequest {
    uint64_t DebugRequestId = 0;
    int FileHandle = -1;
    int ClientRank = -1;
    long Offset = 0;
    size_t Size = 0;
    void *Buffer = nullptr;
    uint64_t PathKey = 0;
    bool HasPathKey = false;
    ompfile::OmpFileIOHint Hint{};

    int Status = 0;
    int Errno = 0;
    bool Done = false;
  };

  std::mutex two_phase_mutex;
  std::condition_variable two_phase_queue_cv;
  std::deque<TwoPhaseReadRequest *> two_phase_queue;

  struct WriteBatchRequest {
    uint64_t DebugRequestId = 0;
    int FileHandle = -1;
    long Offset = 0;
    uint64_t GroupKey = 0;
    std::vector<char> Data;
    int Status = 0;
    int Errno = 0;
    bool Done = false;
  };

  std::mutex write_batch_mutex;
  std::condition_variable write_batch_queue_cv;
  std::deque<WriteBatchRequest *> write_batch_queue;

  struct DirtyOwnerBatchRequest {
    int FileId = -1;
    int RemoteHandle = -1;
    int SourceRank = -1;
    uint64_t ExpectedVersion = 0;
    long Offset = 0;
    size_t Size = 0;
    void *Buffer = nullptr;
    uint64_t DebugRequestId = 0;
    size_t BytesRead = 0;
    int Status = 0;
    int Errno = 0;
    bool Done = false;
  };

  std::mutex dirty_owner_batch_mutex;
  std::condition_variable dirty_owner_batch_cv;
  std::deque<DirtyOwnerBatchRequest *> dirty_owner_batch_queue;
  struct TwoPhaseReadCacheEntry {
    long Start = 0;
    std::vector<char> Data;
  };
  std::unordered_map<uint64_t, TwoPhaseReadCacheEntry> two_phase_read_cache;

public:
  // Shared with file_interface.cpp's free-function env parsing so every
  // LIBOMPFILE_OPT_*-style boolean knob (including LIBOMPFILE_MPP_OPEN and
  // LIBOMPFILE_MPP_IO) rejects malformed values the same way instead of
  // silently falling back with no diagnostic.
  static bool parseBoolEnv(const char *name, bool default_value);

  MPIIOBackend();
  ~MPIIOBackend();

  void recordWriteEpochForTesting(int file_id, long offset, size_t size,
                                  const ompfile::OmpFileIOHint &hint) {
    recordWriteEpochForContext(file_id, offset, size, hint);
  }
  bool canApplyRebalancedReadForTesting(const ompfile::OmpFileIOHint &hint,
                                        int file_id, long start, size_t size,
                                        const char *&reason_out,
                                        int &reason_errno_out) {
    return canApplyRebalancedRead(hint, file_id, start, size, reason_out,
                                  reason_errno_out);
  }
  void rememberWritableFileContextForTesting(int file_id, const char *path) {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    logical_handle_set.insert(file_id);
    if (path && path[0] != '\0') {
      file_path_map[file_id] = path;
      file_path_key_map[file_id] = computePathKey(path);
    }
  }
  static bool plannerStatusForcesFallbackForTesting(
      int planner_errno, const char *&reason_out, int &reason_errno_out) {
    return plannerStatusForcesFallback(planner_errno, reason_out,
                                       reason_errno_out);
  }
  static size_t computeTwoPhaseTargetReadSizeForTesting(
      long start, long end, size_t request_count, bool remote_only,
      uint64_t sieve_bytes, uint64_t max_batch_bytes) {
    return computeTwoPhaseTargetReadSize(start, end, request_count, remote_only,
                                         sieve_bytes, max_batch_bytes);
  }
  size_t computeForecastTargetReadSizeForTesting(
      const ompfile::OmpFileIOHint &hint, long start, long end,
      size_t request_count, bool remote_only, uint64_t sieve_bytes,
      uint64_t max_batch_bytes) const {
    return computeForecastTargetReadSize(hint, start, end, request_count,
                                         remote_only, sieve_bytes,
                                         max_batch_bytes);
  }
  struct PlannerCounterSnapshotForTesting {
    uint64_t Rebalanced = 0;
    uint64_t Applied = 0;
    uint64_t Skipped = 0;
    uint64_t Conflicts = 0;
    uint64_t Errors = 0;
  };
  struct WritableReadRebalanceCountersForTesting {
    uint64_t Candidates = 0;
    uint64_t Eligible = 0;
    uint64_t Applied = 0;
    uint64_t Blocked = 0;
    uint64_t BlockedDisabled = 0;
    uint64_t BlockedMissingMetadata = 0;
    uint64_t BlockedStaleMetadata = 0;
    uint64_t BlockedUnsafeOverlap = 0;
    uint64_t BlockedInvalidDestination = 0;
    uint64_t BlockedPendingInvalidation = 0;
    uint64_t AfterInvalidationEligible = 0;
    uint64_t AfterInvalidationApplied = 0;
    uint64_t StageInvalidationsRequested = 0;
    uint64_t StageInvalidationsCompleted = 0;
    uint64_t StageInvalidationsFailed = 0;
  };
  bool globallyInvalidateStageForTesting(int file_id) {
    return globallyInvalidateStageForFile(file_id);
  }
  bool sourceStageAffinityPreReadInvalidateForTesting(int file_id) {
    return sourceStageAffinityPreReadInvalidateForFile(file_id);
  }
  bool hasCompletedStageInvalidationForTesting(int file_id) const;
  void markStageInvalidationCompleteForTesting(int file_id) {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto &coherence = writable_stage_coherence[file_id];
    coherence.InvalidatedGeneration = coherence.WriteGeneration;
    coherence.InvalidationFailed = false;
  }
  uint64_t writableStageWriteGenerationForTesting(int file_id) const {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = writable_stage_coherence.find(file_id);
    return it == writable_stage_coherence.end() ? 0 : it->second.WriteGeneration;
  }
  void completeStageInvalidationForTesting(int file_id, uint64_t generation,
                                           bool ok) {
    completeStageInvalidationForFile(file_id, generation, ok);
  }
  void noteAppliedRebalancedWritableReadForTesting(int file_id) {
    noteAppliedRebalancedReadForFile(file_id);
  }
  void triggerPostSuccessfulRemoteWriteInvalidationForTesting(int file_id) {
    triggerStageInvalidationAfterSuccessfulRemoteWrite(file_id);
  }
  void addPlannerCountersForTesting(uint64_t rebalanced, uint64_t applied,
                                    uint64_t skipped, uint64_t conflicts,
                                    uint64_t errors) {
    two_phase_planner_rebalanced_count.fetch_add(rebalanced,
                                                 std::memory_order_relaxed);
    two_phase_planner_rebalanced_applied_count.fetch_add(
        applied, std::memory_order_relaxed);
    two_phase_planner_rebalanced_skipped_count.fetch_add(
        skipped, std::memory_order_relaxed);
    two_phase_planner_rebalanced_conflict_count.fetch_add(
        conflicts, std::memory_order_relaxed);
    two_phase_planner_error_count.fetch_add(errors, std::memory_order_relaxed);
  }
  PlannerCounterSnapshotForTesting plannerCountersForTesting() const {
    return {two_phase_planner_rebalanced_count.load(std::memory_order_relaxed),
            two_phase_planner_rebalanced_applied_count.load(
                std::memory_order_relaxed),
            two_phase_planner_rebalanced_skipped_count.load(
                std::memory_order_relaxed),
            two_phase_planner_rebalanced_conflict_count.load(
                std::memory_order_relaxed),
            two_phase_planner_error_count.load(std::memory_order_relaxed)};
  }
  WritableReadRebalanceCountersForTesting
  writableReadRebalanceCountersForTesting() const {
    return {writable_read_rebalance_candidate_count.load(
                std::memory_order_relaxed),
            writable_read_rebalance_eligible_count.load(
                std::memory_order_relaxed),
            writable_read_rebalance_applied_count.load(
                std::memory_order_relaxed),
            writable_read_rebalance_blocked_count.load(
                std::memory_order_relaxed),
            writable_read_rebalance_blocked_disabled_count.load(
                std::memory_order_relaxed),
            writable_read_rebalance_blocked_missing_metadata_count.load(
                std::memory_order_relaxed),
            writable_read_rebalance_blocked_stale_metadata_count.load(
                std::memory_order_relaxed),
            writable_read_rebalance_blocked_unsafe_overlap_count.load(
                std::memory_order_relaxed),
            writable_read_rebalance_blocked_invalid_destination_count.load(
                std::memory_order_relaxed),
            writable_read_rebalance_blocked_pending_invalidation.load(
                std::memory_order_relaxed),
            writable_read_rebalance_after_invalidation_eligible.load(
                std::memory_order_relaxed),
            writable_read_rebalance_after_invalidation_applied.load(
                std::memory_order_relaxed),
            stage_global_invalidations_requested.load(
                std::memory_order_relaxed),
            stage_global_invalidations_completed.load(
                std::memory_order_relaxed),
            stage_global_invalidations_failed.load(std::memory_order_relaxed)};
  }
  struct QueueCoordinationCountersForTesting {
    uint64_t TwoPhaseQueueMaxDepth = 0;
    uint64_t TwoPhaseLeaderTurns = 0;
    uint64_t TwoPhaseFollowerWaitCount = 0;
    uint64_t TwoPhaseFollowerWaitUsTotal = 0;
    uint64_t TwoPhaseWindowWaitUsTotal = 0;
    uint64_t TwoPhaseBatchExecUsTotal = 0;
    uint64_t TwoPhaseRequestWaitUsTotal = 0;
    uint64_t WriteQueueMaxDepth = 0;
    uint64_t WriteLeaderTurns = 0;
    uint64_t WriteFollowerWaitCount = 0;
    uint64_t WriteFollowerWaitUsTotal = 0;
    uint64_t WriteWindowWaitUsTotal = 0;
    uint64_t WriteBatchExecUsTotal = 0;
    uint64_t WriteRequestWaitUsTotal = 0;
  };
  void addQueueCoordinationCountersForTesting(
      const QueueCoordinationCountersForTesting &delta) {
    two_phase_queue_max_depth.store(delta.TwoPhaseQueueMaxDepth,
                                    std::memory_order_relaxed);
    two_phase_leader_turn_count.fetch_add(delta.TwoPhaseLeaderTurns,
                                          std::memory_order_relaxed);
    two_phase_follower_wait_count.fetch_add(delta.TwoPhaseFollowerWaitCount,
                                            std::memory_order_relaxed);
    two_phase_follower_wait_us_total.fetch_add(delta.TwoPhaseFollowerWaitUsTotal,
                                               std::memory_order_relaxed);
    two_phase_window_wait_us_total.fetch_add(delta.TwoPhaseWindowWaitUsTotal,
                                             std::memory_order_relaxed);
    two_phase_batch_exec_us_total.fetch_add(delta.TwoPhaseBatchExecUsTotal,
                                            std::memory_order_relaxed);
    two_phase_request_wait_us_total.fetch_add(delta.TwoPhaseRequestWaitUsTotal,
                                              std::memory_order_relaxed);
    write_queue_max_depth.store(delta.WriteQueueMaxDepth,
                                std::memory_order_relaxed);
    write_leader_turn_count.fetch_add(delta.WriteLeaderTurns,
                                      std::memory_order_relaxed);
    write_follower_wait_count.fetch_add(delta.WriteFollowerWaitCount,
                                        std::memory_order_relaxed);
    write_follower_wait_us_total.fetch_add(delta.WriteFollowerWaitUsTotal,
                                           std::memory_order_relaxed);
    write_window_wait_us_total.fetch_add(delta.WriteWindowWaitUsTotal,
                                         std::memory_order_relaxed);
    write_batch_exec_us_total.fetch_add(delta.WriteBatchExecUsTotal,
                                        std::memory_order_relaxed);
    write_request_wait_us_total.fetch_add(delta.WriteRequestWaitUsTotal,
                                          std::memory_order_relaxed);
  }
  QueueCoordinationCountersForTesting
  queueCoordinationCountersForTesting() const {
    return {two_phase_queue_max_depth.load(std::memory_order_relaxed),
            two_phase_leader_turn_count.load(std::memory_order_relaxed),
            two_phase_follower_wait_count.load(std::memory_order_relaxed),
            two_phase_follower_wait_us_total.load(std::memory_order_relaxed),
            two_phase_window_wait_us_total.load(std::memory_order_relaxed),
            two_phase_batch_exec_us_total.load(std::memory_order_relaxed),
            two_phase_request_wait_us_total.load(std::memory_order_relaxed),
            write_queue_max_depth.load(std::memory_order_relaxed),
            write_leader_turn_count.load(std::memory_order_relaxed),
            write_follower_wait_count.load(std::memory_order_relaxed),
            write_follower_wait_us_total.load(std::memory_order_relaxed),
            write_window_wait_us_total.load(std::memory_order_relaxed),
            write_batch_exec_us_total.load(std::memory_order_relaxed),
            write_request_wait_us_total.load(std::memory_order_relaxed)};
  }
  struct ForecastCountersForTesting {
    uint64_t HintReads = 0;
    uint64_t HintWrites = 0;
    uint64_t HintReadBytes = 0;
    uint64_t HintWriteBytes = 0;
    uint64_t SourceReadStageable = 0;
    uint64_t SourceReadStageableBytes = 0;
    uint64_t TargetReadBypasses = 0;
    uint64_t TargetReadBypassBytes = 0;
    uint64_t RoleUnknown = 0;
    uint64_t RoleUnknownBytes = 0;
  };
  void recordForecastReadForTesting(const ompfile::OmpFileIOHint &hint,
                                    size_t size) {
    recordForecastRead(hint, size);
  }
  void recordForecastWriteForTesting(const ompfile::OmpFileIOHint &hint,
                                     size_t size) {
    recordForecastWrite(hint, size);
  }
  bool shouldBypassStageForForecastReadForTesting(
      const ompfile::OmpFileIOHint &hint) const {
    return shouldBypassStageForForecastRead(hint);
  }
  bool shouldUseNoStageFallbackForTesting(
      const ompfile::OmpFileIOHint &hint, bool force_no_stage) const {
    return shouldUseNoStageFallback(hint, force_no_stage);
  }
  bool shouldAttemptPlannerRebalancedReadForTesting(
      int rc, bool force_no_stage_fallback, bool planner_rebalanced,
      int planned_rank) const {
    return shouldAttemptPlannerRebalancedRead(
        rc, force_no_stage_fallback, planner_rebalanced, planned_rank);
  }
  bool isSourceStageAffinityEligibleForTesting(
      const ompfile::OmpFileIOHint &hint, bool has_path_key,
      bool remote_only) const {
    return isSourceStageAffinityEligible(hint, has_path_key, remote_only);
  }
  bool canAttemptSourceStageAffinityReadForTesting(
      const ompfile::OmpFileIOHint &hint, int file_id, long start, size_t size,
      bool has_path_key, bool remote_only, bool planner_force_fallback) {
    return canAttemptSourceStageAffinityRead(
        hint, file_id, start, size, has_path_key, remote_only,
        planner_force_fallback);
  }
  bool selectSourceStageAffinityRankForTesting(
      uint64_t path_key, const ompfile::OmpFileIOHint &hint,
      const std::vector<int> &worker_ranks, int &rank_out) const {
    return selectSourceStageAffinityRank(path_key, hint, worker_ranks,
                                          rank_out);
  }
  static std::vector<int> workerRanksForTesting(int world_size,
                                                int headnode_rank) {
    return computeWorkerRanks(world_size, headnode_rank);
  }
  static uint64_t computePathKeyForTesting(const char *path) {
    return computePathKey(path);
  }
  bool commitTileFreshnessWriteForTesting(uint64_t path_key, int writer_rank,
                                          bool write_through_mode,
                                          uint64_t &committed_version_out) {
    return commitTileFreshnessWrite(path_key, writer_rank, write_through_mode,
                                    committed_version_out);
  }
  bool markTileFreshForTesting(uint64_t path_key, int rank,
                               uint64_t version) {
    return markTileFresh(path_key, rank, version);
  }
  bool isTileStaleForTesting(uint64_t path_key, int rank,
                             uint64_t local_version) const {
    return isTileStale(path_key, rank, local_version);
  }
  TileFreshnessEntry tileFreshnessEntryForTesting(uint64_t path_key) const {
    return tileFreshnessEntry(path_key);
  }
  ForecastCountersForTesting forecastCountersForTesting() const {
    return {forecast_hint_read_count.load(std::memory_order_relaxed),
            forecast_hint_write_count.load(std::memory_order_relaxed),
            forecast_hint_read_bytes_total.load(std::memory_order_relaxed),
            forecast_hint_write_bytes_total.load(std::memory_order_relaxed),
            forecast_source_read_stageable_count.load(
                std::memory_order_relaxed),
            forecast_source_read_stageable_bytes_total.load(
                std::memory_order_relaxed),
            forecast_target_read_bypass_count.load(std::memory_order_relaxed),
            forecast_target_read_bypass_bytes_total.load(
                std::memory_order_relaxed),
            forecast_role_unknown_count.load(std::memory_order_relaxed),
            forecast_role_unknown_bytes_total.load(
                std::memory_order_relaxed)};
  }
  struct StageAffinityCountersForTesting {
    uint64_t Candidates = 0;
    uint64_t CandidateBytes = 0;
    uint64_t Applied = 0;
    uint64_t AppliedBytes = 0;
    uint64_t Fallbacks = 0;
  };
  void addStageAffinityCountersForTesting(uint64_t candidates,
                                          uint64_t candidate_bytes,
                                          uint64_t applied,
                                          uint64_t applied_bytes,
                                          uint64_t fallbacks) {
    stage_affinity_source_candidate_count.fetch_add(
        candidates, std::memory_order_relaxed);
    stage_affinity_source_candidate_bytes.fetch_add(
        candidate_bytes, std::memory_order_relaxed);
    stage_affinity_source_applied_count.fetch_add(applied,
                                                  std::memory_order_relaxed);
    stage_affinity_source_applied_bytes.fetch_add(
        applied_bytes, std::memory_order_relaxed);
    stage_affinity_source_fallback_count.fetch_add(
        fallbacks, std::memory_order_relaxed);
  }
  StageAffinityCountersForTesting stageAffinityCountersForTesting() const {
    return {stage_affinity_source_candidate_count.load(
                std::memory_order_relaxed),
            stage_affinity_source_candidate_bytes.load(
                std::memory_order_relaxed),
            stage_affinity_source_applied_count.load(
                std::memory_order_relaxed),
            stage_affinity_source_applied_bytes.load(
                std::memory_order_relaxed),
            stage_affinity_source_fallback_count.load(
                std::memory_order_relaxed)};
  }

  int open(const char *filename) override;
  int openWithPlan(const char *filename,
                   const ompfile::OmpFileIOPlan *plan) override;
  int write(int file_id, const void *data, size_t size) override;
  int read(int file_id, void *data, size_t size) override;
  int close(int file_id) override;
  int seek(int file_id, long offset) override;
  int readAt(int file_id, long offset, void *data, size_t size) override;
  int readAtWithContext(const ompfile::OmpFileReadRequestContext &context,
                        void *data, size_t size) override;
  int writeAtWithContext(const ompfile::OmpFileWriteRequestContext &context,
                         const void *data, size_t size) override;
  int writeAt(int file_id, long offset, const void *data, size_t size) override;

private:
  int writeAtRemoteHandle(int remote_handle, long offset, const void *data,
                          size_t size, size_t &bytes_written);
  int writeAtBatched(int file_id, long offset, const void *data, size_t size,
                     uint64_t group_key);
  int readAtFallback(int file_id, long offset, void *data, size_t size);
  int readAtFallbackWithBytes(int file_id, long offset, void *data, size_t size,
                               size_t &bytes_read,
                               const ompfile::OmpFileIOHint *hint = nullptr,
                               bool force_no_stage = false);
  int readAtTwoPhase(const ompfile::OmpFileReadRequestContext &context,
                     void *data, size_t size);
  void processTwoPhaseBatch(std::vector<TwoPhaseReadRequest *> &batch);
  void processTwoPhaseGroup(std::vector<TwoPhaseReadRequest *> &group);
  void processWriteBatch(std::vector<WriteBatchRequest *> &batch);
  void processWriteGroup(std::vector<WriteBatchRequest *> &group);
  uint64_t getTwoPhaseGroupKey(const TwoPhaseReadRequest &request) const;
  uint64_t getWriteBatchGroupKey(const WriteBatchRequest &request) const;
  void completeTwoPhaseRequest(TwoPhaseReadRequest &request, int status,
                               int errnum);
  void completeWriteBatchRequest(WriteBatchRequest &request, int status,
                                 int errnum);
  bool isTwoPhaseActive() const;
  bool isWriteBatchActive() const;
  bool hasUsablePlannedRead(const ompfile::OmpFileReadRequestContext &context) const;
  int getNextFileHandle();
  static TwoPhasePolicy parseTwoPhasePolicy(const char *env_value);
  static ForecastMode parseForecastMode(const char *env_value);
  static StageAffinityMode parseStageAffinityMode(const char *env_value);
  static const char *twoPhasePolicyToString(TwoPhasePolicy policy);
  static const char *forecastModeToString(ForecastMode mode);
  static const char *stageAffinityModeToString(StageAffinityMode mode);
  static uint64_t parseUint64Env(const char *name, uint64_t default_value);
  static void updateAtomicMax(std::atomic<uint64_t> &counter,
                              uint64_t candidate);
  static size_t computeTwoPhaseTargetReadSize(long start, long end,
                                               size_t request_count,
                                               bool remote_only,
                                               uint64_t sieve_bytes,
                                               uint64_t max_batch_bytes);
  size_t computeForecastTargetReadSize(const ompfile::OmpFileIOHint &hint,
                                       long start, long end,
                                       size_t request_count, bool remote_only,
                                       uint64_t sieve_bytes,
                                       uint64_t max_batch_bytes) const;
  static uint64_t computePathKey(const char *path);
  static uint64_t mixHintIntoKey(uint64_t base_key,
                                 const ompfile::OmpFileIOHint &hint);
  static uint64_t mixWriteHintIntoKey(uint64_t base_key,
                                      const ompfile::OmpFileIOHint &hint);
  static bool shouldReportStats();
  bool getFilePathKey(int file_id, uint64_t &path_key_out);
  bool isForecastHintValid(const ompfile::OmpFileIOHint &hint) const;
  bool isForecastRoleHintValid(const ompfile::OmpFileIOHint &hint) const;
  bool shouldBypassStageForForecastRead(
      const ompfile::OmpFileIOHint &hint) const;
  bool shouldUseNoStageFallback(const ompfile::OmpFileIOHint &hint,
                                bool force_no_stage) const;
  bool shouldAttemptPlannerRebalancedRead(int rc, bool force_no_stage_fallback,
                                          bool planner_rebalanced,
                                          int planned_rank) const;
  bool isSourceStageAffinityEligible(const ompfile::OmpFileIOHint &hint,
                                     bool has_path_key,
                                     bool remote_only) const;
  bool canAttemptSourceStageAffinityRead(
      const ompfile::OmpFileIOHint &hint, int file_id, long start, size_t size,
      bool has_path_key, bool remote_only, bool planner_force_fallback);
  static uint64_t computeSourceStageAffinityKey(
      uint64_t path_key, const ompfile::OmpFileIOHint &hint);
  static std::vector<int> computeWorkerRanks(int world_size, int headnode_rank);
  // worker_ranks must be the canonical ordered worker list.
  static bool selectSourceStageAffinityRank(
      uint64_t path_key, const ompfile::OmpFileIOHint &hint,
      const std::vector<int> &worker_ranks, int &rank_out);
  void recordForecastRead(const ompfile::OmpFileIOHint &hint, size_t size);
  void recordForecastWrite(const ompfile::OmpFileIOHint &hint, size_t size);
  void rememberFilePathKey(int file_id, const char *path);
  void rememberFilePath(int file_id, const char *path);
  void forgetFilePath(int file_id);
  void forgetFilePathKey(int file_id);
  uint64_t resolveTwoPhaseKey(
      const ompfile::OmpFileReadRequestContext &context);
  bool rangesOverlap(long a_start, long a_end, long b_start, long b_end) const;
  void recordWriteEpochForContext(int file_id, long offset, size_t size,
                                   const ompfile::OmpFileIOHint &hint);
  bool globallyInvalidateStageForFile(int file_id);
  bool sourceStageAffinityPreReadInvalidateForFile(int file_id);
  void completeStageInvalidationForFile(int file_id, uint64_t generation,
                                         bool ok);
  void triggerStageInvalidationAfterSuccessfulRemoteWrite(int file_id);
  void noteAppliedRebalancedReadForFile(int file_id);
  bool canApplyRebalancedRead(const ompfile::OmpFileIOHint &hint, int file_id,
                               long start, size_t size,
                               const char *&reason_out,
                               int &reason_errno_out,
                               bool record_stats = true);
  void noteWritableReadRebalanceBlocked(const char *reason);
  bool getOrCreateRemoteReadHandleForRank(int file_id, int target_rank,
                                          int &remote_handle_out);
  void evictAndCloseRemoteReadHandle(int file_id, int target_rank);
  bool readAtRemoteRankWithBytes(int file_id, int target_rank, long offset,
                                 void *data, size_t size,
                                 size_t &bytes_read);
  bool readAtRemoteRankNoStageWithBytes(int file_id, int target_rank,
                                        long offset, void *data, size_t size,
                                        size_t &bytes_read);
  bool dirtyOwnerPreadMaybeBatch(int file_id, int remote_handle,
                                 int source_rank, uint64_t expected_version,
                                 long offset, void *data, size_t size,
                                 size_t &bytes_read, int &read_errno,
                                 uint64_t debug_request_id);
  bool writeAtRemoteRankWithBytes(int file_id, int target_rank, long offset,
                                  const void *data, size_t size,
                                  size_t &bytes_written);
  bool readAtSourceStageAffinityRankWithBytes(
      int file_id, uint64_t path_key, const ompfile::OmpFileIOHint &hint,
      long offset, void *data, size_t size, size_t &bytes_read,
      int &rank_out);
  bool commitTileFreshnessWrite(uint64_t path_key, int writer_rank,
                                bool write_through_mode,
                                uint64_t &committed_version_out);
  bool markTileFresh(uint64_t path_key, int rank, uint64_t version);
  bool isTileStale(uint64_t path_key, int rank, uint64_t local_version) const;
  TileFreshnessEntry tileFreshnessEntry(uint64_t path_key) const;
  static bool plannerStatusForcesFallback(int planner_errno,
                                          const char *&reason_out,
                                          int &reason_errno_out) {
    reason_out = "none";
    reason_errno_out = 0;
    if (planner_errno == ESTALE) {
      reason_out = "planner-stale-version";
      reason_errno_out = planner_errno;
      return true;
    }
    if (planner_errno == ENOKEY) {
      reason_out = "planner-missing-metadata";
      reason_errno_out = planner_errno;
      return true;
    }
    if (planner_errno == EAGAIN) {
      reason_out = "planner-owner-retry";
      reason_errno_out = planner_errno;
      return true;
    }
    return false;
  }
  bool tryServeTwoPhaseReadCache(uint64_t key, long offset, void *data,
                                 size_t size);
  void updateTwoPhaseReadCache(uint64_t key, long start,
                               const std::vector<char> &data);
  void invalidateTwoPhaseReadCacheKey(uint64_t key);
  void invalidateTwoPhaseReadCacheForFile(int file_id);
  int failStrictMpp(const char *op_name) const;
  void traceHandleStateLocked(const char *where, int file_id,
                              int remote_handle) const;
  void traceHandleState(const char *where, int file_id, int remote_handle);
  void reportPhase0Stats() const;
};

#endif // MPI_IO_BACKEND_H
