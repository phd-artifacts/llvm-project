#ifndef OMPFILE_MPP_BACKEND_H
#define OMPFILE_MPP_BACKEND_H

#include "abstract_backend.h"
#include "ompfile_sched.h"
#include "ompfile_keys.h"
#include "ompfile_config.h"
#include "ompfile_batcher.h"
#include "ompfile_route.h"
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include <cstddef> // for size_t
#include <cstdint>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class MppBackendTestAccess;

class MppBackend : public IOBackend {
private:
  friend class MppBackendTestAccess;

  using TwoPhasePolicy = ompfile::config::BatchPolicy;
  // Every knob this backend acts on, parsed once at construction
  // (ompfile_config.h); the members below hold the values resolved from it.
  const ompfile::config::ClientConfig config_;

  struct WriteEpochEntry {
    long Start = 0;
    long End = 0;
    uint64_t EpochId = 0;
    uint64_t TileId = 0;
    uint64_t Sequence = 0;
    bool HasEpoch = false;
    bool HasTile = false;
  };
  static constexpr uint64_t kDefaultTwoPhaseWindowUs = 2000;
  static constexpr uint64_t kDefaultTwoPhaseMaxBatchBytes = 8 * 1024 * 1024;
  static constexpr uint64_t kDefaultTwoPhaseSieveBytes = 256 * 1024;
  static constexpr uint64_t kDefaultWriteBatchWindowUs = 2000;
  static constexpr uint64_t kDefaultWriteBatchMaxBatchBytes =
      16 * 1024 * 1024;

  struct Settings {
    // Set once by the constructor, read-only afterwards (no lock).
    int mpi_provided_thread_level = -1;
    // When false (default) the InstrumentedLock guards are plain mutex locks
    // with no timing/counting overhead. Set from LIBOMPFILE_OPT_STATS at
    // construction so production runs pay nothing.
    bool lock_stats_enabled = false;
    bool mpp_init_succeeded = false;
    bool strict_mpp_init_failed = false;
    int open_flags = O_RDWR;
    bool writable_read_rebalance_enabled = false;
    bool dirty_owner_forwarding_enabled = false;
    TwoPhasePolicy two_phase_policy = TwoPhasePolicy::Disabled;
    bool two_phase_enabled = false;
    TwoPhasePolicy write_batch_policy = TwoPhasePolicy::Disabled;
    bool write_batch_enabled = false;
    uint64_t two_phase_window_us = 0;
    uint64_t two_phase_max_batch_bytes = 0;
    uint64_t two_phase_sieve_bytes = 0;
    uint64_t write_batch_window_us = 0;
    // LIBOMPFILE_OPT_WRITE_BATCH_WINDOW_ADAPTIVE=1: a leader skips the window
    // when it is the only thread inside write_batch_batcher (the adaptive check
    // uses the batcher's active count) and nothing else is queued, since no
    // follower can arrive to join the batch. A window it does not skip is held
    // to its full length. Default off.
    bool write_batch_window_adaptive = false;
    // LIBOMPFILE_OPT_TWO_PHASE_WINDOW_ADAPTIVE=1: the two-phase read leader
    // skips its collection window when it is the only thread inside
    // two_phase_batcher (the adaptive check uses the batcher's active count)
    // and nothing else is queued. Default off.
    bool two_phase_window_adaptive = false;
    // LIBOMPFILE_OPT_SCHED_ONCE_PER_HANDLE=1: no scheduler round trip per read
    // batch (the two-phase planner) - the owner was fixed at open and the
    // client routes by the opened handle anyway. Only honoured with the stage
    // off, since the planner's answer is also where rebalanced/dirty-owner
    // routing comes from. The per-write round trip is skipped on the scheduler
    // side (HeadnodeScheduler::per_op_scheduling_skipped) under the same knob.
    // Default off.
    bool sched_once_per_handle = false;
    uint64_t write_batch_max_batch_bytes = 0;
    // Set once at init from LIBOMPFILE_OPT_MPP_CALL_LOCK_FREE and the granted
    // MPI thread level; never flipped afterwards, so it needs no
    // synchronisation.
    bool mpp_call_lock_bypass = false;
  };
  struct HandleTable {
    std::unordered_set<int> logical_handle_set;
    std::unordered_map<int, std::string> file_path_map;
    std::unordered_map<int, uint64_t> file_path_key_map;
  };
  struct RemoteHandleTable {
    std::unordered_map<int, int> remote_file_handle_map;
    std::unordered_map<int, int> remote_file_owner_rank_map;
    std::unordered_map<int, std::unordered_map<int, int>>
        remote_read_handle_cache;
  };
  struct WriteEpochHistory {
    std::unordered_map<int, std::vector<WriteEpochEntry>>
        file_write_epoch_history;
    uint64_t next_write_sequence = 1;
  };
  struct Counters {
    // Stats counters. The io_report format in metrics.cpp, not this order, is
    // the parser contract.
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
    std::atomic<uint64_t>
        writable_read_rebalance_blocked_missing_metadata_count{0};
    std::atomic<uint64_t>
        writable_read_rebalance_blocked_stale_metadata_count{0};
    std::atomic<uint64_t>
        writable_read_rebalance_blocked_unsafe_overlap_count{0};
    std::atomic<uint64_t>
        writable_read_rebalance_blocked_invalid_destination_count{0};
    std::atomic<uint64_t> dirty_owner_forward_attempt_count{0};
    std::atomic<uint64_t> dirty_owner_forward_success_count{0};
    std::atomic<uint64_t> dirty_owner_forward_failure_count{0};
    std::atomic<uint64_t> dirty_owner_forward_bytes_total{0};
    std::atomic<uint64_t>
        writable_read_rebalance_blocked_pending_invalidation{0};
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
    // mutable so the instrumented lock helpers can be const (const members
    // such as instrumentedHandleLock() acquire handle_mutex).
    mutable std::atomic<uint64_t> handle_mutex_acquire_count{0};
    mutable std::atomic<uint64_t> handle_mutex_contended_count{0};
    mutable std::atomic<uint64_t> handle_mutex_wait_us_total{0};
    mutable std::atomic<uint64_t> handle_mutex_hold_us_total{0};
    mutable std::atomic<uint64_t> mpp_call_mutex_acquire_count{0};
    mutable std::atomic<uint64_t> mpp_call_mutex_contended_count{0};
    mutable std::atomic<uint64_t> mpp_call_mutex_wait_us_total{0};
    mutable std::atomic<uint64_t> mpp_call_mutex_hold_us_total{0};
  };
  Settings settings;
  HandleTable handles;            // guarded by handle_mutex
  RemoteHandleTable remote;       // guarded by handle_mutex
  WriteEpochHistory write_epochs; // guarded by handle_mutex
  Counters counters;

  std::atomic<int> next_file_handle{0};
  mutable std::mutex handle_mutex; // guards handles, remote and write_epochs
  mutable std::mutex mpp_call_mutex;
  mutable std::atomic<bool> strict_failure_logged{false};

  // RAII lock guard that optionally records contention/wait/hold-time into the
  // Counters lock fields. When instrumentation is disabled (the default, unless
  // LIBOMPFILE_OPT_STATS=1) it is a plain lock_guard-equivalent with zero extra
  // overhead on the hot path. When enabled it try_lock()s first: if that
  // succeeds the lock was uncontended (no waiting); if it fails another thread
  // held it, so we count a contended acquisition and time the blocking lock().
  // All ~68 backend acquisition sites route through this guard, so the
  // contended ratio and hold time are complete (not sampled) when enabled.
  class InstrumentedLock {
  public:
    InstrumentedLock(std::mutex &m, bool enabled,
                     std::atomic<uint64_t> &acquires,
                     std::atomic<uint64_t> &contended,
                     std::atomic<uint64_t> &wait_us,
                     std::atomic<uint64_t> &hold_us, bool bypass = false)
        : mutex_(m), enabled_(enabled), bypass_(bypass), hold_us_(hold_us) {
      // bypass_ still records acquires and hold time, so a bypassed run stays
      // directly comparable with a locked one; only the mutex is skipped.
      if (bypass_) {
        if (enabled_) {
          acquires.fetch_add(1, std::memory_order_relaxed);
          hold_start_ = std::chrono::steady_clock::now();
        }
        return;
      }
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
      if (!bypass_)
        mutex_.unlock();
    }
    InstrumentedLock(const InstrumentedLock &) = delete;
    InstrumentedLock &operator=(const InstrumentedLock &) = delete;

  private:
    std::mutex &mutex_;
    bool enabled_;
    bool bypass_ = false;
    std::atomic<uint64_t> &hold_us_;
    std::chrono::steady_clock::time_point hold_start_;
  };

  InstrumentedLock instrumentedHandleLock() const {
    return InstrumentedLock(handle_mutex, settings.lock_stats_enabled,
                            counters.handle_mutex_acquire_count,
                            counters.handle_mutex_contended_count,
                            counters.handle_mutex_wait_us_total,
                            counters.handle_mutex_hold_us_total);
  }
  // The global outbound-call lock exists because concurrent MPI calls from
  // several client threads are only legal at MPI_THREAD_MULTIPLE. Measured on
  // sorgan it is the reason host execution is slower than target execution: the
  // lanes serialize on it (wait 0 -> 2.15 s once both issue from one rank).
  // LIBOMPFILE_OPT_MPP_CALL_LOCK_FREE=1 drops it, and only ever takes effect
  // when MPI actually granted MPI_THREAD_MULTIPLE - otherwise the bypass stays
  // off no matter what the environment asks for.
  InstrumentedLock instrumentedMppCallLock() const {
    return InstrumentedLock(mpp_call_mutex, settings.lock_stats_enabled,
                            counters.mpp_call_mutex_acquire_count,
                            counters.mpp_call_mutex_contended_count,
                            counters.mpp_call_mutex_wait_us_total,
                            counters.mpp_call_mutex_hold_us_total,
                            settings.mpp_call_lock_bypass);
  }

  // Request and planner batch ids (not stats counters).
  std::atomic<uint64_t> two_phase_planner_batch_id{1};
  std::atomic<uint64_t> two_phase_request_id{1};
  std::atomic<uint64_t> write_batch_request_id{1};

  struct TwoPhaseReadRequest : ompfile::batch::Completion {
    uint64_t DebugRequestId = 0;
    int FileHandle = -1;
    int ClientRank = -1;
    long Offset = 0;
    size_t Size = 0;
    void *Buffer = nullptr;
    uint64_t PathKey = 0;
    ompfile::OmpFileIOHint Hint{};
  };

  ompfile::batch::Batcher<TwoPhaseReadRequest> two_phase_batcher{
      ompfile::batch::Shape::LeaderPerBatch,
      ompfile::batch::Stats{
          .QueueMaxDepth = &counters.two_phase_queue_max_depth,
          .LeaderTurns = &counters.two_phase_leader_turn_count,
          .FollowerWaits = &counters.two_phase_follower_wait_count,
          .FollowerWaitUs = &counters.two_phase_follower_wait_us_total,
          .WindowWaitUs = &counters.two_phase_window_wait_us_total,
          .BatchExecUs = &counters.two_phase_batch_exec_us_total,
          .RequestWaitUs = &counters.two_phase_request_wait_us_total}};

  struct WriteBatchRequest : ompfile::batch::Completion {
    uint64_t DebugRequestId = 0;
    int FileHandle = -1;
    long Offset = 0;
    uint64_t GroupKey = 0;
    std::vector<char> Data;
  };

  ompfile::batch::Batcher<WriteBatchRequest> write_batch_batcher{
      ompfile::batch::Shape::LeaderPerBatch,
      ompfile::batch::Stats{
          .QueueMaxDepth = &counters.write_queue_max_depth,
          .LeaderTurns = &counters.write_leader_turn_count,
          .FollowerWaits = &counters.write_follower_wait_count,
          .FollowerWaitUs = &counters.write_follower_wait_us_total,
          .WindowWaitUs = &counters.write_window_wait_us_total,
          .BatchExecUs = &counters.write_batch_exec_us_total,
          .RequestWaitUs = &counters.write_request_wait_us_total}};

  struct DirtyOwnerBatchRequest : ompfile::batch::Completion {
    int FileId = -1;
    int RemoteHandle = -1;
    int SourceRank = -1;
    uint64_t ExpectedVersion = 0;
    long Offset = 0;
    size_t Size = 0;
    void *Buffer = nullptr;
    uint64_t DebugRequestId = 0;
    size_t BytesRead = 0;
  };

  ompfile::batch::Batcher<DirtyOwnerBatchRequest> dirty_owner_batcher{
      ompfile::batch::Shape::LeaderUntilEmpty,
      ompfile::batch::Stats{
          .QueueMaxDepth = &counters.dirty_owner_queue_max_depth,
          .LeaderTurns = nullptr,
          .FollowerWaits = &counters.dirty_owner_follower_wait_count,
          .FollowerWaitUs = &counters.dirty_owner_follower_wait_us_total,
          .WindowWaitUs = &counters.dirty_owner_batch_window_wait_us_total,
          .BatchExecUs = nullptr,
          .RequestWaitUs = nullptr}};

public:
  // Snapshots the environment; the direct-linked regressions construct one
  // backend per environment they set up.
  MppBackend();
  explicit MppBackend(const ompfile::config::ClientConfig &config);
  ~MppBackend();

#if defined(OMPFILE_ENABLE_TEST_ACCESS)
  // Direct-linked regression seam. Production libompfile builds do not define
  // OMPFILE_ENABLE_TEST_ACCESS, so these helpers are absent from the runtime
  // class interface.
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
    handles.logical_handle_set.insert(file_id);
    if (path && path[0] != '\0') {
      handles.file_path_map[file_id] = path;
      handles.file_path_key_map[file_id] = ompfile::keys::pathKey(path);
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
  };
  void noteAppliedRebalancedWritableReadForTesting(int file_id) {
    noteAppliedRebalancedReadForFile(file_id);
  }
  void addPlannerCountersForTesting(uint64_t rebalanced, uint64_t applied,
                                    uint64_t skipped, uint64_t conflicts,
                                    uint64_t errors) {
    counters.two_phase_planner_rebalanced_count.fetch_add(
        rebalanced, std::memory_order_relaxed);
    counters.two_phase_planner_rebalanced_applied_count.fetch_add(
        applied, std::memory_order_relaxed);
    counters.two_phase_planner_rebalanced_skipped_count.fetch_add(
        skipped, std::memory_order_relaxed);
    counters.two_phase_planner_rebalanced_conflict_count.fetch_add(
        conflicts, std::memory_order_relaxed);
    counters.two_phase_planner_error_count.fetch_add(errors,
                                                     std::memory_order_relaxed);
  }
  PlannerCounterSnapshotForTesting plannerCountersForTesting() const {
    return {counters.two_phase_planner_rebalanced_count.load(
                std::memory_order_relaxed),
            counters.two_phase_planner_rebalanced_applied_count.load(
                std::memory_order_relaxed),
            counters.two_phase_planner_rebalanced_skipped_count.load(
                std::memory_order_relaxed),
            counters.two_phase_planner_rebalanced_conflict_count.load(
                std::memory_order_relaxed),
            counters.two_phase_planner_error_count.load(
                std::memory_order_relaxed)};
  }
  WritableReadRebalanceCountersForTesting
  writableReadRebalanceCountersForTesting() const {
    return {counters.writable_read_rebalance_candidate_count.load(
                std::memory_order_relaxed),
            counters.writable_read_rebalance_eligible_count.load(
                std::memory_order_relaxed),
            counters.writable_read_rebalance_applied_count.load(
                std::memory_order_relaxed),
            counters.writable_read_rebalance_blocked_count.load(
                std::memory_order_relaxed),
            counters.writable_read_rebalance_blocked_disabled_count.load(
                std::memory_order_relaxed),
            counters.writable_read_rebalance_blocked_missing_metadata_count
                .load(std::memory_order_relaxed),
            counters.writable_read_rebalance_blocked_stale_metadata_count.load(
                std::memory_order_relaxed),
            counters.writable_read_rebalance_blocked_unsafe_overlap_count.load(
                std::memory_order_relaxed),
            counters.writable_read_rebalance_blocked_invalid_destination_count.load(
                std::memory_order_relaxed),
            counters.writable_read_rebalance_blocked_pending_invalidation.load(
                std::memory_order_relaxed)};
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
    counters.two_phase_queue_max_depth.store(delta.TwoPhaseQueueMaxDepth,
                                             std::memory_order_relaxed);
    counters.two_phase_leader_turn_count.fetch_add(delta.TwoPhaseLeaderTurns,
                                                   std::memory_order_relaxed);
    counters.two_phase_follower_wait_count.fetch_add(
        delta.TwoPhaseFollowerWaitCount, std::memory_order_relaxed);
    counters.two_phase_follower_wait_us_total.fetch_add(
        delta.TwoPhaseFollowerWaitUsTotal, std::memory_order_relaxed);
    counters.two_phase_window_wait_us_total.fetch_add(
        delta.TwoPhaseWindowWaitUsTotal, std::memory_order_relaxed);
    counters.two_phase_batch_exec_us_total.fetch_add(
        delta.TwoPhaseBatchExecUsTotal, std::memory_order_relaxed);
    counters.two_phase_request_wait_us_total.fetch_add(
        delta.TwoPhaseRequestWaitUsTotal, std::memory_order_relaxed);
    counters.write_queue_max_depth.store(delta.WriteQueueMaxDepth,
                                         std::memory_order_relaxed);
    counters.write_leader_turn_count.fetch_add(delta.WriteLeaderTurns,
                                               std::memory_order_relaxed);
    counters.write_follower_wait_count.fetch_add(delta.WriteFollowerWaitCount,
                                                 std::memory_order_relaxed);
    counters.write_follower_wait_us_total.fetch_add(
        delta.WriteFollowerWaitUsTotal, std::memory_order_relaxed);
    counters.write_window_wait_us_total.fetch_add(delta.WriteWindowWaitUsTotal,
                                                  std::memory_order_relaxed);
    counters.write_batch_exec_us_total.fetch_add(delta.WriteBatchExecUsTotal,
                                                 std::memory_order_relaxed);
    counters.write_request_wait_us_total.fetch_add(
        delta.WriteRequestWaitUsTotal, std::memory_order_relaxed);
  }
  QueueCoordinationCountersForTesting
  queueCoordinationCountersForTesting() const {
    return {counters.two_phase_queue_max_depth.load(std::memory_order_relaxed),
            counters.two_phase_leader_turn_count.load(
                std::memory_order_relaxed),
            counters.two_phase_follower_wait_count.load(
                std::memory_order_relaxed),
            counters.two_phase_follower_wait_us_total.load(
                std::memory_order_relaxed),
            counters.two_phase_window_wait_us_total.load(
                std::memory_order_relaxed),
            counters.two_phase_batch_exec_us_total.load(
                std::memory_order_relaxed),
            counters.two_phase_request_wait_us_total.load(
                std::memory_order_relaxed),
            counters.write_queue_max_depth.load(std::memory_order_relaxed),
            counters.write_leader_turn_count.load(std::memory_order_relaxed),
            counters.write_follower_wait_count.load(std::memory_order_relaxed),
            counters.write_follower_wait_us_total.load(
                std::memory_order_relaxed),
            counters.write_window_wait_us_total.load(std::memory_order_relaxed),
            counters.write_batch_exec_us_total.load(std::memory_order_relaxed),
            counters.write_request_wait_us_total.load(
                std::memory_order_relaxed)};
  }
#endif // OMPFILE_ENABLE_TEST_ACCESS

  int open(const char *filename) override;
  int openWithPlan(const char *filename,
                   const ompfile::OmpFileIOPlan *plan) override;
  int write(int file_id, const void *data, size_t size) override;
  int read(int file_id, void *data, size_t size) override;
  int close(int file_id) override;
  int commit(int file_id) override;
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
  int readAtFallbackWithBytes(int file_id, long offset, void *data, size_t size,
                              size_t &bytes_read);
  // Read route execution: decisions come from ompfile_route.h; these run them
  // (MPP calls, locks, retries, counters, logs, errno). A value means the read
  // finished with that return code; nullopt falls through to the next route
  // (PlannedOwner, then Fallback).
  ompfile::route::ReadRouteConfig snapshotReadRouteConfig() const;
  ompfile::route::DistributeConfig snapshotDistributeConfig() const;
  ompfile::route::CoherentObservation
  observeCoherentRead(const ompfile::OmpFileReadRequestContext &context);
  ompfile::route::RebalanceVerdict
  observeRebalanceVerdict(const ompfile::OmpFileReadRequestContext &context,
                          size_t size);
  std::optional<int>
  readCoherent(const ompfile::OmpFileReadRequestContext &context,
               const ompfile::route::ReadRouteConfig &route_config, void *data,
               size_t size);
  std::optional<int>
  readWritebackOracle(const ompfile::OmpFileReadRequestContext &context,
                      ompfile::route::ReadRouteState &state,
                      ompfile::route::ReadDecision &decision, void *data,
                      size_t size);
  std::optional<int>
  readWritebackDirtyForward(const ompfile::OmpFileReadRequestContext &context,
                            ompfile::route::ReadRouteState &state,
                            const ompfile::route::ReadDecision &decision,
                            int remote_handle, void *data, size_t size);
  std::optional<int>
  executeCoherentRoute(const ompfile::OmpFileReadRequestContext &context,
                       const ompfile::route::ReadRouteState &state,
                       const ompfile::route::ReadDecision &decision,
                       void *data, size_t size);
  std::optional<int>
  readDirtyOwnerRetry(const ompfile::OmpFileReadRequestContext &context,
                      const ompfile::route::ReadRouteState &state, void *data,
                      size_t size);
  std::optional<int>
  readPlannedOwner(const ompfile::OmpFileReadRequestContext &context,
                   void *data, size_t size);
  int readFallbackTail(const ompfile::OmpFileReadRequestContext &context,
                       void *data, size_t size);
  // Write route execution: writeAtWithContext resolves the route with
  // ompfile::route::resolveWriteRoute and executeWrite runs it (MPP calls,
  // counters, logs, errno). A DistributedRank write that fails falls back to
  // the owner handle. reportFreshnessWriteCommit commits a tiled write's
  // freshness version; a negative writer_rank_override means the file's
  // remote owner rank. On false the caller turns errno 0 into EIO.
  int executeWrite(const ompfile::OmpFileWriteRequestContext &context,
                   const ompfile::route::WriteDecision &decision,
                   uint64_t group_key, const void *data, size_t size);
  bool reportFreshnessWriteCommit(
      const ompfile::OmpFileWriteRequestContext &context,
      int writer_rank_override);
  int readAtTwoPhase(const ompfile::OmpFileReadRequestContext &context,
                     void *data, size_t size);
  // Batcher sites (close policy, drain key, execute() and trace hooks) of
  // two_phase_batcher, write_batch_batcher and dirty_owner_batcher, defined
  // in mpp_backend_two_phase_scheduler.cpp, mpp_backend_write_path.cpp
  // and mpp_backend_read_path.cpp.
  struct TwoPhaseBatchSite;
  struct WriteBatchSite;
  struct DirtyOwnerBatchSite;
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
  int getNextFileHandle();
  static TwoPhasePolicy
  resolvePolicyKnob(const ompfile::config::PolicyKnob &knob);
  static const char *twoPhasePolicyToString(TwoPhasePolicy policy);
  static uint64_t resolveU64Knob(const ompfile::config::U64Knob &knob,
                                 uint64_t default_value);
  static bool resolveBoolKnob(const ompfile::config::BoolKnob &knob);
  static size_t computeTwoPhaseTargetReadSize(long start, long end,
                                               size_t request_count,
                                               bool remote_only,
                                               uint64_t sieve_bytes,
                                               uint64_t max_batch_bytes);
  bool shouldReportStats() const;
  bool getFilePathKey(int file_id, uint64_t &path_key_out);
  void rememberFilePathKey(int file_id, const char *path);
  void rememberFilePath(int file_id, const char *path);
  uint64_t resolveTwoPhaseKey(
      const ompfile::OmpFileReadRequestContext &context);
  bool rangesOverlap(long a_start, long a_end, long b_start, long b_end) const;
  void recordWriteEpochForContext(int file_id, long offset, size_t size,
                                   const ompfile::OmpFileIOHint &hint);
  void noteAppliedRebalancedReadForFile(int file_id);
  bool canApplyRebalancedRead(const ompfile::OmpFileIOHint &hint, int file_id,
                               long start, size_t size,
                               const char *&reason_out,
                               int &reason_errno_out);
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
  // One drained dirty-owner batch: one batched pread when it has more than
  // one member, else (or when that fails) one scalar pread per member. Writes
  // each member's BytesRead, Status and Errno; the batcher sets Done.
  void executeDirtyOwnerBatch(std::vector<DirtyOwnerBatchRequest *> &Batch);
  bool writeAtRemoteRankWithBytes(int file_id, int target_rank, long offset,
                                  const void *data, size_t size,
                                  size_t &bytes_written);
  // Tail shared by every completed remote pread: more bytes than requested is
  // a protocol error (errno EPROTO, false); a short read into a non-null
  // buffer is counted and its tail zero-filled; short_bytes_out is that count,
  // else 0.
  bool finishRemoteRead(void *data, size_t size, size_t bytes_read,
                        size_t &short_bytes_out);
  // The remote handle / owner rank recorded for file_id at open, -1 when
  // absent; one instrumented handle_mutex acquisition each.
  int lookupRemoteHandle(int file_id) const;
  int lookupRemoteOwnerRank(int file_id) const;
  // Up to two tries of attempt(remote_handle) on file_id's cached handle for
  // target_rank; the second first evicts and closes it; a try that gets no
  // handle records errno and continues; a failed attempt stops unless errno
  // is 0 or EBADF; on exhaustion errno = the last recorded value (EIO if
  // none). Defined and used only in mpp_backend_remote_read_handles.cpp.
  template <typename AttemptFn>
  bool runOnRemoteRankHandle(int file_id, int target_rank,
                             AttemptFn &&attempt);
  using RemotePreadFn = bool (*)(int, int64_t, void *, size_t, size_t &);
  bool readAtRemoteRankUsing(RemotePreadFn pread_fn, int file_id,
                             int target_rank, long offset, void *data,
                             size_t size, size_t &bytes_read);
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
  int failStrictMpp(const char *op_name) const;
  void traceHandleStateLocked(const char *where, int file_id,
                              int remote_handle) const;
  void traceHandleState(const char *where, int file_id, int remote_handle);
  void reportPhase0Stats() const;
};

#endif // OMPFILE_MPP_BACKEND_H
