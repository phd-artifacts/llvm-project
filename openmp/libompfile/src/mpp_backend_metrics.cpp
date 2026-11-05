#include "mpp_backend.h"
#include "debug_log.h"

#include <atomic>

bool MppBackend::shouldReportStats() const {
  return resolveBoolKnob(config_.OptStats);
}

void MppBackend::reportPhase0Stats() const {
  if (!settings.two_phase_enabled && !shouldReportStats())
    return;

  const uint64_t pread_reqs =
      counters.pread_request_count.load(std::memory_order_relaxed);
  const uint64_t remote_events =
      counters.remote_pread_event_count.load(std::memory_order_relaxed);
  const uint64_t remote_bytes =
      counters.remote_pread_bytes_total.load(std::memory_order_relaxed);
  const uint64_t pwrite_reqs =
      counters.pwrite_request_count.load(std::memory_order_relaxed);
  const uint64_t remote_pwrite_events =
      counters.remote_pwrite_event_count.load(std::memory_order_relaxed);
  const uint64_t remote_pwrite_bytes =
      counters.remote_pwrite_bytes_total.load(std::memory_order_relaxed);
  const uint64_t short_reads =
      counters.short_read_count.load(std::memory_order_relaxed);
  const uint64_t short_read_bytes =
      counters.short_read_bytes_total.load(std::memory_order_relaxed);
  const uint64_t short_writes =
      counters.short_write_count.load(std::memory_order_relaxed);
  const uint64_t short_write_bytes =
      counters.short_write_bytes_total.load(std::memory_order_relaxed);
  const uint64_t fallback_count =
      counters.two_phase_fallback_count.load(std::memory_order_relaxed);
  const uint64_t batch_count =
      counters.two_phase_batch_count.load(std::memory_order_relaxed);
  const uint64_t coalesced_reads =
      counters.two_phase_coalesced_read_count.load(std::memory_order_relaxed);
  const uint64_t coalesced_bytes =
      counters.two_phase_coalesced_bytes_total.load(std::memory_order_relaxed);
  const uint64_t planner_batches =
      counters.two_phase_planner_batch_count.load(std::memory_order_relaxed);
  const uint64_t planner_segments =
      counters.two_phase_planner_segment_count.load(std::memory_order_relaxed);
  const uint64_t planner_affinity =
      counters.two_phase_planner_affinity_count.load(std::memory_order_relaxed);
  const uint64_t planner_rebalanced =
      counters.two_phase_planner_rebalanced_count.load(
          std::memory_order_relaxed);
  const uint64_t planner_rebalanced_applied =
      counters.two_phase_planner_rebalanced_applied_count.load(
          std::memory_order_relaxed);
  const uint64_t planner_rebalanced_skipped =
      counters.two_phase_planner_rebalanced_skipped_count.load(
          std::memory_order_relaxed);
  const uint64_t planner_rebalanced_conflicts =
      counters.two_phase_planner_rebalanced_conflict_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_candidates =
      counters.writable_read_rebalance_candidate_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_eligible =
      counters.writable_read_rebalance_eligible_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_applied =
      counters.writable_read_rebalance_applied_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked =
      counters.writable_read_rebalance_blocked_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked_disabled =
      counters.writable_read_rebalance_blocked_disabled_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked_missing_metadata =
      counters.writable_read_rebalance_blocked_missing_metadata_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked_stale_metadata =
      counters.writable_read_rebalance_blocked_stale_metadata_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked_unsafe_overlap =
      counters.writable_read_rebalance_blocked_unsafe_overlap_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked_invalid_destination =
      counters.writable_read_rebalance_blocked_invalid_destination_count.load(
          std::memory_order_relaxed);
  const uint64_t dirty_owner_forward_attempts =
      counters.dirty_owner_forward_attempt_count.load(
          std::memory_order_relaxed);
  const uint64_t dirty_owner_forward_successes =
      counters.dirty_owner_forward_success_count.load(
          std::memory_order_relaxed);
  const uint64_t dirty_owner_forward_failures =
      counters.dirty_owner_forward_failure_count.load(
          std::memory_order_relaxed);
  const uint64_t dirty_owner_forward_bytes =
      counters.dirty_owner_forward_bytes_total.load(std::memory_order_relaxed);
  const uint64_t writable_read_blocked_pending_invalidation =
      counters.writable_read_rebalance_blocked_pending_invalidation.load(
          std::memory_order_relaxed);
  const uint64_t planner_scalar_fallbacks =
      counters.two_phase_planner_scalar_fallback_count.load(
          std::memory_order_relaxed);
  const uint64_t planner_errors =
      counters.two_phase_planner_error_count.load(std::memory_order_relaxed);
  const uint64_t write_batches =
      counters.write_batch_count.load(std::memory_order_relaxed);
  const uint64_t coalesced_writes =
      counters.write_batch_coalesced_write_count.load(
          std::memory_order_relaxed);
  const uint64_t coalesced_write_bytes =
      counters.write_batch_coalesced_bytes_total.load(
          std::memory_order_relaxed);
  const uint64_t write_saved_events =
      counters.write_batch_saved_events.load(std::memory_order_relaxed);
  const uint64_t write_batch_failures =
      counters.write_batch_failure_count.load(std::memory_order_relaxed);
  const uint64_t two_phase_queue_depth_max =
      counters.two_phase_queue_max_depth.load(std::memory_order_relaxed);
  const uint64_t two_phase_leader_turns =
      counters.two_phase_leader_turn_count.load(std::memory_order_relaxed);
  const uint64_t two_phase_follower_waits =
      counters.two_phase_follower_wait_count.load(std::memory_order_relaxed);
  const uint64_t two_phase_follower_wait_us =
      counters.two_phase_follower_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t two_phase_window_wait_us =
      counters.two_phase_window_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t two_phase_batch_exec_us =
      counters.two_phase_batch_exec_us_total.load(std::memory_order_relaxed);
  const uint64_t two_phase_request_wait_us =
      counters.two_phase_request_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t write_queue_depth_max =
      counters.write_queue_max_depth.load(std::memory_order_relaxed);
  const uint64_t write_leader_turns =
      counters.write_leader_turn_count.load(std::memory_order_relaxed);
  const uint64_t write_follower_waits =
      counters.write_follower_wait_count.load(std::memory_order_relaxed);
  const uint64_t write_follower_wait_us =
      counters.write_follower_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t write_window_wait_us =
      counters.write_window_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t write_batch_exec_us =
      counters.write_batch_exec_us_total.load(std::memory_order_relaxed);
  const uint64_t write_request_wait_us =
      counters.write_request_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_candidates =
      counters.dirty_owner_batch_candidate_count.load(
          std::memory_order_relaxed);
  const uint64_t dirty_owner_batches =
      counters.dirty_owner_batch_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_single_reads =
      counters.dirty_owner_single_read_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_remote_read_events =
      counters.dirty_owner_remote_read_event_count.load(
          std::memory_order_relaxed);
  const uint64_t dirty_owner_remote_read_bytes =
      counters.dirty_owner_remote_read_bytes_total.load(
          std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_segments =
      counters.dirty_owner_batch_segment_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_saved =
      counters.dirty_owner_batch_saved_events.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_total_bytes =
      counters.dirty_owner_batch_bytes.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_failures =
      counters.dirty_owner_batch_failure_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_queue_depth_max =
      counters.dirty_owner_queue_max_depth.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_follower_waits =
      counters.dirty_owner_follower_wait_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_follower_wait_us =
      counters.dirty_owner_follower_wait_us_total.load(
          std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_window_wait_us =
      counters.dirty_owner_batch_window_wait_us_total.load(
          std::memory_order_relaxed);

  const double avg_remote_bytes =
      remote_events == 0 ? 0.0
                         : static_cast<double>(remote_bytes) /
                               static_cast<double>(remote_events);
  const double avg_remote_pwrite_bytes =
      remote_pwrite_events == 0
          ? 0.0
          : static_cast<double>(remote_pwrite_bytes) /
                static_cast<double>(remote_pwrite_events);
  const double avg_coalesced_bytes =
      coalesced_reads == 0 ? 0.0
                           : static_cast<double>(coalesced_bytes) /
                                 static_cast<double>(coalesced_reads);
  const double avg_coalesced_write_bytes =
      coalesced_writes == 0
          ? 0.0
          : static_cast<double>(coalesced_write_bytes) /
                static_cast<double>(coalesced_writes);

  io_report("Two-phase stats: pread_requests=%llu remote_pread_events=%llu "
         "remote_pread_bytes_total=%llu remote_pread_avg_bytes=%.2f "
         "pwrite_requests=%llu remote_pwrite_events=%llu "
         "remote_pwrite_bytes_total=%llu remote_pwrite_avg_bytes=%.2f "
         "short_reads=%llu short_read_bytes=%llu "
         "short_writes=%llu short_write_bytes=%llu "
         "batch_count=%llu coalesced_reads=%llu coalesced_bytes=%llu "
         "coalesced_avg_bytes=%.2f planner_batches=%llu "
         "planner_segments=%llu planner_affinity=%llu "
         "planner_rebalanced=%llu planner_rebalanced_applied=%llu "
         "planner_rebalanced_skipped=%llu "
         "planner_rebalanced_conflicts=%llu "
         "writable_read_rebalance_candidates=%llu "
         "writable_read_rebalance_eligible=%llu "
         "writable_read_rebalance_applied=%llu "
         "writable_read_rebalance_blocked=%llu "
         "writable_read_rebalance_blocked_disabled=%llu "
         "writable_read_rebalance_blocked_missing_metadata=%llu "
         "writable_read_rebalance_blocked_stale_metadata=%llu "
         "writable_read_rebalance_blocked_unsafe_overlap=%llu "
         "writable_read_rebalance_blocked_invalid_destination=%llu "
         "dirty_owner_forward_attempts=%llu "
         "dirty_owner_forward_successes=%llu "
         "dirty_owner_forward_failures=%llu "
         "dirty_owner_forward_bytes=%llu "
         "writable_read_rebalance_blocked_pending_invalidation=%llu "
         "planner_scalar_fallbacks=%llu planner_errors=%llu "
         "fallbacks=%llu "
         "write_batch_count=%llu coalesced_writes=%llu "
         "coalesced_write_bytes=%llu coalesced_write_avg_bytes=%.2f "
         "write_batch_saved_events=%llu write_batch_failures=%llu "
         "two_phase_queue_max_depth=%llu two_phase_leader_turns=%llu "
         "two_phase_follower_waits=%llu two_phase_follower_wait_us=%llu "
         "two_phase_window_wait_us=%llu two_phase_batch_exec_us=%llu "
         "two_phase_request_wait_us=%llu "
         "write_queue_max_depth=%llu write_leader_turns=%llu "
         "write_follower_waits=%llu write_follower_wait_us=%llu "
         "write_window_wait_us=%llu write_batch_exec_us=%llu "
         "write_request_wait_us=%llu "
          "dirty_owner_batch_candidates=%llu "
          "dirty_owner_batch_count=%llu "
          "dirty_owner_single_reads=%llu "
          "dirty_owner_remote_read_events=%llu "
          "dirty_owner_remote_read_bytes=%llu "
          "dirty_owner_batch_segments=%llu "
          "dirty_owner_batch_saved_events=%llu "
          "dirty_owner_batch_bytes=%llu "
          "dirty_owner_batch_failures=%llu "
          "dirty_owner_queue_max_depth=%llu "
          "dirty_owner_follower_waits=%llu "
          "dirty_owner_follower_wait_us=%llu "
          "dirty_owner_batch_window_wait_us=%llu "
          "write_batch_enabled=%d "
          "two_phase_enabled=%d "
         "two_phase_active=%d two_phase_policy=%s window_us=%llu "
         "max_batch_bytes=%llu sieve_bytes=%llu write_batch_policy=%s "
         "write_batch_window_us=%llu write_batch_max_batch_bytes=%llu "
         "mpi_thread_level=%d "
         "handle_mutex_acquires=%llu handle_mutex_contended=%llu "
         "handle_mutex_wait_us=%llu handle_mutex_hold_us=%llu "
         "mpp_call_mutex_acquires=%llu mpp_call_mutex_contended=%llu "
         "mpp_call_mutex_wait_us=%llu mpp_call_mutex_hold_us=%llu\n",
         static_cast<unsigned long long>(pread_reqs),
         static_cast<unsigned long long>(remote_events),
         static_cast<unsigned long long>(remote_bytes), avg_remote_bytes,
         static_cast<unsigned long long>(pwrite_reqs),
         static_cast<unsigned long long>(remote_pwrite_events),
         static_cast<unsigned long long>(remote_pwrite_bytes),
         avg_remote_pwrite_bytes,
         static_cast<unsigned long long>(short_reads),
         static_cast<unsigned long long>(short_read_bytes),
         static_cast<unsigned long long>(short_writes),
         static_cast<unsigned long long>(short_write_bytes),
         static_cast<unsigned long long>(batch_count),
         static_cast<unsigned long long>(coalesced_reads),
         static_cast<unsigned long long>(coalesced_bytes), avg_coalesced_bytes,
         static_cast<unsigned long long>(planner_batches),
         static_cast<unsigned long long>(planner_segments),
         static_cast<unsigned long long>(planner_affinity),
         static_cast<unsigned long long>(planner_rebalanced),
         static_cast<unsigned long long>(planner_rebalanced_applied),
         static_cast<unsigned long long>(planner_rebalanced_skipped),
         static_cast<unsigned long long>(planner_rebalanced_conflicts),
         static_cast<unsigned long long>(writable_read_candidates),
         static_cast<unsigned long long>(writable_read_eligible),
         static_cast<unsigned long long>(writable_read_applied),
         static_cast<unsigned long long>(writable_read_blocked),
         static_cast<unsigned long long>(writable_read_blocked_disabled),
         static_cast<unsigned long long>(
             writable_read_blocked_missing_metadata),
         static_cast<unsigned long long>(writable_read_blocked_stale_metadata),
         static_cast<unsigned long long>(writable_read_blocked_unsafe_overlap),
         static_cast<unsigned long long>(
             writable_read_blocked_invalid_destination),
         static_cast<unsigned long long>(dirty_owner_forward_attempts),
         static_cast<unsigned long long>(dirty_owner_forward_successes),
         static_cast<unsigned long long>(dirty_owner_forward_failures),
         static_cast<unsigned long long>(dirty_owner_forward_bytes),
         static_cast<unsigned long long>(
             writable_read_blocked_pending_invalidation),
         static_cast<unsigned long long>(planner_scalar_fallbacks),
         static_cast<unsigned long long>(planner_errors),
         static_cast<unsigned long long>(fallback_count),
         static_cast<unsigned long long>(write_batches),
         static_cast<unsigned long long>(coalesced_writes),
         static_cast<unsigned long long>(coalesced_write_bytes),
         avg_coalesced_write_bytes,
         static_cast<unsigned long long>(write_saved_events),
         static_cast<unsigned long long>(write_batch_failures),
         static_cast<unsigned long long>(two_phase_queue_depth_max),
         static_cast<unsigned long long>(two_phase_leader_turns),
         static_cast<unsigned long long>(two_phase_follower_waits),
         static_cast<unsigned long long>(two_phase_follower_wait_us),
         static_cast<unsigned long long>(two_phase_window_wait_us),
         static_cast<unsigned long long>(two_phase_batch_exec_us),
         static_cast<unsigned long long>(two_phase_request_wait_us),
         static_cast<unsigned long long>(write_queue_depth_max),
         static_cast<unsigned long long>(write_leader_turns),
         static_cast<unsigned long long>(write_follower_waits),
         static_cast<unsigned long long>(write_follower_wait_us),
         static_cast<unsigned long long>(write_window_wait_us),
         static_cast<unsigned long long>(write_batch_exec_us),
         static_cast<unsigned long long>(write_request_wait_us),
          static_cast<unsigned long long>(dirty_owner_batch_candidates),
          static_cast<unsigned long long>(dirty_owner_batches),
          static_cast<unsigned long long>(dirty_owner_single_reads),
          static_cast<unsigned long long>(dirty_owner_remote_read_events),
          static_cast<unsigned long long>(dirty_owner_remote_read_bytes),
          static_cast<unsigned long long>(dirty_owner_batch_segments),
          static_cast<unsigned long long>(dirty_owner_batch_saved),
          static_cast<unsigned long long>(dirty_owner_batch_total_bytes),
          static_cast<unsigned long long>(dirty_owner_batch_failures),
          static_cast<unsigned long long>(dirty_owner_queue_depth_max),
          static_cast<unsigned long long>(dirty_owner_follower_waits),
          static_cast<unsigned long long>(dirty_owner_follower_wait_us),
          static_cast<unsigned long long>(dirty_owner_batch_window_wait_us),
          static_cast<int>(settings.write_batch_enabled),
         static_cast<int>(settings.two_phase_enabled),
         static_cast<int>(isTwoPhaseActive()),
         twoPhasePolicyToString(settings.two_phase_policy),
         static_cast<unsigned long long>(settings.two_phase_window_us),
         static_cast<unsigned long long>(settings.two_phase_max_batch_bytes),
         static_cast<unsigned long long>(settings.two_phase_sieve_bytes),
         twoPhasePolicyToString(settings.write_batch_policy),
         static_cast<unsigned long long>(settings.write_batch_window_us),
         static_cast<unsigned long long>(settings.write_batch_max_batch_bytes),
         settings.mpi_provided_thread_level,
         static_cast<unsigned long long>(
             counters.handle_mutex_acquire_count.load(
                 std::memory_order_relaxed)),
         static_cast<unsigned long long>(
             counters.handle_mutex_contended_count.load(
                 std::memory_order_relaxed)),
         static_cast<unsigned long long>(
             counters.handle_mutex_wait_us_total.load(
                 std::memory_order_relaxed)),
         static_cast<unsigned long long>(
             counters.handle_mutex_hold_us_total.load(
                 std::memory_order_relaxed)),
         static_cast<unsigned long long>(
             counters.mpp_call_mutex_acquire_count.load(
                 std::memory_order_relaxed)),
         static_cast<unsigned long long>(
             counters.mpp_call_mutex_contended_count.load(
                 std::memory_order_relaxed)),
         static_cast<unsigned long long>(
             counters.mpp_call_mutex_wait_us_total.load(
                 std::memory_order_relaxed)),
         static_cast<unsigned long long>(
             counters.mpp_call_mutex_hold_us_total.load(
                 std::memory_order_relaxed)));
}
