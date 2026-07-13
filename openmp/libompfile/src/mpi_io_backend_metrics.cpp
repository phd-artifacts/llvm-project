#include "mpi_io_backend.h"
#include "debug_log.h"

#include <atomic>

bool MPIIOBackend::shouldReportStats() {
  return parseBoolEnv("LIBOMPFILE_OPT_STATS", false);
}

void MPIIOBackend::reportPhase0Stats() const {
  if (!two_phase_enabled && !shouldReportStats())
    return;

  const uint64_t pread_reqs =
      pread_request_count.load(std::memory_order_relaxed);
  const uint64_t remote_events =
      remote_pread_event_count.load(std::memory_order_relaxed);
  const uint64_t remote_bytes =
      remote_pread_bytes_total.load(std::memory_order_relaxed);
  const uint64_t pwrite_reqs =
      pwrite_request_count.load(std::memory_order_relaxed);
  const uint64_t remote_pwrite_events =
      remote_pwrite_event_count.load(std::memory_order_relaxed);
  const uint64_t remote_pwrite_bytes =
      remote_pwrite_bytes_total.load(std::memory_order_relaxed);
  const uint64_t short_reads =
      short_read_count.load(std::memory_order_relaxed);
  const uint64_t short_read_bytes =
      short_read_bytes_total.load(std::memory_order_relaxed);
  const uint64_t short_writes =
      short_write_count.load(std::memory_order_relaxed);
  const uint64_t short_write_bytes =
      short_write_bytes_total.load(std::memory_order_relaxed);
  const uint64_t fallback_count =
      two_phase_fallback_count.load(std::memory_order_relaxed);
  const uint64_t batch_count =
      two_phase_batch_count.load(std::memory_order_relaxed);
  const uint64_t coalesced_reads =
      two_phase_coalesced_read_count.load(std::memory_order_relaxed);
  const uint64_t coalesced_bytes =
      two_phase_coalesced_bytes_total.load(std::memory_order_relaxed);
  const uint64_t planner_batches =
      two_phase_planner_batch_count.load(std::memory_order_relaxed);
  const uint64_t planner_segments =
      two_phase_planner_segment_count.load(std::memory_order_relaxed);
  const uint64_t planner_affinity =
      two_phase_planner_affinity_count.load(std::memory_order_relaxed);
  const uint64_t planner_rebalanced =
      two_phase_planner_rebalanced_count.load(std::memory_order_relaxed);
  const uint64_t planner_rebalanced_applied =
      two_phase_planner_rebalanced_applied_count.load(
          std::memory_order_relaxed);
  const uint64_t planner_rebalanced_skipped =
      two_phase_planner_rebalanced_skipped_count.load(
          std::memory_order_relaxed);
  const uint64_t planner_rebalanced_conflicts =
      two_phase_planner_rebalanced_conflict_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_candidates =
      writable_read_rebalance_candidate_count.load(std::memory_order_relaxed);
  const uint64_t writable_read_eligible =
      writable_read_rebalance_eligible_count.load(std::memory_order_relaxed);
  const uint64_t writable_read_applied =
      writable_read_rebalance_applied_count.load(std::memory_order_relaxed);
  const uint64_t writable_read_blocked =
      writable_read_rebalance_blocked_count.load(std::memory_order_relaxed);
  const uint64_t writable_read_blocked_disabled =
      writable_read_rebalance_blocked_disabled_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked_missing_metadata =
      writable_read_rebalance_blocked_missing_metadata_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked_stale_metadata =
      writable_read_rebalance_blocked_stale_metadata_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked_unsafe_overlap =
      writable_read_rebalance_blocked_unsafe_overlap_count.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_blocked_invalid_destination =
      writable_read_rebalance_blocked_invalid_destination_count.load(
          std::memory_order_relaxed);
  const uint64_t dirty_owner_forward_attempts =
      dirty_owner_forward_attempt_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_forward_successes =
      dirty_owner_forward_success_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_forward_failures =
      dirty_owner_forward_failure_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_forward_bytes =
      dirty_owner_forward_bytes_total.load(std::memory_order_relaxed);
  const uint64_t stage_invalidations_requested =
      stage_global_invalidations_requested.load(std::memory_order_relaxed);
  const uint64_t stage_invalidations_completed =
      stage_global_invalidations_completed.load(std::memory_order_relaxed);
  const uint64_t stage_invalidations_failed =
      stage_global_invalidations_failed.load(std::memory_order_relaxed);
  const uint64_t writable_read_blocked_pending_invalidation =
      writable_read_rebalance_blocked_pending_invalidation.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_after_invalidation_eligible =
      writable_read_rebalance_after_invalidation_eligible.load(
          std::memory_order_relaxed);
  const uint64_t writable_read_after_invalidation_applied =
      writable_read_rebalance_after_invalidation_applied.load(
          std::memory_order_relaxed);
  const uint64_t planner_scalar_fallbacks =
      two_phase_planner_scalar_fallback_count.load(std::memory_order_relaxed);
  const uint64_t planner_errors =
      two_phase_planner_error_count.load(std::memory_order_relaxed);
  const uint64_t cache_hits =
      two_phase_cache_hit_count.load(std::memory_order_relaxed);
  const uint64_t write_batches =
      write_batch_count.load(std::memory_order_relaxed);
  const uint64_t coalesced_writes =
      write_batch_coalesced_write_count.load(std::memory_order_relaxed);
  const uint64_t coalesced_write_bytes =
      write_batch_coalesced_bytes_total.load(std::memory_order_relaxed);
  const uint64_t write_saved_events =
      write_batch_saved_events.load(std::memory_order_relaxed);
  const uint64_t write_batch_failures =
      write_batch_failure_count.load(std::memory_order_relaxed);
  const uint64_t two_phase_queue_depth_max =
      two_phase_queue_max_depth.load(std::memory_order_relaxed);
  const uint64_t two_phase_leader_turns =
      two_phase_leader_turn_count.load(std::memory_order_relaxed);
  const uint64_t two_phase_follower_waits =
      two_phase_follower_wait_count.load(std::memory_order_relaxed);
  const uint64_t two_phase_follower_wait_us =
      two_phase_follower_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t two_phase_window_wait_us =
      two_phase_window_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t two_phase_batch_exec_us =
      two_phase_batch_exec_us_total.load(std::memory_order_relaxed);
  const uint64_t two_phase_request_wait_us =
      two_phase_request_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t write_queue_depth_max =
      write_queue_max_depth.load(std::memory_order_relaxed);
  const uint64_t write_leader_turns =
      write_leader_turn_count.load(std::memory_order_relaxed);
  const uint64_t write_follower_waits =
      write_follower_wait_count.load(std::memory_order_relaxed);
  const uint64_t write_follower_wait_us =
      write_follower_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t write_window_wait_us =
      write_window_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t write_batch_exec_us =
      write_batch_exec_us_total.load(std::memory_order_relaxed);
  const uint64_t write_request_wait_us =
      write_request_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t forecast_hint_reads =
      forecast_hint_read_count.load(std::memory_order_relaxed);
  const uint64_t forecast_hint_writes =
      forecast_hint_write_count.load(std::memory_order_relaxed);
  const uint64_t forecast_hint_read_bytes =
      forecast_hint_read_bytes_total.load(std::memory_order_relaxed);
  const uint64_t forecast_hint_write_bytes =
      forecast_hint_write_bytes_total.load(std::memory_order_relaxed);
  const uint64_t forecast_source_read_stageable =
      forecast_source_read_stageable_count.load(std::memory_order_relaxed);
  const uint64_t forecast_source_read_stageable_bytes =
      forecast_source_read_stageable_bytes_total.load(std::memory_order_relaxed);
  const uint64_t forecast_target_read_bypasses =
      forecast_target_read_bypass_count.load(std::memory_order_relaxed);
  const uint64_t forecast_target_read_bypass_bytes =
      forecast_target_read_bypass_bytes_total.load(std::memory_order_relaxed);
  const uint64_t forecast_role_unknown =
      forecast_role_unknown_count.load(std::memory_order_relaxed);
  const uint64_t forecast_role_unknown_bytes =
      forecast_role_unknown_bytes_total.load(std::memory_order_relaxed);
  const uint64_t stage_affinity_candidates =
      stage_affinity_source_candidate_count.load(std::memory_order_relaxed);
  const uint64_t stage_affinity_candidate_bytes =
      stage_affinity_source_candidate_bytes.load(std::memory_order_relaxed);
  const uint64_t stage_affinity_applied =
      stage_affinity_source_applied_count.load(std::memory_order_relaxed);
  const uint64_t stage_affinity_applied_bytes =
      stage_affinity_source_applied_bytes.load(std::memory_order_relaxed);
  const uint64_t stage_affinity_fallbacks =
      stage_affinity_source_fallback_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_candidates =
      dirty_owner_batch_candidate_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batches =
      dirty_owner_batch_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_segments =
      dirty_owner_batch_segment_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_saved =
      dirty_owner_batch_saved_events.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_total_bytes =
      dirty_owner_batch_bytes.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_failures =
      dirty_owner_batch_failure_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_prefetch_attempts =
      dirty_owner_prefetch_attempt_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_prefetch_cached =
      dirty_owner_prefetch_cached_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_prefetch_disabled =
      dirty_owner_prefetch_disabled_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_queue_depth_max =
      dirty_owner_queue_max_depth.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_follower_waits =
      dirty_owner_follower_wait_count.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_follower_wait_us =
      dirty_owner_follower_wait_us_total.load(std::memory_order_relaxed);
  const uint64_t dirty_owner_batch_window_wait_us =
      dirty_owner_batch_window_wait_us_total.load(std::memory_order_relaxed);

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

  io_log("Two-phase stats: pread_requests=%llu remote_pread_events=%llu "
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
         "stage_global_invalidations_requested=%llu "
         "stage_global_invalidations_completed=%llu "
         "stage_global_invalidations_failed=%llu "
         "writable_read_rebalance_blocked_pending_invalidation=%llu "
         "writable_read_rebalance_after_invalidation_eligible=%llu "
         "writable_read_rebalance_after_invalidation_applied=%llu "
         "planner_scalar_fallbacks=%llu planner_errors=%llu "
         "cache_hits=%llu fallbacks=%llu "
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
         "forecast_mode=%s forecast_hint_reads=%llu "
         "forecast_hint_writes=%llu forecast_hint_read_bytes=%llu "
         "forecast_hint_write_bytes=%llu "
         "forecast_source_read_stageable=%llu "
         "forecast_source_read_stageable_bytes=%llu "
         "forecast_target_read_bypasses=%llu "
          "forecast_target_read_bypass_bytes=%llu "
          "forecast_role_unknown=%llu forecast_role_unknown_bytes=%llu "
          "stage_affinity_mode=%s "
          "stage_affinity_source_candidates=%llu "
          "stage_affinity_source_candidate_bytes=%llu "
          "stage_affinity_source_applied=%llu "
          "stage_affinity_source_applied_bytes=%llu "
          "stage_affinity_source_fallbacks=%llu "
          "dirty_owner_batch_candidates=%llu "
          "dirty_owner_batch_count=%llu "
          "dirty_owner_batch_segments=%llu "
          "dirty_owner_batch_saved_events=%llu "
          "dirty_owner_batch_bytes=%llu "
          "dirty_owner_batch_failures=%llu "
          "dirty_owner_prefetch_attempts=%llu "
          "dirty_owner_prefetch_cached_segments=%llu "
          "dirty_owner_prefetch_disabled=%llu "
          "dirty_owner_queue_max_depth=%llu "
          "dirty_owner_follower_waits=%llu "
          "dirty_owner_follower_wait_us=%llu "
          "dirty_owner_batch_window_wait_us=%llu "
          "write_batch_enabled=%d "
          "two_phase_enabled=%d "
         "two_phase_active=%d two_phase_policy=%s window_us=%llu "
         "max_batch_bytes=%llu sieve_bytes=%llu write_batch_policy=%s "
         "write_batch_window_us=%llu write_batch_max_batch_bytes=%llu\n",
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
         static_cast<unsigned long long>(stage_invalidations_requested),
         static_cast<unsigned long long>(stage_invalidations_completed),
         static_cast<unsigned long long>(stage_invalidations_failed),
         static_cast<unsigned long long>(
             writable_read_blocked_pending_invalidation),
         static_cast<unsigned long long>(
             writable_read_after_invalidation_eligible),
         static_cast<unsigned long long>(
             writable_read_after_invalidation_applied),
         static_cast<unsigned long long>(planner_scalar_fallbacks),
         static_cast<unsigned long long>(planner_errors),
         static_cast<unsigned long long>(cache_hits),
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
         forecastModeToString(forecast_mode),
         static_cast<unsigned long long>(forecast_hint_reads),
         static_cast<unsigned long long>(forecast_hint_writes),
         static_cast<unsigned long long>(forecast_hint_read_bytes),
         static_cast<unsigned long long>(forecast_hint_write_bytes),
         static_cast<unsigned long long>(forecast_source_read_stageable),
         static_cast<unsigned long long>(forecast_source_read_stageable_bytes),
         static_cast<unsigned long long>(forecast_target_read_bypasses),
          static_cast<unsigned long long>(forecast_target_read_bypass_bytes),
          static_cast<unsigned long long>(forecast_role_unknown),
          static_cast<unsigned long long>(forecast_role_unknown_bytes),
          stageAffinityModeToString(stage_affinity_mode),
          static_cast<unsigned long long>(stage_affinity_candidates),
          static_cast<unsigned long long>(stage_affinity_candidate_bytes),
          static_cast<unsigned long long>(stage_affinity_applied),
          static_cast<unsigned long long>(stage_affinity_applied_bytes),
          static_cast<unsigned long long>(stage_affinity_fallbacks),
          static_cast<unsigned long long>(dirty_owner_batch_candidates),
          static_cast<unsigned long long>(dirty_owner_batches),
          static_cast<unsigned long long>(dirty_owner_batch_segments),
          static_cast<unsigned long long>(dirty_owner_batch_saved),
          static_cast<unsigned long long>(dirty_owner_batch_total_bytes),
          static_cast<unsigned long long>(dirty_owner_batch_failures),
          static_cast<unsigned long long>(dirty_owner_prefetch_attempts),
          static_cast<unsigned long long>(dirty_owner_prefetch_cached),
          static_cast<unsigned long long>(dirty_owner_prefetch_disabled),
          static_cast<unsigned long long>(dirty_owner_queue_depth_max),
          static_cast<unsigned long long>(dirty_owner_follower_waits),
          static_cast<unsigned long long>(dirty_owner_follower_wait_us),
          static_cast<unsigned long long>(dirty_owner_batch_window_wait_us),
          static_cast<int>(write_batch_enabled),
         static_cast<int>(two_phase_enabled),
         static_cast<int>(isTwoPhaseActive()),
         twoPhasePolicyToString(two_phase_policy),
         static_cast<unsigned long long>(two_phase_window_us),
         static_cast<unsigned long long>(two_phase_max_batch_bytes),
         static_cast<unsigned long long>(two_phase_sieve_bytes),
         twoPhasePolicyToString(write_batch_policy),
         static_cast<unsigned long long>(write_batch_window_us),
         static_cast<unsigned long long>(write_batch_max_batch_bytes));
}
