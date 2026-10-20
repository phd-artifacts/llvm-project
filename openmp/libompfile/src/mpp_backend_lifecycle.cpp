#include "mpp_backend.h"
#include "debug_log.h"
#include "mpp_shim.h"
#include "ompfile_config.h"
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mpi.h>
#include <mutex>
#include <string>
#include <vector>

MppBackend::MppBackend()
    : MppBackend(ompfile::config::ClientConfig::fromEnvironment()) {}

MppBackend::MppBackend(const ompfile::config::ClientConfig &config)
    : config_(config) {
  // Remote-only MPP is libompfile's only mode (Oct 2026). A removed mode
  // requested through LIBOMPFILE_BACKEND / LIBOMPFILE_MPP_OPEN /
  // LIBOMPFILE_MPP_IO takes the MPP-init-failure path below, and every
  // removed knob still set is named once (both lists live in ompfile_env.h).
  const std::string &rejected_mode_knobs = config_.RejectedModeKnobs;
  const bool mode_knobs_valid = rejected_mode_knobs.empty();
  if (!mode_knobs_valid)
    io_report("libompfile init failed: %s requests a removed mode; "
              "remote-only MPP is the only mode (leave LIBOMPFILE_BACKEND, "
              "LIBOMPFILE_MPP_OPEN and LIBOMPFILE_MPP_IO unset, or set them "
              "to MPI, 1 and 1); all I/O calls will fail.\n",
              rejected_mode_knobs.c_str());
  const std::string &removed_knobs = config_.RemovedKnobsSet;
  if (!removed_knobs.empty())
    io_report("removed knob ignored: %s\n", removed_knobs.c_str());

  int provided = -1;
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized && mode_knobs_valid) {
    // The runtime owns MPI end to end: libompfile never calls MPI_Init.
    // Bringing the MPP bridge up initializes the MPI plugin, which
    // initializes MPI at MPI_THREAD_MULTIPLE.
    (void)ompfile::mpp::init();
    MPI_Initialized(&initialized);
  }
  if (initialized) {
    // MPI was initialized elsewhere (offload runtime / launcher), so we did
    // not choose the thread level. Query the level actually granted, since
    // it decides whether mpp_call_mutex is load-bearing for correctness: if
    // the runtime only got MPI_THREAD_SERIALIZED (or lower), concurrent
    // outbound MPP/MPI calls from multiple client threads MUST stay
    // serialized and mpp_call_mutex cannot be removed or per-rank sharded.
    MPI_Query_thread(&provided);
  } else {
    io_log("MPI is not initialized; libompfile does not initialize it.\n");
  }
  settings.mpi_provided_thread_level = provided;
  settings.lock_stats_enabled = resolveBoolKnob(config_.OptStats);
  // Fail-safe: the opt-in only engages when MPI actually granted
  // MPI_THREAD_MULTIPLE. At any lower level concurrent outbound MPI calls from
  // multiple client threads are illegal and mpp_call_mutex is load-bearing, so
  // the request is ignored rather than honoured unsafely.
  settings.mpp_call_lock_bypass =
      resolveBoolKnob(config_.MppCallLockFree) &&
      provided == MPI_THREAD_MULTIPLE;
  io_log("libompfile mpp_call_lock_bypass=%d (requested=%d thread_level=%d "
         "multiple=%d)\n",
         static_cast<int>(settings.mpp_call_lock_bypass),
         static_cast<int>(resolveBoolKnob(config_.MppCallLockFree)),
         provided, MPI_THREAD_MULTIPLE);
  io_log("libompfile MPI thread level: provided=%d (SINGLE=%d FUNNELED=%d "
         "SERIALIZED=%d MULTIPLE=%d) "
         "lock_stats_enabled=%d\n",
         provided, MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED,
         MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE,
         static_cast<int>(settings.lock_stats_enabled));

  if (config_.OpenModeMalformed)
    io_log("Invalid LIBOMPFILE_OPEN_MODE='%s'; using readwrite mode.\n",
           config_.OpenModeRaw.c_str());
  settings.open_flags =
      config_.Open == ompfile::config::OpenMode::ReadOnly ? O_RDONLY : O_RDWR;
  if (settings.open_flags == O_RDONLY)
    io_log("LIBOMPFILE_OPEN_MODE=readonly: opening files read-only.\n");
  settings.writable_read_rebalance_enabled =
      resolveBoolKnob(config_.WritableReadRebalance);
  settings.dirty_owner_forwarding_enabled =
      resolveBoolKnob(config_.DirtyOwnerForwarding);
  if (config_.Shared.WritebackDistributeMisconfigured) {
    io_report("Error: LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_WRITES=1 requires "
              "LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_RANKS > 1 (got '%s'); "
              "distributed write-back routing stays off.\n",
              config_.WritebackDistributeRanksSet
                  ? config_.WritebackDistributeRanksRaw.c_str()
                  : "unset");
  }

  if (!mode_knobs_valid) {
    settings.strict_mpp_init_failed = true;
  } else if (ompfile::mpp::init()) {
    settings.mpp_init_succeeded = true;
    if (config_.MppPing) {
      if (!ompfile::mpp::ping())
        io_log("MPP shim ping failed.\n");
    }
  } else {
    settings.strict_mpp_init_failed = true;
    io_log("MPP init failed; all I/O calls will fail.\n");
  }

  const bool scheduler_headnode = config_.Shared.SchedulerHeadnode;

  settings.two_phase_policy = resolvePolicyKnob(config_.TwoPhase);
  if (settings.two_phase_policy == TwoPhasePolicy::Enabled) {
    settings.two_phase_enabled = true;
  } else if (settings.two_phase_policy == TwoPhasePolicy::Auto) {
    settings.two_phase_enabled = scheduler_headnode;
  } else {
    settings.two_phase_enabled = false;
  }
  settings.two_phase_window_us = resolveU64Knob(
      config_.TwoPhaseWindowUs,
      settings.two_phase_enabled ? kDefaultTwoPhaseWindowUs : 0);
  settings.two_phase_max_batch_bytes = resolveU64Knob(
      config_.TwoPhaseMaxBatchBytes,
      settings.two_phase_enabled ? kDefaultTwoPhaseMaxBatchBytes : 0);
  settings.two_phase_sieve_bytes = resolveU64Knob(
      config_.TwoPhaseSieveBytes,
      settings.two_phase_enabled ? kDefaultTwoPhaseSieveBytes : 0);
  settings.write_batch_policy = resolvePolicyKnob(config_.WriteBatch);
  if (settings.write_batch_policy == TwoPhasePolicy::Enabled) {
    settings.write_batch_enabled = true;
  } else if (settings.write_batch_policy == TwoPhasePolicy::Auto) {
    settings.write_batch_enabled = true;
  } else {
    settings.write_batch_enabled = false;
  }
  settings.write_batch_window_us = resolveU64Knob(
      config_.WriteBatchWindowUs,
      settings.write_batch_enabled ? kDefaultWriteBatchWindowUs : 0);
  settings.write_batch_window_adaptive =
      resolveBoolKnob(config_.WriteBatchWindowAdaptive);
  io_log("libompfile write_batch_window_adaptive=%d\n",
         static_cast<int>(settings.write_batch_window_adaptive));
  settings.two_phase_window_adaptive =
      resolveBoolKnob(config_.TwoPhaseWindowAdaptive);
  settings.sched_once_per_handle =
      resolveBoolKnob(config_.SchedOncePerHandle) && config_.Shared.stageOff();
  io_log("libompfile two_phase_window_adaptive=%d sched_once_per_handle=%d\n",
         static_cast<int>(settings.two_phase_window_adaptive),
         static_cast<int>(settings.sched_once_per_handle));
  settings.write_batch_max_batch_bytes = resolveU64Knob(
      config_.WriteBatchMaxBatchBytes,
      settings.write_batch_enabled ? kDefaultWriteBatchMaxBatchBytes : 0);

  io_log("Two-phase guard config: policy=%s enabled=%d window_us=%llu "
         "max_batch_bytes=%llu sieve_bytes=%llu write_batch_policy=%s "
         "write_batch_enabled=%d write_batch_window_us=%llu "
         "write_batch_max_batch_bytes=%llu scheduler=%s "
         "writable_read_rebalance=%d dirty_owner_forwarding=%d\n",
         twoPhasePolicyToString(settings.two_phase_policy),
         static_cast<int>(settings.two_phase_enabled),
         static_cast<unsigned long long>(settings.two_phase_window_us),
         static_cast<unsigned long long>(settings.two_phase_max_batch_bytes),
         static_cast<unsigned long long>(settings.two_phase_sieve_bytes),
         twoPhasePolicyToString(settings.write_batch_policy),
         static_cast<int>(settings.write_batch_enabled),
         static_cast<unsigned long long>(settings.write_batch_window_us),
         static_cast<unsigned long long>(settings.write_batch_max_batch_bytes),
         scheduler_headnode ? "HEADNODE" : "LOCAL",
         static_cast<int>(settings.writable_read_rebalance_enabled),
         static_cast<int>(settings.dirty_owner_forwarding_enabled));

  if (isTwoPhaseActive()) {
    io_log("Two-phase batching active (leader/follower mode).\n");
  } else if (settings.two_phase_policy == TwoPhasePolicy::Auto) {
    io_log("Two-phase auto policy resolved to disabled for this run: "
           "scheduler_headnode=%d.\n",
           static_cast<int>(scheduler_headnode));
  }

  int rank = -1;
  int mpi_initialized_for_rank = 0;
  MPI_Initialized(&mpi_initialized_for_rank);
  if (mpi_initialized_for_rank)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  io_trace("MPIIOBackend ctor this=%p rank=%d mode_knobs_valid=%d "
           "mpp_init_ok=%d strict_init_failed=%d "
           "two_phase_policy=%s two_phase_enabled=%d "
           "two_phase_window_us=%llu "
           "two_phase_max_batch_bytes=%llu two_phase_sieve_bytes=%llu\n",
           static_cast<void *>(this), rank,
           static_cast<int>(mode_knobs_valid),
           static_cast<int>(settings.mpp_init_succeeded),
           static_cast<int>(settings.strict_mpp_init_failed),
           twoPhasePolicyToString(settings.two_phase_policy),
           static_cast<int>(settings.two_phase_enabled),
           static_cast<unsigned long long>(settings.two_phase_window_us),
           static_cast<unsigned long long>(settings.two_phase_max_batch_bytes),
           static_cast<unsigned long long>(settings.two_phase_sieve_bytes));
  io_trace("MPIIOBackend ctor write-batch this=%p policy=%s enabled=%d "
           "window_us=%llu max_batch_bytes=%llu\n",
           static_cast<void *>(this),
           twoPhasePolicyToString(settings.write_batch_policy),
           static_cast<int>(settings.write_batch_enabled),
           static_cast<unsigned long long>(settings.write_batch_window_us),
           static_cast<unsigned long long>(
               settings.write_batch_max_batch_bytes));

}

MppBackend::~MppBackend() {
  reportPhase0Stats();
  {
    std::vector<int> remote_handles;
    {
      const auto lock = instrumentedHandleLock();
      remote_handles.reserve(remote.remote_file_handle_map.size());
      for (const auto &entry : remote.remote_file_handle_map) {
        if (entry.second >= 0)
          remote_handles.push_back(entry.second);
      }
      remote.remote_file_handle_map.clear();
      remote.remote_file_owner_rank_map.clear();
    }
    for (int remote_handle : remote_handles) {
      const auto lock = instrumentedMppCallLock();
      if (!ompfile::mpp::close(remote_handle)) {
        io_log("MPP close retry failed during backend teardown (handle=%d)\n",
               remote_handle);
      }
    }
  }
  {
    std::vector<int> remote_read_handles;
    {
      const auto lock = instrumentedHandleLock();
      for (const auto &per_file : remote.remote_read_handle_cache) {
        for (const auto &per_rank : per_file.second) {
          if (per_rank.second >= 0)
            remote_read_handles.push_back(per_rank.second);
        }
      }
      remote.remote_read_handle_cache.clear();
    }
    for (int remote_handle : remote_read_handles) {
      const auto lock = instrumentedMppCallLock();
      if (!ompfile::mpp::close(remote_handle)) {
        io_log("MPP close retry failed during backend teardown for cached "
               "remote read handle=%d\n",
               remote_handle);
      }
    }
  }
  {
    std::vector<uint64_t> path_keys;
    {
      const auto lock = instrumentedHandleLock();
      path_keys.reserve(handles.file_path_key_map.size());
      for (const auto &entry : handles.file_path_key_map) {
        if (entry.second != 0)
          path_keys.push_back(entry.second);
      }
    }
    for (uint64_t path_key : path_keys) {
      int source_rank = -1;
      uint64_t flushed_version = 0;
      if (!ompfile::mpp::flushDirtyTile(path_key, source_rank,
                                        flushed_version)) {
        io_log("Dirty flush failed path_key=%llu errno=%d\n",
               static_cast<unsigned long long>(path_key), errno);
      }
    }
  }
  ompfile::mpp::finalize();
}


