#include "mpi_io_backend.h"
#include "debug_log.h"
#include "mpp_shim.h"
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mpi.h>
#include <mutex>
#include <vector>

namespace {

struct OpenModeConfig {
  int PosixFlags = O_RDWR;
  int MpiMode = MPI_MODE_RDWR;
};

OpenModeConfig parseOpenModeConfig() {
  const char *mode = std::getenv("LIBOMPFILE_OPEN_MODE");
  if (!mode || !mode[0] || std::strcmp(mode, "readwrite") == 0)
    return {};
  if (std::strcmp(mode, "readonly") == 0 ||
      std::strcmp(mode, "read-only") == 0 || std::strcmp(mode, "read") == 0)
    return {O_RDONLY, MPI_MODE_RDONLY};

  io_log("Invalid LIBOMPFILE_OPEN_MODE='%s'; using readwrite mode.\n", mode);
  return {};
}

} // namespace

MPIIOBackend::MPIIOBackend() {
  int provided = 0;
  int initialized = 0;

  // Check if MPI is already initialized
  MPI_Initialized(&initialized);
  if (!initialized) {
    // Initialize MPI with thread support
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    io_log("MPI_Init_thread called, provided = %d\n", provided);
    externally_initialized = 0;
  } else {
    io_log("MPI already initialized by user program.\n");
    externally_initialized = 1;
    // MPI was initialized elsewhere (offload runtime / launcher), so we did
    // not choose the thread level. Query the level actually granted, since
    // it decides whether mpp_call_mutex is load-bearing for correctness: if
    // the runtime only got MPI_THREAD_SERIALIZED (or lower), concurrent
    // outbound MPP/MPI calls from multiple client threads MUST stay
    // serialized and mpp_call_mutex cannot be removed or per-rank sharded.
    MPI_Query_thread(&provided);
  }
  mpi_provided_thread_level = provided;
  lock_stats_enabled = parseBoolEnv("LIBOMPFILE_OPT_STATS", false);
  // Fail-safe: the opt-in only engages when MPI actually granted
  // MPI_THREAD_MULTIPLE. At any lower level concurrent outbound MPI calls from
  // multiple client threads are illegal and mpp_call_mutex is load-bearing, so
  // the request is ignored rather than honoured unsafely.
  mpp_call_lock_bypass =
      parseBoolEnv("LIBOMPFILE_OPT_MPP_CALL_LOCK_FREE", false) &&
      provided == MPI_THREAD_MULTIPLE;
  io_log("libompfile mpp_call_lock_bypass=%d (requested=%d thread_level=%d "
         "multiple=%d)\n",
         static_cast<int>(mpp_call_lock_bypass),
         static_cast<int>(parseBoolEnv("LIBOMPFILE_OPT_MPP_CALL_LOCK_FREE", false)),
         provided, MPI_THREAD_MULTIPLE);
  io_log("libompfile MPI thread level: provided=%d (SINGLE=%d FUNNELED=%d "
         "SERIALIZED=%d MULTIPLE=%d) externally_initialized=%d "
         "lock_stats_enabled=%d\n",
         provided, MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED,
         MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE,
         static_cast<int>(externally_initialized),
         static_cast<int>(lock_stats_enabled));

  if (parseBoolEnv("LIBOMPFILE_MPP_OPEN", false)) {
    mpp_open_enabled = true;
    io_log("LIBOMPFILE_MPP_OPEN=1: enabling MPP open/close path.\n");
  }

  if (parseBoolEnv("LIBOMPFILE_MPP_IO", false)) {
    if (mpp_open_enabled) {
      mpp_io_enabled = true;
      io_log("LIBOMPFILE_MPP_IO=1: enabling MPP I/O path.\n");
    } else {
      io_log("LIBOMPFILE_MPP_IO set without MPP open enabled.\n");
    }
  }

  mpp_requested = mpp_open_enabled || mpp_io_enabled;
  mpp_remote_only = mpp_open_enabled && mpp_io_enabled;
  const OpenModeConfig open_mode = parseOpenModeConfig();
  open_flags = open_mode.PosixFlags;
  mpi_open_mode = open_mode.MpiMode;
  if (open_flags == O_RDONLY)
    io_log("LIBOMPFILE_OPEN_MODE=readonly: opening files read-only.\n");
  writable_read_rebalance_enabled =
      parseBoolEnv("LIBOMPFILE_OPT_WRITABLE_READ_REBALANCE", false);
  dirty_owner_forwarding_enabled =
      parseBoolEnv("LIBOMPFILE_OPT_DIRTY_OWNER_FORWARDING", false);
  dirty_clean_pfs_rebalance_enabled =
      parseBoolEnv("LIBOMPFILE_OPT_DIRTY_CLEAN_PFS_REBALANCE", false);
  dirty_clean_pfs_allow_clean_pfs_enabled = parseBoolEnv(
      "LIBOMPFILE_OPT_DIRTY_CLEAN_PFS_REBALANCE_ALLOW_CLEAN_PFS", false);
  stage_global_invalidation_enabled =
      parseBoolEnv("LIBOMPFILE_OPT_STAGE_GLOBAL_INVALIDATION", false);
  allow_fallback = parseBoolEnv("LIBOMPFILE_ALLOWFALLBACK", false);
  strict_mpp_required = mpp_requested && !allow_fallback;

  if (!mpp_remote_only) {
    MPI_Comm source_comm = MPI_COMM_WORLD;
    const char *comm_self_env = std::getenv("LIBOMPFILE_MPI_COMM_SELF");
    if (comm_self_env && comm_self_env[0] == '1' && comm_self_env[1] == '\0') {
      source_comm = MPI_COMM_SELF;
      io_log("LIBOMPFILE_MPI_COMM_SELF=1: using MPI_COMM_SELF for file I/O.\n");
    }

    // Duplicate the chosen communicator for file I/O.
    MPI_Comm_dup(source_comm, &file_comm);
    io_log("MPI_Comm_dup completed for file I/O.\n");
  } else {
    io_log("Remote-only MPP mode: skipping MPI communicator duplication.\n");
  }

  if (mpp_requested) {
    if (ompfile::mpp::init()) {
      mpp_init_succeeded = true;
      const char *env = std::getenv("LIBOMPFILE_MPP_PING");
      if (env && env[0] == '1' && env[1] == '\0') {
        if (!ompfile::mpp::ping())
          io_log("MPP shim ping failed.\n");
      }
    } else {
      strict_mpp_init_failed = strict_mpp_required;
      if (strict_mpp_required) {
        io_log("MPP mode requested but MPP init failed and "
               "LIBOMPFILE_ALLOWFALLBACK is disabled; all I/O calls will "
               "fail.\n");
      } else {
        io_log("MPP mode requested but MPP init failed; MPP operations may "
               "fallback or fail.\n");
      }
    }
  }

  const char *scheduler_env = std::getenv("LIBOMPFILE_SCHEDULER");
  const bool scheduler_headnode =
      scheduler_env && std::strcmp(scheduler_env, "HEADNODE") == 0;

  two_phase_policy = parseTwoPhasePolicy(std::getenv("LIBOMPFILE_OPT_TWO_PHASE"));
  if (two_phase_policy == TwoPhasePolicy::Enabled) {
    two_phase_enabled = true;
  } else if (two_phase_policy == TwoPhasePolicy::Auto) {
    two_phase_enabled = mpp_remote_only && scheduler_headnode;
  } else {
    two_phase_enabled = false;
  }
  two_phase_window_us =
      parseUint64Env("LIBOMPFILE_OPT_TWO_PHASE_WINDOW_US",
                     two_phase_enabled ? kDefaultTwoPhaseWindowUs : 0);
  two_phase_max_batch_bytes =
      parseUint64Env("LIBOMPFILE_OPT_TWO_PHASE_MAX_BATCH_BYTES",
                     two_phase_enabled ? kDefaultTwoPhaseMaxBatchBytes : 0);
  two_phase_sieve_bytes =
      parseUint64Env("LIBOMPFILE_OPT_TWO_PHASE_SIEVE_BYTES",
                     two_phase_enabled ? kDefaultTwoPhaseSieveBytes : 0);
  write_batch_policy =
      parseTwoPhasePolicy(std::getenv("LIBOMPFILE_OPT_WRITE_BATCH"));
  if (write_batch_policy == TwoPhasePolicy::Enabled) {
    write_batch_enabled = true;
  } else if (write_batch_policy == TwoPhasePolicy::Auto) {
    write_batch_enabled = mpp_remote_only;
  } else {
    write_batch_enabled = false;
  }
  write_batch_window_us =
      parseUint64Env("LIBOMPFILE_OPT_WRITE_BATCH_WINDOW_US",
                     write_batch_enabled ? kDefaultWriteBatchWindowUs : 0);
  write_batch_max_batch_bytes =
      parseUint64Env("LIBOMPFILE_OPT_WRITE_BATCH_MAX_BATCH_BYTES",
                     write_batch_enabled ? kDefaultWriteBatchMaxBatchBytes : 0);
  forecast_mode = parseForecastMode(std::getenv("LIBOMPFILE_IO_FORECAST"));
  stage_affinity_mode = parseStageAffinityMode(
      std::getenv("LIBOMPFILE_IO_STAGE_AFFINITY"));

  io_log("Two-phase guard config: policy=%s enabled=%d window_us=%llu "
         "max_batch_bytes=%llu sieve_bytes=%llu write_batch_policy=%s "
         "write_batch_enabled=%d write_batch_window_us=%llu "
         "write_batch_max_batch_bytes=%llu scheduler=%s remote_only=%d "
         "writable_read_rebalance=%d dirty_owner_forwarding=%d "
         "dirty_clean_pfs_rebalance=%d dirty_clean_pfs_allow_clean_pfs=%d "
         "stage_global_invalidation=%d forecast_mode=%s stage_affinity_mode=%s\n",
         twoPhasePolicyToString(two_phase_policy),
         static_cast<int>(two_phase_enabled),
         static_cast<unsigned long long>(two_phase_window_us),
         static_cast<unsigned long long>(two_phase_max_batch_bytes),
         static_cast<unsigned long long>(two_phase_sieve_bytes),
         twoPhasePolicyToString(write_batch_policy),
         static_cast<int>(write_batch_enabled),
         static_cast<unsigned long long>(write_batch_window_us),
         static_cast<unsigned long long>(write_batch_max_batch_bytes),
         scheduler_env ? scheduler_env : "(unset)",
         static_cast<int>(mpp_remote_only),
         static_cast<int>(writable_read_rebalance_enabled),
         static_cast<int>(dirty_owner_forwarding_enabled),
         static_cast<int>(dirty_clean_pfs_rebalance_enabled),
         static_cast<int>(dirty_clean_pfs_allow_clean_pfs_enabled),
         static_cast<int>(stage_global_invalidation_enabled),
         forecastModeToString(forecast_mode),
         stageAffinityModeToString(stage_affinity_mode));

  if (isTwoPhaseActive()) {
    io_log("Two-phase batching active (leader/follower mode).\n");
  } else if (two_phase_enabled) {
    io_log("Two-phase requested but inactive (requires remote-only MPP mode).\n");
  } else if (two_phase_policy == TwoPhasePolicy::Auto) {
    io_log("Two-phase auto policy resolved to disabled for this run: "
           "remote_only=%d scheduler_headnode=%d.\n",
           static_cast<int>(mpp_remote_only),
           static_cast<int>(scheduler_headnode));
  }

  int rank = -1;
  int mpi_initialized_for_rank = 0;
  MPI_Initialized(&mpi_initialized_for_rank);
  if (mpi_initialized_for_rank)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  io_trace("MPIIOBackend ctor this=%p rank=%d external_mpi=%d mpp_open=%d "
           "mpp_io=%d mpp_requested=%d mpp_remote_only=%d mpp_init_ok=%d "
           "allow_fallback=%d strict_mpp_required=%d strict_init_failed=%d "
           "two_phase_policy=%s two_phase_enabled=%d "
           "two_phase_window_us=%llu "
           "two_phase_max_batch_bytes=%llu two_phase_sieve_bytes=%llu\n",
           static_cast<void *>(this), rank, externally_initialized,
           static_cast<int>(mpp_open_enabled), static_cast<int>(mpp_io_enabled),
           static_cast<int>(mpp_requested), static_cast<int>(mpp_remote_only),
           static_cast<int>(mpp_init_succeeded),
           static_cast<int>(allow_fallback),
           static_cast<int>(strict_mpp_required),
           static_cast<int>(strict_mpp_init_failed),
           twoPhasePolicyToString(two_phase_policy),
           static_cast<int>(two_phase_enabled),
           static_cast<unsigned long long>(two_phase_window_us),
           static_cast<unsigned long long>(two_phase_max_batch_bytes),
           static_cast<unsigned long long>(two_phase_sieve_bytes));
  io_trace("MPIIOBackend ctor write-batch this=%p policy=%s enabled=%d "
           "window_us=%llu max_batch_bytes=%llu\n",
           static_cast<void *>(this),
           twoPhasePolicyToString(write_batch_policy),
           static_cast<int>(write_batch_enabled),
           static_cast<unsigned long long>(write_batch_window_us),
           static_cast<unsigned long long>(write_batch_max_batch_bytes));

}

MPIIOBackend::~MPIIOBackend() {
  reportPhase0Stats();
  {
    std::vector<int> remote_handles;
    {
      const auto lock = instrumentedHandleLock();
      remote_handles.reserve(remote_file_handle_map.size());
      for (const auto &entry : remote_file_handle_map) {
        if (entry.second >= 0)
          remote_handles.push_back(entry.second);
      }
      remote_file_handle_map.clear();
      remote_file_owner_rank_map.clear();
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
      for (const auto &per_file : remote_read_handle_cache) {
        for (const auto &per_rank : per_file.second) {
          if (per_rank.second >= 0)
            remote_read_handles.push_back(per_rank.second);
        }
      }
      remote_read_handle_cache.clear();
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
      path_keys.reserve(file_path_key_map.size());
      for (const auto &entry : file_path_key_map) {
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
  if (file_comm != MPI_COMM_NULL) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
      MPI_Comm_free(&file_comm);
  }
  // if(!externally_initialized) {
  //   // Free the duplicated communicator
  //   io_log("MPI_Comm_free completed.\n");
  // }
}


