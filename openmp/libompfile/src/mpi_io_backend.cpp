#include "mpi_io_backend.h"
#include "debug_log.h"
#include "mpp_shim.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <climits>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mpi.h>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace {

bool freshnessWriteThroughModeEnabled() {
  const char *mode = std::getenv("LIBOMPFILE_TILE_FRESHNESS_WRITE_MODE");
  if (mode && std::strcmp(mode, "write-back") == 0)
    return false;
  return true;
}

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
  }

  const char *mpp_open_env = std::getenv("LIBOMPFILE_MPP_OPEN");
  if (mpp_open_env && mpp_open_env[0] == '1' && mpp_open_env[1] == '\0') {
    mpp_open_enabled = true;
    io_log("LIBOMPFILE_MPP_OPEN=1: enabling MPP open/close path.\n");
  }

  const char *mpp_io_env = std::getenv("LIBOMPFILE_MPP_IO");
  if (mpp_io_env && mpp_io_env[0] == '1' && mpp_io_env[1] == '\0') {
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
         "writable_read_rebalance=%d forecast_mode=%s "
         "stage_affinity_mode=%s\n",
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
      const std::lock_guard<std::mutex> lock(handle_mutex);
      remote_handles.reserve(remote_file_handle_map.size());
      for (const auto &entry : remote_file_handle_map) {
        if (entry.second >= 0)
          remote_handles.push_back(entry.second);
      }
      remote_file_handle_map.clear();
      remote_file_owner_rank_map.clear();
    }
    for (int remote_handle : remote_handles) {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      if (!ompfile::mpp::close(remote_handle)) {
        io_log("MPP close retry failed during backend teardown (handle=%d)\n",
               remote_handle);
      }
    }
  }
  {
    std::vector<uint64_t> path_keys;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
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

int MPIIOBackend::open(const char *filename) {
  return openWithPlan(filename, nullptr);
}

int MPIIOBackend::openWithPlan(const char *filename,
                               const ompfile::OmpFileIOPlan *plan) {
  if (strict_mpp_init_failed)
    return failStrictMpp("open");
  int file_id = getNextFileHandle();
  const bool remote_only = mpp_remote_only;
  const bool have_open_plan = plan && plan->Status == 0 &&
                              plan->AggregatorRank >= 0;

  io_log("Opening file %s with file_id %d\n", filename, file_id);
  io_trace("MPIIOBackend::open enter this=%p file_id=%d remote_only=%d "
           "filename=%s have_plan=%d aggregator=%d remote_handle=%d\n",
           static_cast<void *>(this), file_id, static_cast<int>(remote_only),
           filename ? filename : "(null)", static_cast<int>(have_open_plan),
           have_open_plan ? plan->AggregatorRank : -1,
           have_open_plan ? plan->RemoteHandle : -1);

  auto openRemoteHandle = [&](int &remote_handle_out, int &owner_rank_out) {
    bool open_ok = false;
    owner_rank_out = -1;
    const std::lock_guard<std::mutex> lock(mpp_call_mutex);
    if (have_open_plan) {
      open_ok = ompfile::mpp::openOnRank(filename, open_flags, 0666,
                                         plan->AggregatorRank,
                                         remote_handle_out);
      if (open_ok)
        owner_rank_out = plan->AggregatorRank;
      if (!open_ok) {
        io_log("Planned MPP open failed for %s on rank %d; falling back to "
               "default open path.\n",
               filename, plan->AggregatorRank);
      }
    }
    if (!open_ok) {
      open_ok = ompfile::mpp::open(filename, open_flags, 0666,
                                   remote_handle_out);
      if (open_ok &&
          !ompfile::mpp::handleOwnerRank(remote_handle_out, owner_rank_out)) {
        io_log("MPP open owner-rank query failed for %s (handle=%d errno=%d)\n",
               filename ? filename : "(null)", remote_handle_out, errno);
        open_ok = false;
        if (errno == 0)
          errno = ENOKEY;
      }
    }
    return open_ok;
  };

  if (remote_only) {
    int remote_handle = -1;
    int owner_rank = -1;
    const bool open_ok = openRemoteHandle(remote_handle, owner_rank);
    if (!open_ok) {
      io_log("MPP open failed for %s\n", filename);
      errno = EIO;
      return -1;
    }
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      remote_file_handle_map[file_id] = remote_handle;
      remote_file_owner_rank_map[file_id] = owner_rank;
      remote_read_handle_cache[file_id].clear();
      logical_handle_set.insert(file_id);
      traceHandleStateLocked("open.remote_only.insert", file_id, remote_handle);
    }
    io_log("Remote-only open completed for file_id %d\n", file_id);
    rememberFilePath(file_id, filename);
    rememberFilePathKey(file_id, filename);
    traceHandleState("open.remote_only.done", file_id, remote_handle);
    return file_id;
  }

  MPI_File file_handle;
  int ret = MPI_File_open(file_comm, filename, mpi_open_mode, MPI_INFO_NULL,
                          &file_handle);
  if (ret != MPI_SUCCESS) {
    char err_str[MPI_MAX_ERROR_STRING];
    int err_len = 0;
    MPI_Error_string(ret, err_str, &err_len);
    io_log("MPI_File_open failed for %s: %.*s (code %d)\n", filename,
           err_len, err_str, ret);
    errno = EIO;
    return -1;
  }
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    file_handle_map[file_id] = file_handle;
    remote_read_handle_cache[file_id].clear();
    logical_handle_set.insert(file_id);
    traceHandleStateLocked("open.mpi.insert", file_id, -1);
  }

  if (mpp_open_enabled) {
    int remote_handle = -1;
    int owner_rank = -1;
    const bool open_ok = openRemoteHandle(remote_handle, owner_rank);
    if (!open_ok) {
      io_log("MPP open failed for %s\n", filename);
      MPI_File_close(&file_handle);
      {
        const std::lock_guard<std::mutex> lock(handle_mutex);
        file_handle_map.erase(file_id);
        logical_handle_set.erase(file_id);
        traceHandleStateLocked("open.mpp.fail.cleanup", file_id, -1);
      }
      errno = EIO;
      return -1;
    }
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      remote_file_handle_map[file_id] = remote_handle;
      remote_file_owner_rank_map[file_id] = owner_rank;
      traceHandleStateLocked("open.mpp.remote.insert", file_id, remote_handle);
    }
  }
  rememberFilePath(file_id, filename);
  rememberFilePathKey(file_id, filename);
  traceHandleState("open.done", file_id, -1);
  return file_id;
}

int MPIIOBackend::write(int file_id, const void *data, size_t size) {
  if (strict_mpp_init_failed)
    return failStrictMpp("write");
  if (mpp_open_enabled && mpp_io_enabled) {
    io_log("Error: write() without explicit offset is unsupported in "
           "remote-only MPP mode.\n");
    errno = ENOTSUP;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    errno = EOVERFLOW;
    return -1;
  }
  invalidateTwoPhaseReadCacheForFile(file_id);

  io_log("Writing %zu bytes to file %d\n", size, file_id);

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    file = it->second;
  }

  MPI_Status status{};
  const int ret = MPI_File_write(file, data, static_cast<int>(size), MPI_BYTE,
                                 &status);

  if (ret != MPI_SUCCESS) {
    io_log("Error: Write failed\n");
    return -1;
  }

  int count = 0;
  if (MPI_Get_count(&status, MPI_BYTE, &count) != MPI_SUCCESS || count < 0) {
    errno = EIO;
    return -1;
  }
  const size_t bytes_written = static_cast<size_t>(count);
  if (bytes_written > size) {
    errno = EPROTO;
    return -1;
  }
  if (bytes_written < size) {
    const size_t short_bytes = size - bytes_written;
    short_write_count.fetch_add(1, std::memory_order_relaxed);
    short_write_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
    errno = EIO;
    return -1;
  }

  io_log("Write completed\n");

  return 0;
}

int MPIIOBackend::close(int file_id) {
  if (strict_mpp_init_failed)
    return failStrictMpp("close");
  io_log("Closing file %d\n", file_id);
  std::vector<int> rebalanced_handles;
  uint64_t close_path_key = 0;
  const bool remote_only = mpp_remote_only;
  int remote_handle = -1;
  bool has_remote_handle = false;
  MPI_File mpi_file = MPI_FILE_NULL;
  io_trace("MPIIOBackend::close enter this=%p file_id=%d remote_only=%d\n",
           static_cast<void *>(this), file_id, static_cast<int>(remote_only));

  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    if (logical_handle_set.find(file_id) == logical_handle_set.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      traceHandleStateLocked("close.invalid_handle", file_id, -1);
      errno = EBADF;
      return -1;
    }

    auto path_key_it = file_path_key_map.find(file_id);
    if (path_key_it != file_path_key_map.end())
      close_path_key = path_key_it->second;

    auto read_cache_it = remote_read_handle_cache.find(file_id);
    if (read_cache_it != remote_read_handle_cache.end()) {
      for (const auto &entry : read_cache_it->second)
        rebalanced_handles.push_back(entry.second);
      remote_read_handle_cache.erase(read_cache_it);
    }

    file_path_map.erase(file_id);
    file_path_key_map.erase(file_id);
    file_write_epoch_history.erase(file_id);

    if (mpp_open_enabled) {
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end()) {
        remote_handle = it->second;
        has_remote_handle = true;
      } else {
        remote_file_owner_rank_map.erase(file_id);
      }
    }

    if (!remote_only) {
      auto it = file_handle_map.find(file_id);
      if (it != file_handle_map.end()) {
        mpi_file = it->second;
        file_handle_map.erase(it);
      }
    }

    logical_handle_set.erase(file_id);
    traceHandleStateLocked("close.after_local_erase", file_id, remote_handle);
  }

  invalidateTwoPhaseReadCacheForFile(file_id);

  int mpp_ret = 0;
  if (has_remote_handle) {
    bool close_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      close_ok = ompfile::mpp::close(remote_handle);
    }
    if (!close_ok) {
      io_log("MPP close failed for file %d\n", file_id);
      mpp_ret = -1;
    } else {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end() && it->second == remote_handle)
        remote_file_handle_map.erase(it);
      remote_file_owner_rank_map.erase(file_id);
    }
  }

  for (int cached_handle : rebalanced_handles) {
    if (cached_handle < 0)
      continue;
    if (has_remote_handle && cached_handle == remote_handle)
      continue;
    bool close_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      close_ok = ompfile::mpp::close(cached_handle);
    }
    if (!close_ok) {
      io_log("MPP close failed for cached rebalanced handle (file %d, handle=%d)\n",
             file_id, cached_handle);
      mpp_ret = -1;
    }
  }

  if (remote_only) {
    if (mpp_ret != 0)
      return -1;
    if (close_path_key != 0) {
      int source_rank = -1;
      uint64_t flushed_version = 0;
      (void)ompfile::mpp::flushDirtyTile(close_path_key, source_rank,
                                         flushed_version);
    }
    io_log("Remote-only close completed\n");
    traceHandleState("close.remote_only.done", file_id, remote_handle);
    return 0;
  }

  if (mpi_file == MPI_FILE_NULL) {
    io_log("Error: Missing MPI file handle for file %d\n", file_id);
    return -1;
  }

  int ret = MPI_File_close(&mpi_file);

  if (ret != MPI_SUCCESS) {
    io_log("Error: Close failed\n");
    return -1;
  }

  if (close_path_key != 0) {
    int source_rank = -1;
    uint64_t flushed_version = 0;
    (void)ompfile::mpp::flushDirtyTile(close_path_key, source_rank,
                                       flushed_version);
  }

  if (mpp_ret != 0)
    return -1;

  io_log("Close completed\n");
  traceHandleState("close.done", file_id, remote_handle);

  return 0;
}

int MPIIOBackend::read(int file_id, void *data, size_t size) {
  if (strict_mpp_init_failed)
    return failStrictMpp("read");
  if (mpp_open_enabled && mpp_io_enabled) {
    io_log("Error: read() without explicit offset is unsupported in "
           "remote-only MPP mode.\n");
    errno = ENOTSUP;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    errno = EOVERFLOW;
    return -1;
  }

  io_log("Reading %zu bytes from file %d\n", size, file_id);

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    file = it->second;
  }

  MPI_Status status{};
  const int ret =
      MPI_File_read(file, data, static_cast<int>(size), MPI_BYTE, &status);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Read failed\n");
    errno = EIO;
    return -1;
  }

  int count = 0;
  if (MPI_Get_count(&status, MPI_BYTE, &count) != MPI_SUCCESS || count < 0) {
    errno = EIO;
    return -1;
  }
  const size_t bytes_read = static_cast<size_t>(count);
  if (bytes_read > size) {
    errno = EPROTO;
    return -1;
  }
  if (bytes_read < size && data) {
    const size_t short_bytes = size - bytes_read;
    short_read_count.fetch_add(1, std::memory_order_relaxed);
    short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
    std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
  }

  io_log("Read completed\n");

  return 0;
}

int MPIIOBackend::seek(int file_id, long offset) {
  if (strict_mpp_init_failed)
    return failStrictMpp("seek");
  if (mpp_open_enabled && mpp_io_enabled) {
    io_log("Error: seek() is unsupported in remote-only MPP mode.\n");
    errno = ENOTSUP;
    return -1;
  }

  io_log("Setting file pointer to offset %ld\n", offset);

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    file = it->second;
  }

  int ret = MPI_File_seek(file, offset, MPI_SEEK_SET);

  if (ret != MPI_SUCCESS) {
    io_log("Error: Seek failed\n");
    return -1;
  }

  io_log("Seek completed\n");

  return 0;
}

int MPIIOBackend::readAt(int file_id, long offset, void *data, size_t size) {
  ompfile::OmpFileReadRequestContext context{};
  context.FileHandle = file_id;
  context.Offset = static_cast<int64_t>(offset);
  context.Size = static_cast<uint64_t>(size);
  return readAtWithContext(context, data, size);
}

bool MPIIOBackend::hasUsablePlannedRead(
    const ompfile::OmpFileReadRequestContext &context) const {
  if ((context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PLAN) == 0)
    return false;
  if (context.Plan.Status != 0)
    return false;
  if (!mpp_open_enabled)
    return false;
  return true;
}

int MPIIOBackend::readAtWithContext(
    const ompfile::OmpFileReadRequestContext &context, void *data,
    size_t size) {
  if (strict_mpp_init_failed)
    return failStrictMpp("pread");
  if (context.Offset <
          static_cast<int64_t>(std::numeric_limits<long>::min()) ||
      context.Offset >
          static_cast<int64_t>(std::numeric_limits<long>::max())) {
    errno = EOVERFLOW;
    return -1;
  }
  if (context.Size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    errno = EOVERFLOW;
    return -1;
  }

  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  assert(size == static_cast<size_t>(context.Size) || context.Size == 0);
  if (offset < 0) {
    errno = EINVAL;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }
  pread_request_count.fetch_add(1, std::memory_order_relaxed);
  recordForecastRead(context.Hint, size);
  io_trace("MPIIOBackend::readAtWithContext enter this=%p req_id=%llu file=%d "
           "offset=%ld size=%zu ctx_flags=0x%x has_plan=%d has_path_key=%d "
           "plan_status=%d aggregator=%d remote_handle=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(context.RequestId), file_id, offset,
           size, context.ContextFlags,
           (context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PLAN) != 0,
           (context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY) != 0,
           context.Plan.Status, context.Plan.AggregatorRank,
           context.Plan.RemoteHandle);

  io_log("Reading %zu bytes from file %d at offset %lld\n", size, file_id,
         static_cast<long long>(offset));

  if (isTwoPhaseActive())
    return readAtTwoPhase(context, data, size);

  if (hasUsablePlannedRead(context)) {
    int remote_handle = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end())
        remote_handle = it->second;
    }
    if (remote_handle >= 0) {
      remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
      remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);
      io_log("Phase 1 planned read route: request_id=%llu file=%d "
             "path_key=%llu aggregator_rank=%d\n",
             static_cast<unsigned long long>(context.RequestId), file_id,
             static_cast<unsigned long long>(context.PathKey),
             context.Plan.AggregatorRank);
      bool pread_ok = false;
      size_t bytes_read = 0;
      {
        const std::lock_guard<std::mutex> lock(mpp_call_mutex);
        pread_ok =
            ompfile::mpp::preadEx(remote_handle, offset, data, size, bytes_read);
      }
      if (!pread_ok) {
        io_log("MPP pread failed for file %d\n", file_id);
        return -1;
      }
      if (bytes_read > size) {
        errno = EPROTO;
        return -1;
      }
      if (bytes_read < size && data) {
        const size_t short_bytes = size - bytes_read;
        short_read_count.fetch_add(1, std::memory_order_relaxed);
        short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
        std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
        io_trace("MPIIOBackend::readAtWithContext short-read req_id=%llu "
                 "file=%d requested=%zu read=%zu short=%zu\n",
                 static_cast<unsigned long long>(context.RequestId), file_id,
                 size, bytes_read, short_bytes);
      }

      io_log("MPP read at offset completed.\n");
      io_trace("MPIIOBackend::readAtWithContext planned read success req_id=%llu "
               "file=%d remote_handle=%d\n",
               static_cast<unsigned long long>(context.RequestId), file_id,
               remote_handle);
      return 0;
    }
    io_log("Phase 1 planner produced a route but remote handle is missing for "
           "file %d; falling back to baseline pread path.\n",
           file_id);
    traceHandleState("readAtWithContext.plan_missing_remote", file_id, -1);
  }

  if (two_phase_enabled)
    two_phase_fallback_count.fetch_add(1, std::memory_order_relaxed);

  io_trace("MPIIOBackend::readAtWithContext fallback req_id=%llu file=%d "
           "offset=%ld size=%zu\n",
           static_cast<unsigned long long>(context.RequestId), file_id, offset,
           size);
  size_t bytes_read = 0;
  return readAtFallbackWithBytes(file_id, offset, data, size, bytes_read,
                                 &context.Hint);
}

int MPIIOBackend::readAtFallback(int file_id, long offset, void *data,
                                 size_t size) {
  size_t bytes_read = 0;
  return readAtFallbackWithBytes(file_id, offset, data, size, bytes_read);
}

int MPIIOBackend::readAtFallbackWithBytes(int file_id, long offset, void *data,
                                           size_t size, size_t &bytes_read,
                                           const ompfile::OmpFileIOHint *hint,
                                           bool force_no_stage) {
  bytes_read = 0;
  if (offset < 0) {
    errno = EINVAL;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }

  // Check if the file handle is valid
  if (mpp_io_enabled) {
    int remote_handle = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end())
        remote_handle = it->second;
    }
    if (remote_handle < 0) {
      io_log("Error: Missing remote handle for file %d\n", file_id);
      traceHandleState("readAtFallback.mpp_io.missing_remote", file_id,
                       remote_handle);
      errno = EBADF;
      return -1;
    }
    remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
    remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);
    bool pread_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      if (hint && shouldUseNoStageFallback(*hint, force_no_stage)) {
        pread_ok = ompfile::mpp::preadNoStageEx(remote_handle, offset, data,
                                                size, bytes_read);
      } else {
        pread_ok =
            ompfile::mpp::preadEx(remote_handle, offset, data, size, bytes_read);
      }
    }
    if (!pread_ok) {
      io_log("MPP pread failed for file %d\n", file_id);
      traceHandleState("readAtFallback.mpp_io.pread_fail", file_id,
                       remote_handle);
      return -1;
    }
    if (bytes_read > size) {
      errno = EPROTO;
      return -1;
    }
    if (bytes_read < size && data) {
      const size_t short_bytes = size - bytes_read;
      short_read_count.fetch_add(1, std::memory_order_relaxed);
      short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
      std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
    }

    io_log("MPP read at offset completed.\n");
    return 0;
  }

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    file = it->second;
  }

  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    errno = EOVERFLOW;
    return -1;
  }

  MPI_Status status{};
  const int ret = MPI_File_read_at(file, offset, data, static_cast<int>(size),
                                   MPI_BYTE, &status);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Read at offset failed\n");
    return -1;
  }
  int count = 0;
  if (MPI_Get_count(&status, MPI_BYTE, &count) != MPI_SUCCESS || count < 0) {
    errno = EIO;
    return -1;
  }

  bytes_read = static_cast<size_t>(count);
  if (bytes_read > size) {
    errno = EPROTO;
    return -1;
  }
  if (bytes_read < size && data) {
    const size_t short_bytes = size - bytes_read;
    short_read_count.fetch_add(1, std::memory_order_relaxed);
    short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
    std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
  }

  io_log("Read at offset completed\n");
  return 0;
}

int MPIIOBackend::readAtTwoPhase(
    const ompfile::OmpFileReadRequestContext &context, void *data,
    size_t size) {
  // Keep zero-length reads on the validated fallback path so invalid handles
  // still fail consistently with the non-cached pread flow.
  if (size == 0)
    return readAtFallback(context.FileHandle, static_cast<long>(context.Offset),
                          data, size);

  const uint64_t request_key = resolveTwoPhaseKey(context);
  const long request_offset = static_cast<long>(context.Offset);
  if (!mpp_remote_only &&
      tryServeTwoPhaseReadCache(request_key, request_offset, data, size)) {
    io_trace("MPIIOBackend::readAtTwoPhase cache-hit file=%d offset=%ld "
             "size=%zu key=%llu\n",
             context.FileHandle, request_offset, size,
             static_cast<unsigned long long>(request_key));
    return 0;
  }

  TwoPhaseReadRequest request{};
  request.DebugRequestId =
      two_phase_request_id.fetch_add(1, std::memory_order_relaxed);
  request.FileHandle = context.FileHandle;
  request.ClientRank = context.ClientRank;
  request.Offset = request_offset;
  request.Size = size;
  request.Buffer = data;
  request.PathKey = request_key;
  request.HasPathKey = true;
  request.Hint = context.Hint;

  const auto enqueue_ts = std::chrono::steady_clock::now();
  std::vector<TwoPhaseReadRequest *> batch;
  std::unique_lock<std::mutex> lock(two_phase_mutex);
  const size_t queue_size_before = two_phase_queue.size();
  updateAtomicMax(two_phase_queue_max_depth,
                  static_cast<uint64_t>(queue_size_before + 1));
  io_trace("MPIIOBackend::readAtTwoPhase enqueue req=%llu file=%d offset=%ld "
           "size=%zu queue_before=%zu in_progress=%d\n",
           static_cast<unsigned long long>(request.DebugRequestId),
           request.FileHandle, request.Offset, request.Size, queue_size_before,
           static_cast<int>(two_phase_batch_in_progress));
  two_phase_queue.push_back(&request);
  two_phase_queue_cv.notify_all();

  while (!request.Done) {
    if (!two_phase_batch_in_progress) {
      two_phase_batch_in_progress = true;
      two_phase_leader_turn_count.fetch_add(1, std::memory_order_relaxed);
      io_trace("MPIIOBackend::readAtTwoPhase leader req=%llu queue_size=%zu "
               "window_us=%llu\n",
               static_cast<unsigned long long>(request.DebugRequestId),
               two_phase_queue.size(),
               static_cast<unsigned long long>(two_phase_window_us));

      if (two_phase_window_us > 0) {
        const auto window_wait_begin = std::chrono::steady_clock::now();
        two_phase_queue_cv.wait_for(lock,
                                    std::chrono::microseconds(two_phase_window_us));
        const auto window_wait_end = std::chrono::steady_clock::now();
        const auto window_wait_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                window_wait_end - window_wait_begin)
                .count();
        if (window_wait_us > 0) {
          two_phase_window_wait_us_total.fetch_add(
              static_cast<uint64_t>(window_wait_us),
              std::memory_order_relaxed);
        }
      }

      while (!two_phase_queue.empty()) {
        batch.push_back(two_phase_queue.front());
        two_phase_queue.pop_front();
      }
      io_trace("MPIIOBackend::readAtTwoPhase leader req=%llu draining batch "
               "size=%zu queue_after_drain=%zu\n",
               static_cast<unsigned long long>(request.DebugRequestId),
               batch.size(), two_phase_queue.size());

      lock.unlock();
      const auto batch_exec_begin = std::chrono::steady_clock::now();
      processTwoPhaseBatch(batch);
      const auto batch_exec_end = std::chrono::steady_clock::now();
      const auto batch_exec_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              batch_exec_end - batch_exec_begin)
              .count();
      if (batch_exec_us > 0) {
        two_phase_batch_exec_us_total.fetch_add(
            static_cast<uint64_t>(batch_exec_us), std::memory_order_relaxed);
      }
      batch.clear();
      lock.lock();

      two_phase_batch_in_progress = false;
      two_phase_queue_cv.notify_all();
    } else {
      two_phase_follower_wait_count.fetch_add(1, std::memory_order_relaxed);
      io_trace("MPIIOBackend::readAtTwoPhase follower wait req=%llu "
               "queue_size=%zu\n",
               static_cast<unsigned long long>(request.DebugRequestId),
               two_phase_queue.size());
      const auto follower_wait_begin = std::chrono::steady_clock::now();
      two_phase_queue_cv.wait(
          lock, [this, &request] { return request.Done || !two_phase_batch_in_progress; });
      const auto follower_wait_end = std::chrono::steady_clock::now();
      const auto follower_wait_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              follower_wait_end - follower_wait_begin)
              .count();
      if (follower_wait_us > 0) {
        two_phase_follower_wait_us_total.fetch_add(
            static_cast<uint64_t>(follower_wait_us),
            std::memory_order_relaxed);
      }
    }
  }

  const auto done_ts = std::chrono::steady_clock::now();
  const auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           done_ts - enqueue_ts)
                           .count();
  if (wait_us > 0) {
    two_phase_request_wait_us_total.fetch_add(static_cast<uint64_t>(wait_us),
                                              std::memory_order_relaxed);
  }
  if (request.Status != 0) {
    errno = request.Errno;
    io_trace("MPIIOBackend::readAtTwoPhase done req=%llu status=%d errno=%d "
             "wait_us=%lld\n",
             static_cast<unsigned long long>(request.DebugRequestId),
             request.Status, request.Errno, static_cast<long long>(wait_us));
    return -1;
  }

  io_trace("MPIIOBackend::readAtTwoPhase done req=%llu status=0 wait_us=%lld\n",
           static_cast<unsigned long long>(request.DebugRequestId),
           static_cast<long long>(wait_us));
  return 0;
}

void MPIIOBackend::processTwoPhaseBatch(std::vector<TwoPhaseReadRequest *> &batch) {
  if (batch.empty())
    return;

  io_trace("MPIIOBackend::processTwoPhaseBatch enter this=%p batch_size=%zu\n",
           static_cast<void *>(this), batch.size());
  two_phase_batch_count.fetch_add(1, std::memory_order_relaxed);

  std::unordered_map<uint64_t, std::vector<TwoPhaseReadRequest *>> grouped;
  grouped.reserve(batch.size());
  for (TwoPhaseReadRequest *request : batch)
    grouped[getTwoPhaseGroupKey(*request)].push_back(request);

  io_trace("MPIIOBackend::processTwoPhaseBatch grouped=%zu\n", grouped.size());

  for (auto &entry : grouped)
    processTwoPhaseGroup(entry.second);
}

void MPIIOBackend::processTwoPhaseGroup(std::vector<TwoPhaseReadRequest *> &group) {
  if (group.empty())
    return;

  io_trace("MPIIOBackend::processTwoPhaseGroup enter this=%p group_size=%zu "
           "first_req=%llu first_file=%d first_offset=%ld first_size=%zu\n",
           static_cast<void *>(this), group.size(),
           static_cast<unsigned long long>(group.front()->DebugRequestId),
           group.front()->FileHandle, group.front()->Offset,
           group.front()->Size);

  // Debug: log all request offsets before sorting
  for (size_t dbg_i = 0; dbg_i < group.size(); ++dbg_i) {
    io_log("  [pre-sort] req[%zu] offset=%ld size=%zu client_rank=%d\n",
           dbg_i, group[dbg_i]->Offset, group[dbg_i]->Size,
           group[dbg_i]->ClientRank);
  }

  std::sort(group.begin(), group.end(),
            [](const TwoPhaseReadRequest *lhs, const TwoPhaseReadRequest *rhs) {
              if (lhs->Offset != rhs->Offset)
                return lhs->Offset < rhs->Offset;
              return lhs->Size < rhs->Size;
            });

  struct CoalescedRead {
    long Start = 0;
    long End = 0;
    std::vector<TwoPhaseReadRequest *> Requests;
    int PlannedAggregatorRank = -1;
    bool PlannerRebalanced = false;
    bool PlannerForceFallback = false;
    int PlannerFallbackErrno = 0;
    bool Skip = false;
    int ErrorStatus = 0;
    int ErrorErrno = 0;
  };

  std::vector<CoalescedRead> coalesced;
  coalesced.reserve(group.size());
  for (TwoPhaseReadRequest *request : group) {
    if (request->Offset < 0) {
      completeTwoPhaseRequest(*request, -1, EINVAL);
      continue;
    }
    if (request->Size >
            static_cast<size_t>(std::numeric_limits<long>::max()) ||
        request->Offset >
            std::numeric_limits<long>::max() -
                static_cast<long>(request->Size)) {
      completeTwoPhaseRequest(*request, -1, EOVERFLOW);
      continue;
    }

    const long start = request->Offset;
    const long end = start + static_cast<long>(request->Size);
    if (coalesced.empty()) {
      CoalescedRead item{};
      item.Start = start;
      item.End = end;
      item.Requests.push_back(request);
      coalesced.push_back(std::move(item));
      continue;
    }

    CoalescedRead &tail = coalesced.back();
    const bool contiguous = start <= tail.End;
    const long merged_end = std::max(tail.End, end);
    const uint64_t merged_size =
        static_cast<uint64_t>(merged_end - tail.Start);
    const bool exceeds_max_batch =
        two_phase_max_batch_bytes > 0 && merged_size > two_phase_max_batch_bytes;

    if (contiguous && !exceeds_max_batch) {
      tail.End = merged_end;
      tail.Requests.push_back(request);
      continue;
    }

    CoalescedRead item{};
    item.Start = start;
    item.End = end;
    item.Requests.push_back(request);
    coalesced.push_back(std::move(item));
  }
  if (coalesced.empty())
    return;

  // Debug: log coalesced ranges after grouping
  for (size_t dbg_i = 0; dbg_i < coalesced.size(); ++dbg_i) {
    io_log("  [coalesced] range[%zu] start=%ld end=%ld requests=%zu\n",
           dbg_i, coalesced[dbg_i].Start, coalesced[dbg_i].End,
           coalesced[dbg_i].Requests.size());
  }

  ompfile::OmpFileIOBatchRequest batch_request{};
  batch_request.AbiVersion = ompfile::OMPFILE_SCHED_BATCH_ABI_VERSION;
  batch_request.SegmentCount = static_cast<uint32_t>(coalesced.size());
  batch_request.RequestFlags =
      ompfile::OMPFILE_BATCH_REQ_FAIL_ON_ANY_ERROR |
      ompfile::OMPFILE_BATCH_REQ_DISABLE_SCALAR_FALLBACK;
  batch_request.BatchId =
      two_phase_planner_batch_id.fetch_add(1, std::memory_order_relaxed);
  std::vector<ompfile::OmpFileIOBatchSegment> batch_segments;
  batch_segments.reserve(coalesced.size());
  for (size_t i = 0; i < coalesced.size(); ++i) {
    const CoalescedRead &item = coalesced[i];
    ompfile::OmpFileIOBatchSegment segment{};
    segment.SegmentId = static_cast<uint64_t>(i);
    segment.FileHandle = item.Requests.front()->FileHandle;
    segment.ClientRank = item.Requests.front()->ClientRank;
    segment.Offset = item.Start;
    segment.Size = static_cast<uint64_t>(computeForecastTargetReadSize(
        item.Requests.front()->Hint, item.Start, item.End,
        item.Requests.size(), mpp_remote_only, two_phase_sieve_bytes,
        two_phase_max_batch_bytes));
    segment.PathKey = item.Requests.front()->HasPathKey
                          ? item.Requests.front()->PathKey
                          : static_cast<uint64_t>(0);
    segment.SegmentFlags = item.Requests.front()->Hint.HintFlags;
    segment.EpochId = item.Requests.front()->Hint.EpochId;
    segment.StreamId = item.Requests.front()->Hint.StreamId;
    segment.TileId = item.Requests.front()->Hint.TileId;
    batch_segments.push_back(segment);
  }

  ompfile::OmpFileIOBatchPlan batch_plan{};
  std::vector<ompfile::OmpFileIOBatchPlanEntry> batch_entries;
  io_trace("MPIIOBackend::processTwoPhaseGroup coalesced_ranges=%zu "
           "batch_id=%llu\n",
           coalesced.size(),
           static_cast<unsigned long long>(batch_request.BatchId));
  if (ompfile::mpp::schedBatchRequest(batch_request, batch_segments, batch_plan,
                                      batch_entries)) {
    two_phase_planner_batch_count.fetch_add(1, std::memory_order_relaxed);
    two_phase_planner_segment_count.fetch_add(batch_entries.size(),
                                              std::memory_order_relaxed);
    if ((batch_plan.PlanFlags & ompfile::OMPFILE_BATCH_PLAN_SCALAR_FALLBACK) !=
        0) {
      two_phase_planner_scalar_fallback_count.fetch_add(
          1, std::memory_order_relaxed);
    }

    if (batch_entries.size() != coalesced.size()) {
      two_phase_planner_error_count.fetch_add(1, std::memory_order_relaxed);
      io_log("Two-phase planner returned mismatched entry count: expected=%zu "
             "actual=%zu\n",
             coalesced.size(), batch_entries.size());
    } else {
      io_trace("MPIIOBackend::processTwoPhaseGroup planner success batch_id=%llu "
               "entries=%zu flags=0x%x status=%d\n",
               static_cast<unsigned long long>(batch_request.BatchId),
               batch_entries.size(), batch_plan.PlanFlags, batch_plan.Status);
      for (size_t i = 0; i < batch_entries.size(); ++i) {
        const ompfile::OmpFileIOBatchPlanEntry &entry = batch_entries[i];
        CoalescedRead &item = coalesced[i];
        if ((entry.PlanFlags & ompfile::OMPFILE_BATCH_PLAN_FILE_AFFINITY) != 0)
          two_phase_planner_affinity_count.fetch_add(1,
                                                     std::memory_order_relaxed);
        if ((entry.PlanFlags & ompfile::OMPFILE_BATCH_PLAN_REBALANCED) != 0)
          two_phase_planner_rebalanced_count.fetch_add(
              1, std::memory_order_relaxed);
        if ((entry.PlanFlags & ompfile::OMPFILE_BATCH_PLAN_REBALANCED) != 0) {
          item.PlannerRebalanced = true;
          item.PlannedAggregatorRank = entry.AggregatorRank;
        }
        if ((entry.PlanFlags & ompfile::OMPFILE_BATCH_PLAN_REBALANCED) != 0) {
          io_trace("MPIIOBackend::processTwoPhaseGroup planner rebalanced "
                   "segment batch_id=%llu segment_id=%llu aggregator=%d; "
                   "continuing on the opened-handle owner path\n",
                   static_cast<unsigned long long>(batch_request.BatchId),
                   static_cast<unsigned long long>(entry.SegmentId),
                   entry.AggregatorRank);
        }
        if (entry.Status != 0) {
          const int entry_errno = entry.Errno != 0 ? entry.Errno : EIO;
          if (item.PlannerRebalanced &&
              (entry_errno == ESTALE || entry_errno == ENOKEY ||
               entry_errno == EAGAIN)) {
            item.PlannerForceFallback = true;
            item.PlannerFallbackErrno = entry_errno;
            io_trace("MPIIOBackend::processTwoPhaseGroup planner requested "
                     "owner fallback segment=%llu errno=%d\n",
                     static_cast<unsigned long long>(entry.SegmentId),
                     entry_errno);
          } else {
            item.Skip = true;
            item.ErrorStatus = -1;
            item.ErrorErrno = entry_errno;
          }
        }
      }
    }
  } else {
    two_phase_planner_error_count.fetch_add(1, std::memory_order_relaxed);
    io_trace("MPIIOBackend::processTwoPhaseGroup planner failed batch_id=%llu\n",
             static_cast<unsigned long long>(batch_request.BatchId));
  }

  two_phase_coalesced_read_count.fetch_add(coalesced.size(),
                                           std::memory_order_relaxed);

  for (CoalescedRead &item : coalesced) {
    if (item.Skip) {
      io_trace("MPIIOBackend::processTwoPhaseGroup skipping range start=%ld "
               "end=%ld requests=%zu status=%d errno=%d\n",
               item.Start, item.End, item.Requests.size(), item.ErrorStatus,
               item.ErrorErrno);
      for (TwoPhaseReadRequest *request : item.Requests)
        completeTwoPhaseRequest(*request, item.ErrorStatus, item.ErrorErrno);
      continue;
    }

    const size_t read_size = computeForecastTargetReadSize(
        item.Requests.front()->Hint, item.Start, item.End,
        item.Requests.size(), mpp_remote_only, two_phase_sieve_bytes,
        two_phase_max_batch_bytes);
    two_phase_coalesced_bytes_total.fetch_add(read_size,
                                              std::memory_order_relaxed);
    std::vector<char> buffer(read_size);
    size_t bytes_read = 0;

    int rc = -1;
    bool remote_rebalanced_ok = false;
    int read_errno = 0;
    const int file_id = item.Requests.front()->FileHandle;
    int source_affinity_rank = -1;
    bool force_no_stage_fallback = false;
    const bool source_affinity_candidate = isSourceStageAffinityEligible(
        item.Requests.front()->Hint, item.Requests.front()->HasPathKey,
        mpp_remote_only);
    if (source_affinity_candidate) {
      stage_affinity_source_candidate_count.fetch_add(
          1, std::memory_order_relaxed);
      stage_affinity_source_candidate_bytes.fetch_add(
          read_size, std::memory_order_relaxed);
    }
    const bool source_affinity_safe = canAttemptSourceStageAffinityRead(
        item.Requests.front()->Hint, file_id, item.Start, read_size,
        item.Requests.front()->HasPathKey, mpp_remote_only,
        item.PlannerForceFallback);
    if (source_affinity_safe) {
      if (sourceStageAffinityPreReadInvalidateForFile(file_id)) {
        bool ok = readAtSourceStageAffinityRankWithBytes(
            file_id, item.Requests.front()->PathKey,
            item.Requests.front()->Hint, item.Start, buffer.data(), read_size,
            bytes_read, source_affinity_rank);
        if (ok) {
          rc = 0;
          stage_affinity_source_applied_count.fetch_add(
              1, std::memory_order_relaxed);
          stage_affinity_source_applied_bytes.fetch_add(
              read_size, std::memory_order_relaxed);
        }
      } else {
        force_no_stage_fallback = true;
      }
    }
    if (source_affinity_candidate && rc != 0) {
      stage_affinity_source_fallback_count.fetch_add(
          1, std::memory_order_relaxed);
    }

    if (shouldAttemptPlannerRebalancedRead(
            rc, force_no_stage_fallback, item.PlannerRebalanced,
            item.PlannedAggregatorRank)) {
      const char *fallback_reason = "none";
      int fallback_errno = 0;
      bool eligible_for_rebalance = false;
      if (item.PlannerForceFallback) {
        const int planner_errno =
            item.PlannerFallbackErrno != 0 ? item.PlannerFallbackErrno : ESTALE;
        if (!plannerStatusForcesFallback(planner_errno, fallback_reason,
                                         fallback_errno)) {
          fallback_reason = "scheduler-version-mismatch";
          fallback_errno = planner_errno;
        }
      } else {
        eligible_for_rebalance =
            canApplyRebalancedRead(item.Requests.front()->Hint, file_id,
                                   item.Start, read_size, fallback_reason,
                                   fallback_errno);
      }

      if (eligible_for_rebalance) {
        remote_rebalanced_ok =
            readAtRemoteRankWithBytes(file_id, item.PlannedAggregatorRank,
                                      item.Start, buffer.data(), read_size,
                                      bytes_read);
        rc = remote_rebalanced_ok ? 0 : -1;
      } else {
        rc = -1;
        read_errno = fallback_errno != 0 ? fallback_errno : EAGAIN;
        two_phase_planner_rebalanced_conflict_count.fetch_add(
            1, std::memory_order_relaxed);
        two_phase_planner_rebalanced_skipped_count.fetch_add(
            1, std::memory_order_relaxed);
        io_log("Rebalanced segment fallback file=%d target_rank=%d offset=%ld "
               "size=%zu reason=%s errno=%d\n",
               file_id, item.PlannedAggregatorRank, item.Start, read_size,
               fallback_reason, read_errno);
      }

      if (remote_rebalanced_ok) {
        two_phase_planner_rebalanced_applied_count.fetch_add(
            1, std::memory_order_relaxed);
        noteAppliedRebalancedReadForFile(file_id);
      } else if (eligible_for_rebalance) {
        read_errno = errno != 0 ? errno : EIO;
        two_phase_planner_rebalanced_skipped_count.fetch_add(
            1, std::memory_order_relaxed);
        io_log("Rebalanced segment fallback file=%d target_rank=%d offset=%ld "
               "size=%zu reason=remote-read-fail errno=%d\n",
               file_id, item.PlannedAggregatorRank, item.Start, read_size,
               read_errno);
      }
    }

    if (rc != 0) {
      rc = readAtFallbackWithBytes(file_id, item.Start, buffer.data(), read_size,
                                   bytes_read, &item.Requests.front()->Hint,
                                   force_no_stage_fallback);
    }
    if (rc != 0) {
      const int errnum = errno;
      io_trace("MPIIOBackend::processTwoPhaseGroup coalesced read failed "
               "start=%ld end=%ld size=%zu rc=%d errno=%d\n",
               item.Start, item.End, read_size, rc, errnum);
      for (TwoPhaseReadRequest *request : item.Requests)
        completeTwoPhaseRequest(*request, rc, errnum);
      continue;
    }

    if (!mpp_remote_only) {
      const uint64_t cache_key = getTwoPhaseGroupKey(*item.Requests.front());
      updateTwoPhaseReadCache(cache_key, item.Start, buffer);
    }

    for (TwoPhaseReadRequest *request : item.Requests) {
      const size_t scatter_offset =
          static_cast<size_t>(request->Offset - item.Start);
      assert(scatter_offset <= buffer.size() &&
             "Scatter offset must stay within coalesced read buffer.");
      if (!request->Buffer && request->Size > 0) {
        completeTwoPhaseRequest(*request, -1, EINVAL);
        continue;
      }
      if (scatter_offset > buffer.size() ||
          request->Size > (buffer.size() - scatter_offset)) {
        completeTwoPhaseRequest(*request, -1, EPROTO);
        continue;
      }
      if (request->Size > 0) {
        std::memcpy(request->Buffer, buffer.data() + scatter_offset,
                    request->Size);
      }
      completeTwoPhaseRequest(*request, 0, 0);
    }
    io_trace("MPIIOBackend::processTwoPhaseGroup coalesced read success "
             "start=%ld end=%ld buffered_end=%ld size=%zu bytes_read=%zu "
             "scattered=%zu\n",
             item.Start, item.End,
             item.Start + static_cast<long>(read_size),
             read_size, bytes_read, item.Requests.size());
  }
}

uint64_t MPIIOBackend::getTwoPhaseGroupKey(
    const TwoPhaseReadRequest &request) const {
  if (request.HasPathKey)
    return request.PathKey;
  return static_cast<uint64_t>(static_cast<uint32_t>(request.FileHandle));
}

uint64_t MPIIOBackend::getWriteBatchGroupKey(
    const WriteBatchRequest &request) const {
  return request.GroupKey;
}

void MPIIOBackend::completeTwoPhaseRequest(TwoPhaseReadRequest &request,
                                           int status, int errnum) {
  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  request.Status = status;
  request.Errno = errnum;
  request.Done = true;
  io_trace("MPIIOBackend::completeTwoPhaseRequest req=%llu file=%d offset=%ld "
           "size=%zu status=%d errno=%d\n",
           static_cast<unsigned long long>(request.DebugRequestId),
           request.FileHandle, request.Offset, request.Size, status, errnum);
  two_phase_queue_cv.notify_all();
}

void MPIIOBackend::completeWriteBatchRequest(WriteBatchRequest &request,
                                             int status, int errnum) {
  const std::lock_guard<std::mutex> lock(write_batch_mutex);
  request.Status = status;
  request.Errno = errnum;
  request.Done = true;
  write_batch_queue_cv.notify_all();
}

bool MPIIOBackend::isTwoPhaseActive() const {
  return two_phase_enabled && mpp_open_enabled && mpp_io_enabled;
}

bool MPIIOBackend::isWriteBatchActive() const {
  return write_batch_enabled && mpp_remote_only;
}

int MPIIOBackend::writeAtRemoteHandle(int remote_handle, long offset,
                                      const void *data, size_t size,
                                      size_t &bytes_written) {
  bytes_written = 0;
  if (size == 0)
    return 0;

  const char *cursor = static_cast<const char *>(data);
  long current_offset = offset;
  size_t remaining = size;

  while (remaining > 0) {
    remote_pwrite_event_count.fetch_add(1, std::memory_order_relaxed);
    remote_pwrite_bytes_total.fetch_add(remaining, std::memory_order_relaxed);

    bool pwrite_ok = false;
    size_t call_bytes_written = 0;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      pwrite_ok = ompfile::mpp::pwriteEx(remote_handle, current_offset, cursor,
                                         remaining, call_bytes_written);
    }
    if (!pwrite_ok) {
      const int saved_errno = errno;
      io_trace("MPIIOBackend::writeAtRemoteHandle pwriteEx fail "
               "remote_handle=%d offset=%ld remaining=%zu cursor=%p errno=%d\n",
               remote_handle, current_offset, remaining,
               static_cast<const void *>(cursor), saved_errno);
      if (errno == EIO && remaining > 0) {
        short_write_count.fetch_add(1, std::memory_order_relaxed);
        short_write_bytes_total.fetch_add(remaining, std::memory_order_relaxed);
      }
      errno = saved_errno;
      return -1;
    }
    if (call_bytes_written > remaining) {
      errno = EPROTO;
      return -1;
    }
    if (call_bytes_written == 0) {
      short_write_count.fetch_add(1, std::memory_order_relaxed);
      short_write_bytes_total.fetch_add(remaining, std::memory_order_relaxed);
      errno = EIO;
      return -1;
    }

    bytes_written += call_bytes_written;
    if (call_bytes_written < remaining) {
      short_write_count.fetch_add(1, std::memory_order_relaxed);
      short_write_bytes_total.fetch_add(remaining - call_bytes_written,
                                        std::memory_order_relaxed);
    }

    if (call_bytes_written >
            static_cast<size_t>(std::numeric_limits<long>::max()) ||
        current_offset > std::numeric_limits<long>::max() -
                             static_cast<long>(call_bytes_written)) {
      errno = EOVERFLOW;
      return -1;
    }
    current_offset += static_cast<long>(call_bytes_written);
    cursor += call_bytes_written;
    remaining -= call_bytes_written;
  }

  assert(bytes_written == size &&
         "writeAtRemoteHandle must account for all requested bytes.");
  return 0;
}

int MPIIOBackend::writeAtWithContext(
    const ompfile::OmpFileWriteRequestContext &context, const void *data,
    size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  uint64_t group_key = static_cast<uint64_t>(static_cast<uint32_t>(file_id));
  if ((context.ContextFlags & ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY) != 0)
    group_key = context.PathKey;
  else {
    uint64_t path_key = 0;
    if (getFilePathKey(file_id, path_key))
      group_key = path_key;
  }
  group_key = mixWriteHintIntoKey(group_key, context.Hint);

  if (strict_mpp_init_failed)
    return failStrictMpp("pwrite");
  const bool remote_only = mpp_remote_only;
  pwrite_request_count.fetch_add(1, std::memory_order_relaxed);
  if (offset < 0) {
    errno = EINVAL;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }
  recordForecastWrite(context.Hint, size);
  invalidateTwoPhaseReadCacheForFile(file_id);

  const auto report_freshness_write_commit = [&]() {
    if ((context.Hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) == 0)
      return true;

    uint64_t path_key = 0;
    if ((context.ContextFlags & ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY) != 0)
      path_key = context.PathKey;
    else if (!getFilePathKey(file_id, path_key))
      return true;

    if (path_key == 0)
      return true;

    int writer_rank = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto owner_it = remote_file_owner_rank_map.find(file_id);
      if (owner_it != remote_file_owner_rank_map.end())
        writer_rank = owner_it->second;
    }
    if (writer_rank < 0) {
      io_log("Freshness write-commit failed: missing remote owner rank for file %d.\n",
             file_id);
      errno = ENOKEY;
      return false;
    }

    const bool write_through_mode = freshnessWriteThroughModeEnabled();
    uint64_t committed_version = 0;
    if (!ompfile::mpp::freshnessWriteCommit(path_key, writer_rank,
                                            context.Hint.TileId,
                                            write_through_mode,
                                            committed_version)) {
      io_log("Freshness write-commit failed path_key=%llu writer=%d tile=%llu "
             "mode=%s errno=%d\n",
             static_cast<unsigned long long>(path_key), writer_rank,
             static_cast<unsigned long long>(context.Hint.TileId),
             write_through_mode ? "write-through" : "write-back", errno);
      return false;
    }

    io_trace("MPIIOBackend::writeAtWithContext freshness write-commit "
             "path_key=%llu writer=%d tile=%llu mode=%s version=%llu\n",
             static_cast<unsigned long long>(path_key), writer_rank,
             static_cast<unsigned long long>(context.Hint.TileId),
             write_through_mode ? "write-through" : "write-back",
             static_cast<unsigned long long>(committed_version));
    return true;
  };

  io_log("Writing %zu bytes to file %d at offset %lld\n", size, file_id,
         static_cast<long long>(offset));

  if (isWriteBatchActive()) {
    const int batched_rc = writeAtBatched(file_id, offset, data, size, group_key);
    if (batched_rc == 0) {
      recordWriteEpochForContext(file_id, offset, size, context.Hint);
      triggerStageInvalidationAfterSuccessfulRemoteWrite(file_id);
      if (!report_freshness_write_commit()) {
        if (errno == 0)
          errno = EIO;
        return -1;
      }
    }
    return batched_rc;
  }

  if (remote_only) {
    int remote_handle = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end())
        remote_handle = it->second;
    }
    if (remote_handle < 0) {
      io_log("Error: Missing remote handle for file %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    size_t bytes_written = 0;
    if (writeAtRemoteHandle(remote_handle, offset, data, size, bytes_written) !=
        0) {
      io_log("MPP pwrite failed for file %d (remote_handle=%d offset=%lld "
             "size=%zu errno=%d)\n",
             file_id, remote_handle, static_cast<long long>(offset), size,
             errno);
      return -1;
    }
    assert(bytes_written == size && "Remote pwrite must complete full request");

    recordWriteEpochForContext(file_id, offset, size, context.Hint);
    triggerStageInvalidationAfterSuccessfulRemoteWrite(file_id);
    if (!report_freshness_write_commit()) {
      if (errno == 0)
        errno = EIO;
      return -1;
    }
    io_log("MPP write at offset completed.\n");
    return 0;
  }

  // Check if the file handle is valid
  if (mpp_io_enabled) {
    int remote_handle = -1;
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end())
        remote_handle = it->second;
    }
    if (remote_handle < 0) {
      io_log("Error: Missing remote handle for file %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    size_t bytes_written = 0;
    if (writeAtRemoteHandle(remote_handle, offset, data, size, bytes_written) !=
        0) {
      io_log("MPP pwrite failed for file %d (remote_handle=%d offset=%lld "
             "size=%zu errno=%d)\n",
             file_id, remote_handle, static_cast<long long>(offset), size,
             errno);
      return -1;
    }
    assert(bytes_written == size && "Remote pwrite must complete full request");

    recordWriteEpochForContext(file_id, offset, size, context.Hint);
    if (!report_freshness_write_commit()) {
      if (errno == 0)
        errno = EIO;
      return -1;
    }
    io_log("MPP write at offset completed.\n");
    return 0;
  }

  MPI_File file = MPI_FILE_NULL;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = file_handle_map.find(file_id);
    if (it == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      errno = EBADF;
      return -1;
    }
    file = it->second;
  }

  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    errno = EOVERFLOW;
    return -1;
  }

  // Perform the actual write at the specified offset
  MPI_Status status{};
  const int ret = MPI_File_write_at(file, offset, data, static_cast<int>(size),
                                    MPI_BYTE, &status);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Write at offset failed\n");
    return -1;
  }

  int count = 0;
  if (MPI_Get_count(&status, MPI_BYTE, &count) != MPI_SUCCESS || count < 0) {
    errno = EIO;
    return -1;
  }
  const size_t bytes_written = static_cast<size_t>(count);
  if (bytes_written > size) {
    errno = EPROTO;
    return -1;
  }
  if (bytes_written < size) {
    const size_t short_bytes = size - bytes_written;
    short_write_count.fetch_add(1, std::memory_order_relaxed);
    short_write_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
    errno = EIO;
    return -1;
  }

  recordWriteEpochForContext(file_id, offset, size, context.Hint);
  io_log("Write at offset completed\n");
  return 0;
}

int MPIIOBackend::writeAt(int file_id, long offset, const void *data,
                          size_t size) {
  ompfile::OmpFileWriteRequestContext context{};
  context.FileHandle = file_id;
  context.Offset = static_cast<int64_t>(offset);
  context.Size = static_cast<uint64_t>(size);
  uint64_t path_key = 0;
  if (getFilePathKey(file_id, path_key)) {
    context.PathKey = path_key;
    context.ContextFlags |= ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY;
  }
  return writeAtWithContext(context, data, size);
}

int MPIIOBackend::writeAtBatched(int file_id, long offset, const void *data,
                                 size_t size, uint64_t group_key) {
  WriteBatchRequest request{};
  request.DebugRequestId =
      write_batch_request_id.fetch_add(1, std::memory_order_relaxed);
  request.FileHandle = file_id;
  request.Offset = offset;
  request.GroupKey = group_key;
  request.Data.resize(size);
  if (size > 0)
    std::memcpy(request.Data.data(), data, size);

  std::vector<WriteBatchRequest *> batch;
  const auto enqueue_ts = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(write_batch_mutex);
  const size_t queue_size_before = write_batch_queue.size();
  updateAtomicMax(write_queue_max_depth,
                  static_cast<uint64_t>(queue_size_before + 1));
  write_batch_queue.push_back(&request);
  write_batch_queue_cv.notify_all();

  while (!request.Done) {
    if (!write_batch_in_progress) {
      write_batch_in_progress = true;
      write_leader_turn_count.fetch_add(1, std::memory_order_relaxed);
      if (write_batch_window_us > 0) {
        const auto window_wait_begin = std::chrono::steady_clock::now();
        write_batch_queue_cv.wait_for(
            lock, std::chrono::microseconds(write_batch_window_us));
        const auto window_wait_end = std::chrono::steady_clock::now();
        const auto window_wait_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                window_wait_end - window_wait_begin)
                .count();
        if (window_wait_us > 0) {
          write_window_wait_us_total.fetch_add(
              static_cast<uint64_t>(window_wait_us), std::memory_order_relaxed);
        }
      }

      while (!write_batch_queue.empty()) {
        batch.push_back(write_batch_queue.front());
        write_batch_queue.pop_front();
      }

      lock.unlock();
      const auto batch_exec_begin = std::chrono::steady_clock::now();
      processWriteBatch(batch);
      const auto batch_exec_end = std::chrono::steady_clock::now();
      const auto batch_exec_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              batch_exec_end - batch_exec_begin)
              .count();
      if (batch_exec_us > 0) {
        write_batch_exec_us_total.fetch_add(static_cast<uint64_t>(batch_exec_us),
                                            std::memory_order_relaxed);
      }
      batch.clear();
      lock.lock();

      write_batch_in_progress = false;
      write_batch_queue_cv.notify_all();
    } else {
      write_follower_wait_count.fetch_add(1, std::memory_order_relaxed);
      const auto follower_wait_begin = std::chrono::steady_clock::now();
      write_batch_queue_cv.wait(lock, [this, &request] {
        return request.Done || !write_batch_in_progress;
      });
      const auto follower_wait_end = std::chrono::steady_clock::now();
      const auto follower_wait_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              follower_wait_end - follower_wait_begin)
              .count();
      if (follower_wait_us > 0) {
        write_follower_wait_us_total.fetch_add(
            static_cast<uint64_t>(follower_wait_us), std::memory_order_relaxed);
      }
    }
  }

  const auto done_ts = std::chrono::steady_clock::now();
  const auto request_wait_us =
      std::chrono::duration_cast<std::chrono::microseconds>(done_ts - enqueue_ts)
          .count();
  if (request_wait_us > 0) {
    write_request_wait_us_total.fetch_add(
        static_cast<uint64_t>(request_wait_us), std::memory_order_relaxed);
  }

  if (request.Status != 0) {
    errno = request.Errno;
    return -1;
  }
  return 0;
}

void MPIIOBackend::processWriteBatch(std::vector<WriteBatchRequest *> &batch) {
  if (batch.empty())
    return;

  write_batch_count.fetch_add(1, std::memory_order_relaxed);
  std::unordered_map<uint64_t, std::vector<WriteBatchRequest *>> grouped;
  grouped.reserve(batch.size());
  for (WriteBatchRequest *request : batch)
    grouped[getWriteBatchGroupKey(*request)].push_back(request);

  for (auto &entry : grouped)
    processWriteGroup(entry.second);
}

void MPIIOBackend::processWriteGroup(std::vector<WriteBatchRequest *> &group) {
  if (group.empty())
    return;

  // Debug: log all write request offsets before sorting
  for (size_t dbg_i = 0; dbg_i < group.size(); ++dbg_i) {
    io_log("[write-batch] req[%zu] offset=%ld size=%zu\n",
           dbg_i, group[dbg_i]->Offset, group[dbg_i]->Data.size());
  }

  std::sort(group.begin(), group.end(),
            [](const WriteBatchRequest *lhs, const WriteBatchRequest *rhs) {
              if (lhs->Offset != rhs->Offset)
                return lhs->Offset < rhs->Offset;
              return lhs->Data.size() < rhs->Data.size();
            });

  struct CoalescedWrite {
    long Start = 0;
    long End = 0;
    std::vector<char> Data;
    std::vector<WriteBatchRequest *> Requests;
  };

  std::vector<CoalescedWrite> coalesced;
  coalesced.reserve(group.size());
  for (WriteBatchRequest *request : group) {
    if (request->Offset < 0) {
      completeWriteBatchRequest(*request, -1, EINVAL);
      continue;
    }
    if (request->Data.size() >
            static_cast<size_t>(std::numeric_limits<long>::max()) ||
        request->Offset >
            std::numeric_limits<long>::max() -
                static_cast<long>(request->Data.size())) {
      completeWriteBatchRequest(*request, -1, EOVERFLOW);
      continue;
    }

    const long start = request->Offset;
    const long end = start + static_cast<long>(request->Data.size());
    if (coalesced.empty()) {
      CoalescedWrite item{};
      item.Start = start;
      item.End = end;
      item.Data = request->Data;
      item.Requests.push_back(request);
      coalesced.push_back(std::move(item));
      continue;
    }

    CoalescedWrite &tail = coalesced.back();
    const bool mergeable = start <= tail.End;
    const long merged_end = std::max(tail.End, end);
    const uint64_t merged_size =
        static_cast<uint64_t>(merged_end - tail.Start);
    const bool exceeds_cap =
        write_batch_max_batch_bytes > 0 &&
        merged_size > write_batch_max_batch_bytes;
    if (!mergeable || exceeds_cap) {
      CoalescedWrite item{};
      item.Start = start;
      item.End = end;
      item.Data = request->Data;
      item.Requests.push_back(request);
      coalesced.push_back(std::move(item));
      continue;
    }

    tail.Data.resize(static_cast<size_t>(merged_end - tail.Start), 0);
    const size_t copy_offset = static_cast<size_t>(start - tail.Start);
    if (!request->Data.empty()) {
      std::memcpy(tail.Data.data() + copy_offset, request->Data.data(),
                  request->Data.size());
    }
    tail.End = merged_end;
    tail.Requests.push_back(request);
  }

  // Debug: log coalesced write ranges after grouping
  for (size_t dbg_i = 0; dbg_i < coalesced.size(); ++dbg_i) {
    io_log("[write-coalesced] range[%zu] start=%ld end=%ld size=%zu requests=%zu\n",
             dbg_i, coalesced[dbg_i].Start, coalesced[dbg_i].End,
             coalesced[dbg_i].Data.size(), coalesced[dbg_i].Requests.size());
  }

  int remote_handle = -1;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto it = remote_file_handle_map.find(group.front()->FileHandle);
    if (it != remote_file_handle_map.end())
      remote_handle = it->second;
  }
  if (remote_handle < 0) {
    write_batch_failure_count.fetch_add(1, std::memory_order_relaxed);
    for (WriteBatchRequest *request : group)
      completeWriteBatchRequest(*request, -1, EBADF);
    return;
  }

  for (CoalescedWrite &item : coalesced) {
    write_batch_coalesced_write_count.fetch_add(1, std::memory_order_relaxed);
    write_batch_coalesced_bytes_total.fetch_add(item.Data.size(),
                                                std::memory_order_relaxed);
    if (item.Requests.size() > 1) {
      write_batch_saved_events.fetch_add(item.Requests.size() - 1,
                                         std::memory_order_relaxed);
    }

    size_t bytes_written = 0;
    const int rc = writeAtRemoteHandle(remote_handle, item.Start,
                                       item.Data.data(), item.Data.size(),
                                       bytes_written);
    if (rc != 0 || bytes_written != item.Data.size()) {
      const int errnum = errno != 0 ? errno : EIO;
      write_batch_failure_count.fetch_add(1, std::memory_order_relaxed);
      for (WriteBatchRequest *request : item.Requests)
        completeWriteBatchRequest(*request, -1, errnum);
      continue;
    }

    for (WriteBatchRequest *request : item.Requests)
      completeWriteBatchRequest(*request, 0, 0);
  }
}

int MPIIOBackend::getNextFileHandle() {
  return next_file_handle.fetch_add(1, std::memory_order_relaxed);
}

uint64_t MPIIOBackend::computePathKey(const char *path) {
  constexpr uint64_t offset_basis = 1469598103934665603ULL;
  constexpr uint64_t prime = 1099511628211ULL;
  uint64_t hash = offset_basis;
  if (!path)
    return hash;

  for (const unsigned char *p =
           reinterpret_cast<const unsigned char *>(path);
       *p != '\0'; ++p) {
    hash ^= static_cast<uint64_t>(*p);
    hash *= prime;
  }
  return hash;
}

uint64_t MPIIOBackend::mixHintIntoKey(uint64_t base_key,
                                      const ompfile::OmpFileIOHint &hint) {
  uint64_t mixed = base_key;
  auto mix = [&mixed](uint64_t value) {
    mixed ^= value + 0x9e3779b97f4a7c15ULL + (mixed << 6) + (mixed >> 2);
  };

  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_STREAM) != 0)
    mix(hint.StreamId);
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_EPOCH) != 0)
    mix(hint.EpochId);
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0)
    mix(hint.TileId);
  return mixed;
}

uint64_t MPIIOBackend::computeSourceStageAffinityKey(
    uint64_t path_key, const ompfile::OmpFileIOHint &hint) {
  uint64_t key = path_key ^ 0x9e3779b97f4a7c15ULL;
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_STREAM) != 0)
    key ^= hint.StreamId + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
  key ^= hint.TileId + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
  return key;
}

bool MPIIOBackend::selectSourceStageAffinityRank(
    uint64_t path_key, const ompfile::OmpFileIOHint &hint,
    const std::vector<int> &worker_ranks, int &rank_out) {
  rank_out = -1;
  if (path_key == 0 || worker_ranks.empty())
    return false;
  const uint64_t key = computeSourceStageAffinityKey(path_key, hint);
  rank_out = worker_ranks[static_cast<size_t>(key % worker_ranks.size())];
  if (rank_out < 0) {
    rank_out = -1;
    return false;
  }
  return true;
}

std::vector<int> MPIIOBackend::computeWorkerRanks(int world_size,
                                                  int headnode_rank) {
  std::vector<int> ranks;
  if (world_size <= 0)
    return ranks;
  for (int rank = 0; rank < world_size; ++rank) {
    if (rank != headnode_rank)
      ranks.push_back(rank);
  }
  return ranks;
}

uint64_t MPIIOBackend::mixWriteHintIntoKey(
    uint64_t base_key, const ompfile::OmpFileIOHint &hint) {
  uint64_t mixed = base_key;
  auto mix = [&mixed](uint64_t value) {
    mixed ^= value + 0x9e3779b97f4a7c15ULL + (mixed << 6) + (mixed >> 2);
  };

  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_STREAM) != 0)
    mix(hint.StreamId);
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0)
    mix(hint.TileId);
  return mixed;
}

bool MPIIOBackend::getFilePathKey(int file_id, uint64_t &path_key_out) {
  const std::lock_guard<std::mutex> lock(handle_mutex);
  const auto it = file_path_key_map.find(file_id);
  if (it == file_path_key_map.end())
    return false;
  path_key_out = it->second;
  return true;
}

bool MPIIOBackend::isForecastHintValid(
    const ompfile::OmpFileIOHint &hint) const {
  if (forecast_mode != ForecastMode::CholeskyOracle)
    return false;
  return (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0;
}

bool MPIIOBackend::isForecastRoleHintValid(
    const ompfile::OmpFileIOHint &hint) const {
  return isForecastHintValid(hint) &&
         (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_ROLE) != 0;
}

bool MPIIOBackend::shouldBypassStageForForecastRead(
    const ompfile::OmpFileIOHint &hint) const {
  return isForecastRoleHintValid(hint) &&
         hint.Role == ompfile::OMPFILE_IO_ROLE_TARGET_READ;
}

bool MPIIOBackend::shouldUseNoStageFallback(
    const ompfile::OmpFileIOHint &hint, bool force_no_stage) const {
  return force_no_stage || shouldBypassStageForForecastRead(hint);
}

bool MPIIOBackend::shouldAttemptPlannerRebalancedRead(
    int rc, bool force_no_stage_fallback, bool planner_rebalanced,
    int planned_rank) const {
  return rc != 0 && !force_no_stage_fallback && planner_rebalanced &&
         planned_rank >= 0;
}

bool MPIIOBackend::isSourceStageAffinityEligible(
    const ompfile::OmpFileIOHint &hint, bool has_path_key,
    bool remote_only) const {
  if (!remote_only || !has_path_key)
    return false;
  if (forecast_mode != ForecastMode::CholeskyOracle ||
      stage_affinity_mode != StageAffinityMode::CholeskyOracle)
    return false;
  const char *stage_mode = std::getenv("LIBOMPFILE_STAGE_MODE");
  if (!stage_mode || std::strcmp(stage_mode, "readthrough") != 0)
    return false;
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_ROLE) == 0 ||
      hint.Role != ompfile::OMPFILE_IO_ROLE_SOURCE_READ)
    return false;
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) == 0)
    return false;
  return true;
}

bool MPIIOBackend::canAttemptSourceStageAffinityRead(
    const ompfile::OmpFileIOHint &hint, int file_id, long start, size_t size,
    bool has_path_key, bool remote_only, bool planner_force_fallback) {
  if (planner_force_fallback)
    return false;
  if (!isSourceStageAffinityEligible(hint, has_path_key, remote_only))
    return false;
  const char *reason = "none";
  int reason_errno = 0;
  return canApplyRebalancedRead(hint, file_id, start, size, reason,
                                reason_errno, /*record_stats=*/false);
}

void MPIIOBackend::recordForecastRead(const ompfile::OmpFileIOHint &hint,
                                      size_t size) {
  if (!isForecastHintValid(hint))
    return;
  forecast_hint_read_count.fetch_add(1, std::memory_order_relaxed);
  forecast_hint_read_bytes_total.fetch_add(size, std::memory_order_relaxed);
  if (!isForecastRoleHintValid(hint)) {
    forecast_role_unknown_count.fetch_add(1, std::memory_order_relaxed);
    forecast_role_unknown_bytes_total.fetch_add(size,
                                                std::memory_order_relaxed);
    return;
  }
  if (hint.Role == ompfile::OMPFILE_IO_ROLE_SOURCE_READ) {
    forecast_source_read_stageable_count.fetch_add(1,
                                                   std::memory_order_relaxed);
    forecast_source_read_stageable_bytes_total.fetch_add(
        size, std::memory_order_relaxed);
    return;
  }
  if (shouldBypassStageForForecastRead(hint)) {
    forecast_target_read_bypass_count.fetch_add(1, std::memory_order_relaxed);
    forecast_target_read_bypass_bytes_total.fetch_add(
        size, std::memory_order_relaxed);
    return;
  }
  forecast_role_unknown_count.fetch_add(1, std::memory_order_relaxed);
  forecast_role_unknown_bytes_total.fetch_add(size, std::memory_order_relaxed);
}

void MPIIOBackend::recordForecastWrite(const ompfile::OmpFileIOHint &hint,
                                       size_t size) {
  if (!isForecastHintValid(hint))
    return;
  forecast_hint_write_count.fetch_add(1, std::memory_order_relaxed);
  forecast_hint_write_bytes_total.fetch_add(size, std::memory_order_relaxed);
}

void MPIIOBackend::rememberFilePathKey(int file_id, const char *path) {
  const uint64_t path_key = computePathKey(path);
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    file_path_key_map[file_id] = path_key;
  }
  invalidateTwoPhaseReadCacheKey(path_key);
}

void MPIIOBackend::rememberFilePath(int file_id, const char *path) {
  if (!path)
    return;
  const std::lock_guard<std::mutex> lock(handle_mutex);
  file_path_map[file_id] = path;
  file_write_epoch_history.erase(file_id);
}

void MPIIOBackend::forgetFilePath(int file_id) {
  const std::lock_guard<std::mutex> lock(handle_mutex);
  file_path_map.erase(file_id);
  file_write_epoch_history.erase(file_id);
}

void MPIIOBackend::forgetFilePathKey(int file_id) {
  const std::lock_guard<std::mutex> lock(handle_mutex);
  file_path_key_map.erase(file_id);
}

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

  const std::lock_guard<std::mutex> lock(handle_mutex);
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

bool MPIIOBackend::hasCompletedStageInvalidationForTesting(
    int file_id) const {
  const std::lock_guard<std::mutex> lock(handle_mutex);
  auto it = writable_stage_coherence.find(file_id);
  if (it == writable_stage_coherence.end())
    return false;
  const WritableStageCoherenceState &coherence = it->second;
  return coherence.WriteGeneration != 0 && !coherence.InvalidationFailed &&
         coherence.InvalidatedGeneration >= coherence.WriteGeneration;
}

bool MPIIOBackend::globallyInvalidateStageForFile(int file_id) {
  uint64_t path_key = 0;
  uint64_t generation = 0;
  std::string path;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    auto path_key_it = file_path_key_map.find(file_id);
    auto path_it = file_path_map.find(file_id);
    auto coherence_it = writable_stage_coherence.find(file_id);
    if (path_key_it == file_path_key_map.end() || path_key_it->second == 0 ||
        path_it == file_path_map.end() || path_it->second.empty() ||
        coherence_it == writable_stage_coherence.end() ||
        coherence_it->second.WriteGeneration == 0) {
      stage_global_invalidations_failed.fetch_add(1,
                                                  std::memory_order_relaxed);
      return false;
    }
    path_key = path_key_it->second;
    path = path_it->second;
    generation = coherence_it->second.WriteGeneration;
  }

  stage_global_invalidations_requested.fetch_add(1, std::memory_order_relaxed);
  bool ok = false;
  {
    const std::lock_guard<std::mutex> lock(mpp_call_mutex);
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
    const std::lock_guard<std::mutex> lock(handle_mutex);
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

  const std::lock_guard<std::mutex> lock(mpp_call_mutex);
  return ompfile::mpp::stageInvalidatePathKey(path_key, /*generation=*/0,
                                             path.c_str());
}

void MPIIOBackend::completeStageInvalidationForFile(int file_id,
                                                    uint64_t generation,
                                                    bool ok) {
  const std::lock_guard<std::mutex> lock(handle_mutex);
  auto &coherence = writable_stage_coherence[file_id];
  if (!ok) {
    if (generation == coherence.WriteGeneration)
      coherence.InvalidationFailed = true;
    stage_global_invalidations_failed.fetch_add(1,
                                                std::memory_order_relaxed);
    return;
  }

  if (generation == coherence.WriteGeneration) {
    coherence.InvalidatedGeneration = generation;
    coherence.InvalidationFailed = false;
  }
  stage_global_invalidations_completed.fetch_add(1,
                                                 std::memory_order_relaxed);
}

void MPIIOBackend::triggerStageInvalidationAfterSuccessfulRemoteWrite(
    int file_id) {
  if (writable_read_rebalance_enabled && mpp_remote_only)
    (void)globallyInvalidateStageForFile(file_id);
}

void MPIIOBackend::noteAppliedRebalancedReadForFile(int file_id) {
  bool count_writable_read_applied = false;
  bool count_after_invalidation_applied = false;
  if (mpp_remote_only) {
    const std::lock_guard<std::mutex> lock(handle_mutex);
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
    writable_read_rebalance_applied_count.fetch_add(
        1, std::memory_order_relaxed);
  }
  if (count_after_invalidation_applied) {
    writable_read_rebalance_after_invalidation_applied.fetch_add(
        1, std::memory_order_relaxed);
  }
}

void MPIIOBackend::noteWritableReadRebalanceBlocked(const char *reason) {
  writable_read_rebalance_blocked_count.fetch_add(1,
                                                  std::memory_order_relaxed);
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

bool MPIIOBackend::canApplyRebalancedRead(
    const ompfile::OmpFileIOHint &hint, int file_id, long start, size_t size,
    const char *&reason_out, int &reason_errno_out, bool record_stats) {
  reason_out = "none";
  reason_errno_out = 0;
  const std::lock_guard<std::mutex> lock(handle_mutex);
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
  for (const WriteEpochEntry &entry : it->second) {
    if (!rangesOverlap(start, end, entry.Start, entry.End))
      continue;
    if (read_has_tile && entry.HasTile && hint.TileId != entry.TileId)
      continue;
    saw_overlap = true;
    if (!entry.HasEpoch) {
      saw_missing_write_epoch = true;
      continue;
    }
    if (read_has_epoch && entry.EpochId > read_epoch)
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
    writable_read_rebalance_eligible_count.fetch_add(
        1, std::memory_order_relaxed);
  }
  return true;
}

bool MPIIOBackend::getOrCreateRemoteReadHandleForRank(int file_id,
                                                      int target_rank,
                                                      int &remote_handle_out) {
  remote_handle_out = -1;
  std::string path;

  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    if (logical_handle_set.find(file_id) == logical_handle_set.end()) {
      errno = EBADF;
      return false;
    }
    auto cache_it = remote_read_handle_cache.find(file_id);
    if (cache_it != remote_read_handle_cache.end()) {
      auto rank_it = cache_it->second.find(target_rank);
      if (rank_it != cache_it->second.end()) {
        remote_handle_out = rank_it->second;
        return true;
      }
    }

    auto path_it = file_path_map.find(file_id);
    if (path_it == file_path_map.end()) {
      errno = ENOENT;
      return false;
    }
    path = path_it->second;
  }

  if (path.empty()) {
    errno = ENOENT;
    return false;
  }

  int opened_handle = -1;
  {
    const std::lock_guard<std::mutex> lock(mpp_call_mutex);
    if (!ompfile::mpp::openOnRank(path.c_str(), open_flags, 0666,
                                  target_rank, opened_handle)) {
      return false;
    }
  }

  bool must_close_opened_handle = false;
  bool closed_during_open = false;
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    if (logical_handle_set.find(file_id) == logical_handle_set.end()) {
      must_close_opened_handle = true;
      closed_during_open = true;
      errno = EBADF;
    } else {
      auto &rank_cache = remote_read_handle_cache[file_id];
      auto [it, inserted] = rank_cache.try_emplace(target_rank, opened_handle);
      if (!inserted) {
        must_close_opened_handle = true;
        opened_handle = it->second;
      }
    }
  }

  if (must_close_opened_handle) {
    const std::lock_guard<std::mutex> lock(mpp_call_mutex);
    (void)ompfile::mpp::close(opened_handle);
    if (closed_during_open)
      return false;
  }

  remote_handle_out = opened_handle;
  return true;
}

bool MPIIOBackend::readAtRemoteRankWithBytes(int file_id, int target_rank,
                                              long offset, void *data,
                                              size_t size, size_t &bytes_read) {
  bytes_read = 0;
  if (!mpp_remote_only) {
    errno = ENOTSUP;
    return false;
  }
  if (target_rank < 0) {
    errno = EINVAL;
    return false;
  }
  int last_errno = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt > 0) {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      auto cache_it = remote_read_handle_cache.find(file_id);
      if (cache_it != remote_read_handle_cache.end())
        cache_it->second.erase(target_rank);
    }

    int remote_handle = -1;
    if (!getOrCreateRemoteReadHandleForRank(file_id, target_rank,
                                            remote_handle)) {
      last_errno = errno;
      continue;
    }

    remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
    remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);

    bytes_read = 0;
    bool ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      ok = ompfile::mpp::preadEx(remote_handle, offset, data, size, bytes_read);
    }
    if (ok)
      goto pread_ok;

    last_errno = errno;
    if (last_errno != 0 && last_errno != EBADF)
      break;
  }

  errno = last_errno != 0 ? last_errno : EIO;
  return false;

pread_ok:
  if (bytes_read > size) {
    errno = EPROTO;
    return false;
  }
  if (bytes_read < size && data) {
    const size_t short_bytes = size - bytes_read;
    short_read_count.fetch_add(1, std::memory_order_relaxed);
    short_read_bytes_total.fetch_add(short_bytes, std::memory_order_relaxed);
    std::memset(static_cast<char *>(data) + bytes_read, 0, short_bytes);
  }
  return true;
}

bool MPIIOBackend::readAtSourceStageAffinityRankWithBytes(
    int file_id, uint64_t path_key, const ompfile::OmpFileIOHint &hint,
    long offset, void *data, size_t size, size_t &bytes_read, int &rank_out) {
  rank_out = -1;
  int finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
    return false;
  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS)
    return false;
  if (!initialized)
    return false;
  int world_size = 0;
  if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS)
    return false;
  const int headnode_rank = world_size > 0 ? world_size - 1 : -1;
  const std::vector<int> workers = computeWorkerRanks(world_size, headnode_rank);
  if (!selectSourceStageAffinityRank(path_key, hint, workers, rank_out))
    return false;
  return readAtRemoteRankWithBytes(file_id, rank_out, offset, data, size,
                                   bytes_read);
}

bool MPIIOBackend::commitTileFreshnessWrite(uint64_t path_key, int writer_rank,
                                            bool write_through_mode,
                                            uint64_t &committed_version_out) {
  committed_version_out = 0;
  if (path_key == 0 || writer_rank < 0) {
    errno = EINVAL;
    return false;
  }

  const std::lock_guard<std::mutex> lock(handle_mutex);
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

  const std::lock_guard<std::mutex> lock(handle_mutex);
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

  const std::lock_guard<std::mutex> lock(handle_mutex);
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
  const std::lock_guard<std::mutex> lock(handle_mutex);
  auto it = tile_freshness_table.find(path_key);
  if (it == tile_freshness_table.end())
    return {};
  return it->second;
}

uint64_t MPIIOBackend::resolveTwoPhaseKey(
    const ompfile::OmpFileReadRequestContext &context) {
  uint64_t base_key = 0;
  if ((context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY) != 0) {
    base_key = context.PathKey;
    return mixHintIntoKey(base_key, context.Hint);
  }

  uint64_t path_key = 0;
  if (getFilePathKey(context.FileHandle, path_key))
    base_key = path_key;
  else
    base_key = static_cast<uint64_t>(static_cast<uint32_t>(context.FileHandle));
  return mixHintIntoKey(base_key, context.Hint);
}

bool MPIIOBackend::tryServeTwoPhaseReadCache(uint64_t key, long offset,
                                             void *data, size_t size) {
  if (!data || size == 0)
    return false;

  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  const auto it = two_phase_read_cache.find(key);
  if (it == two_phase_read_cache.end())
    return false;

  const TwoPhaseReadCacheEntry &entry = it->second;
  if (offset < entry.Start)
    return false;
  const uint64_t delta = static_cast<uint64_t>(offset - entry.Start);
  if (delta > static_cast<uint64_t>(entry.Data.size()))
    return false;
  if (size > entry.Data.size() - static_cast<size_t>(delta))
    return false;

  std::memcpy(data, entry.Data.data() + static_cast<size_t>(delta), size);
  two_phase_cache_hit_count.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void MPIIOBackend::updateTwoPhaseReadCache(uint64_t key, long start,
                                           const std::vector<char> &data) {
  if (data.empty())
    return;

  TwoPhaseReadCacheEntry entry{};
  entry.Start = start;
  entry.Data = data;

  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  two_phase_read_cache[key] = std::move(entry);
}

void MPIIOBackend::invalidateTwoPhaseReadCacheKey(uint64_t key) {
  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  two_phase_read_cache.erase(key);
}

void MPIIOBackend::invalidateTwoPhaseReadCacheForFile(int file_id) {
  uint64_t path_key = 0;
  const bool has_path_key = getFilePathKey(file_id, path_key);
  const uint64_t fallback_key =
      static_cast<uint64_t>(static_cast<uint32_t>(file_id));
  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  const size_t cleared_entries = two_phase_read_cache.size();
  if (cleared_entries == 0)
    return;

  // Two-phase cache entries are stored under hint-mixed keys. Once writes happen,
  // the current cache key space is not invertible back to every mixed variant for
  // the file, so clear conservatively instead of serving stale read-after-write
  // data under an old epoch/stream/tile key.
  two_phase_read_cache.clear();
  io_trace("MPIIOBackend::invalidateTwoPhaseReadCacheForFile file_id=%d "
           "has_path_key=%d path_key=%llu fallback_key=%llu cleared=%zu\n",
           file_id, static_cast<int>(has_path_key),
           static_cast<unsigned long long>(path_key),
           static_cast<unsigned long long>(fallback_key), cleared_entries);
}

void MPIIOBackend::traceHandleStateLocked(const char *where, int file_id,
                                          int remote_handle) const {
  if (!io_trace_enabled())
    return;

  const bool logical_present =
      logical_handle_set.find(file_id) != logical_handle_set.end();
  const auto remote_it = remote_file_handle_map.find(file_id);
  const bool remote_present = remote_it != remote_file_handle_map.end();
  const int remote_mapped_handle =
      remote_present ? remote_it->second : -1;
  const bool mpi_present = file_handle_map.find(file_id) != file_handle_map.end();

  std::ostringstream logical_ids;
  int logical_emitted = 0;
  for (int id : logical_handle_set) {
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
  for (const auto &entry : remote_file_handle_map) {
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
           "mpi_present=%d logical_count=%zu remote_count=%zu mpi_count=%zu "
           "logical_ids=[%s] remote_ids=[%s]\n",
           where ? where : "(unknown)", static_cast<const void *>(this), file_id,
           remote_handle, static_cast<int>(logical_present),
           static_cast<int>(remote_present), remote_mapped_handle,
           static_cast<int>(mpi_present), logical_handle_set.size(),
           remote_file_handle_map.size(), file_handle_map.size(),
           logical_ids.str().c_str(), remote_ids.str().c_str());
}

void MPIIOBackend::traceHandleState(const char *where, int file_id,
                                    int remote_handle) {
  if (!io_trace_enabled())
    return;
  const std::lock_guard<std::mutex> lock(handle_mutex);
  traceHandleStateLocked(where, file_id, remote_handle);
}

int MPIIOBackend::failStrictMpp(const char *op_name) const {
  errno = EIO;
  if (!strict_failure_logged.exchange(true, std::memory_order_relaxed)) {
    io_log("Error: '%s' failed because MPP init did not complete and fallback "
           "is disabled. Set LIBOMPFILE_ALLOWFALLBACK=1 to allow fallback "
           "behavior.\n",
           op_name ? op_name : "(unknown)");
  }
  io_trace("MPIIOBackend::failStrictMpp this=%p op=%s mpp_requested=%d "
           "allow_fallback=%d strict_mpp_init_failed=%d\n",
           static_cast<const void *>(this), op_name ? op_name : "(unknown)",
           static_cast<int>(mpp_requested), static_cast<int>(allow_fallback),
           static_cast<int>(strict_mpp_init_failed));
  return -1;
}

bool MPIIOBackend::parseBoolEnv(const char *name, bool default_value) {
  const char *env = std::getenv(name);
  if (!env)
    return default_value;

  if (env[0] == '1' && env[1] == '\0')
    return true;
  if (env[0] == '0' && env[1] == '\0')
    return false;

  io_log("Invalid boolean value for %s='%s'; using default=%d\n", name, env,
         static_cast<int>(default_value));
  return default_value;
}

MPIIOBackend::TwoPhasePolicy
MPIIOBackend::parseTwoPhasePolicy(const char *env_value) {
  if (!env_value || env_value[0] == '\0')
    return TwoPhasePolicy::Auto;

  if (std::strcmp(env_value, "1") == 0)
    return TwoPhasePolicy::Enabled;
  if (std::strcmp(env_value, "0") == 0)
    return TwoPhasePolicy::Disabled;
  if (std::strcmp(env_value, "enabled") == 0 ||
      std::strcmp(env_value, "ENABLED") == 0) {
    return TwoPhasePolicy::Enabled;
  }
  if (std::strcmp(env_value, "disabled") == 0 ||
      std::strcmp(env_value, "DISABLED") == 0) {
    return TwoPhasePolicy::Disabled;
  }
  if (std::strcmp(env_value, "auto") == 0 ||
      std::strcmp(env_value, "AUTO") == 0) {
    return TwoPhasePolicy::Auto;
  }

  io_log("Invalid LIBOMPFILE_OPT_TWO_PHASE='%s'; using disabled policy.\n",
         env_value);
  return TwoPhasePolicy::Disabled;
}

MPIIOBackend::ForecastMode
MPIIOBackend::parseForecastMode(const char *env_value) {
  if (!env_value || env_value[0] == '\0' || std::strcmp(env_value, "off") == 0 ||
      std::strcmp(env_value, "OFF") == 0 || std::strcmp(env_value, "0") == 0) {
    return ForecastMode::Off;
  }

  if (std::strcmp(env_value, "cholesky-oracle") == 0)
    return ForecastMode::CholeskyOracle;

  io_log("Invalid LIBOMPFILE_IO_FORECAST='%s'; using off mode.\n", env_value);
  return ForecastMode::Off;
}

MPIIOBackend::StageAffinityMode
MPIIOBackend::parseStageAffinityMode(const char *env_value) {
  if (!env_value || env_value[0] == '\0' || std::strcmp(env_value, "off") == 0)
    return StageAffinityMode::Off;
  if (std::strcmp(env_value, "cholesky-oracle") == 0)
    return StageAffinityMode::CholeskyOracle;
  io_log("Invalid LIBOMPFILE_IO_STAGE_AFFINITY='%s'; using off mode.\n",
         env_value);
  return StageAffinityMode::Off;
}

const char *
MPIIOBackend::twoPhasePolicyToString(TwoPhasePolicy policy) {
  switch (policy) {
  case TwoPhasePolicy::Disabled:
    return "disabled";
  case TwoPhasePolicy::Enabled:
    return "enabled";
  case TwoPhasePolicy::Auto:
    return "auto";
  }
  return "unknown";
}

const char *MPIIOBackend::forecastModeToString(ForecastMode mode) {
  switch (mode) {
  case ForecastMode::Off:
    return "off";
  case ForecastMode::CholeskyOracle:
    return "cholesky-oracle";
  }
  return "unknown";
}

const char *MPIIOBackend::stageAffinityModeToString(StageAffinityMode mode) {
  switch (mode) {
  case StageAffinityMode::Off:
    return "off";
  case StageAffinityMode::CholeskyOracle:
    return "cholesky-oracle";
  }
  return "off";
}

uint64_t MPIIOBackend::parseUint64Env(const char *name, uint64_t default_value) {
  const char *env = std::getenv(name);
  if (!env)
    return default_value;

  errno = 0;
  char *end_ptr = nullptr;
  const unsigned long long value = std::strtoull(env, &end_ptr, 10);
  if (errno != 0 || end_ptr == env || (end_ptr && *end_ptr != '\0')) {
    io_log("Invalid integer value for %s='%s'; using default=%llu\n", name, env,
           static_cast<unsigned long long>(default_value));
    return default_value;
  }

  return static_cast<uint64_t>(value);
}

void MPIIOBackend::updateAtomicMax(std::atomic<uint64_t> &counter,
                                   uint64_t candidate) {
  uint64_t observed = counter.load(std::memory_order_relaxed);
  while (candidate > observed &&
         !counter.compare_exchange_weak(observed, candidate,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
  }
}

size_t MPIIOBackend::computeTwoPhaseTargetReadSize(
    long start, long end, size_t request_count, bool remote_only,
    uint64_t sieve_bytes, uint64_t max_batch_bytes) {
  assert(start >= 0 && end >= start &&
         "Coalesced read range must be monotonic.");
  const size_t read_size = static_cast<size_t>(end - start);
  if (sieve_bytes == 0)
    return read_size;

  // Remote-only MPP intentionally does not cache two-phase overreads across
  // proxies, so data sieving isolated requests only adds transfer volume.
  if (remote_only && request_count <= 1)
    return read_size;

  uint64_t target_size =
      std::max<uint64_t>(static_cast<uint64_t>(read_size), sieve_bytes);
  if (max_batch_bytes > 0)
    target_size = std::min<uint64_t>(target_size, max_batch_bytes);

  const uint64_t size_t_max =
      static_cast<uint64_t>(std::numeric_limits<size_t>::max());
  target_size = std::min<uint64_t>(target_size, size_t_max);
  const uint64_t long_max =
      static_cast<uint64_t>(std::numeric_limits<long>::max());
  const uint64_t start_u64 = static_cast<uint64_t>(start);
  if (start_u64 >= long_max)
    return read_size;
  const uint64_t max_extent = long_max - start_u64;
  target_size = std::min<uint64_t>(target_size, max_extent);

  if (target_size < static_cast<uint64_t>(read_size))
    return read_size;
  return static_cast<size_t>(target_size);
}

size_t MPIIOBackend::computeForecastTargetReadSize(
    const ompfile::OmpFileIOHint &hint, long start, long end,
    size_t request_count, bool remote_only, uint64_t sieve_bytes,
    uint64_t max_batch_bytes) const {
  assert(start >= 0 && end >= start &&
         "Forecast read range must be monotonic.");
  if (isForecastHintValid(hint))
    return static_cast<size_t>(end - start);
  return computeTwoPhaseTargetReadSize(start, end, request_count, remote_only,
                                       sieve_bytes, max_batch_bytes);
}
