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
#include <string>
#include <unordered_set>
#include <unordered_map>
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
      if (errno == 0)
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


