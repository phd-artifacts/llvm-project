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
#include <unordered_map>
#include <vector>

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

  io_log("Two-phase guard config: policy=%s enabled=%d window_us=%llu "
         "max_batch_bytes=%llu sieve_bytes=%llu write_batch_policy=%s "
         "write_batch_enabled=%d write_batch_window_us=%llu "
         "write_batch_max_batch_bytes=%llu scheduler=%s remote_only=%d\n",
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
         static_cast<int>(mpp_remote_only));

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
  if (strict_mpp_init_failed)
    return failStrictMpp("open");
  int file_id = getNextFileHandle();
  const bool remote_only = mpp_remote_only;

  io_log("Opening file %s with file_id %d\n", filename, file_id);
  io_trace("MPIIOBackend::open enter this=%p file_id=%d remote_only=%d "
           "filename=%s\n",
           static_cast<void *>(this), file_id, static_cast<int>(remote_only),
           filename ? filename : "(null)");

  if (remote_only) {
    int remote_handle = -1;
    bool open_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      open_ok = ompfile::mpp::open(filename, O_RDWR, 0666, remote_handle);
    }
    if (!open_ok) {
      io_log("MPP open failed for %s\n", filename);
      errno = EIO;
      return -1;
    }
    {
      const std::lock_guard<std::mutex> lock(handle_mutex);
      remote_file_handle_map[file_id] = remote_handle;
      logical_handle_set.insert(file_id);
      traceHandleStateLocked("open.remote_only.insert", file_id, remote_handle);
    }
    io_log("Remote-only open completed for file_id %d\n", file_id);
    rememberFilePathKey(file_id, filename);
    traceHandleState("open.remote_only.done", file_id, remote_handle);
    return file_id;
  }

  MPI_File file_handle;
  int ret = MPI_File_open(file_comm, filename, MPI_MODE_RDWR, MPI_INFO_NULL,
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
    logical_handle_set.insert(file_id);
    traceHandleStateLocked("open.mpi.insert", file_id, -1);
  }

  if (mpp_open_enabled) {
    int remote_handle = -1;
    bool open_ok = false;
    {
      const std::lock_guard<std::mutex> lock(mpp_call_mutex);
      open_ok = ompfile::mpp::open(filename, O_RDWR, 0666, remote_handle);
    }
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
      traceHandleStateLocked("open.mpp.remote.insert", file_id, remote_handle);
    }
  }
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
  invalidateTwoPhaseReadCacheForFile(file_id);
  forgetFilePathKey(file_id);
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

    if (mpp_open_enabled) {
      auto it = remote_file_handle_map.find(file_id);
      if (it != remote_file_handle_map.end()) {
        remote_handle = it->second;
        has_remote_handle = true;
        remote_file_handle_map.erase(it);
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
    }
  }

  if (remote_only) {
    if (mpp_ret != 0)
      return -1;
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
  return readAtFallback(file_id, offset, data, size);
}

int MPIIOBackend::readAtFallback(int file_id, long offset, void *data,
                                 size_t size) {
  size_t bytes_read = 0;
  return readAtFallbackWithBytes(file_id, offset, data, size, bytes_read);
}

int MPIIOBackend::readAtFallbackWithBytes(int file_id, long offset, void *data,
                                          size_t size, size_t &bytes_read) {
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
      pread_ok =
          ompfile::mpp::preadEx(remote_handle, offset, data, size, bytes_read);
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
  if (tryServeTwoPhaseReadCache(request_key, request_offset, data, size)) {
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

  const auto enqueue_ts = std::chrono::steady_clock::now();
  std::vector<TwoPhaseReadRequest *> batch;
  std::unique_lock<std::mutex> lock(two_phase_mutex);
  const size_t queue_size_before = two_phase_queue.size();
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
      io_trace("MPIIOBackend::readAtTwoPhase leader req=%llu queue_size=%zu "
               "window_us=%llu\n",
               static_cast<unsigned long long>(request.DebugRequestId),
               two_phase_queue.size(),
               static_cast<unsigned long long>(two_phase_window_us));

      if (two_phase_window_us > 0) {
        two_phase_queue_cv.wait_for(lock,
                                    std::chrono::microseconds(two_phase_window_us));
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
      processTwoPhaseBatch(batch);
      batch.clear();
      lock.lock();

      two_phase_batch_in_progress = false;
      two_phase_queue_cv.notify_all();
    } else {
      io_trace("MPIIOBackend::readAtTwoPhase follower wait req=%llu "
               "queue_size=%zu\n",
               static_cast<unsigned long long>(request.DebugRequestId),
               two_phase_queue.size());
      two_phase_queue_cv.wait(
          lock, [this, &request] { return request.Done || !two_phase_batch_in_progress; });
    }
  }

  const auto done_ts = std::chrono::steady_clock::now();
  const auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           done_ts - enqueue_ts)
                           .count();
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

  auto computeTargetReadSize = [this](long start, long end) -> size_t {
    assert(start >= 0 && end >= start &&
           "Coalesced read range must be monotonic.");
    size_t read_size = static_cast<size_t>(end - start);
    if (two_phase_sieve_bytes == 0)
      return read_size;

    uint64_t target_size = std::max<uint64_t>(
        static_cast<uint64_t>(read_size), two_phase_sieve_bytes);
    if (two_phase_max_batch_bytes > 0) {
      target_size = std::min<uint64_t>(target_size, two_phase_max_batch_bytes);
    }
    const uint64_t size_t_max = static_cast<uint64_t>(
        std::numeric_limits<size_t>::max());
    target_size = std::min<uint64_t>(target_size, size_t_max);
    const uint64_t long_max = static_cast<uint64_t>(
        std::numeric_limits<long>::max());
    const uint64_t start_u64 = static_cast<uint64_t>(start);
    if (start_u64 >= long_max)
      return read_size;
    const uint64_t max_extent = long_max - start_u64;
    target_size = std::min<uint64_t>(target_size, max_extent);

    if (target_size < static_cast<uint64_t>(read_size))
      return read_size;
    return static_cast<size_t>(target_size);
  };

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
    segment.Size = static_cast<uint64_t>(
        computeTargetReadSize(item.Start, item.End));
    segment.PathKey = item.Requests.front()->HasPathKey
                          ? item.Requests.front()->PathKey
                          : static_cast<uint64_t>(0);
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
          two_phase_planner_error_count.fetch_add(1, std::memory_order_relaxed);
          item.Skip = true;
          item.ErrorStatus = -1;
          item.ErrorErrno = ENOTSUP;
          io_log("Two-phase planner returned an unsupported rebalanced batch "
                 "segment: batch_id=%llu segment_id=%llu aggregator=%d\n",
                 static_cast<unsigned long long>(batch_request.BatchId),
                 static_cast<unsigned long long>(entry.SegmentId),
                 entry.AggregatorRank);
        }
        if (entry.Status != 0) {
          item.Skip = true;
          item.ErrorStatus = -1;
          item.ErrorErrno = entry.Errno != 0 ? entry.Errno : EIO;
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

    const size_t read_size = computeTargetReadSize(item.Start, item.End);
    two_phase_coalesced_bytes_total.fetch_add(read_size,
                                              std::memory_order_relaxed);
    std::vector<char> buffer(read_size);
    size_t bytes_read = 0;

    const int rc = readAtFallbackWithBytes(item.Requests.front()->FileHandle,
                                           item.Start, buffer.data(), read_size,
                                           bytes_read);
    if (rc != 0) {
      const int errnum = errno;
      io_trace("MPIIOBackend::processTwoPhaseGroup coalesced read failed "
               "start=%ld end=%ld size=%zu rc=%d errno=%d\n",
               item.Start, item.End, read_size, rc, errnum);
      for (TwoPhaseReadRequest *request : item.Requests)
        completeTwoPhaseRequest(*request, rc, errnum);
      continue;
    }

    const uint64_t cache_key = getTwoPhaseGroupKey(*item.Requests.front());
    updateTwoPhaseReadCache(cache_key, item.Start, buffer);

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
  return static_cast<uint64_t>(static_cast<uint32_t>(request.FileHandle));
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

int MPIIOBackend::writeAt(int file_id, long offset, const void *data,
                          size_t size) {
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
  invalidateTwoPhaseReadCacheForFile(file_id);

  io_log("Writing %zu bytes to file %d at offset %lld\n", size, file_id,
         static_cast<long long>(offset));

  if (isWriteBatchActive())
    return writeAtBatched(file_id, offset, data, size);

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

  io_log("Write at offset completed\n");
  return 0;
}

int MPIIOBackend::writeAtBatched(int file_id, long offset, const void *data,
                                 size_t size) {
  WriteBatchRequest request{};
  request.DebugRequestId =
      write_batch_request_id.fetch_add(1, std::memory_order_relaxed);
  request.FileHandle = file_id;
  request.Offset = offset;
  request.Data.resize(size);
  if (size > 0)
    std::memcpy(request.Data.data(), data, size);

  std::vector<WriteBatchRequest *> batch;
  std::unique_lock<std::mutex> lock(write_batch_mutex);
  write_batch_queue.push_back(&request);
  write_batch_queue_cv.notify_all();

  while (!request.Done) {
    if (!write_batch_in_progress) {
      write_batch_in_progress = true;
      if (write_batch_window_us > 0) {
        write_batch_queue_cv.wait_for(
            lock, std::chrono::microseconds(write_batch_window_us));
      }

      while (!write_batch_queue.empty()) {
        batch.push_back(write_batch_queue.front());
        write_batch_queue.pop_front();
      }

      lock.unlock();
      processWriteBatch(batch);
      batch.clear();
      lock.lock();

      write_batch_in_progress = false;
      write_batch_queue_cv.notify_all();
    } else {
      write_batch_queue_cv.wait(lock, [this, &request] {
        return request.Done || !write_batch_in_progress;
      });
    }
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

bool MPIIOBackend::getFilePathKey(int file_id, uint64_t &path_key_out) {
  const std::lock_guard<std::mutex> lock(handle_mutex);
  const auto it = file_path_key_map.find(file_id);
  if (it == file_path_key_map.end())
    return false;
  path_key_out = it->second;
  return true;
}

void MPIIOBackend::rememberFilePathKey(int file_id, const char *path) {
  const uint64_t path_key = computePathKey(path);
  {
    const std::lock_guard<std::mutex> lock(handle_mutex);
    file_path_key_map[file_id] = path_key;
  }
  invalidateTwoPhaseReadCacheKey(path_key);
}

void MPIIOBackend::forgetFilePathKey(int file_id) {
  const std::lock_guard<std::mutex> lock(handle_mutex);
  file_path_key_map.erase(file_id);
}

uint64_t MPIIOBackend::resolveTwoPhaseKey(
    const ompfile::OmpFileReadRequestContext &context) {
  if ((context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY) != 0) {
    return context.PathKey;
  }

  uint64_t path_key = 0;
  if (getFilePathKey(context.FileHandle, path_key))
    return path_key;

  return static_cast<uint64_t>(static_cast<uint32_t>(context.FileHandle));
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
  if (getFilePathKey(file_id, path_key)) {
    invalidateTwoPhaseReadCacheKey(path_key);
    return;
  }
  invalidateTwoPhaseReadCacheKey(
      static_cast<uint64_t>(static_cast<uint32_t>(file_id)));
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
         "planner_rebalanced=%llu planner_scalar_fallbacks=%llu "
         "planner_errors=%llu cache_hits=%llu fallbacks=%llu "
         "write_batch_count=%llu coalesced_writes=%llu "
         "coalesced_write_bytes=%llu coalesced_write_avg_bytes=%.2f "
         "write_batch_saved_events=%llu write_batch_failures=%llu "
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
