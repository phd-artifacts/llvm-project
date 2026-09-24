#include "abstract_backend.h"
#include "debug_log.h"
#include "ompfile_env.h"
#include "file_interface.h"
#include "mpp_shim.h"
#include "ompfile_sched.h"
#include "ompfile_trace.h"
#include "mpi.h"
#include "mpi_io_backend.h"
#include "posix_backend.h"
#include "io_uring_io_backend.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <string>
#include <condition_variable>
#include <deque>
#include <functional>
#include <vector>

enum class IOSchedulerTy {
  LOCAL,
  HEADNODE,
};

static const char *schedulerOpToString(ompfile::OmpFileIOOp op) {
  switch (op) {
  case ompfile::OmpFileIOOp::OPEN:
    return "OPEN";
  case ompfile::OmpFileIOOp::CLOSE:
    return "CLOSE";
  case ompfile::OmpFileIOOp::PREAD:
    return "PREAD";
  case ompfile::OmpFileIOOp::PWRITE:
    return "PWRITE";
  case ompfile::OmpFileIOOp::PREFETCH:
    return "PREFETCH";
  }
  return "UNKNOWN";
}

static const char *backendTypeToString(IOBackendTy backend_type) {
  switch (backend_type) {
  case IOBackendTy::MPI:
    return "MPI";
  case IOBackendTy::POSIX:
    return "POSIX";
  case IOBackendTy::IO_URING:
    return "IO_URING";
  case IOBackendTy::HDF5:
    return "HDF5";
  }
  return "UNKNOWN";
}

static bool envFlagEnabled(const char *name) {
  return ompfile::env::flag(name);
}

// LIBOMPFILE_ASYNC_READ_WAIT=range: a read (or a synchronous write) waits only
// for the queued async writes that could cover its own byte range of the same
// file, instead of draining the whole engine. Off (drain) by default; see the
// AsyncWriteEngine::waitForRange comment for the contract.
static bool asyncReadWaitIsRange() {
  static const bool range = [] {
    const char *value = std::getenv("LIBOMPFILE_ASYNC_READ_WAIT");
    return value && std::string(value) == "range";
  }();
  return range;
}

static bool isMppRemoteOnlyEnabled() {
  return MPIIOBackend::parseBoolEnv("LIBOMPFILE_MPP_OPEN", false) &&
         MPIIOBackend::parseBoolEnv("LIBOMPFILE_MPP_IO", false);
}

static void applyIoHint(ompfile::OmpFileIOHint &dst,
                        const omp_file_io_hint_v1 *src) {
  dst = {};
  if (!src)
    return;
  if (src->abi_version != OMPFILE_IO_HINT_ABI_VERSION)
    return;
  dst.AbiVersion = src->abi_version;
  dst.HintFlags = src->hint_flags;
  dst.EpochId = src->epoch_id;
  dst.StreamId = src->stream_id;
  dst.TileId = src->tile_id;
  dst.Role = src->role;
}

class IOScheduler {
public:
  virtual ~IOScheduler() {}
  virtual int open(const char *filename) = 0;
  virtual int write(int file_handle, const void *data, size_t size) = 0;
  virtual int read(int file_handle, void *data, size_t size) = 0;
  virtual int close(int file_handle) = 0;
  virtual int commit(int file_handle) = 0;
  virtual int seek(int file_handle, long offset) = 0;
  virtual int readAt(int file_handle, long offset, void *data, size_t size) = 0;
  virtual int readAtHint(int file_handle, long offset, void *data, size_t size,
                         const ompfile::OmpFileIOHint *hint) = 0;
  virtual int writeAt(int file_handle, long offset, const void *data,
                      size_t size) = 0;
  virtual int writeAtHint(int file_handle, long offset, const void *data,
                          size_t size,
                          const ompfile::OmpFileIOHint *hint) = 0;
};

class LocalScheduler final : public IOScheduler {
public:
  explicit LocalScheduler(IOBackend &backend) : backend(backend) {}

  int open(const char *filename) override { return backend.open(filename); }
  int write(int file_handle, const void *data, size_t size) override {
    return backend.write(file_handle, data, size);
  }
  int read(int file_handle, void *data, size_t size) override {
    return backend.read(file_handle, data, size);
  }
  int close(int file_handle) override { return backend.close(file_handle); }
  int commit(int file_handle) override { return backend.commit(file_handle); }
  int seek(int file_handle, long offset) override {
    return backend.seek(file_handle, offset);
  }
  int readAt(int file_handle, long offset, void *data, size_t size) override {
    return backend.readAt(file_handle, offset, data, size);
  }
  int readAtHint(int file_handle, long offset, void *data, size_t size,
                 const ompfile::OmpFileIOHint *hint) override {
    ompfile::OmpFileReadRequestContext context{};
    context.FileHandle = file_handle;
    context.Offset = static_cast<int64_t>(offset);
    context.Size = static_cast<uint64_t>(size);
    if (hint)
      context.Hint = *hint;
    return backend.readAtWithContext(context, data, size);
  }
  int writeAt(int file_handle, long offset, const void *data,
              size_t size) override {
    return backend.writeAt(file_handle, offset, data, size);
  }
  int writeAtHint(int file_handle, long offset, const void *data, size_t size,
                  const ompfile::OmpFileIOHint *hint) override {
    ompfile::OmpFileWriteRequestContext context{};
    context.FileHandle = file_handle;
    context.Offset = static_cast<int64_t>(offset);
    context.Size = static_cast<uint64_t>(size);
    if (hint)
      context.Hint = *hint;
    return backend.writeAtWithContext(context, data, size);
  }

private:
  IOBackend &backend;
};

// Express lane for the owner-bypass contract: when the app asserts disjoint
// accesses (LIBOMPFILE_OPT_LOCAL_DISJOINT_*), the headnode's per-op
// scheduling round trip (scheduleWrite / read planning, one MPI event each,
// serialized on sched_request_mutex) only computes owner/aggregator routing
// that the bypass then ignores — a pure ~2 ms fixed tax per op (AMD job
// 359359). With express on, per-op writeAt/readAt skip scheduling and go
// straight to the backend; open/close scheduling stays (handles and owner
// state still need it), and the proxy-side bypass keeps its fail-closed
// owner-forward fallback.
static bool localDisjointExpressEnabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("LIBOMPFILE_OPT_LOCAL_DISJOINT_EXPRESS");
    return env && env[0] == '1' && env[1] == '\0';
  }();
  return enabled;
}

class HeadnodeScheduler final : public IOScheduler {
public:
  explicit HeadnodeScheduler(IOBackend &backend)
      : backend(backend), client_rank(resolveClientRank()),
        mpp_remote_only_enabled(isMppRemoteOnlyEnabled()),
        allow_fallback(envFlagEnabled("LIBOMPFILE_ALLOWFALLBACK")),
        strict_mpp_required(mpp_remote_only_enabled && !allow_fallback),
        mpp_sched_active(mpp_remote_only_enabled && ompfile::mpp::init()),
        two_phase_batch_preferred(mpp_sched_active &&
                                  shouldPreferBatchReadPlanner()) {
    if (!mpp_remote_only_enabled) {
      io_log("HEADNODE scheduler requested but MPP remote-only mode is "
             "disabled (requires LIBOMPFILE_MPP_OPEN=1 and "
             "LIBOMPFILE_MPP_IO=1); using LOCAL dispatch.\n");
    } else if (!mpp_sched_active) {
      if (allow_fallback) {
        io_log("HEADNODE scheduler requested but MPP init failed; "
               "LIBOMPFILE_ALLOWFALLBACK=1, using LOCAL dispatch.\n");
      } else {
      io_log("HEADNODE scheduler requested but MPP init failed; fallback is "
             "disabled, all scheduler operations will fail.\n");
      }
    }
    if (two_phase_batch_preferred) {
      io_log("HEADNODE scheduler: two-phase batch planner active; skipping "
             "scalar per-request scheduleRead path.\n");
    }
  }

  int open(const char *filename) override {
    if (!ensureSchedulerReady("open", /*log_once=*/true))
      return -1;
    ompfile::OmpFileIOPlan open_plan{};
    ompfile::OmpFileIOPlan *open_plan_ptr = nullptr;
    if (mpp_sched_active) {
      if (!scheduleOpen(filename, &open_plan))
        return failStrict("open");
      if (open_plan.Status == 0 && open_plan.AggregatorRank >= 0)
        open_plan_ptr = &open_plan;
    }
    const int file_handle = backend.openWithPlan(filename, open_plan_ptr);
    if (file_handle >= 0 && filename)
      rememberOpenPath(file_handle, filename);
    return file_handle;
  }

  int write(int file_handle, const void *data, size_t size) override {
    if (!ensureSchedulerReady("write", /*log_once=*/true))
      return -1;
    if (mpp_sched_active &&
        !scheduleWrite(file_handle, /*offset=*/0, size, nullptr))
      return failStrict("write");
    return backend.write(file_handle, data, size);
  }

  int read(int file_handle, void *data, size_t size) override {
    if (!ensureSchedulerReady("read", /*log_once=*/true))
      return -1;
    if (mpp_sched_active &&
        !scheduleRead(file_handle, /*offset=*/0, size, nullptr))
      return failStrict("read");
    return backend.read(file_handle, data, size);
  }

  // Commit needs no scheduler round trip: it changes no open/close state and
  // the backend owns the headnode dirty-tile flush protocol it delegates to.
  int commit(int file_handle) override { return backend.commit(file_handle); }

  int close(int file_handle) override {
    bool schedule_ok = true;
    if (!ensureSchedulerReady("close", /*log_once=*/true))
      schedule_ok = false;
    if (mpp_sched_active)
      schedule_ok = scheduleClose(file_handle);
    const int rc = backend.close(file_handle);
    if (rc == 0)
      forgetOpenPath(file_handle);
    if (!schedule_ok && strict_mpp_required)
      return failStrict("close");
    return rc;
  }

  int seek(int file_handle, long offset) override {
    if (!ensureSchedulerReady("seek", /*log_once=*/true))
      return -1;
    return backend.seek(file_handle, offset);
  }

  int readAt(int file_handle, long offset, void *data, size_t size) override {
    if (!mpp_sched_active) {
      if (!ensureSchedulerReady("readAt", /*log_once=*/true))
        return -1;
      return backend.readAt(file_handle, offset, data, size);
    }
    if (localDisjointExpressEnabled())
      return backend.readAt(file_handle, offset, data, size);
    ompfile::OmpFileReadRequestContext context{};
    if (!buildReadContext(file_handle, offset, size, nullptr, context))
      return failStrict("readAt");
    return backend.readAtWithContext(context, data, size);
  }

  int readAtHint(int file_handle, long offset, void *data, size_t size,
                 const ompfile::OmpFileIOHint *hint) override {
    ompfile::OmpFileReadRequestContext context{};
    if (!buildReadContext(file_handle, offset, size, hint, context))
      return failStrict("readAtHint");
    return backend.readAtWithContext(context, data, size);
  }

  int writeAt(int file_handle, long offset, const void *data,
              size_t size) override {
    if (!mpp_sched_active) {
      if (!ensureSchedulerReady("writeAt", /*log_once=*/true))
        return -1;
      return backend.writeAt(file_handle, offset, data, size);
    }
    if (!localDisjointExpressEnabled() &&
        !scheduleWrite(file_handle, offset, size, nullptr))
      return failStrict("writeAt");
    return backend.writeAt(file_handle, offset, data, size);
  }

  int writeAtHint(int file_handle, long offset, const void *data, size_t size,
                  const ompfile::OmpFileIOHint *hint) override {
    ompfile::OmpFileWriteRequestContext context{};
    if (!buildWriteContext(file_handle, offset, size, hint, context))
      return -1;
    if (!mpp_sched_active) {
      if (!ensureSchedulerReady("writeAtHint", /*log_once=*/true))
        return -1;
      return backend.writeAtWithContext(context, data, size);
    }
    if (mpp_sched_active && !localDisjointExpressEnabled() &&
        !scheduleWrite(file_handle, offset, size, hint))
      return failStrict("writeAtHint");
    return backend.writeAtWithContext(context, data, size);
  }

private:
  struct TrackedFileMetadata {
    std::string Path;
    uint64_t PathKey = 0;
  };

  IOBackend &backend;
  int client_rank = -1;
  const bool mpp_remote_only_enabled = false;
  const bool allow_fallback = false;
  const bool strict_mpp_required = false;
  const bool mpp_sched_active = false;
  const bool two_phase_batch_preferred = false;
  std::atomic<uint64_t> request_id{1};
  std::atomic<bool> strict_failure_logged{false};
  std::mutex tracked_file_mutex;
  std::mutex sched_request_mutex;
  std::unordered_map<int, TrackedFileMetadata> tracked_file_map;

  static int resolveClientRank() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
      return -1;
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  static bool shouldPreferBatchReadPlanner() {
    const char *raw = std::getenv("LIBOMPFILE_OPT_TWO_PHASE");
    if (!raw || !raw[0])
      return true; // auto policy under HEADNODE+remote-only

    if ((raw[0] == '0' && raw[1] == '\0') ||
        std::strcmp(raw, "disabled") == 0 ||
        std::strcmp(raw, "DISABLED") == 0)
      return false;

    if ((raw[0] == '1' && raw[1] == '\0') ||
        std::strcmp(raw, "enabled") == 0 ||
        std::strcmp(raw, "ENABLED") == 0 ||
        std::strcmp(raw, "auto") == 0 ||
        std::strcmp(raw, "AUTO") == 0)
      return true;

    io_log("Invalid LIBOMPFILE_OPT_TWO_PHASE='%s' for HEADNODE scheduler; "
           "keeping scalar scheduleRead path.\n",
           raw);
    return false;
  }

  int failStrict(const char *op_name) {
    errno = EIO;
    if (!strict_failure_logged.exchange(true, std::memory_order_relaxed)) {
      io_log("HEADNODE scheduler operation '%s' failed because MPP is required "
             "and fallback is disabled. Set LIBOMPFILE_ALLOWFALLBACK=1 to "
             "restore local fallback.\n",
             op_name ? op_name : "(unknown)");
    }
    io_trace("HeadnodeScheduler::failStrict scheduler=%p op=%s\n",
             static_cast<void *>(this), op_name ? op_name : "(unknown)");
    return -1;
  }

  bool ensureSchedulerReady(const char *op_name, bool log_once) {
    if (!strict_mpp_required)
      return true;
    if (mpp_sched_active)
      return true;
    if (!log_once || !strict_failure_logged.load(std::memory_order_relaxed)) {
      io_trace("HeadnodeScheduler::ensureSchedulerReady fail scheduler=%p op=%s "
               "strict=%d mpp_remote_only=%d mpp_sched_active=%d\n",
               static_cast<void *>(this), op_name ? op_name : "(unknown)",
               static_cast<int>(strict_mpp_required),
               static_cast<int>(mpp_remote_only_enabled),
               static_cast<int>(mpp_sched_active));
    }
    failStrict(op_name);
    return false;
  }

  static uint64_t computePathKey(const char *path) {
    // 64-bit FNV-1a hash for a stable per-path key in scheduler context.
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

  void rememberOpenPath(int file_handle, const char *path) {
    if (!path)
      return;
    TrackedFileMetadata metadata{};
    metadata.Path = path;
    metadata.PathKey = computePathKey(path);
    const std::lock_guard<std::mutex> lock(tracked_file_mutex);
    tracked_file_map[file_handle] = std::move(metadata);
  }

  void forgetOpenPath(int file_handle) {
    const std::lock_guard<std::mutex> lock(tracked_file_mutex);
    tracked_file_map.erase(file_handle);
  }

  bool getTrackedPath(int file_handle, std::string &path_out,
                      uint64_t &path_key_out) {
    const std::lock_guard<std::mutex> lock(tracked_file_mutex);
    auto it = tracked_file_map.find(file_handle);
    if (it == tracked_file_map.end())
      return false;
    path_out = it->second.Path;
    path_key_out = it->second.PathKey;
    return true;
  }

  bool schedRequest(const ompfile::OmpFileIORequest &req,
                    const char *path, ompfile::OmpFileIOPlan *plan_out) {
    if (!mpp_sched_active)
      return false;
    ompfile::OmpFileIOPlan plan{};
    io_trace("HeadnodeScheduler::schedRequest enter scheduler=%p req_id=%llu "
             "op=%s client_rank=%d file=%d offset=%lld size=%llu path_size=%u "
             "path=%s\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(req.RequestId),
             schedulerOpToString(req.Op), req.ClientRank, req.FileHandle,
             static_cast<long long>(req.Offset),
             static_cast<unsigned long long>(req.Size), req.PathSize,
             path ? path : "(null)");
    {
      const std::lock_guard<std::mutex> lock(sched_request_mutex);
      if (!ompfile::mpp::schedRequest(req, path, plan)) {
        if (strict_mpp_required) {
          errno = EIO;
          io_log("HEADNODE scheduler request failed (global manager "
                 "unavailable); fallback is disabled.\n");
        } else {
          io_log("HEADNODE scheduler request failed (global manager "
                 "unavailable); falling back to LOCAL.\n");
        }
        io_trace("HeadnodeScheduler::schedRequest mpp call failed req_id=%llu\n",
                 static_cast<unsigned long long>(req.RequestId));
        return false;
      }
    }

    if (plan.Status != 0) {
      errno = plan.Errno;
      io_trace("HeadnodeScheduler::schedRequest plan error req_id=%llu "
               "status=%d errno=%d\n",
               static_cast<unsigned long long>(req.RequestId), plan.Status,
               plan.Errno);
      return false;
    }

    if (plan_out)
      *plan_out = plan;
    io_trace("HeadnodeScheduler::schedRequest success req_id=%llu "
             "aggregator_rank=%d remote_handle=%d plan_flags=0x%x\n",
             static_cast<unsigned long long>(req.RequestId),
             plan.AggregatorRank, plan.RemoteHandle, plan.PlanFlags);
    return true;
  }

  bool scheduleOpen(const char *path, ompfile::OmpFileIOPlan *plan_out) {
    if (!path)
      return true;
    ompfile::OmpFileIORequest req{};
    req.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    req.Op = ompfile::OmpFileIOOp::OPEN;
    req.ClientRank = client_rank;
    req.PathSize = static_cast<uint32_t>(std::strlen(path) + 1);
    io_trace("HeadnodeScheduler::scheduleOpen scheduler=%p req_id=%llu path=%s\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(req.RequestId), path);
    return schedRequest(req, path, plan_out) || !strict_mpp_required;
  }

  bool scheduleClose(int file_handle) {
    ompfile::OmpFileIORequest req{};
    req.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    req.Op = ompfile::OmpFileIOOp::CLOSE;
    req.ClientRank = client_rank;
    req.FileHandle = file_handle;
    io_trace("HeadnodeScheduler::scheduleClose scheduler=%p req_id=%llu "
             "file=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(req.RequestId), file_handle);
    return schedRequest(req, nullptr, nullptr) || !strict_mpp_required;
  }

  bool buildReadContext(int file_handle, long offset, size_t size,
                        const ompfile::OmpFileIOHint *hint,
                        ompfile::OmpFileReadRequestContext &context) {
    context = {};
    context.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    context.FileHandle = file_handle;
    context.ClientRank = client_rank;
    context.Offset = static_cast<int64_t>(offset);
    context.Size = static_cast<uint64_t>(size);
    context.PathKey = static_cast<uint64_t>(static_cast<uint32_t>(file_handle));
    if (hint)
      context.Hint = *hint;

    std::string tracked_path;
    if (getTrackedPath(file_handle, tracked_path, context.PathKey))
      context.ContextFlags |= ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY;

    ompfile::OmpFileIORequest req{};
    req.RequestId = context.RequestId;
    req.Op = ompfile::OmpFileIOOp::PREAD;
    req.ClientRank = client_rank;
    req.FileHandle = file_handle;
    req.Offset = static_cast<int64_t>(offset);
    req.Size = static_cast<uint64_t>(size);
    if (hint) {
      req.HintFlags = hint->HintFlags;
      req.EpochId = hint->EpochId;
      req.StreamId = hint->StreamId;
      req.TileId = hint->TileId;
    }
    io_trace("HeadnodeScheduler::buildReadContext scheduler=%p req_id=%llu "
             "file=%d offset=%lld size=%llu has_path_key=%d path_key=%llu\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(context.RequestId), file_handle,
             static_cast<long long>(offset),
             static_cast<unsigned long long>(size),
             (context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY) !=
                 0,
             static_cast<unsigned long long>(context.PathKey));

    if (two_phase_batch_preferred || localDisjointExpressEnabled()) {
      io_trace("HeadnodeScheduler::buildReadContext skip-scalar req_id=%llu "
               "file=%d offset=%lld size=%llu\n",
               static_cast<unsigned long long>(context.RequestId), file_handle,
               static_cast<long long>(offset),
               static_cast<unsigned long long>(size));
      return true;
    }

    ompfile::OmpFileIOPlan plan{};
    if (!schedRequest(req, nullptr, &plan))
      return !strict_mpp_required;

    context.ContextFlags |= ompfile::OMPFILE_READ_CTX_HAS_PLAN;
    context.Plan = plan;
    return true;
  }

  bool scheduleRead(int file_handle, long offset, size_t size,
                    const ompfile::OmpFileIOHint *hint) {
    ompfile::OmpFileReadRequestContext context{};
    return buildReadContext(file_handle, offset, size, hint, context);
  }

  bool scheduleWrite(int file_handle, long offset, size_t size,
                     const ompfile::OmpFileIOHint *hint) {
    ompfile::OmpFileIORequest req{};
    req.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    req.Op = ompfile::OmpFileIOOp::PWRITE;
    req.ClientRank = client_rank;
    req.FileHandle = file_handle;
    req.Offset = static_cast<int64_t>(offset);
    req.Size = static_cast<uint64_t>(size);
    if (hint) {
      req.HintFlags = hint->HintFlags;
      req.EpochId = hint->EpochId;
      req.StreamId = hint->StreamId;
      req.TileId = hint->TileId;
    }

    std::string tracked_path;
    uint64_t ignored_path_key = 0;
    const bool has_path = getTrackedPath(file_handle, tracked_path, ignored_path_key);
    if (has_path && !tracked_path.empty())
      req.PathSize = static_cast<uint32_t>(tracked_path.size() + 1);

    return schedRequest(req, has_path ? tracked_path.c_str() : nullptr,
                        nullptr) ||
           !strict_mpp_required;
  }

  bool buildWriteContext(int file_handle, long offset, size_t size,
                         const ompfile::OmpFileIOHint *hint,
                         ompfile::OmpFileWriteRequestContext &context) {
    context = {};
    context.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    context.FileHandle = file_handle;
    context.ClientRank = client_rank;
    context.Offset = static_cast<int64_t>(offset);
    context.Size = static_cast<uint64_t>(size);
    context.PathKey = static_cast<uint64_t>(static_cast<uint32_t>(file_handle));
    if (hint)
      context.Hint = *hint;

    std::string tracked_path;
    if (getTrackedPath(file_handle, tracked_path, context.PathKey))
      context.ContextFlags |= ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY;
    return true;
  }
};

static int resolveDefaultIOTokens() {
  const char *env = std::getenv("LIBOMPFILE_IO_TOKENS");
  if (env) {
    try {
      const int parsed = std::stoi(env);
      if (parsed > 0)
        return parsed;
    } catch (...) {
    }
    io_log("Invalid LIBOMPFILE_IO_TOKENS='%s'; using runtime default.\n", env);
  }

  int tokens = 4;
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized)
    return tokens;

  int thread_level = MPI_THREAD_SINGLE;
  MPI_Query_thread(&thread_level);
  if (thread_level < MPI_THREAD_MULTIPLE) {
    tokens = 1;
    io_log("MPI thread level=%d; defaulting LIBOMPFILE_IO_TOKENS to %d for "
           "safe scheduler serialization (override with LIBOMPFILE_IO_TOKENS).\n",
           thread_level, tokens);
  }

  return tokens;
}

static IOSchedulerTy getSchedulerType() {
  const char *env = std::getenv("LIBOMPFILE_SCHEDULER");
  if (!env) {
    return IOSchedulerTy::LOCAL;
  }

  std::string envStr(env);
  if (envStr == "LOCAL") {
    return IOSchedulerTy::LOCAL;
  }
  if (envStr == "HEADNODE") {
    return IOSchedulerTy::HEADNODE;
  }

  io_log("Unknown LIBOMPFILE_SCHEDULER '%s', defaulting to LOCAL\n", env);
  return IOSchedulerTy::LOCAL;
}

static std::mutex &getClientContextInitMutex() {
  static std::mutex init_mutex;
  return init_mutex;
}

// Background write engine that turns async-issued writes into a producer/
// consumer pipeline: the issuing (target-region) thread copies the payload,
// enqueues it, and returns immediately, so it can dispatch the next write while
// a single worker thread drains the previous one through the normal scheduler
// path. This attacks overhead #3 ("no pipelining") — issue of write N+1 now
// overlaps the drain of write N instead of stalling behind it.
//
// Safety: the proxy initializes MPI at MPI_THREAD_MULTIPLE, so the worker may
// call MPI concurrently. Even so, we keep exactly one worker and never let the
// issuing thread touch the scheduler while writes are queued: it only enqueues
// during a write wave and drains the engine at every sync point (open/close/
// read/seek). That keeps scheduler/headnode state single-threaded in practice
// while still overlapping issue with drain. If the runtime is ever initialized
// below MPI_THREAD_SERIALIZED, async falls back to synchronous execution.
class AsyncWriteEngine {
public:
  struct Task {
    int handle = -1;
    long offset = 0;
    bool has_offset = false; // pwrite when true, sequential write when false
    bool has_hint = false;
    ompfile::OmpFileIOHint hint{};
    // The payload, one of two ways. `data` is the copy `enqueue` takes on the
    // omp_file_pwrite path. `owned` is the caller's own buffer on the
    // omp_file_pwrite_owned path: written from in place, then handed to
    // `release` once the write has run, so the issuing thread never pays for
    // a memcpy of the payload. Exactly one of the two carries the bytes.
    std::vector<char> data;
    void *owned = nullptr;
    size_t owned_size = 0;
    void (*release)(void *) = nullptr;

    const void *payload() const { return owned ? owned : data.data(); }
    size_t payloadSize() const { return owned ? owned_size : data.size(); }

    // Hands an owned buffer back. Called by the worker after the executor
    // returns (success or failure alike), never under the engine mutex: the
    // callback is the application's. Not called from the destructor, because
    // a task that was refused by `enqueue` is still the caller's buffer.
    void releaseOwned() {
      if (!owned)
        return;
      assert(release && "owned task without a release callback");
      release(owned);
      owned = nullptr;
      owned_size = 0;
    }
  };

  // Executes one queued task synchronously; returns the backend rc.
  using Executor = std::function<int(const Task &)>;

  // A completion unit the caller can wait on: one handle plus one epoch id
  // taken from the write's hint. Writes issued without an epoch belong to the
  // handle alone and are waited on by flushHandle, never by flushEpoch.
  using EpochKey = std::pair<int, uint64_t>;

  static bool taskHasEpoch(const Task &task) {
    return task.has_hint &&
           (task.hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_EPOCH) != 0;
  }

  static EpochKey taskEpochKey(const Task &task) {
    return EpochKey{task.handle, task.hint.EpochId};
  }

  AsyncWriteEngine() = default;
  ~AsyncWriteEngine() { shutdown(); }

  AsyncWriteEngine(const AsyncWriteEngine &) = delete;
  AsyncWriteEngine &operator=(const AsyncWriteEngine &) = delete;

  void configure(Executor exec) {
    executor_ = std::move(exec);
    max_depth_ = resolveDepth();
  }

  // Lazily decides (once) whether the async worker path is usable. When it
  // returns false the caller must run the write synchronously.
  bool available() {
    std::call_once(available_once_, [this] { enabled_ = resolveEnabled(); });
    return enabled_;
  }

  // Copies and queues a write. Returns 0 on accept, or the rc of an earlier
  // failure *within this write's own completion unit* — its epoch when the
  // write carries one, otherwise its handle — so the caller can stop early
  // without an unrelated branch's failure stopping it.
  int enqueue(Task &&task) {
    std::unique_lock<std::mutex> lock(mtx_);
    // Refuse only writes whose own completion unit already failed: an epoched
    // write is blocked by its epoch, an un-epoched one by its handle (that is
    // the scope it can be waited on at). A failure on an unrelated branch must
    // not stop this one; `drain()` at close still reports engine-wide.
    const bool has_epoch = taskHasEpoch(task);
    if (has_epoch) {
      auto it = epoch_error_.find(taskEpochKey(task));
      if (it != epoch_error_.end() && it->second != 0)
        return it->second;
    } else {
      auto it = handle_error_.find(task.handle);
      if (it != handle_error_.end() && it->second != 0)
        return it->second;
    }
    startWorkerLocked();
    if (queue_.size() >= max_depth_ && !stop_) {
      // Back-pressure: the issuing thread blocks here when the bounded queue
      // is full, which is the other place (besides drain) where async issue
      // serializes against the drain; report it so a lane can see which.
      ++enqueue_waits_;
      const auto start = std::chrono::steady_clock::now();
      not_full_.wait(lock,
                     [&] { return queue_.size() < max_depth_ || stop_; });
      enqueue_wait_ns_ += static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count());
    }
    if (stop_)
      return -1;
    pending_[task.handle]++;
    if (has_epoch)
      pending_epoch_[taskEpochKey(task)]++;
    if (task.owned) {
      ++owned_enqueued_;
      owned_bytes_ += task.owned_size;
    }
    queue_.push_back(std::move(task));
    ++enqueued_;
    max_queue_depth_ = std::max(max_queue_depth_, queue_.size());
    has_work_.notify_one();
    return 0;
  }

  bool active() {
    std::lock_guard<std::mutex> lock(mtx_);
    return worker_started_ && (!queue_.empty() || busy_);
  }

  // Waits until every queued write for `handle` has drained, then reports
  // only that handle's own first failing rc. A failure queued on an unrelated
  // handle must not be charged here; `drain()` (the close/read/seek join point)
  // still reports the engine-wide sticky rc.
  int flushHandle(int handle) {
    assert(handle >= 0 && "flushHandle expects a caller-validated handle");
    std::unique_lock<std::mutex> lock(mtx_);
    if (!worker_started_)
      return 0; // nothing was ever queued, so nothing can be outstanding
    handle_done_.wait(lock, [&] {
      auto it = pending_.find(handle);
      return it == pending_.end() || it->second == 0;
    });
    auto err = handle_error_.find(handle);
    return err == handle_error_.end() ? 0 : err->second;
  }

  // Waits until every queued write tagged with `epoch` on `handle` has
  // drained, then reports only that epoch's own first failing rc. Writes
  // issued without an epoch hint are not waited on here — they have no epoch
  // to belong to, so flushHandle is their boundary.
  int flushEpoch(int handle, uint64_t epoch) {
    assert(handle >= 0 && "flushEpoch expects a caller-validated handle");
    const EpochKey key{handle, epoch};
    std::unique_lock<std::mutex> lock(mtx_);
    if (!worker_started_)
      return 0; // nothing was ever queued, so nothing can be outstanding
    handle_done_.wait(lock, [&] {
      auto it = pending_epoch_.find(key);
      return it == pending_epoch_.end() || it->second == 0;
    });
    auto err = epoch_error_.find(key);
    return err == epoch_error_.end() ? 0 : err->second;
  }

  // Drops the failure records for a handle and every epoch under it. Called
  // when a handle is closed so a recycled file id does not inherit the
  // previous owner's errors.
  void forgetHandle(int handle) {
    std::lock_guard<std::mutex> lock(mtx_);
    handle_error_.erase(handle);
    eraseHandleRange(pending_epoch_, handle);
    eraseHandleRange(epoch_error_, handle);
  }

  // Range-scoped alternative to drain() for a read or synchronous write of
  // [offset, offset+size) on a file that `handles` are open on: waits until
  // no queued or in-flight task on one of those handles could cover that
  // range, and returns at once when none can. A sequential (no-offset) write
  // has no known range and always counts. This is the wait a dependency
  // actually needs — a read of bytes nobody has queued a write to proceeds
  // while the worker drains unrelated ranges of the same file — and the
  // engine keeps FIFO order among the writes themselves, so two queued writes
  // to one range still land in issue order. Opt-in through
  // LIBOMPFILE_ASYNC_READ_WAIT=range; the caller passes every handle open on
  // the read's path, since a file open twice is still one file.
  void waitForRange(const std::vector<int> &handles, long offset,
                    size_t size) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!worker_started_)
      return;
    const auto covers = [&](int handle, bool has_offset, long task_offset,
                            size_t task_size) {
      if (std::find(handles.begin(), handles.end(), handle) == handles.end())
        return false;
      if (!has_offset)
        return true; // cursor write: range unknown, be conservative
      const long lo = task_offset;
      const long hi = task_offset + static_cast<long>(task_size);
      return lo < offset + static_cast<long>(size) && offset < hi;
    };
    const auto pending = [&] {
      if (current_.active &&
          covers(current_.handle, current_.has_offset, current_.offset,
                 current_.size))
        return true;
      for (const Task &t : queue_)
        if (covers(t.handle, t.has_offset, t.offset, t.payloadSize()))
          return true;
      return false;
    };
    if (!pending()) {
      // Counted only when something was outstanding: these are the reads a
      // drain would have stalled and this wait let through.
      if (current_.active || !queue_.empty())
        ++range_wait_free_;
      return;
    }
    ++range_waits_;
    const auto start = std::chrono::steady_clock::now();
    handle_done_.wait(lock, [&] { return !pending(); });
    range_wait_ns_ += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  }

  // Waits until the whole queue has drained and the worker is idle.
  int drain() {
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mtx_);
    if (!worker_started_)
      return sticky_error_;
    ++drain_calls_;
    idle_.wait(lock, [&] { return queue_.empty() && !busy_; });
    drain_wait_ns_ += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    return sticky_error_;
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stop_ = true;
      has_work_.notify_all();
      not_full_.notify_all();
    }
    if (worker_.joinable())
      worker_.join();
    // Under LIBOMPFILE_OPT_STATS as well as the trace flag, so a benchmark
    // lane can assert from its log that the owned (zero-copy) submit engaged.
    // Once only: shutdown runs from the client context's destructor and again
    // from this engine's own, and a lane sums the token over the log.
    if (stats_reported_)
      return;
    stats_reported_ = true;
    if (envFlagEnabled("LIBOMPFILE_ASYNC_TRACE") ||
        envFlagEnabled("LIBOMPFILE_OPT_STATS"))
      io_report("Async IO stats: enqueued=%llu completed=%llu "
             "max_queue_depth=%zu drain_calls=%llu drain_wait_ns=%llu "
             "owned_enqueued=%llu owned_bytes=%llu range_waits=%llu "
             "range_wait_ns=%llu range_wait_free=%llu enqueue_waits=%llu "
             "enqueue_wait_ns=%llu\n",
             static_cast<unsigned long long>(enqueued_),
             static_cast<unsigned long long>(completed_), max_queue_depth_,
             static_cast<unsigned long long>(drain_calls_),
             static_cast<unsigned long long>(drain_wait_ns_),
             static_cast<unsigned long long>(owned_enqueued_),
             static_cast<unsigned long long>(owned_bytes_),
             static_cast<unsigned long long>(range_waits_),
             static_cast<unsigned long long>(range_wait_ns_),
             static_cast<unsigned long long>(range_wait_free_),
             static_cast<unsigned long long>(enqueue_waits_),
             static_cast<unsigned long long>(enqueue_wait_ns_));
  }

private:
  static size_t resolveDepth() {
    const char *env = std::getenv("LIBOMPFILE_ASYNC_QUEUE_DEPTH");
    if (env && env[0]) {
      char *end = nullptr;
      long v = std::strtol(env, &end, 10);
      if (end != env && v > 0)
        return static_cast<size_t>(v);
    }
    return 2; // double-buffer: one draining, one queued ahead
  }

  static bool resolveEnabled() {
    if (envFlagEnabled("LIBOMPFILE_ASYNC_DISABLE"))
      return false;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
      return true; // no MPI in flight => local backend, a worker is always safe
    int level = MPI_THREAD_SINGLE;
    MPI_Query_thread(&level);
    if (level >= MPI_THREAD_SERIALIZED)
      return true;
    io_log("Async IO worker disabled: MPI thread level below SERIALIZED; "
           "async writes run synchronously.\n");
    return false;
  }

  // Erases every entry belonging to `handle` from a (handle, epoch) map.
  template <typename MapTy> static void eraseHandleRange(MapTy &m, int handle) {
    m.erase(m.lower_bound(EpochKey{handle, 0}),
            m.upper_bound(EpochKey{handle, UINT64_MAX}));
  }

  void startWorkerLocked() {
    if (worker_started_)
      return;
    worker_started_ = true;
    worker_ = std::thread([this] {
      ompfile::trace::nameThisThread("ompfile-async-worker");
      workerLoop();
    });
  }

  void workerLoop() {
    std::unique_lock<std::mutex> lock(mtx_);
    for (;;) {
      has_work_.wait(lock, [&] { return !queue_.empty() || stop_; });
      if (queue_.empty()) {
        if (stop_)
          return;
        continue;
      }
      Task task = std::move(queue_.front());
      queue_.pop_front();
      busy_ = true;
      // Keep the in-flight task's range visible to waitForRange while it is
      // no longer in the queue.
      current_ = {true, task.handle, task.has_offset, task.offset,
                  task.payloadSize()};
      not_full_.notify_one();
      lock.unlock();
      int rc = -1;
      {
        // The queued write actually running: its width against the issuing
        // thread's short omp_file_pwrite range is the async overlap.
        ompfile::trace::Scope Trace(ompfile::trace::Domain::LibOmpFile,
                                    "async-write",
                                    ompfile::trace::Color::Io,
                                    task.payloadSize());
        rc = executor_ ? executor_(task) : -1;
      }
      // The write has run, so the caller's buffer goes back now — before the
      // completion is published, so a flush that returns cannot race a
      // release still in flight.
      task.releaseOwned();
      lock.lock();
      busy_ = false;
      current_.active = false;
      auto it = pending_.find(task.handle);
      if (it != pending_.end() && --it->second == 0)
        pending_.erase(it);
      const bool has_epoch = taskHasEpoch(task);
      if (has_epoch) {
        auto ep = pending_epoch_.find(taskEpochKey(task));
        if (ep != pending_epoch_.end() && --ep->second == 0)
          pending_epoch_.erase(ep);
      }
      if (rc != 0) {
        if (sticky_error_ == 0)
          sticky_error_ = rc;
        int &handle_rc = handle_error_[task.handle];
        if (handle_rc == 0)
          handle_rc = rc;
        if (has_epoch) {
          int &epoch_rc = epoch_error_[taskEpochKey(task)];
          if (epoch_rc == 0)
            epoch_rc = rc;
        }
      }
      ++completed_;
      handle_done_.notify_all();
      idle_.notify_all();
    }
  }

  Executor executor_;
  std::mutex mtx_;
  std::condition_variable has_work_;
  std::condition_variable not_full_;
  std::condition_variable handle_done_;
  std::condition_variable idle_;
  std::deque<Task> queue_;
  struct InFlight {
    bool active = false;
    int handle = -1;
    bool has_offset = false;
    long offset = 0;
    size_t size = 0;
  };
  InFlight current_;
  std::unordered_map<int, uint64_t> pending_;
  // First failing rc per handle; entries are dropped on close (see
  // forgetHandle) so a reused file id starts clean.
  std::unordered_map<int, int> handle_error_;
  // The same two, keyed by completion unit. Ordered so forgetHandle can erase
  // a handle's whole epoch range in one pass; both stay small.
  std::map<EpochKey, uint64_t> pending_epoch_;
  std::map<EpochKey, int> epoch_error_;
  std::thread worker_;
  std::once_flag available_once_;
  size_t max_depth_ = 2;
  bool enabled_ = false;
  bool worker_started_ = false;
  bool busy_ = false;
  bool stop_ = false;
  bool stats_reported_ = false;
  int sticky_error_ = 0;
  uint64_t enqueued_ = 0;
  uint64_t completed_ = 0;
  uint64_t owned_enqueued_ = 0;
  uint64_t owned_bytes_ = 0;
  size_t max_queue_depth_ = 0;
  uint64_t drain_calls_ = 0;
  uint64_t drain_wait_ns_ = 0;
  uint64_t range_waits_ = 0;
  uint64_t range_wait_ns_ = 0;
  uint64_t range_wait_free_ = 0;
  uint64_t enqueue_waits_ = 0;
  uint64_t enqueue_wait_ns_ = 0;
};

class OmpFileClientContext {

private:
  static OmpFileClientContext *instance;
  std::unique_ptr<IOBackend> io_backend;
  std::unique_ptr<IOScheduler> io_scheduler;
  std::atomic<int> io_resource_token;
  std::atomic<uint64_t> api_call_id{1};
  AsyncWriteEngine async_engine;
  // Handles handed out by a successful open and not yet closed. The scheduler
  // owns handle lifetime, but flush has no scheduler round trip of its own, so
  // the client keeps this set to reject unknown/stale ids instead of reporting
  // "all writes completed" for a file that was never open.
  std::mutex open_handles_mutex;
  std::unordered_set<int> open_handles;
  // The path each open handle was opened with, so a range-scoped wait can
  // treat every handle on one file as that file (the OOC apps open a path
  // once per lane). Compared as the strings the app passed; two spellings of
  // one file are two files here, which errs toward waiting less than a
  // drain would but never toward a stale read of a range this process wrote
  // through the same string.
  std::unordered_map<int, std::string> handle_paths;

  // Every open handle on the same path as `file_handle`, itself included.
  std::vector<int> sameFileHandles(int file_handle) {
    std::lock_guard<std::mutex> lock(open_handles_mutex);
    std::vector<int> out;
    auto it = handle_paths.find(file_handle);
    if (it == handle_paths.end()) {
      out.push_back(file_handle);
      return out;
    }
    for (const auto &entry : handle_paths)
      if (entry.second == it->second)
        out.push_back(entry.first);
    return out;
  }

  // The wait a read or synchronous write of [offset, offset+size) on
  // `file_handle` owes the async queue: everything (the historical drain), or
  // only the queued writes that could cover that range of the same file.
  void waitForQueuedWrites(int file_handle, long offset, size_t size) {
    if (!async_engine.active())
      return;
    if (asyncReadWaitIsRange())
      async_engine.waitForRange(sameFileHandles(file_handle), offset, size);
    else
      async_engine.drain();
  }

  // RAII guard for IO resource token
  class IOResourceGuard {
    std::atomic<int> &token;
    std::chrono::milliseconds delay{1};
    static constexpr std::chrono::milliseconds max_delay{100};
  public:
    IOResourceGuard(std::atomic<int> &tok) : token(tok) {
      // Acquire a token with exponential backoff when none available
      while (true) {
        int current = token.load(std::memory_order_relaxed);
        if (current > 0) {
          if (token.compare_exchange_strong(current, current - 1,
                                            std::memory_order_acquire)) {
            break;
          }
        } else {
          std::this_thread::sleep_for(delay);
          // Exponential backoff capped at max_delay
          delay = std::min(delay * 2, max_delay);
        }
      }
    }

    ~IOResourceGuard() {
      token.fetch_add(1, std::memory_order_release);
    }

    // disable copying and moving
    IOResourceGuard(const IOResourceGuard &) = delete;
    IOResourceGuard &operator=(const IOResourceGuard &) = delete;
  };

public:
  OmpFileClientContext(IOBackendTy backend_type)
      : io_resource_token(resolveDefaultIOTokens()) {
    io_log("IO resource slots set to %d\n", io_resource_token.load());
    io_trace("OmpFileClientContext ctor enter this=%p backend=%s tokens=%d\n",
             static_cast<void *>(this),
             backendTypeToString(backend_type), io_resource_token.load());

    switch (backend_type) {
    case IOBackendTy::MPI:
      io_log("MPI backend selected\n");
      io_backend = std::make_unique<MPIIOBackend>();
      break;
    case IOBackendTy::POSIX:
      io_log("POSIX backend selected\n");
      io_backend = std::make_unique<POSIXIOBackend>();
      break;
    case IOBackendTy::IO_URING:
      io_log("IO_URING selected\n");
      io_backend = std::make_unique<IoUringIOBackend>();
      break;
    case IOBackendTy::HDF5:
      io_log("HDF5 backend not implemented yet\n");
      break;
    }

    IOSchedulerTy scheduler_type = getSchedulerType();
    if (scheduler_type == IOSchedulerTy::HEADNODE) {
      io_log("HEADNODE scheduler selected (client mode)\n");
      io_scheduler = std::make_unique<HeadnodeScheduler>(*io_backend);
    } else {
      io_log("LOCAL scheduler selected (client mode)\n");
      io_scheduler = std::make_unique<LocalScheduler>(*io_backend);
    }

    // Wire the async engine to drain queued writes through the same scheduler
    // path (and token semaphore) a synchronous write would take.
    async_engine.configure([this](const AsyncWriteEngine::Task &task) -> int {
      IOResourceGuard guard(io_resource_token);
      const void *bytes = task.payload();
      const size_t size = task.payloadSize();
      if (task.has_offset) {
        if (task.has_hint)
          return io_scheduler->writeAtHint(task.handle, task.offset, bytes,
                                           size, &task.hint);
        return io_scheduler->writeAt(task.handle, task.offset, bytes, size);
      }
      return io_scheduler->write(task.handle, bytes, size);
    });

    io_log("OmpFileClientContext constructor called\n");
    io_trace("OmpFileClientContext ctor done this=%p io_backend=%p "
             "io_scheduler=%p\n",
             static_cast<void *>(this), static_cast<void *>(io_backend.get()),
             static_cast<void *>(io_scheduler.get()));
  }

  ~OmpFileClientContext() {
    io_trace("OmpFileClientContext dtor this=%p io_backend=%p io_scheduler=%p\n",
             static_cast<void *>(this), static_cast<void *>(io_backend.get()),
             static_cast<void *>(io_scheduler.get()));
    // Queue must already be drained by the last close/read; join the idle
    // worker before tearing down the scheduler it calls into.
    async_engine.shutdown();
    io_log("Destroying OmpFileClientContext\n");
  }

  static OmpFileClientContext &getInstance() {
    io_trace("OmpFileClientContext::getInstance enter instance=%p\n",
             static_cast<void *>(instance));
    const std::lock_guard<std::mutex> lock(getClientContextInitMutex());
    if (instance == nullptr) {
      io_log("Creating new libompfile client instance\n");

      IOBackendTy backend = IOBackendTy::MPI; // Default
      const char *env = std::getenv("LIBOMPFILE_BACKEND");
      if (env) {
        std::string envStr(env);
        if (envStr == "MPI") {
          backend = IOBackendTy::MPI;
        } else if (envStr == "POSIX") {
          backend = IOBackendTy::POSIX;
        } else if (envStr == "IO_URING") {
          backend = IOBackendTy::IO_URING;
        } else if (envStr == "HDF5") {
          backend = IOBackendTy::HDF5;
        } else {
          io_log("Unknown LIBOMPFILE_BACKEND '%s', defaulting to MPI\n", env);
        }
      } else {
        io_log("LIBOMPFILE_BACKEND not set, defaulting to MPI\n");
      }

      instance = new OmpFileClientContext(backend);
      io_trace_symbol_owner(
          "OmpFileClientContext::getInstance",
          reinterpret_cast<const void *>(&OmpFileClientContext::getInstance));
      io_trace("OmpFileClientContext::getInstance created instance=%p\n",
               static_cast<void *>(instance));
      std::atexit(&OmpFileClientContext::finalize);
    } else {
      io_trace("OmpFileClientContext::getInstance reuse instance=%p\n",
               static_cast<void *>(instance));
    }

    return *instance;
  }

  int openFile(const char *filename) {
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu openFile enter filename=%s tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id),
             filename ? filename : "(null)", io_resource_token.load());
    if (async_engine.active())
      async_engine.drain();
    IOResourceGuard guard(io_resource_token);
    const int rc = io_scheduler->open(filename);
    if (rc >= 0) {
      {
        std::lock_guard<std::mutex> lock(open_handles_mutex);
        open_handles.insert(rc);
        handle_paths[rc] = filename ? filename : "";
      }
      // A recycled file id must not inherit the previous owner's async rc.
      async_engine.forgetHandle(rc);
    }
    io_trace("ctx=%p call=%llu openFile exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    return rc;
  }

  int writeFile(int file_handle, const void *data, size_t size) {
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->write(file_handle, data, size);
  }

  int readFile(int file_handle, void *data, size_t size) {
    if (async_engine.active())
      async_engine.drain();
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->read(file_handle, data, size);
  }

  int closeFile(int file_handle) {
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu closeFile enter file_handle=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), file_handle,
             io_resource_token.load());
    // Every queued async write must physically complete before the file is
    // closed; this is the pipeline's join point. Draining all (not just this
    // handle) also keeps scheduler/headnode state single-threaded. Always drain
    // (even if the queue already emptied) so a sticky async-write failure is
    // surfaced here rather than silently swallowed.
    const int async_rc = async_engine.drain();
    if (async_rc != 0)
      io_log("closeFile: a queued async write failed (rc=%d) before close of "
             "handle %d\n",
             async_rc, file_handle);
    IOResourceGuard guard(io_resource_token);
    const int rc = io_scheduler->close(file_handle);
    if (rc == 0) {
      {
        std::lock_guard<std::mutex> lock(open_handles_mutex);
        open_handles.erase(file_handle);
        handle_paths.erase(file_handle);
      }
      async_engine.forgetHandle(file_handle);
    }
    io_trace("ctx=%p call=%llu closeFile exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    if (rc == 0 && async_rc != 0)
      return -1; // surface the async write failure the close would otherwise hide
    return rc;
  }

  int seekFile(int file_handle, long offset) {
    if (async_engine.active())
      async_engine.drain();
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->seek(file_handle, offset);
  }

  bool isOpenHandle(int file_handle) {
    if (file_handle < 0)
      return false;
    std::lock_guard<std::mutex> lock(open_handles_mutex);
    return open_handles.count(file_handle) != 0;
  }

  // Completion boundary for async writes queued on one handle. This drains the
  // client-side queue only: it does not fsync, and under proxy write-back
  // staging the bytes may still sit in the proxy stage when it returns.
  int flushFile(int file_handle) {
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu flushFile enter file_handle=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), file_handle,
             io_resource_token.load());
    if (!isOpenHandle(file_handle)) {
      io_log("flushFile: invalid file handle %d\n", file_handle);
      io_trace("ctx=%p call=%llu flushFile exit rc=-1 reason=bad-handle\n",
               static_cast<void *>(this),
               static_cast<unsigned long long>(call_id));
      errno = EBADF;
      return -1;
    }
    const int rc = async_engine.flushHandle(file_handle);
    if (rc != 0)
      io_log("flushFile: a queued async write failed (rc=%d) for handle %d\n",
             rc, file_handle);
    io_trace("ctx=%p call=%llu flushFile exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    return rc;
  }

  // Same boundary as flushFile, narrowed to the writes tagged with `epoch`.
  // Same contract: client-side queue completion only, no fsync, no stage
  // flush. Writes issued without an epoch hint are not waited on here.
  int flushFileEpoch(int file_handle, uint64_t epoch) {
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu flushFileEpoch enter file_handle=%d epoch=%llu "
             "tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), file_handle,
             static_cast<unsigned long long>(epoch),
             io_resource_token.load());
    if (!isOpenHandle(file_handle)) {
      io_log("flushFileEpoch: invalid file handle %d\n", file_handle);
      io_trace("ctx=%p call=%llu flushFileEpoch exit rc=-1 reason=bad-handle\n",
               static_cast<void *>(this),
               static_cast<unsigned long long>(call_id));
      errno = EBADF;
      return -1;
    }
    const int rc = async_engine.flushEpoch(file_handle, epoch);
    if (rc != 0)
      io_log("flushFileEpoch: a queued async write failed (rc=%d) for handle "
             "%d epoch %llu\n",
             rc, file_handle, static_cast<unsigned long long>(epoch));
    io_trace("ctx=%p call=%llu flushFileEpoch exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    return rc;
  }

  // Visibility boundary, as opposed to flushFile's queue boundary: when this
  // returns 0, everything written on this handle has reached the source
  // filesystem and another reader can observe it. It is deliberately the
  // expensive one - it drains the handle's queue *and* pushes the proxy stage
  // to the source - so it belongs at a phase boundary, not inside a write wave.
  int commitFile(int file_handle) {
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu commitFile enter file_handle=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), file_handle,
             io_resource_token.load());
    if (!isOpenHandle(file_handle)) {
      io_log("commitFile: invalid file handle %d\n", file_handle);
      io_trace("ctx=%p call=%llu commitFile exit rc=-1 reason=bad-handle\n",
               static_cast<void *>(this),
               static_cast<unsigned long long>(call_id));
      errno = EBADF;
      return -1;
    }
    // Queued async writes are part of what the caller means by "what I wrote",
    // so they have to land before the stage is pushed to the source. A failure
    // here means the data is incomplete, and committing it would be a lie.
    const int flush_rc = async_engine.flushHandle(file_handle);
    if (flush_rc != 0) {
      io_log("commitFile: a queued async write failed (rc=%d) for handle %d\n",
             flush_rc, file_handle);
      io_trace("ctx=%p call=%llu commitFile exit rc=%d reason=queued-write\n",
               static_cast<void *>(this),
               static_cast<unsigned long long>(call_id), flush_rc);
      return flush_rc;
    }
    IOResourceGuard guard(io_resource_token);
    const int rc = io_scheduler->commit(file_handle);
    io_trace("ctx=%p call=%llu commitFile exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    return rc;
  }

  int writeFileAt(int file_handle, const void *data, size_t size, long offset) {
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->writeAt(file_handle, offset, data, size);
  }

  int writeFileAtHint(int file_handle, const void *data, size_t size,
                      long offset, const omp_file_io_hint_v1 *hint) {
    ompfile::OmpFileIOHint internal_hint{};
    applyIoHint(internal_hint, hint);
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->writeAtHint(file_handle, offset, data, size,
                                     hint ? &internal_hint : nullptr);
  }

  // Unified write entry for the C API. When async is requested and the worker
  // path is available, the payload is copied and queued so the caller can
  // pipeline the next issue; otherwise it runs synchronously after draining any
  // in-flight async writes (preserving program order and single-threaded
  // scheduler access). Returns 0/queued-accept or a negative error.
  int submitWrite(int file_handle, long offset, bool has_offset,
                  const void *data, size_t size,
                  const omp_file_io_hint_v1 *hint, bool async) {
    if (async && async_engine.available()) {
      AsyncWriteEngine::Task task;
      task.handle = file_handle;
      task.offset = offset;
      task.has_offset = has_offset;
      if (hint) {
        task.has_hint = true;
        applyIoHint(task.hint, hint);
      }
      const char *bytes = static_cast<const char *>(data);
      task.data.assign(bytes, bytes + size);
      return async_engine.enqueue(std::move(task));
    }
    if (has_offset) {
      // A synchronous pwrite behind queued async ones must land after any of
      // them that cover its range; the rest may keep draining.
      waitForQueuedWrites(file_handle, offset, size);
      if (hint)
        return writeFileAtHint(file_handle, data, size, offset, hint);
      return writeFileAt(file_handle, data, size, offset);
    }
    if (async_engine.active())
      async_engine.drain();
    return writeFile(file_handle, data, size);
  }

  // The owned form of submitWrite: no payload copy — the engine writes from
  // the caller's buffer and hands it to `release` once the write has run.
  // Ownership follows the return value and nothing else: zero means the
  // runtime has the buffer (released after the write, whatever its rc, which
  // then surfaces through flush/close like any queued write); non-zero means
  // nothing was queued, `release` was not called and the buffer is still the
  // caller's. Without an async worker the write runs inline under the same
  // rule: released on success, left with the caller on failure.
  int submitWriteOwned(int file_handle, long offset, void *data, size_t size,
                       void (*release)(void *),
                       const omp_file_io_hint_v1 *hint) {
    assert(release && "submitWriteOwned expects a resolved release callback");
    if (!isOpenHandle(file_handle)) {
      io_log("submitWriteOwned: invalid file handle %d\n", file_handle);
      errno = EBADF;
      return -1;
    }
    if (async_engine.available()) {
      AsyncWriteEngine::Task task;
      task.handle = file_handle;
      task.offset = offset;
      task.has_offset = true;
      if (hint) {
        task.has_hint = true;
        applyIoHint(task.hint, hint);
      }
      task.owned = data;
      task.owned_size = size;
      task.release = release;
      // A refused task is never moved into the queue, so `data` is untouched
      // and stays the caller's, as the contract above promises.
      return async_engine.enqueue(std::move(task));
    }
    if (async_engine.active())
      async_engine.drain();
    const int rc = hint ? writeFileAtHint(file_handle, data, size, offset, hint)
                        : writeFileAt(file_handle, data, size, offset);
    if (rc == 0)
      release(data);
    return rc;
  }

  int readFileAt(int file_handle, void *data, size_t size, long offset) {
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu readFileAt enter file_handle=%d offset=%ld "
             "size=%zu tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), file_handle, offset,
             size, io_resource_token.load());
    waitForQueuedWrites(file_handle, offset, size);
    IOResourceGuard guard(io_resource_token);
    const int rc = io_scheduler->readAt(file_handle, offset, data, size);
    io_trace("ctx=%p call=%llu readFileAt exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    return rc;
  }

  int readFileAtHint(int file_handle, void *data, size_t size, long offset,
                     const omp_file_io_hint_v1 *hint) {
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu readFileAtHint enter file_handle=%d offset=%ld "
             "size=%zu tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), file_handle, offset,
             size, io_resource_token.load());
    ompfile::OmpFileIOHint internal_hint{};
    applyIoHint(internal_hint, hint);
    waitForQueuedWrites(file_handle, offset, size);
    IOResourceGuard guard(io_resource_token);
    const int rc = io_scheduler->readAtHint(file_handle, offset, data, size,
                                            hint ? &internal_hint : nullptr);
    io_trace("ctx=%p call=%llu readFileAtHint exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    return rc;
  }

  int getFileHandle(int file_handle) { return file_handle; }

  static void finalize() {
    io_trace("OmpFileClientContext::finalize enter instance=%p\n",
             static_cast<void *>(instance));
    if (instance != nullptr) {
      io_log("Finalizing OmpFileClientContext\n");
      delete instance;
      instance = nullptr;
    }
    io_trace("OmpFileClientContext::finalize exit instance=%p\n",
             static_cast<void *>(instance));
  }
};

OmpFileClientContext *OmpFileClientContext::instance = nullptr;

// One range per public call, on the calling thread (thread-local, so
// push/pop). Payload = the request size where there is one.
#define OMPFILE_API_TRACE(Name, Size)                                          \
  ompfile::trace::Scope OmpFileApiTrace(ompfile::trace::Domain::LibOmpFile,   \
                                        Name, ompfile::trace::Color::Io,       \
                                        static_cast<uint64_t>(Size))

extern "C" {

// Reads cannot be fire-and-forget through the fixed C ABI (there is no
// completion handle to return the buffer through later), so an async-requested
// read is served synchronously. The read paths already drain any in-flight
// async writes first, so the data is coherent. Log the downgrade once.
static void note_async_read_synchronous(int async) {
  if (!async)
    return;
  static std::once_flag once;
  std::call_once(once, [] {
    io_log("Async read requested; served synchronously (no async-read "
           "completion handle in the ABI).\n");
  });
}

int omp_file_open(const char *filename) {
  OMPFILE_API_TRACE("omp_file_open", 0);
  io_trace("omp_file_open api enter filename=%s\n",
           filename ? filename : "(null)");
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.openFile(filename);
  io_trace("omp_file_open api exit rc=%d\n", rc);
  return rc;
}

int omp_file_write(int file_handle, const void *data, size_t size, int async) {
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.submitWrite(file_handle, /*offset=*/0, /*has_offset=*/false, data,
                         size, /*hint=*/nullptr, async != 0);
}

int omp_file_pwrite(int file_handle, long offset, const void *data, size_t size,
                    int async) {
  OMPFILE_API_TRACE("omp_file_pwrite", size);
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.submitWrite(file_handle, offset, /*has_offset=*/true, data, size,
                         /*hint=*/nullptr, async != 0);
}

int omp_file_pwrite_hint(int file_handle, long offset, const void *data,
                         size_t size, int async,
                         const omp_file_io_hint_v1 *hint) {
  OMPFILE_API_TRACE("omp_file_pwrite_hint", size);
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.submitWrite(file_handle, offset, /*has_offset=*/true, data, size,
                         hint, async != 0);
}

int omp_file_pwrite_owned(int file_handle, long offset, void *data,
                          size_t size, void (*release)(void *)) {
  OMPFILE_API_TRACE("omp_file_pwrite_owned", size);
  io_trace("omp_file_pwrite_owned api enter file_handle=%d offset=%ld "
           "size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.submitWriteOwned(file_handle, offset, data, size,
                                      release ? release : &::free,
                                      /*hint=*/nullptr);
  io_trace("omp_file_pwrite_owned api exit rc=%d\n", rc);
  return rc;
}

int omp_file_pwrite_owned_hint(int file_handle, long offset, void *data,
                               size_t size, void (*release)(void *),
                               const omp_file_io_hint_v1 *hint) {
  OMPFILE_API_TRACE("omp_file_pwrite_owned_hint", size);
  io_trace("omp_file_pwrite_owned_hint api enter file_handle=%d offset=%ld "
           "size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.submitWriteOwned(file_handle, offset, data, size,
                                      release ? release : &::free, hint);
  io_trace("omp_file_pwrite_owned_hint api exit rc=%d\n", rc);
  return rc;
}

int omp_file_flush(int file_handle) {
  OMPFILE_API_TRACE("omp_file_flush", 0);
  io_trace("omp_file_flush api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.flushFile(file_handle);
  io_trace("omp_file_flush api exit rc=%d\n", rc);
  return rc;
}

int omp_file_flush_epoch(int file_handle, uint64_t epoch) {
  OMPFILE_API_TRACE("omp_file_flush_epoch", epoch);
  io_trace("omp_file_flush_epoch api enter file_handle=%d epoch=%llu\n",
           file_handle, static_cast<unsigned long long>(epoch));
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.flushFileEpoch(file_handle, epoch);
  io_trace("omp_file_flush_epoch api exit rc=%d\n", rc);
  return rc;
}

int omp_file_commit(int file_handle) {
  OMPFILE_API_TRACE("omp_file_commit", 0);
  io_trace("omp_file_commit api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.commitFile(file_handle);
  io_trace("omp_file_commit api exit rc=%d\n", rc);
  return rc;
}

int omp_file_pread(int file_handle, long offset, void *data, size_t size,
                   int async) {
  OMPFILE_API_TRACE("omp_file_pread", size);
  note_async_read_synchronous(async);
  io_trace("omp_file_pread api enter file_handle=%d offset=%ld size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.readFileAt(file_handle, data, size, offset);
  io_trace("omp_file_pread api exit rc=%d\n", rc);
  return rc;
}

int omp_file_pread_hint(int file_handle, long offset, void *data, size_t size,
                        int async, const omp_file_io_hint_v1 *hint) {
  OMPFILE_API_TRACE("omp_file_pread_hint", size);
  note_async_read_synchronous(async);
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.readFileAtHint(file_handle, data, size, offset, hint);
}

int omp_file_read(int file_handle, void *data, size_t size, int async) {
  note_async_read_synchronous(async);
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.readFile(file_handle, data, size);
}

int omp_file_close(int file_handle) {
  OMPFILE_API_TRACE("omp_file_close", 0);
  io_trace("omp_file_close api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.closeFile(file_handle);
  io_trace("omp_file_close api exit rc=%d\n", rc);
  return rc;
}

int omp_file_seek(int file_handle, long offset) {
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.seekFile(file_handle, offset);
}

} // extern "C"
