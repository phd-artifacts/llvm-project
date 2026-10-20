#include "io_scheduler.h"
#include "debug_log.h"
#include "mpi.h"
#include "mpp_shim.h"
#include "ompfile_config.h"
#include "ompfile_keys.h"
#include "ompfile_sched.h"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

class HeadnodeScheduler final : public IOScheduler {
public:
  explicit HeadnodeScheduler(IOBackend &backend,
                             const ompfile::config::ClientConfig &config)
      : backend(backend), client_rank(resolveClientRank()),
        mpp_remote_only_enabled(config.RejectedModeKnobs.empty()),
        mpp_sched_active(mpp_remote_only_enabled && ompfile::mpp::init()),
        two_phase_batch_preferred(mpp_sched_active &&
                                  shouldPreferBatchReadPlanner(config)),
        per_op_scheduling_skipped(config.LocalDisjointExpress ||
                                  (config.SchedOncePerHandle.Value &&
                                   config.Shared.stageOff())) {
    if (!mpp_remote_only_enabled) {
      io_log("HEADNODE scheduler requested but a removed libompfile mode was "
             "requested (see the 'libompfile init failed' line); all "
             "scheduler operations will fail.\n");
    } else if (!mpp_sched_active) {
      io_log("HEADNODE scheduler requested but MPP init failed; all scheduler "
             "operations will fail.\n");
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
    if (file_handle >= 0 && filename) {
      rememberOpenPath(file_handle, filename,
                       open_plan_ptr ? open_plan_ptr->AggregatorRank : -1);
    } else if (file_handle < 0 && open_plan_ptr) {
      // The headnode counted this open and nothing will close it.
      const int saved_errno = errno;
      (void)scheduleClose(/*file_handle=*/-1, open_plan_ptr->AggregatorRank);
      errno = saved_errno;
    }
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

  // Commit needs no scheduler round trip: it changes no open/close state.
  // The backend broadcasts a version-free stage flush for the file
  // (commitStagePathKey), not the headnode dirty-tile flush protocol.
  int commit(int file_handle) override { return backend.commit(file_handle); }

  int close(int file_handle) override {
    bool schedule_ok = true;
    if (!ensureSchedulerReady("close", /*log_once=*/true))
      schedule_ok = false;
    if (mpp_sched_active)
      schedule_ok = scheduleClose(file_handle, takePlannedRank(file_handle));
    const int rc = backend.close(file_handle);
    if (rc == 0)
      forgetOpenPath(file_handle);
    if (!schedule_ok)
      return failStrict("close");
    return rc;
  }

  int seek(int file_handle, long offset) override {
    if (!ensureSchedulerReady("seek", /*log_once=*/true))
      return -1;
    return backend.seek(file_handle, offset);
  }

  int readAt(int file_handle, long offset, void *data, size_t size) override {
    if (!ensureSchedulerReady("readAt", /*log_once=*/true))
      return -1;
    if (per_op_scheduling_skipped)
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
    if (!ensureSchedulerReady("writeAt", /*log_once=*/true))
      return -1;
    if (!per_op_scheduling_skipped &&
        !scheduleWrite(file_handle, offset, size, nullptr))
      return failStrict("writeAt");
    return backend.writeAt(file_handle, offset, data, size);
  }

  int writeAtHint(int file_handle, long offset, const void *data, size_t size,
                  const ompfile::OmpFileIOHint *hint) override {
    ompfile::OmpFileWriteRequestContext context{};
    if (!buildWriteContext(file_handle, offset, size, hint, context))
      return -1;
    if (!ensureSchedulerReady("writeAtHint", /*log_once=*/true))
      return -1;
    if (!per_op_scheduling_skipped &&
        !scheduleWrite(file_handle, offset, size, hint))
      return failStrict("writeAtHint");
    return backend.writeAtWithContext(context, data, size);
  }

private:
  struct TrackedFileMetadata {
    std::string Path;
    uint64_t PathKey = 0;
    // AggregatorRank of the OPEN plan that counted this handle on the
    // headnode, sent back with its CLOSE; -1 once released or if unplanned.
    int PlannedRank = -1;
  };

  IOBackend &backend;
  int client_rank = -1;
  const bool mpp_remote_only_enabled = false;
  const bool mpp_sched_active = false;
  // per_op_scheduling_skipped (the member after two_phase_batch_preferred) is
  // set by either of two knobs:
  //
  // Express lane for the owner-bypass contract: when the app asserts disjoint
  // accesses (LIBOMPFILE_OPT_LOCAL_DISJOINT_*), the headnode's per-op
  // scheduling round trip (scheduleWrite / read planning, one MPI event each,
  // serialized on sched_request_mutex) only computes owner/aggregator routing
  // that the bypass then ignores — a pure ~2 ms fixed tax per op (AMD job
  // 359359). With express on, per-op writeAt/readAt skip scheduling and go
  // straight to the backend; open/close scheduling stays (handles and owner
  // state still need it), and the proxy-side bypass keeps its fail-closed
  // owner-forward fallback.
  //
  // LIBOMPFILE_OPT_SCHED_ONCE_PER_HANDLE=1: the same skip of the per-op
  // PWRITE / scalar-PREAD round trips, without asserting the disjoint-access
  // contract. The owner is fixed at open and the client routes by the opened
  // handle; what the per-write trip adds on the headnode is freshness-version
  // bookkeeping for staged coherence, so the knob is honoured only with the
  // stage off.
  const bool two_phase_batch_preferred = false;
  const bool per_op_scheduling_skipped = false;
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

  static bool
  shouldPreferBatchReadPlanner(const ompfile::config::ClientConfig &config) {
    const ompfile::config::PolicyKnob &knob = config.TwoPhase;
    if (knob.Malformed) {
      io_log("Invalid LIBOMPFILE_OPT_TWO_PHASE='%s' "
             "for HEADNODE scheduler; keeping scalar scheduleRead path.\n",
             knob.Raw.c_str());
      return false;
    }
    return knob.Policy != ompfile::config::BatchPolicy::Disabled;
  }

  int failStrict(const char *op_name) {
    errno = EIO;
    if (!strict_failure_logged.exchange(true, std::memory_order_relaxed)) {
      io_log("HEADNODE scheduler operation '%s' failed: MPP is not available "
             "(remote-only MPP is libompfile's only mode).\n",
             op_name ? op_name : "(unknown)");
    }
    io_trace("HeadnodeScheduler::failStrict scheduler=%p op=%s\n",
             static_cast<void *>(this), op_name ? op_name : "(unknown)");
    return -1;
  }

  bool ensureSchedulerReady(const char *op_name, bool log_once) {
    if (mpp_sched_active)
      return true;
    if (!log_once || !strict_failure_logged.load(std::memory_order_relaxed)) {
      io_trace("HeadnodeScheduler::ensureSchedulerReady fail scheduler=%p op=%s "
               "mpp_remote_only_enabled=%d mpp_sched_active=%d\n",
               static_cast<void *>(this), op_name ? op_name : "(unknown)",
               static_cast<int>(mpp_remote_only_enabled),
               static_cast<int>(mpp_sched_active));
    }
    failStrict(op_name);
    return false;
  }

  void rememberOpenPath(int file_handle, const char *path, int planned_rank) {
    if (!path)
      return;
    TrackedFileMetadata metadata{};
    metadata.Path = path;
    metadata.PathKey = ompfile::keys::pathKey(path);
    metadata.PlannedRank = planned_rank;
    const std::lock_guard<std::mutex> lock(tracked_file_mutex);
    tracked_file_map[file_handle] = std::move(metadata);
  }

  void forgetOpenPath(int file_handle) {
    const std::lock_guard<std::mutex> lock(tracked_file_mutex);
    tracked_file_map.erase(file_handle);
  }

  // The planned rank recorded at open, cleared so a retried close of the
  // same handle cannot release the headnode count twice.
  int takePlannedRank(int file_handle) {
    const std::lock_guard<std::mutex> lock(tracked_file_mutex);
    auto it = tracked_file_map.find(file_handle);
    if (it == tracked_file_map.end())
      return -1;
    const int rank = it->second.PlannedRank;
    it->second.PlannedRank = -1;
    return rank;
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
        errno = EIO;
        io_log("HEADNODE scheduler request failed (global manager "
               "unavailable).\n");
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
    return schedRequest(req, path, plan_out);
  }

  bool scheduleClose(int file_handle, int planned_rank) {
    ompfile::OmpFileIORequest req{};
    req.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    req.Op = ompfile::OmpFileIOOp::CLOSE;
    req.ClientRank = client_rank;
    req.FileHandle = file_handle;
    req.AggregatorRank = planned_rank;
    io_trace("HeadnodeScheduler::scheduleClose scheduler=%p req_id=%llu "
             "file=%d planned_rank=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(req.RequestId), file_handle,
             planned_rank);
    return schedRequest(req, nullptr, nullptr);
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
    context.PathKey = ompfile::keys::handleKey(file_handle);
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

    if (two_phase_batch_preferred || per_op_scheduling_skipped) {
      io_trace("HeadnodeScheduler::buildReadContext skip-scalar req_id=%llu "
               "file=%d offset=%lld size=%llu\n",
               static_cast<unsigned long long>(context.RequestId), file_handle,
               static_cast<long long>(offset),
               static_cast<unsigned long long>(size));
      return true;
    }

    ompfile::OmpFileIOPlan plan{};
    if (!schedRequest(req, nullptr, &plan))
      return false;

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
                        nullptr);
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
    context.PathKey = ompfile::keys::handleKey(file_handle);
    if (hint)
      context.Hint = *hint;

    std::string tracked_path;
    if (getTrackedPath(file_handle, tracked_path, context.PathKey))
      context.ContextFlags |= ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY;
    return true;
  }
};

std::unique_ptr<IOScheduler> createLocalScheduler(IOBackend &backend) {
  return std::make_unique<LocalScheduler>(backend);
}

std::unique_ptr<IOScheduler>
createHeadnodeScheduler(IOBackend &backend,
                        const ompfile::config::ClientConfig &config) {
  return std::make_unique<HeadnodeScheduler>(backend, config);
}
