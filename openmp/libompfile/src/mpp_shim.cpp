#include "mpp_shim.h"

#include "debug_log.h"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <thread>

namespace {

using MppInitFn = int (*)();
using MppSubmitFn = int (*)(uint64_t);
using MppOpenFn = int (*)(const char *, int, int, int *);
using MppOpenOnRankFn = int (*)(const char *, int, int, int, int *);
using MppCloseFn = int (*)(int);
using MppPreadFn = int (*)(int, int64_t, void *, uint64_t);
using MppPreadExFn = int (*)(int, int64_t, void *, uint64_t, uint64_t *);
using MppPwriteFn = int (*)(int, int64_t, const void *, uint64_t);
using MppPwriteExFn = int (*)(int, int64_t, const void *, uint64_t, uint64_t *);
using MppStageInvalidatePathKeyFn = int (*)(uint64_t, uint64_t, const char *);
using MppSchedRequestFn = int (*)(const ompfile::OmpFileIORequest *,
                                  const char *, ompfile::OmpFileIOPlan *);
using MppSchedBatchRequestFn = int (*)(const ompfile::OmpFileIOBatchRequest *,
                                       const void *, uint64_t,
                                       ompfile::OmpFileIOBatchPlan *, void *,
                                       uint64_t, uint64_t *);
using MppPollFn = int (*)(uint64_t, int *);
using MppFinalizeFn = int (*)();

struct MppApi {
  MppInitFn init = nullptr;
  MppSubmitFn submit = nullptr;
  MppOpenFn open = nullptr;
  MppOpenOnRankFn open_on_rank = nullptr;
  MppCloseFn close = nullptr;
  MppPreadFn pread = nullptr;
  MppPreadExFn pread_ex = nullptr;
  MppPwriteFn pwrite = nullptr;
  MppPwriteExFn pwrite_ex = nullptr;
  MppStageInvalidatePathKeyFn stage_invalidate_path_key = nullptr;
  MppSchedRequestFn sched_request = nullptr;
  MppSchedBatchRequestFn sched_batch_request = nullptr;
  MppPollFn poll = nullptr;
  MppFinalizeFn finalize = nullptr;
};

MppApi loadMppApi() {
  MppApi loaded;
  loaded.init = reinterpret_cast<MppInitFn>(dlsym(RTLD_DEFAULT,
                                                  "ompfile_mpp_init"));
  loaded.submit = reinterpret_cast<MppSubmitFn>(dlsym(RTLD_DEFAULT,
                                                      "ompfile_mpp_submit"));
  loaded.open = reinterpret_cast<MppOpenFn>(dlsym(RTLD_DEFAULT,
                                                       "ompfile_mpp_open"));
  loaded.open_on_rank = reinterpret_cast<MppOpenOnRankFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_open_on_rank"));
  loaded.close = reinterpret_cast<MppCloseFn>(dlsym(RTLD_DEFAULT,
                                                        "ompfile_mpp_close"));
  loaded.pread = reinterpret_cast<MppPreadFn>(dlsym(RTLD_DEFAULT,
                                                       "ompfile_mpp_pread"));
  loaded.pread_ex = reinterpret_cast<MppPreadExFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_pread_ex"));
  loaded.pwrite = reinterpret_cast<MppPwriteFn>(dlsym(RTLD_DEFAULT,
                                                        "ompfile_mpp_pwrite"));
  loaded.pwrite_ex = reinterpret_cast<MppPwriteExFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_pwrite_ex"));
  loaded.stage_invalidate_path_key =
      reinterpret_cast<MppStageInvalidatePathKeyFn>(
          dlsym(RTLD_DEFAULT, "ompfile_mpp_stage_invalidate_path_key"));
  loaded.sched_request = reinterpret_cast<MppSchedRequestFn>(dlsym(
      RTLD_DEFAULT, "ompfile_mpp_sched_request"));
  loaded.sched_batch_request =
      reinterpret_cast<MppSchedBatchRequestFn>(dlsym(
          RTLD_DEFAULT, "ompfile_mpp_sched_request_batch"));
  loaded.poll = reinterpret_cast<MppPollFn>(dlsym(RTLD_DEFAULT,
                                                  "ompfile_mpp_poll"));
  loaded.finalize = reinterpret_cast<MppFinalizeFn>(dlsym(RTLD_DEFAULT,
                                                          "ompfile_mpp_finalize"));
  return loaded;
}

MppApi &getMppApi() {
  static MppApi api = loadMppApi();
  return api;
}

uint64_t nextShimCallId() {
  static std::atomic<uint64_t> call_id{1};
  return call_id.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

namespace ompfile {
namespace mpp {

bool init() {
  static std::atomic<bool> ready{false};
  if (ready.load(std::memory_order_acquire))
    return true;

  io_trace("mpp_shim::init enter\n");

  auto &api = getMppApi();
  if (!api.init)
    api = loadMppApi();

  if (!api.init) {
    io_log("MPP shim not available (ompfile_mpp_init missing).\n");
    io_trace("mpp_shim::init missing init symbol\n");
    return false;
  }

  io_trace_symbol_owner("ompfile_mpp_init", reinterpret_cast<void *>(api.init));
  io_trace_symbol_owner("ompfile_mpp_open", reinterpret_cast<void *>(api.open));
  io_trace_symbol_owner("ompfile_mpp_open_on_rank",
                        reinterpret_cast<void *>(api.open_on_rank));
  io_trace_symbol_owner("ompfile_mpp_close",
                        reinterpret_cast<void *>(api.close));
  io_trace_symbol_owner("ompfile_mpp_pread",
                        reinterpret_cast<void *>(api.pread));
  io_trace_symbol_owner("ompfile_mpp_pread_ex",
                        reinterpret_cast<void *>(api.pread_ex));
  io_trace_symbol_owner("ompfile_mpp_pwrite",
                        reinterpret_cast<void *>(api.pwrite));
  io_trace_symbol_owner("ompfile_mpp_pwrite_ex",
                        reinterpret_cast<void *>(api.pwrite_ex));
  io_trace_symbol_owner(
      "ompfile_mpp_stage_invalidate_path_key",
      reinterpret_cast<void *>(api.stage_invalidate_path_key));
  io_trace_symbol_owner("ompfile_mpp_sched_request",
                        reinterpret_cast<void *>(api.sched_request));
  io_trace_symbol_owner("ompfile_mpp_sched_request_batch",
                        reinterpret_cast<void *>(api.sched_batch_request));

  int init_rc = api.init();
  if (init_rc != 0) {
    io_log("MPP shim init failed (rc=%d).\n", init_rc);
    io_trace("mpp_shim::init failed rc=%d\n", init_rc);
    return false;
  }

  ready.store(true, std::memory_order_release);
  io_log("MPP shim initialized.\n");
  io_trace("mpp_shim::init success\n");
  return true;
}

bool submit(uint64_t token) {
  if (!init())
    return false;

  const auto &api = getMppApi();
  if (!api.submit)
    return false;
  return api.submit(token) == 0;
}

bool poll(uint64_t token, bool &done) {
  done = false;
  if (!init())
    return false;

  const auto &api = getMppApi();
  if (!api.poll)
    return false;

  int raw_done = 0;
  if (api.poll(token, &raw_done) != 0)
    return false;
  done = raw_done != 0;
  return true;
}

void finalize() {
  const auto &api = getMppApi();
  if (api.finalize)
    api.finalize();
}

bool open(const char *path, int flags, int mode, int &handle) {
  const uint64_t call_id = nextShimCallId();
  handle = -1;
  io_trace("mpp_shim::open enter call=%llu path=%s flags=0x%x mode=%o\n",
           static_cast<unsigned long long>(call_id), path ? path : "(null)",
           flags, mode);
  if (!init())
    return false;

  auto &api = getMppApi();
  if (!api.open)
    api = loadMppApi();

  if (!api.open) {
    io_log("MPP shim not available (ompfile_mpp_open missing).\n");
    io_trace("mpp_shim::open missing symbol call=%llu\n",
             static_cast<unsigned long long>(call_id));
    return false;
  }

  const int rc = api.open(path, flags, mode, &handle);
  if (rc != 0) {
    io_log("MPP shim open failed.\n");
    io_trace("mpp_shim::open failed call=%llu rc=%d handle=%d\n",
             static_cast<unsigned long long>(call_id), rc, handle);
    return false;
  }

  io_trace("mpp_shim::open success call=%llu handle=%d\n",
           static_cast<unsigned long long>(call_id), handle);
  return true;
}

bool openOnRank(const char *path, int flags, int mode, int rank, int &handle) {
  const uint64_t call_id = nextShimCallId();
  handle = -1;
  io_trace("mpp_shim::openOnRank enter call=%llu path=%s flags=0x%x mode=%o "
           "rank=%d\n",
           static_cast<unsigned long long>(call_id), path ? path : "(null)",
           flags, mode, rank);
  if (!init())
    return false;

  auto &api = getMppApi();
  if (!api.open_on_rank)
    api = loadMppApi();

  if (!api.open_on_rank) {
    io_log("MPP shim not available (ompfile_mpp_open_on_rank missing).\n");
    io_trace("mpp_shim::openOnRank missing symbol call=%llu\n",
             static_cast<unsigned long long>(call_id));
    return false;
  }

  const int rc = api.open_on_rank(path, flags, mode, rank, &handle);
  if (rc != 0) {
    io_log("MPP shim openOnRank failed.\n");
    io_trace("mpp_shim::openOnRank failed call=%llu rc=%d handle=%d rank=%d\n",
             static_cast<unsigned long long>(call_id), rc, handle, rank);
    return false;
  }

  io_trace("mpp_shim::openOnRank success call=%llu handle=%d rank=%d\n",
           static_cast<unsigned long long>(call_id), handle, rank);
  return true;
}

bool close(int handle) {
  const uint64_t call_id = nextShimCallId();
  io_trace("mpp_shim::close enter call=%llu handle=%d\n",
           static_cast<unsigned long long>(call_id), handle);
  if (!init())
    return false;

  auto &api = getMppApi();
  if (!api.close)
    api = loadMppApi();

  if (!api.close) {
    io_log("MPP shim not available (ompfile_mpp_close missing).\n");
    io_trace("mpp_shim::close missing symbol call=%llu\n",
             static_cast<unsigned long long>(call_id));
    return false;
  }

  const int rc = api.close(handle);
  io_trace("mpp_shim::close exit call=%llu rc=%d\n",
           static_cast<unsigned long long>(call_id), rc);
  return rc == 0;
}

bool pread(int handle, int64_t offset, void *buffer, size_t size) {
  size_t bytes_read = 0;
  return preadEx(handle, offset, buffer, size, bytes_read);
}

bool preadEx(int handle, int64_t offset, void *buffer, size_t size,
             size_t &bytes_read) {
  const uint64_t call_id = nextShimCallId();
  bytes_read = 0;
  io_trace("mpp_shim::pread enter call=%llu handle=%d offset=%lld size=%zu\n",
           static_cast<unsigned long long>(call_id), handle,
           static_cast<long long>(offset), size);
  if (!buffer && size > 0)
    return false;

  if (!init())
    return false;

  auto &api = getMppApi();
  if (!api.pread)
    api = loadMppApi();

  if (!api.pread) {
    io_log("MPP shim not available (ompfile_mpp_pread missing).\n");
    io_trace("mpp_shim::pread missing symbol call=%llu\n",
             static_cast<unsigned long long>(call_id));
    return false;
  }

  if (api.pread_ex) {
    uint64_t remote_bytes_read = 0;
    const int rc = api.pread_ex(handle, offset, buffer, size, &remote_bytes_read);
    io_trace("mpp_shim::pread_ex exit call=%llu rc=%d bytes=%llu\n",
             static_cast<unsigned long long>(call_id), rc,
             static_cast<unsigned long long>(remote_bytes_read));
    if (rc != 0)
      return false;
    if (remote_bytes_read >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
      return false;
    bytes_read = static_cast<size_t>(remote_bytes_read);
    return true;
  }

  const int rc = api.pread(handle, offset, buffer, size);
  io_trace("mpp_shim::pread legacy exit call=%llu rc=%d\n",
           static_cast<unsigned long long>(call_id), rc);
  if (rc != 0)
    return false;
  bytes_read = size;
  return true;
}

bool pwriteEx(int handle, int64_t offset, const void *buffer, size_t size,
              size_t &bytes_written) {
  const uint64_t call_id = nextShimCallId();
  bytes_written = 0;
  io_trace("mpp_shim::pwrite enter call=%llu handle=%d offset=%lld size=%zu "
           "buffer=%p\n",
           static_cast<unsigned long long>(call_id), handle,
           static_cast<long long>(offset), size, buffer);
  if (!buffer && size > 0)
    return false;

  if (!init())
    return false;

  auto &api = getMppApi();
  if (!api.pwrite && !api.pwrite_ex)
    api = loadMppApi();

  if (!api.pwrite && !api.pwrite_ex) {
    io_log("MPP shim not available (ompfile_mpp_pwrite{,_ex} missing).\n");
    io_trace("mpp_shim::pwrite missing symbol call=%llu\n",
             static_cast<unsigned long long>(call_id));
    return false;
  }

  if (api.pwrite_ex) {
    uint64_t remote_bytes_written = 0;
    const int rc =
        api.pwrite_ex(handle, offset, buffer, size, &remote_bytes_written);
    io_trace("mpp_shim::pwrite_ex exit call=%llu rc=%d bytes=%llu\n",
             static_cast<unsigned long long>(call_id), rc,
             static_cast<unsigned long long>(remote_bytes_written));
    if (rc != 0)
      return false;
    if (remote_bytes_written >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
      return false;
    bytes_written = static_cast<size_t>(remote_bytes_written);
    return true;
  }

  const int rc = api.pwrite(handle, offset, buffer, size);
  io_trace("mpp_shim::pwrite legacy exit call=%llu rc=%d\n",
           static_cast<unsigned long long>(call_id), rc);
  if (rc != 0)
    return false;
  bytes_written = size;
  return true;
}

bool pwrite(int handle, int64_t offset, const void *buffer, size_t size) {
  size_t bytes_written = 0;
  if (!pwriteEx(handle, offset, buffer, size, bytes_written))
    return false;
  if (bytes_written != size) {
    errno = EIO;
    return false;
  }
  return true;
}

bool stageInvalidatePathKey(uint64_t path_key, uint64_t generation,
                            const char *path) {
  const uint64_t call_id = nextShimCallId();
  auto &api = getMppApi();
  if (!api.stage_invalidate_path_key)
    api = loadMppApi();

  if (!api.stage_invalidate_path_key) {
    io_log("MPP shim not available "
           "(ompfile_mpp_stage_invalidate_path_key missing).\n");
    errno = ENOSYS;
    return false;
  }

  if (!init())
    return false;

  const int rc = api.stage_invalidate_path_key(path_key, generation, path);
  io_trace("mpp_shim::stageInvalidatePathKey call=%llu path_key=%llu "
           "generation=%llu rc=%d\n",
           static_cast<unsigned long long>(call_id),
           static_cast<unsigned long long>(path_key),
           static_cast<unsigned long long>(generation), rc);
  if (rc != 0)
    errno = rc < 0 ? EIO : rc;
  return rc == 0;
}

bool schedRequest(const ompfile::OmpFileIORequest &request, const char *path,
                  ompfile::OmpFileIOPlan &plan) {
  const uint64_t call_id = nextShimCallId();
  io_trace("mpp_shim::schedRequest enter call=%llu req_id=%llu op=%u "
           "file=%d client_rank=%d offset=%lld size=%llu path_size=%u\n",
           static_cast<unsigned long long>(call_id),
           static_cast<unsigned long long>(request.RequestId),
           static_cast<unsigned>(request.Op), request.FileHandle,
           request.ClientRank, static_cast<long long>(request.Offset),
           static_cast<unsigned long long>(request.Size), request.PathSize);
  if (request.PathSize > 0 && !path)
    return false;

  if (!init()) {
    io_log("MPP scheduler request aborted because MPP init failed.\n");
    io_trace("mpp_shim::schedRequest init failed call=%llu\n",
             static_cast<unsigned long long>(call_id));
    return false;
  }

  auto &api = getMppApi();
  if (!api.sched_request)
    api = loadMppApi();

  if (!api.sched_request) {
    io_log("MPP shim not available (ompfile_mpp_sched_request missing).\n");
    io_trace("mpp_shim::schedRequest missing symbol call=%llu\n",
             static_cast<unsigned long long>(call_id));
    return false;
  }

  const int rc = api.sched_request(&request, path, &plan);
  io_trace("mpp_shim::schedRequest exit call=%llu rc=%d plan_status=%d "
           "plan_errno=%d aggregator=%d remote_handle=%d flags=0x%x\n",
           static_cast<unsigned long long>(call_id), rc, plan.Status, plan.Errno,
           plan.AggregatorRank, plan.RemoteHandle, plan.PlanFlags);
  return rc == 0;
}

bool schedBatchRequest(
    const ompfile::OmpFileIOBatchRequest &request,
    const std::vector<ompfile::OmpFileIOBatchSegment> &segments,
    ompfile::OmpFileIOBatchPlan &plan,
    std::vector<ompfile::OmpFileIOBatchPlanEntry> &entries) {
  const uint64_t call_id = nextShimCallId();
  io_trace("mpp_shim::schedBatchRequest enter call=%llu batch_id=%llu "
           "segments=%zu flags=0x%x\n",
           static_cast<unsigned long long>(call_id),
           static_cast<unsigned long long>(request.BatchId), segments.size(),
           request.RequestFlags);
  plan = {};
  entries.clear();

  if (request.AbiVersion != ompfile::OMPFILE_SCHED_BATCH_ABI_VERSION) {
    io_log("MPP batch scheduler request has unsupported ABI version=%u\n",
           request.AbiVersion);
    io_trace("mpp_shim::schedBatchRequest abi mismatch call=%llu "
             "abi=%u expected=%u\n",
             static_cast<unsigned long long>(call_id), request.AbiVersion,
             ompfile::OMPFILE_SCHED_BATCH_ABI_VERSION);
    return false;
  }

  if (request.SegmentCount != segments.size()) {
    io_log("MPP batch scheduler segment count mismatch: header=%u vec=%zu\n",
           request.SegmentCount, segments.size());
    io_trace("mpp_shim::schedBatchRequest segment mismatch call=%llu "
             "header=%u vec=%zu\n",
             static_cast<unsigned long long>(call_id), request.SegmentCount,
             segments.size());
    return false;
  }

  std::vector<uint8_t> request_payload;
  if (!ompfile::encodeBatchSegments(segments, request_payload))
    return false;

  ompfile::OmpFileIOBatchRequest normalized_request = request;
  normalized_request.PayloadBytes =
      static_cast<uint32_t>(request_payload.size());
  normalized_request.AbiVersion = ompfile::OMPFILE_SCHED_BATCH_ABI_VERSION;

  if (!init()) {
    io_log("MPP batch scheduler request aborted because MPP init failed.\n");
    io_trace("mpp_shim::schedBatchRequest init failed call=%llu\n",
             static_cast<unsigned long long>(call_id));
    return false;
  }

  auto &api = getMppApi();
  if (!api.sched_batch_request)
    api = loadMppApi();

  if (api.sched_batch_request) {
    std::vector<uint8_t> plan_payload(
        static_cast<size_t>(ompfile::batchPlanPayloadBytes(
            normalized_request.SegmentCount)));
    uint64_t plan_payload_bytes = plan_payload.size();
    int rc = api.sched_batch_request(
        &normalized_request,
        request_payload.empty() ? nullptr : request_payload.data(),
        static_cast<uint64_t>(request_payload.size()), &plan,
        plan_payload.empty() ? nullptr : plan_payload.data(),
        static_cast<uint64_t>(plan_payload.size()), &plan_payload_bytes);
    if (rc == 0) {
      if (plan_payload_bytes >
          static_cast<uint64_t>(plan_payload.size())) {
        io_log("MPP batch scheduler returned payload larger than caller "
               "buffer: returned=%llu buffer=%zu\n",
               static_cast<unsigned long long>(plan_payload_bytes),
               plan_payload.size());
        return false;
      }
      plan_payload.resize(static_cast<size_t>(plan_payload_bytes));
      if (!ompfile::decodeBatchPlanEntries(
              plan_payload.empty() ? nullptr : plan_payload.data(),
              plan_payload.size(), plan.SegmentCount, entries)) {
        io_log("MPP batch scheduler returned malformed plan payload.\n");
        io_trace("mpp_shim::schedBatchRequest malformed payload call=%llu\n",
                 static_cast<unsigned long long>(call_id));
        return false;
      }
      plan.PlanFlags |= ompfile::OMPFILE_BATCH_PLAN_BATCH_API;
      io_trace("mpp_shim::schedBatchRequest native success call=%llu "
               "segment_count=%u plan_flags=0x%x status=%d\n",
               static_cast<unsigned long long>(call_id), plan.SegmentCount,
               plan.PlanFlags, plan.Status);
      return true;
    }

    io_log("MPP batch scheduler API returned rc=%d; falling back to scalar "
           "scheduler path.\n",
           rc);
    io_trace("mpp_shim::schedBatchRequest native rc=%d call=%llu "
             "falling back to scalar path\n",
             rc, static_cast<unsigned long long>(call_id));
  }

  if ((normalized_request.RequestFlags &
       ompfile::OMPFILE_BATCH_REQ_DISABLE_SCALAR_FALLBACK) != 0) {
    io_log("MPP batch scheduler unavailable and scalar fallback disabled.\n");
    io_trace("mpp_shim::schedBatchRequest scalar fallback disabled call=%llu\n",
             static_cast<unsigned long long>(call_id));
    return false;
  }

  plan.AbiVersion = ompfile::OMPFILE_SCHED_BATCH_ABI_VERSION;
  plan.SegmentCount = normalized_request.SegmentCount;
  plan.BatchId = normalized_request.BatchId;
  plan.PlanFlags = ompfile::OMPFILE_BATCH_PLAN_SCALAR_FALLBACK;
  plan.PayloadBytes = static_cast<uint32_t>(
      ompfile::batchPlanPayloadBytes(normalized_request.SegmentCount));
  entries.resize(segments.size());

  bool saw_error = false;
  int first_errno = 0;
  for (size_t i = 0; i < segments.size(); ++i) {
    const ompfile::OmpFileIOBatchSegment &segment = segments[i];
    ompfile::OmpFileIOPlan scalar_plan{};
    ompfile::OmpFileIORequest scalar_request{};
    scalar_request.RequestId =
        segment.SegmentId != 0
            ? segment.SegmentId
            : (normalized_request.BatchId + static_cast<uint64_t>(i) + 1);
    scalar_request.Op = ompfile::OmpFileIOOp::PREAD;
    scalar_request.FileHandle = segment.FileHandle;
    scalar_request.ClientRank = segment.ClientRank;
    scalar_request.Offset = segment.Offset;
    scalar_request.Size = segment.Size;

    const bool ok = schedRequest(scalar_request, nullptr, scalar_plan);

    ompfile::OmpFileIOBatchPlanEntry &entry = entries[i];
    entry.SegmentId = segment.SegmentId;
    entry.AggregatorRank = scalar_plan.AggregatorRank;
    entry.RemoteHandle = scalar_plan.RemoteHandle;
    entry.Status = (ok ? scalar_plan.Status : -1);
    entry.Errno = (ok ? scalar_plan.Errno : (errno != 0 ? errno : EIO));
    entry.Offset = segment.Offset;
    entry.Size = segment.Size;
    entry.PlanFlags = ompfile::OMPFILE_BATCH_PLAN_SCALAR_FALLBACK;

    if (entry.Status != 0) {
      saw_error = true;
      if (first_errno == 0)
        first_errno = entry.Errno;
      if ((normalized_request.RequestFlags &
           ompfile::OMPFILE_BATCH_REQ_FAIL_ON_ANY_ERROR) != 0)
        break;
    }
  }

  if (saw_error) {
    plan.Status = -1;
    plan.Errno = first_errno != 0 ? first_errno : EIO;
  }

  io_trace("mpp_shim::schedBatchRequest scalar done call=%llu "
           "entries=%zu plan_status=%d plan_errno=%d flags=0x%x\n",
           static_cast<unsigned long long>(call_id), entries.size(),
           plan.Status, plan.Errno, plan.PlanFlags);
  return true;
}

bool ping() {
  static std::atomic<uint64_t> token_seed{1};
  const uint64_t token = token_seed.fetch_add(1, std::memory_order_relaxed);

  if (!submit(token))
    return false;

  for (int i = 0; i < 100; ++i) {
    bool done = false;
    if (!poll(token, done))
      return false;
    if (done)
      return true;
    std::this_thread::yield();
  }

  io_log("MPP shim ping timed out.\n");
  return false;
}

} // namespace mpp
} // namespace ompfile
