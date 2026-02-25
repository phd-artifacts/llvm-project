#include "mpp_shim.h"

#include "debug_log.h"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <thread>

namespace {

using MppInitFn = int (*)();
using MppSubmitFn = int (*)(uint64_t);
using MppOpenFn = int (*)(const char *, int, int, int *);
using MppCloseFn = int (*)(int);
using MppPreadFn = int (*)(int, int64_t, void *, uint64_t);
using MppPwriteFn = int (*)(int, int64_t, const void *, uint64_t);
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
  MppCloseFn close = nullptr;
  MppPreadFn pread = nullptr;
  MppPwriteFn pwrite = nullptr;
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
  loaded.close = reinterpret_cast<MppCloseFn>(dlsym(RTLD_DEFAULT,
                                                       "ompfile_mpp_close"));
  loaded.pread = reinterpret_cast<MppPreadFn>(dlsym(RTLD_DEFAULT,
                                                       "ompfile_mpp_pread"));
  loaded.pwrite = reinterpret_cast<MppPwriteFn>(dlsym(RTLD_DEFAULT,
                                                        "ompfile_mpp_pwrite"));
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

} // namespace

namespace ompfile {
namespace mpp {

bool init() {
  static std::atomic<bool> ready{false};
  if (ready.load(std::memory_order_acquire))
    return true;

  auto &api = getMppApi();
  if (!api.init)
    api = loadMppApi();

  if (!api.init) {
    io_log("MPP shim not available (ompfile_mpp_init missing).\n");
    return false;
  }

  int init_rc = api.init();
  if (init_rc != 0) {
    io_log("MPP shim init failed (rc=%d).\n", init_rc);
    return false;
  }

  ready.store(true, std::memory_order_release);
  io_log("MPP shim initialized.\n");
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
  handle = -1;
  if (!init())
    return false;

  auto &api = getMppApi();
  if (!api.open)
    api = loadMppApi();

  if (!api.open) {
    io_log("MPP shim not available (ompfile_mpp_open missing).\n");
    return false;
  }

  if (api.open(path, flags, mode, &handle) != 0) {
    io_log("MPP shim open failed.\n");
    return false;
  }

  return true;
}

bool close(int handle) {
  if (!init())
    return false;

  auto &api = getMppApi();
  if (!api.close)
    api = loadMppApi();

  if (!api.close) {
    io_log("MPP shim not available (ompfile_mpp_close missing).\n");
    return false;
  }

  return api.close(handle) == 0;
}

bool pread(int handle, int64_t offset, void *buffer, size_t size) {
  if (!buffer && size > 0)
    return false;

  if (!init())
    return false;

  auto &api = getMppApi();
  if (!api.pread)
    api = loadMppApi();

  if (!api.pread) {
    io_log("MPP shim not available (ompfile_mpp_pread missing).\n");
    return false;
  }

  return api.pread(handle, offset, buffer, size) == 0;
}

bool pwrite(int handle, int64_t offset, const void *buffer, size_t size) {
  if (!buffer && size > 0)
    return false;

  if (!init())
    return false;

  auto &api = getMppApi();
  if (!api.pwrite)
    api = loadMppApi();

  if (!api.pwrite) {
    io_log("MPP shim not available (ompfile_mpp_pwrite missing).\n");
    return false;
  }

  return api.pwrite(handle, offset, buffer, size) == 0;
}

bool schedRequest(const ompfile::OmpFileIORequest &request, const char *path,
                  ompfile::OmpFileIOPlan &plan) {
  if (request.PathSize > 0 && !path)
    return false;

  if (!init()) {
    io_log("MPP scheduler request aborted because MPP init failed.\n");
    return false;
  }

  auto &api = getMppApi();
  if (!api.sched_request)
    api = loadMppApi();

  if (!api.sched_request) {
    io_log("MPP shim not available (ompfile_mpp_sched_request missing).\n");
    return false;
  }

  return api.sched_request(&request, path, &plan) == 0;
}

bool schedBatchRequest(
    const ompfile::OmpFileIOBatchRequest &request,
    const std::vector<ompfile::OmpFileIOBatchSegment> &segments,
    ompfile::OmpFileIOBatchPlan &plan,
    std::vector<ompfile::OmpFileIOBatchPlanEntry> &entries) {
  plan = {};
  entries.clear();

  if (request.AbiVersion != ompfile::OMPFILE_SCHED_BATCH_ABI_VERSION) {
    io_log("MPP batch scheduler request has unsupported ABI version=%u\n",
           request.AbiVersion);
    return false;
  }

  if (request.SegmentCount != segments.size()) {
    io_log("MPP batch scheduler segment count mismatch: header=%u vec=%zu\n",
           request.SegmentCount, segments.size());
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
        return false;
      }
      plan.PlanFlags |= ompfile::OMPFILE_BATCH_PLAN_BATCH_API;
      return true;
    }

    io_log("MPP batch scheduler API returned rc=%d; falling back to scalar "
           "scheduler path.\n",
           rc);
  }

  if ((normalized_request.RequestFlags &
       ompfile::OMPFILE_BATCH_REQ_DISABLE_SCALAR_FALLBACK) != 0) {
    io_log("MPP batch scheduler unavailable and scalar fallback disabled.\n");
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
