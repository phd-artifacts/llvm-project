#include "mpp_shim.h"

#include "debug_log.h"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace {

extern "C" int ompfile_mpp_freshness_write_commit(uint64_t, int, uint64_t, int,
                                                     uint64_t *)
    __attribute__((weak));
extern "C" int ompfile_mpp_proxy_copy_tile(uint64_t, uint64_t, int, int,
                                             uint64_t) __attribute__((weak));
extern "C" int ompfile_mpp_freshness_mark_fresh(uint64_t, int, uint64_t)
    __attribute__((weak));
extern "C" int ompfile_mpp_flush_dirty_tile(uint64_t, int *, uint64_t *)
    __attribute__((weak));

using MppInitFn = int (*)();
using MppSubmitFn = int (*)(uint64_t);
using MppOpenFn = int (*)(const char *, int, int, int *);
using MppOpenOnRankFn = int (*)(const char *, int, int, int, int *);
using MppHandleOwnerRankFn = int (*)(int, int *);
using MppCloseFn = int (*)(int);
using MppPreadFn = int (*)(int, int64_t, void *, uint64_t);
using MppPreadExFn = int (*)(int, int64_t, void *, uint64_t, uint64_t *);
using MppPreadNoStageExFn = int (*)(int, int64_t, void *, uint64_t, uint64_t *);
using MppDirtyOwnerPreadExFn = int (*)(int, int, uint64_t, int64_t, void *,
                                       uint64_t, uint64_t *);
struct MppDirtyOwnerPreadBatchAbiSegment {
  int64_t Offset = 0;
  uint64_t Size = 0;
  uint64_t ExpectedVersion = 0;
  uint64_t ClientSegmentId = 0;
};
using MppDirtyOwnerPreadBatchExFn = int (*)(
    int, int, const MppDirtyOwnerPreadBatchAbiSegment *, uint64_t,
    void *const *, uint64_t *, int *, int *);
using MppDirtyOwnerQueryExFn = int (*)(int, int, uint64_t, int64_t, uint64_t,
                                       int *);
using MppPwriteFn = int (*)(int, int64_t, const void *, uint64_t);
using MppPwriteExFn = int (*)(int, int64_t, const void *, uint64_t, uint64_t *);
using MppStageInvalidatePathKeyFn = int (*)(uint64_t, uint64_t, const char *);
using MppFreshnessQueryFn = int (*)(const ompfile::OmpFileFreshnessQueryRequest *,
                                    ompfile::OmpFileFreshnessQueryReply *);
using MppFreshnessWriteCommitFn = int (*)(uint64_t, int, uint64_t, int,
                                          uint64_t *);
using MppProxyCopyTileFn = int (*)(uint64_t, uint64_t, int, int, uint64_t);
using MppFreshnessMarkFreshFn = int (*)(uint64_t, int, uint64_t);
using MppFlushDirtyTileFn = int (*)(uint64_t, int *, uint64_t *);
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
  MppHandleOwnerRankFn handle_owner_rank = nullptr;
  MppCloseFn close = nullptr;
  MppPreadFn pread = nullptr;
  MppPreadExFn pread_ex = nullptr;
  MppPreadNoStageExFn pread_no_stage_ex = nullptr;
  MppDirtyOwnerPreadExFn dirty_owner_pread_ex = nullptr;
  MppDirtyOwnerPreadBatchExFn dirty_owner_pread_batch_ex = nullptr;
  MppDirtyOwnerQueryExFn dirty_owner_query_ex = nullptr;
  MppPwriteFn pwrite = nullptr;
  MppPwriteExFn pwrite_ex = nullptr;
  MppStageInvalidatePathKeyFn stage_invalidate_path_key = nullptr;
  MppFreshnessQueryFn freshness_query = nullptr;
  MppFreshnessWriteCommitFn freshness_write_commit = nullptr;
  MppProxyCopyTileFn proxy_copy_tile = nullptr;
  MppFreshnessMarkFreshFn freshness_mark_fresh = nullptr;
  MppFlushDirtyTileFn flush_dirty_tile = nullptr;
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
  loaded.handle_owner_rank = reinterpret_cast<MppHandleOwnerRankFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_handle_owner_rank"));
  loaded.close = reinterpret_cast<MppCloseFn>(dlsym(RTLD_DEFAULT,
                                                        "ompfile_mpp_close"));
  loaded.pread = reinterpret_cast<MppPreadFn>(dlsym(RTLD_DEFAULT,
                                                       "ompfile_mpp_pread"));
  loaded.pread_ex = reinterpret_cast<MppPreadExFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_pread_ex"));
  loaded.pread_no_stage_ex = reinterpret_cast<MppPreadNoStageExFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_pread_no_stage_ex"));
  loaded.dirty_owner_pread_ex = reinterpret_cast<MppDirtyOwnerPreadExFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_dirty_owner_pread_ex"));
  loaded.dirty_owner_pread_batch_ex =
      reinterpret_cast<MppDirtyOwnerPreadBatchExFn>(
          dlsym(RTLD_DEFAULT, "ompfile_mpp_dirty_owner_pread_batch_ex"));
  loaded.dirty_owner_query_ex = reinterpret_cast<MppDirtyOwnerQueryExFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_dirty_owner_query_ex"));
  loaded.pwrite = reinterpret_cast<MppPwriteFn>(dlsym(RTLD_DEFAULT,
                                                        "ompfile_mpp_pwrite"));
  loaded.pwrite_ex = reinterpret_cast<MppPwriteExFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_pwrite_ex"));
  loaded.stage_invalidate_path_key =
      reinterpret_cast<MppStageInvalidatePathKeyFn>(
          dlsym(RTLD_DEFAULT, "ompfile_mpp_stage_invalidate_path_key"));
  loaded.sched_request = reinterpret_cast<MppSchedRequestFn>(dlsym(
      RTLD_DEFAULT, "ompfile_mpp_sched_request"));
  loaded.freshness_query = reinterpret_cast<MppFreshnessQueryFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_freshness_query"));
  loaded.freshness_write_commit = reinterpret_cast<MppFreshnessWriteCommitFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_freshness_write_commit"));
  loaded.proxy_copy_tile = reinterpret_cast<MppProxyCopyTileFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_proxy_copy_tile"));
  loaded.freshness_mark_fresh = reinterpret_cast<MppFreshnessMarkFreshFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_freshness_mark_fresh"));
  loaded.flush_dirty_tile = reinterpret_cast<MppFlushDirtyTileFn>(
      dlsym(RTLD_DEFAULT, "ompfile_mpp_flush_dirty_tile"));
  loaded.sched_batch_request =
      reinterpret_cast<MppSchedBatchRequestFn>(dlsym(
          RTLD_DEFAULT, "ompfile_mpp_sched_request_batch"));
  loaded.poll = reinterpret_cast<MppPollFn>(dlsym(RTLD_DEFAULT,
                                                  "ompfile_mpp_poll"));
  loaded.finalize = reinterpret_cast<MppFinalizeFn>(dlsym(RTLD_DEFAULT,
                                                          "ompfile_mpp_finalize"));
  return loaded;
}

// The resolved MppApi function-pointer table is process-wide mutable state:
// entrypoints below re-resolve (dlsym) and reassign it whenever a symbol they
// need hasn't appeared yet (e.g. the proxy shim library is dlopen'd after
// this translation unit's first call). That reassignment used to race
// unsynchronized reads/writes of a shared static under MPI_THREAD_MULTIPLE.
// All access now goes through this mutex, and callers get a thread-local
// copy back instead of a reference into the shared static.
std::mutex &mppApiMutex() {
  static std::mutex m;
  return m;
}

MppApi &mppApiStorage() {
  static MppApi api = loadMppApi();
  return api;
}

MppApi getMppApi() {
  const std::lock_guard<std::mutex> lock(mppApiMutex());
  return mppApiStorage();
}

template <typename Fn>
MppApi getMppApiReloadIfMissing(Fn MppApi::*member) {
  const std::lock_guard<std::mutex> lock(mppApiMutex());
  MppApi &api = mppApiStorage();
  if (!(api.*member))
    api = loadMppApi();
  return api;
}

MppApi getMppApiReloadIfPwriteMissing() {
  const std::lock_guard<std::mutex> lock(mppApiMutex());
  MppApi &api = mppApiStorage();
  if (!api.pwrite && !api.pwrite_ex)
    api = loadMppApi();
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

  MppApi api = getMppApiReloadIfMissing(&MppApi::init);

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
  io_trace_symbol_owner("ompfile_mpp_dirty_owner_pread_ex",
                        reinterpret_cast<void *>(api.dirty_owner_pread_ex));
  io_trace_symbol_owner(
      "ompfile_mpp_dirty_owner_pread_batch_ex",
      reinterpret_cast<void *>(api.dirty_owner_pread_batch_ex));
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

  MppApi api = getMppApiReloadIfMissing(&MppApi::open);

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

  MppApi api = getMppApiReloadIfMissing(&MppApi::open_on_rank);

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

bool handleOwnerRank(int handle, int &rank_out) {
  rank_out = -1;
  if (!init())
    return false;

  MppApi api = getMppApiReloadIfMissing(&MppApi::handle_owner_rank);
  if (!api.handle_owner_rank) {
    errno = ENOSYS;
    return false;
  }

  const int rc = api.handle_owner_rank(handle, &rank_out);
  if (rc != 0 || rank_out < 0) {
    errno = rc != 0 ? (rc < 0 ? EIO : rc) : EIO;
    rank_out = -1;
    return false;
  }
  return true;
}

bool close(int handle) {
  const uint64_t call_id = nextShimCallId();
  io_trace("mpp_shim::close enter call=%llu handle=%d\n",
           static_cast<unsigned long long>(call_id), handle);
  if (!init())
    return false;

  MppApi api = getMppApiReloadIfMissing(&MppApi::close);

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

  MppApi api = getMppApiReloadIfMissing(&MppApi::pread);

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

bool preadNoStageEx(int handle, int64_t offset, void *buffer, size_t size,
                    size_t &bytes_read) {
  bytes_read = 0;
  if (!buffer && size > 0)
    return false;

  if (!init())
    return false;

  MppApi api = getMppApiReloadIfMissing(&MppApi::pread_no_stage_ex);

  if (!api.pread_no_stage_ex)
    return preadEx(handle, offset, buffer, size, bytes_read);

  uint64_t remote_bytes_read = 0;
  const int rc = api.pread_no_stage_ex(handle, offset, buffer, size,
                                       &remote_bytes_read);
  if (rc != 0)
    return false;
  if (remote_bytes_read >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return false;
  bytes_read = static_cast<size_t>(remote_bytes_read);
  return true;
}

bool dirtyOwnerPreadEx(int handle, int source_rank, uint64_t expected_version,
                       int64_t offset, void *buffer, size_t size,
                       size_t &bytes_read) {
  bytes_read = 0;
  if ((!buffer && size > 0) || source_rank < 0) {
    errno = EINVAL;
    return false;
  }
  if (!init())
    return false;
  MppApi api = getMppApiReloadIfMissing(&MppApi::dirty_owner_pread_ex);
  if (!api.dirty_owner_pread_ex) {
    errno = ENOSYS;
    return false;
  }
  uint64_t remote_bytes_read = 0;
  const int rc = api.dirty_owner_pread_ex(handle, source_rank, expected_version,
                                          offset, buffer, size,
                                          &remote_bytes_read);
  if (rc != 0) {
    if (errno == 0)
      errno = rc < 0 ? EIO : rc;
    return false;
  }
  if (remote_bytes_read >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    errno = EOVERFLOW;
    return false;
  }
  bytes_read = static_cast<size_t>(remote_bytes_read);
  return true;
}

bool dirtyOwnerPreadBatchEx(int handle, int source_rank,
                            DirtyOwnerPreadBatchSegment *segments,
                            size_t segment_count) {
  if (!segments || segment_count == 0 || source_rank < 0) {
    errno = EINVAL;
    return false;
  }
  if (!init())
    return false;
  MppApi api = getMppApiReloadIfMissing(&MppApi::dirty_owner_pread_batch_ex);
  if (!api.dirty_owner_pread_batch_ex) {
    errno = ENOSYS;
    return false;
  }

  std::vector<MppDirtyOwnerPreadBatchAbiSegment> AbiSegments(segment_count);
  std::vector<void *> Buffers(segment_count, nullptr);
  std::vector<uint64_t> Bytes(segment_count, 0);
  std::vector<int> Statuses(segment_count, -1);
  std::vector<int> Errnos(segment_count, 0);
  for (size_t I = 0; I < segment_count; ++I) {
    if (!segments[I].buffer && segments[I].size > 0) {
      errno = EINVAL;
      return false;
    }
    AbiSegments[I].Offset = segments[I].offset;
    AbiSegments[I].Size = static_cast<uint64_t>(segments[I].size);
    AbiSegments[I].ExpectedVersion = segments[I].expected_version;
    AbiSegments[I].ClientSegmentId = I;
    Buffers[I] = segments[I].buffer;
    segments[I].bytes_read = 0;
    segments[I].status = -1;
    segments[I].error = 0;
  }

  const int rc = api.dirty_owner_pread_batch_ex(
      handle, source_rank, AbiSegments.data(),
      static_cast<uint64_t>(segment_count), Buffers.data(), Bytes.data(),
      Statuses.data(), Errnos.data());
  bool AllOk = rc == 0;
  for (size_t I = 0; I < segment_count; ++I) {
    segments[I].status = Statuses[I];
    segments[I].error = Errnos[I];
    if (Bytes[I] > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      segments[I].status = -1;
      segments[I].error = EOVERFLOW;
      AllOk = false;
      continue;
    }
    segments[I].bytes_read = static_cast<size_t>(Bytes[I]);
    if (segments[I].status != 0)
      AllOk = false;
  }
  if (!AllOk && errno == 0)
    errno = EIO;
  return AllOk;
}

bool dirtyOwnerQueryEx(int handle, int source_rank, uint64_t expected_version,
                       int64_t offset, size_t size, int &state_out) {
  state_out = -1;
  if (source_rank < 0) {
    errno = EINVAL;
    return false;
  }
  if (!init())
    return false;
  MppApi api = getMppApiReloadIfMissing(&MppApi::dirty_owner_query_ex);
  if (!api.dirty_owner_query_ex) {
    errno = ENOSYS;
    return false;
  }
  int remote_state = -1;
  const int rc = api.dirty_owner_query_ex(handle, source_rank, expected_version,
                                          offset, size, &remote_state);
  if (rc != 0) {
    if (errno == 0)
      errno = rc < 0 ? EIO : rc;
    return false;
  }
  state_out = remote_state;
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

  MppApi api = getMppApiReloadIfPwriteMissing();

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
  MppApi api = getMppApiReloadIfMissing(&MppApi::stage_invalidate_path_key);

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

bool freshnessQuery(uint64_t path_key, uint64_t local_version,
                    int requester_rank, uint32_t &decision_out,
                    int &source_rank_out, uint64_t &selected_version_out) {
  decision_out = static_cast<uint32_t>(ompfile::OmpFileFreshnessDecision::WAIT_OR_FAIL);
  source_rank_out = -1;
  selected_version_out = 0;
  if (path_key == 0 || requester_rank < 0) {
    errno = EINVAL;
    return false;
  }
  if (!init())
    return false;

  MppApi api = getMppApiReloadIfMissing(&MppApi::freshness_query);
  if (!api.freshness_query) {
    errno = ENOSYS;
    return false;
  }

  ompfile::OmpFileFreshnessQueryRequest request{};
  request.AbiVersion = ompfile::OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
  request.PathKey = path_key;
  request.LocalVersion = local_version;
  request.RequesterRank = requester_rank;

  ompfile::OmpFileFreshnessQueryReply reply{};
  const int rc = api.freshness_query(&request, &reply);
  if (rc != 0) {
    errno = rc < 0 ? EIO : rc;
    return false;
  }
  if (reply.Status != 0) {
    errno = reply.Errno != 0 ? reply.Errno : EIO;
    return false;
  }

  if (reply.AbiVersion != ompfile::OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
    errno = EPROTO;
    return false;
  }

  if (reply.Decision >
      static_cast<uint32_t>(ompfile::OmpFileFreshnessDecision::COPY_FROM_RANK)) {
    errno = EPROTO;
    return false;
  }

  decision_out = reply.Decision;
  source_rank_out = reply.SourceRank;
  selected_version_out = reply.SelectedVersion;
  return true;
}

bool freshnessWriteCommit(uint64_t path_key, int writer_rank, uint64_t tile_id,
                          bool write_through_mode,
                          uint64_t &committed_version_out) {
  committed_version_out = 0;
  if (path_key == 0 || writer_rank < 0) {
    errno = EINVAL;
    return false;
  }
  const int mode = write_through_mode ? 1 : 0;
  if (ompfile_mpp_freshness_write_commit) {
    const int rc = ompfile_mpp_freshness_write_commit(
        path_key, writer_rank, tile_id, mode, &committed_version_out);
    if (rc != 0) {
      errno = rc < 0 ? EIO : rc;
      committed_version_out = 0;
      return false;
    }
    return true;
  }

  if (!init())
    return false;

  MppApi api = getMppApiReloadIfMissing(&MppApi::freshness_write_commit);
  if (!api.freshness_write_commit) {
    errno = ENOSYS;
    return false;
  }

  const int rc = api.freshness_write_commit(path_key, writer_rank, tile_id,
                                            mode, &committed_version_out);
  if (rc != 0) {
    errno = rc < 0 ? EIO : rc;
    committed_version_out = 0;
    return false;
  }
  return true;
}

bool proxyCopyTile(uint64_t path_key, uint64_t tile_id, int source_rank,
                   int dest_rank, uint64_t version) {
  if (path_key == 0 || source_rank < 0 || dest_rank < 0 || version == 0) {
    errno = EINVAL;
    return false;
  }

  if (ompfile_mpp_proxy_copy_tile) {
    const int rc = ompfile_mpp_proxy_copy_tile(path_key, tile_id, source_rank,
                                               dest_rank, version);
    if (rc != 0) {
      errno = rc < 0 ? EIO : rc;
      return false;
    }
    return true;
  }

  if (!init())
    return false;

  MppApi api = getMppApiReloadIfMissing(&MppApi::proxy_copy_tile);
  if (!api.proxy_copy_tile) {
    errno = ENOSYS;
    return false;
  }

  const int rc =
      api.proxy_copy_tile(path_key, tile_id, source_rank, dest_rank, version);
  if (rc != 0) {
    errno = rc < 0 ? EIO : rc;
    return false;
  }
  return true;
}

bool freshnessMarkFresh(uint64_t path_key, int rank, uint64_t version) {
  if (path_key == 0 || rank < 0 || version == 0) {
    errno = EINVAL;
    return false;
  }

  if (ompfile_mpp_freshness_mark_fresh) {
    const int rc = ompfile_mpp_freshness_mark_fresh(path_key, rank, version);
    if (rc != 0) {
      errno = rc < 0 ? EIO : rc;
      return false;
    }
    return true;
  }

  if (!init())
    return false;

  MppApi api = getMppApiReloadIfMissing(&MppApi::freshness_mark_fresh);
  if (!api.freshness_mark_fresh) {
    errno = ENOSYS;
    return false;
  }

  const int rc = api.freshness_mark_fresh(path_key, rank, version);
  if (rc != 0) {
    errno = rc < 0 ? EIO : rc;
    return false;
  }
  return true;
}

bool flushDirtyTile(uint64_t path_key, int &source_rank_out,
                    uint64_t &flushed_version_out) {
  source_rank_out = -1;
  flushed_version_out = 0;
  if (path_key == 0) {
    errno = EINVAL;
    return false;
  }

  if (ompfile_mpp_flush_dirty_tile) {
    const int rc = ompfile_mpp_flush_dirty_tile(path_key, &source_rank_out,
                                                &flushed_version_out);
    if (rc != 0) {
      errno = rc < 0 ? EIO : rc;
      source_rank_out = -1;
      flushed_version_out = 0;
      return false;
    }
    return true;
  }

  if (!init())
    return false;

  MppApi api = getMppApiReloadIfMissing(&MppApi::flush_dirty_tile);
  if (!api.flush_dirty_tile) {
    errno = ENOSYS;
    return false;
  }

  const int rc =
      api.flush_dirty_tile(path_key, &source_rank_out, &flushed_version_out);
  if (rc != 0) {
    errno = rc < 0 ? EIO : rc;
    source_rank_out = -1;
    flushed_version_out = 0;
    return false;
  }
  return true;
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

  MppApi api = getMppApiReloadIfMissing(&MppApi::sched_request);

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

  MppApi api = getMppApiReloadIfMissing(&MppApi::sched_batch_request);

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
