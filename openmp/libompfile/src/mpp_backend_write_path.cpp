#include "debug_log.h"
#include "ompfile_config.h"
#include "mpp_backend.h"
#include "mpp_shim.h"

#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

bool MppBackend::isWriteBatchActive() const {
  return settings.write_batch_enabled;
}

int MppBackend::writeAtRemoteHandle(int remote_handle, long offset,
                                    const void *data, size_t size,
                                    size_t &bytes_written) {
  bytes_written = 0;
  if (size == 0)
    return 0;

  const char *cursor = static_cast<const char *>(data);
  long current_offset = offset;
  size_t remaining = size;

  while (remaining > 0) {
    counters.remote_pwrite_event_count.fetch_add(1, std::memory_order_relaxed);
    counters.remote_pwrite_bytes_total.fetch_add(remaining,
                                                 std::memory_order_relaxed);

    bool pwrite_ok = false;
    size_t call_bytes_written = 0;
    {
      const auto lock = instrumentedMppCallLock();
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
        counters.short_write_count.fetch_add(1, std::memory_order_relaxed);
        counters.short_write_bytes_total.fetch_add(remaining,
                                                   std::memory_order_relaxed);
      }
      errno = saved_errno;
      return -1;
    }
    if (call_bytes_written > remaining) {
      errno = EPROTO;
      return -1;
    }
    if (call_bytes_written == 0) {
      counters.short_write_count.fetch_add(1, std::memory_order_relaxed);
      counters.short_write_bytes_total.fetch_add(remaining,
                                                 std::memory_order_relaxed);
      errno = EIO;
      return -1;
    }

    bytes_written += call_bytes_written;
    if (call_bytes_written < remaining) {
      counters.short_write_count.fetch_add(1, std::memory_order_relaxed);
      counters.short_write_bytes_total.fetch_add(remaining - call_bytes_written,
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

int MppBackend::writeAtWithContext(
    const ompfile::OmpFileWriteRequestContext &context, const void *data,
    size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  uint64_t group_key = ompfile::keys::handleKey(file_id);
  if ((context.ContextFlags & ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY) != 0)
    group_key = context.PathKey;
  else {
    uint64_t path_key = 0;
    if (getFilePathKey(file_id, path_key))
      group_key = path_key;
  }
  group_key = ompfile::keys::writeGroupKey(group_key, context.Hint);

  if (settings.strict_mpp_init_failed)
    return failStrictMpp("pwrite");
  counters.pwrite_request_count.fetch_add(1, std::memory_order_relaxed);
  if (offset < 0) {
    errno = EINVAL;
    return -1;
  }
  if (!data && size > 0) {
    errno = EINVAL;
    return -1;
  }

  io_log("Writing %zu bytes to file %d at offset %lld\n", size, file_id,
         static_cast<long long>(offset));

  ompfile::route::WriteRouteConfig route_config{};
  route_config.WriteBatchActive = isWriteBatchActive();
  if (!route_config.WriteBatchActive)
    route_config.Distribute = snapshotDistributeConfig();
  const ompfile::route::WriteDecision decision =
      ompfile::route::resolveWriteRoute(context, size, route_config);
  return executeWrite(context, decision, group_key, data, size);
}

int MppBackend::executeWrite(
    const ompfile::OmpFileWriteRequestContext &context,
    const ompfile::route::WriteDecision &decision, uint64_t group_key,
    const void *data, size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  if (decision.Route == ompfile::route::WriteRoute::Batched) {
    const int batched_rc =
        writeAtBatched(file_id, offset, data, size, group_key);
    if (batched_rc == 0) {
      recordWriteEpochForContext(file_id, offset, size, context.Hint);
      if (!reportFreshnessWriteCommit(context, /*writer_rank_override=*/-1)) {
        if (errno == 0)
          errno = EIO;
        return -1;
      }
    }
    return batched_rc;
  }

  if (decision.Route == ompfile::route::WriteRoute::DistributedRank) {
    size_t distributed_bytes_written = 0;
    if (writeAtRemoteRankWithBytes(file_id, decision.TargetRank, offset,
                                   data, size,
                                   distributed_bytes_written)) {
      assert(distributed_bytes_written == size &&
             "Distributed remote pwrite must complete full request");
      recordWriteEpochForContext(file_id, offset, size, context.Hint);
      if (!reportFreshnessWriteCommit(context, decision.TargetRank)) {
        if (errno == 0)
          errno = EIO;
        return -1;
      }
      io_log("MPP distributed write at offset completed: file=%d rank=%d "
             "bytes=%zu.\n",
             file_id, decision.TargetRank, distributed_bytes_written);
      return 0;
    }
    io_log("MPP distributed pwrite failed for file %d rank=%d offset=%lld "
           "size=%zu errno=%d; falling back to owner handle.\n",
           file_id, decision.TargetRank, static_cast<long long>(offset),
           size, errno);
  }

  int remote_handle = lookupRemoteHandle(file_id);
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
  if (!reportFreshnessWriteCommit(context, /*writer_rank_override=*/-1)) {
    if (errno == 0)
      errno = EIO;
    return -1;
  }
  io_log("MPP write at offset completed.\n");
  return 0;
}

bool MppBackend::reportFreshnessWriteCommit(
    const ompfile::OmpFileWriteRequestContext &context,
    int writer_rank_override) {
  const int file_id = context.FileHandle;
  if ((context.Hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) == 0)
    return true;

  uint64_t path_key = 0;
  if ((context.ContextFlags & ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY) != 0)
    path_key = context.PathKey;
  else if (!getFilePathKey(file_id, path_key))
    return true;

  if (path_key == 0)
    return true;

  int writer_rank = writer_rank_override;
  if (writer_rank < 0)
    writer_rank = lookupRemoteOwnerRank(file_id);
  if (writer_rank < 0) {
    io_log("Freshness write-commit failed: missing remote owner rank for "
           "file %d.\n",
           file_id);
    errno = ENOKEY;
    return false;
  }

  const bool write_through_mode = config_.TileFreshnessWriteThrough;
  const uint64_t freshness_key =
      ompfile::keys::tileKeyForHint(path_key, context.Hint);
  uint64_t committed_version = 0;
  if (!ompfile::mpp::freshnessWriteCommit(
          freshness_key, writer_rank, context.Hint.TileId, write_through_mode,
          committed_version)) {
    io_log("Freshness write-commit failed path_key=%llu freshness_key=%llu writer=%d tile=%llu "
           "mode=%s errno=%d\n",
           static_cast<unsigned long long>(path_key),
           static_cast<unsigned long long>(freshness_key), writer_rank,
           static_cast<unsigned long long>(context.Hint.TileId),
           write_through_mode ? "write-through" : "write-back", errno);
    return false;
  }

  io_trace("MPIIOBackend::writeAtWithContext freshness write-commit "
           "path_key=%llu freshness_key=%llu writer=%d tile=%llu mode=%s version=%llu\n",
           static_cast<unsigned long long>(path_key),
           static_cast<unsigned long long>(freshness_key), writer_rank,
           static_cast<unsigned long long>(context.Hint.TileId),
           write_through_mode ? "write-through" : "write-back",
           static_cast<unsigned long long>(committed_version));
  return true;
}

int MppBackend::writeAt(int file_id, long offset, const void *data,
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

// The write batch on ompfile::batch::Batcher (LeaderPerBatch): Fixed or
// Adaptive window from LIBOMPFILE_OPT_WRITE_BATCH_WINDOW_US / _ADAPTIVE, drain
// of the whole queue, execute = processWriteBatch. The leader holds its window
// to an absolute deadline; before Oct 2026 the first follower's enqueue
// notification ended it.
struct MppBackend::WriteBatchSite
    : ompfile::batch::SiteDefaults<MppBackend::WriteBatchRequest> {
  MppBackend &Backend;
  explicit WriteBatchSite(MppBackend &B) : Backend(B) {}
  ompfile::batch::ClosePolicy closePolicy() {
    return {Backend.settings.write_batch_window_adaptive
                ? ompfile::batch::CloseRule::Adaptive
                : ompfile::batch::CloseRule::Fixed,
            Backend.settings.write_batch_window_us, 0};
  }
  void execute(std::vector<WriteBatchRequest *> &batch) {
    Backend.processWriteBatch(batch);
  }
};

int MppBackend::writeAtBatched(int file_id, long offset, const void *data,
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

  WriteBatchSite site(*this);
  (void)write_batch_batcher.submit(request, site);
  if (request.Status != 0) {
    errno = request.Errno;
    return -1;
  }
  return 0;
}
