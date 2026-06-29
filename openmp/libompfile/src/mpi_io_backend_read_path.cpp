#include "debug_log.h"
#include "mpi_io_backend.h"
#include "mpp_shim.h"

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <chrono>
#include <mutex>
#include <thread>

namespace {

bool envEquals(const char *Name, const char *Expected) {
  const char *Value = std::getenv(Name);
  return Value && std::strcmp(Value, Expected) == 0;
}

bool scalarPlannedReadRebalanceEnabledForStage() {
  return envEquals("LIBOMPFILE_STAGE_MODE", "readthrough") &&
         envEquals("LIBOMPFILE_STAGE_WRITE_MODE", "write-back") &&
         !envEquals("LIBOMPFILE_STAGE_FRESHNESS_GUARD", "0");
}

struct FreshnessRouteDecision {
  ompfile::OmpFileFreshnessDecision Decision =
      ompfile::OmpFileFreshnessDecision::WAIT_OR_FAIL;
  int SourceRank = -1;
  uint64_t SelectedVersion = 0;
  bool Valid = false;
};

FreshnessRouteDecision queryFreshnessForRank(uint64_t path_key,
                                             int target_rank) {
  FreshnessRouteDecision Result{};
  if (path_key == 0 || target_rank < 0)
    return Result;

  uint32_t decision = static_cast<uint32_t>(
      ompfile::OmpFileFreshnessDecision::WAIT_OR_FAIL);
  int source_rank = -1;
  uint64_t selected_version = 0;
  if (!ompfile::mpp::freshnessQuery(path_key, /*local_version=*/0,
                                    target_rank, decision, source_rank,
                                    selected_version))
    return Result;

  Result.Decision = static_cast<ompfile::OmpFileFreshnessDecision>(decision);
  Result.SourceRank = source_rank;
  Result.SelectedVersion = selected_version;
  Result.Valid = true;
  return Result;
}

} // namespace

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
    if (context.Plan.AggregatorRank >= 0 && mpp_remote_only &&
        scalarPlannedReadRebalanceEnabledForStage()) {
      const char *rebalance_reason = "none";
      int rebalance_errno = 0;
      const FreshnessRouteDecision InitialFreshnessRoute =
          queryFreshnessForRank(context.PathKey, context.Plan.AggregatorRank);
      bool has_local_write_history = false;
      {
        const std::lock_guard<std::mutex> lock(handle_mutex);
        auto write_it = file_write_epoch_history.find(file_id);
        has_local_write_history =
            write_it != file_write_epoch_history.end() &&
            !write_it->second.empty();
      }
      const bool file_has_writeback_epoch =
          InitialFreshnessRoute.Valid &&
          InitialFreshnessRoute.SelectedVersion != 0;
      if (dirty_owner_forwarding_enabled &&
          (has_local_write_history || file_has_writeback_epoch)) {
        constexpr int ConservativeWritebackOwnerRank = 0;
        size_t source_bytes_read = 0;
        if (readAtRemoteRankWithBytes(file_id, ConservativeWritebackOwnerRank,
                                      offset, data, size, source_bytes_read)) {
          noteAppliedRebalancedReadForFile(file_id);
          io_log("Phase 1 write-back owner-staged read: request_id=%llu "
                 "file=%d aggregator_rank=%d source_rank=%d version=%llu "
                 "bytes=%zu local_write_history=%d file_writeback_epoch=%d\n",
                 static_cast<unsigned long long>(context.RequestId), file_id,
                 context.Plan.AggregatorRank, ConservativeWritebackOwnerRank,
                 static_cast<unsigned long long>(
                     InitialFreshnessRoute.SelectedVersion),
                 source_bytes_read, static_cast<int>(has_local_write_history),
                 static_cast<int>(file_has_writeback_epoch));
          return 0;
        }
        io_log("Phase 1 write-back owner-staged read failed: request_id=%llu "
               "file=%d aggregator_rank=%d errno=%d; fail-closed.\n",
               static_cast<unsigned long long>(context.RequestId), file_id,
               context.Plan.AggregatorRank, errno);
        errno = errno != 0 ? errno : EIO;
        return -1;
      }

      const bool freshness_needs_dirty_owner =
          InitialFreshnessRoute.Valid &&
          InitialFreshnessRoute.Decision ==
              ompfile::OmpFileFreshnessDecision::COPY_FROM_RANK &&
          InitialFreshnessRoute.SourceRank >= 0 &&
          InitialFreshnessRoute.SourceRank != context.Plan.AggregatorRank;
      const bool freshness_allows_rebalance =
          InitialFreshnessRoute.Valid &&
          (InitialFreshnessRoute.Decision ==
               ompfile::OmpFileFreshnessDecision::READ_PFS ||
           InitialFreshnessRoute.Decision ==
               ompfile::OmpFileFreshnessDecision::USE_LOCAL ||
           (InitialFreshnessRoute.Decision ==
                ompfile::OmpFileFreshnessDecision::COPY_FROM_RANK &&
            InitialFreshnessRoute.SourceRank == context.Plan.AggregatorRank));
      const bool metadata_safe =
          canApplyRebalancedRead(context.Hint, file_id, offset, size,
                                 rebalance_reason, rebalance_errno);
      const bool planned_rank_current =
          InitialFreshnessRoute.Valid &&
          (InitialFreshnessRoute.Decision ==
               ompfile::OmpFileFreshnessDecision::USE_LOCAL ||
           (InitialFreshnessRoute.Decision ==
                ompfile::OmpFileFreshnessDecision::COPY_FROM_RANK &&
            InitialFreshnessRoute.SourceRank == context.Plan.AggregatorRank));
      const bool metadata_blocked_only_by_write_history =
          std::strcmp(rebalance_reason, "unsafe-write-history") == 0;
      const bool pfs_current_after_writeback =
          dirty_owner_forwarding_enabled && file_has_writeback_epoch &&
          InitialFreshnessRoute.Valid &&
          InitialFreshnessRoute.Decision ==
              ompfile::OmpFileFreshnessDecision::READ_PFS;
      const bool rebalanced_safe =
          freshness_allows_rebalance && !pfs_current_after_writeback &&
          (metadata_safe ||
           (dirty_owner_forwarding_enabled && file_has_writeback_epoch &&
            planned_rank_current && metadata_blocked_only_by_write_history));
      if (rebalanced_safe) {
        size_t bytes_read = 0;
        if (readAtRemoteRankWithBytes(file_id, context.Plan.AggregatorRank,
                                      offset, data, size, bytes_read)) {
          noteAppliedRebalancedReadForFile(file_id);
          io_log("Phase 1 planned read rebalanced: request_id=%llu file=%d "
                 "path_key=%llu aggregator_rank=%d bytes=%zu\n",
                 static_cast<unsigned long long>(context.RequestId), file_id,
                 static_cast<unsigned long long>(context.PathKey),
                 context.Plan.AggregatorRank, bytes_read);
          return 0;
        }
        io_log("Phase 1 planned read rebalanced route failed: request_id=%llu "
               "file=%d aggregator_rank=%d errno=%d; fail-closed in "
               "write-back dirty-owner mode.\n",
               static_cast<unsigned long long>(context.RequestId), file_id,
               context.Plan.AggregatorRank, errno);
        if (dirty_owner_forwarding_enabled) {
          errno = errno != 0 ? errno : EIO;
          return -1;
        }
      } else if (dirty_owner_forwarding_enabled &&
                 (freshness_needs_dirty_owner ||
                  std::strcmp(rebalance_reason, "unsafe-write-history") == 0)) {
        int remote_handle = -1;
        {
          const std::lock_guard<std::mutex> lock(handle_mutex);
          auto it = remote_file_handle_map.find(file_id);
          if (it != remote_file_handle_map.end())
            remote_handle = it->second;
        }
        if (remote_handle >= 0) {
          constexpr unsigned MaxDirtyOwnerForwardRetries = 8;
          int last_forward_errno = 0;
          bool attempted_forward = false;
          bool resolved_to_pfs = false;
          for (unsigned attempt = 0; attempt <= MaxDirtyOwnerForwardRetries;
               ++attempt) {
            const FreshnessRouteDecision Route = queryFreshnessForRank(
                context.PathKey, context.Plan.AggregatorRank);
            if (!Route.Valid) {
              if (attempted_forward && attempt < MaxDirtyOwnerForwardRetries) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
              }
              break;
            }
            if (Route.Decision == ompfile::OmpFileFreshnessDecision::READ_PFS) {
              if (has_local_write_history) {
                size_t source_bytes_read = 0;
                constexpr int ConservativeWritebackOwnerRank = 0;
                if (readAtRemoteRankWithBytes(file_id,
                                              ConservativeWritebackOwnerRank,
                                              offset, data, size,
                                              source_bytes_read)) {
                  noteAppliedRebalancedReadForFile(file_id);
                  io_log("Phase 1 dirty-owner pfs-current source read: "
                         "request_id=%llu file=%d aggregator_rank=%d "
                         "source_rank=%d version=%llu bytes=%zu attempt=%u\n",
                         static_cast<unsigned long long>(context.RequestId),
                         file_id, context.Plan.AggregatorRank,
                         ConservativeWritebackOwnerRank,
                         static_cast<unsigned long long>(Route.SelectedVersion),
                         source_bytes_read, attempt);
                  return 0;
                }
                last_forward_errno = errno != 0 ? errno : EIO;
                break;
              }
              resolved_to_pfs = true;
              break;
            }
            if (Route.Decision == ompfile::OmpFileFreshnessDecision::USE_LOCAL ||
                (Route.Decision ==
                     ompfile::OmpFileFreshnessDecision::COPY_FROM_RANK &&
                 Route.SourceRank == context.Plan.AggregatorRank)) {
              size_t local_bytes_read = 0;
              if (readAtRemoteRankWithBytes(file_id, context.Plan.AggregatorRank,
                                            offset, data, size,
                                            local_bytes_read)) {
                noteAppliedRebalancedReadForFile(file_id);
                io_log("Phase 1 dirty-owner retry resolved local: "
                       "request_id=%llu file=%d aggregator_rank=%d "
                       "version=%llu bytes=%zu attempt=%u\n",
                       static_cast<unsigned long long>(context.RequestId),
                       file_id, context.Plan.AggregatorRank,
                       static_cast<unsigned long long>(Route.SelectedVersion),
                       local_bytes_read, attempt);
                return 0;
              }
              last_forward_errno = errno != 0 ? errno : EIO;
              break;
            }
            if (Route.Decision !=
                    ompfile::OmpFileFreshnessDecision::COPY_FROM_RANK ||
                Route.SourceRank < 0) {
              if (attempted_forward && attempt < MaxDirtyOwnerForwardRetries) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
              }
              break;
            }

            attempted_forward = true;
            dirty_owner_forward_attempt_count.fetch_add(
                1, std::memory_order_relaxed);
            size_t bytes_read = 0;
            bool forward_ok = false;
            {
              const std::lock_guard<std::mutex> lock(mpp_call_mutex);
              errno = 0;
              forward_ok = ompfile::mpp::dirtyOwnerPreadEx(
                  remote_handle, Route.SourceRank, Route.SelectedVersion,
                  offset, data, size, bytes_read);
              last_forward_errno = errno;
            }
            if (forward_ok) {
              dirty_owner_forward_success_count.fetch_add(
                  1, std::memory_order_relaxed);
              dirty_owner_forward_bytes_total.fetch_add(
                  bytes_read, std::memory_order_relaxed);
              remote_pread_event_count.fetch_add(1,
                                                 std::memory_order_relaxed);
              remote_pread_bytes_total.fetch_add(size,
                                                 std::memory_order_relaxed);
              noteAppliedRebalancedReadForFile(file_id);
              io_log("Phase 1 dirty-owner forward read: request_id=%llu "
                     "file=%d path_key=%llu aggregator_rank=%d source_rank=%d "
                     "version=%llu bytes=%zu attempt=%u\n",
                     static_cast<unsigned long long>(context.RequestId),
                     file_id, static_cast<unsigned long long>(context.PathKey),
                     context.Plan.AggregatorRank, Route.SourceRank,
                     static_cast<unsigned long long>(Route.SelectedVersion),
                     bytes_read, attempt);
              return 0;
            }

            dirty_owner_forward_failure_count.fetch_add(
                1, std::memory_order_relaxed);
            const char *forward_failure_action = "; fail-closed";
            if (last_forward_errno == ESTALE &&
                attempt < MaxDirtyOwnerForwardRetries)
              forward_failure_action = "; retrying freshness";
            else if (last_forward_errno == ENODATA)
              forward_failure_action = "; routing source-rank staged read";
            io_log("Phase 1 dirty-owner forward failed: request_id=%llu "
                   "file=%d aggregator_rank=%d source_rank=%d version=%llu "
                   "errno=%d attempt=%u%s\n",
                   static_cast<unsigned long long>(context.RequestId), file_id,
                   context.Plan.AggregatorRank, Route.SourceRank,
                   static_cast<unsigned long long>(Route.SelectedVersion),
                   last_forward_errno, attempt, forward_failure_action);
            if (last_forward_errno == ENODATA) {
              size_t source_bytes_read = 0;
              if (readAtRemoteRankWithBytes(file_id, context.Plan.AggregatorRank,
                                            offset, data, size,
                                            source_bytes_read)) {
                noteAppliedRebalancedReadForFile(file_id);
                io_log("Phase 1 dirty-owner clean-range planned read: "
                       "request_id=%llu file=%d aggregator_rank=%d "
                       "source_rank=%d version=%llu bytes=%zu attempt=%u\n",
                       static_cast<unsigned long long>(context.RequestId),
                       file_id, context.Plan.AggregatorRank, Route.SourceRank,
                       static_cast<unsigned long long>(Route.SelectedVersion),
                       source_bytes_read, attempt);
                return 0;
              }
              last_forward_errno = errno != 0 ? errno : EIO;
              break;
            }
            if (last_forward_errno != ESTALE)
              break;
            if (attempt < MaxDirtyOwnerForwardRetries)
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
          if (attempted_forward && !resolved_to_pfs) {
            errno = last_forward_errno != 0 ? last_forward_errno : EIO;
            return -1;
          }
        }
      } else if (pfs_current_after_writeback) {
        size_t source_bytes_read = 0;
        constexpr int ConservativeWritebackOwnerRank = 0;
        if (readAtRemoteRankWithBytes(file_id, ConservativeWritebackOwnerRank,
                                      offset, data, size, source_bytes_read)) {
          noteAppliedRebalancedReadForFile(file_id);
          io_log("Phase 1 planned read pfs-current source read: "
                 "request_id=%llu file=%d aggregator_rank=%d source_rank=%d "
                 "version=%llu bytes=%zu\n",
                 static_cast<unsigned long long>(context.RequestId), file_id,
                 context.Plan.AggregatorRank, ConservativeWritebackOwnerRank,
                 static_cast<unsigned long long>(
                     InitialFreshnessRoute.SelectedVersion),
                 source_bytes_read);
          return 0;
        }
        io_log("Phase 1 planned read pfs-current source read failed: "
               "request_id=%llu file=%d aggregator_rank=%d errno=%d; "
               "fail-closed in write-back dirty-owner mode.\n",
               static_cast<unsigned long long>(context.RequestId), file_id,
               context.Plan.AggregatorRank, errno);
        errno = errno != 0 ? errno : EIO;
        return -1;
      } else {
        io_log("Phase 1 planned read rebalanced route blocked: request_id=%llu "
               "file=%d aggregator_rank=%d reason=%s errno=%d; %s.\n",
               static_cast<unsigned long long>(context.RequestId), file_id,
               context.Plan.AggregatorRank, rebalance_reason, rebalance_errno,
               dirty_owner_forwarding_enabled
                   ? "fail-closed in write-back dirty-owner mode"
                   : "falling back to owner handle");
        if (dirty_owner_forwarding_enabled) {
          errno = rebalance_errno != 0 ? rebalance_errno : EIO;
          return -1;
        }
      }
    }

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
