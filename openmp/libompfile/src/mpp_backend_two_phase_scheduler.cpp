#include "mpp_backend.h"
#include "debug_log.h"
#include "mpp_shim.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

// The two-phase read on ompfile::batch::Batcher (LeaderPerBatch): Fixed or
// Adaptive window from LIBOMPFILE_OPT_TWO_PHASE_WINDOW_US / _ADAPTIVE, drain
// of the whole queue, execute = processTwoPhaseBatch. The trace hooks print
// the lines readAtTwoPhase always printed.
struct MppBackend::TwoPhaseBatchSite
    : ompfile::batch::SiteDefaults<MppBackend::TwoPhaseReadRequest> {
  MppBackend &Backend;
  explicit TwoPhaseBatchSite(MppBackend &B) : Backend(B) {}
  ompfile::batch::ClosePolicy closePolicy() {
    return {Backend.settings.two_phase_window_adaptive
                ? ompfile::batch::CloseRule::Adaptive
                : ompfile::batch::CloseRule::Fixed,
            Backend.settings.two_phase_window_us, 0};
  }
  void execute(std::vector<TwoPhaseReadRequest *> &batch) {
    Backend.processTwoPhaseBatch(batch);
  }
  void traceEnqueue(const TwoPhaseReadRequest &r, size_t queue_before,
                    bool in_progress) {
    io_trace(
        "MPIIOBackend::readAtTwoPhase enqueue req=%llu file=%d offset=%ld "
        "size=%zu queue_before=%zu in_progress=%d\n",
        static_cast<unsigned long long>(r.DebugRequestId), r.FileHandle,
        r.Offset, r.Size, queue_before, static_cast<int>(in_progress));
  }
  void traceLeader(const TwoPhaseReadRequest &r, size_t queue_size) {
    io_trace("MPIIOBackend::readAtTwoPhase leader req=%llu queue_size=%zu "
             "window_us=%llu\n",
             static_cast<unsigned long long>(r.DebugRequestId), queue_size,
             static_cast<unsigned long long>(
                 Backend.settings.two_phase_window_us));
  }
  void traceDrained(const TwoPhaseReadRequest &r, size_t batch_size,
                    size_t queue_after) {
    io_trace("MPIIOBackend::readAtTwoPhase leader req=%llu draining batch "
             "size=%zu queue_after_drain=%zu\n",
             static_cast<unsigned long long>(r.DebugRequestId), batch_size,
             queue_after);
  }
  void traceFollowerWait(const TwoPhaseReadRequest &r, size_t queue_size) {
    io_trace("MPIIOBackend::readAtTwoPhase follower wait req=%llu "
             "queue_size=%zu\n",
             static_cast<unsigned long long>(r.DebugRequestId), queue_size);
  }
};

int MppBackend::readAtTwoPhase(
    const ompfile::OmpFileReadRequestContext &context, void *data,
    size_t size) {
  // Keep zero-length reads on the validated fallback path so invalid handles
  // still fail consistently with the scalar pread flow.
  if (size == 0) {
    size_t bytes_read = 0;
    return readAtFallbackWithBytes(context.FileHandle,
                                   static_cast<long>(context.Offset), data,
                                   size, bytes_read);
  }

  const uint64_t request_key = resolveTwoPhaseKey(context);
  const long request_offset = static_cast<long>(context.Offset);

  TwoPhaseReadRequest request{};
  request.DebugRequestId =
      two_phase_request_id.fetch_add(1, std::memory_order_relaxed);
  request.FileHandle = context.FileHandle;
  request.ClientRank = context.ClientRank;
  request.Offset = request_offset;
  request.Size = size;
  request.Buffer = data;
  request.PathKey = request_key;
  request.Hint = context.Hint;

  // Queue, leader election, collection window and per-member completion
  // live in ompfile_batcher.h; TwoPhaseBatchSite binds this path to them.
  TwoPhaseBatchSite site(*this);
  const int64_t wait_us = two_phase_batcher.submit(request, site);
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

void MppBackend::processTwoPhaseBatch(
    std::vector<TwoPhaseReadRequest *> &batch) {
  if (batch.empty())
    return;

  io_trace("MPIIOBackend::processTwoPhaseBatch enter this=%p batch_size=%zu\n",
           static_cast<void *>(this), batch.size());
  counters.two_phase_batch_count.fetch_add(1, std::memory_order_relaxed);

  std::unordered_map<uint64_t, std::vector<TwoPhaseReadRequest *>> grouped;
  grouped.reserve(batch.size());
  for (TwoPhaseReadRequest *request : batch)
    grouped[getTwoPhaseGroupKey(*request)].push_back(request);

  io_trace("MPIIOBackend::processTwoPhaseBatch grouped=%zu\n", grouped.size());

  for (auto &entry : grouped)
    processTwoPhaseGroup(entry.second);
}

void MppBackend::processTwoPhaseGroup(
    std::vector<TwoPhaseReadRequest *> &group) {
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
  ompfile::batch::coalesceSortedRuns(
      group, settings.two_phase_max_batch_bytes, coalesced,
      [](const TwoPhaseReadRequest &r) { return r.Offset; },
      [](const TwoPhaseReadRequest &r) { return r.Size; },
      [this](TwoPhaseReadRequest &r, int errnum) {
        completeTwoPhaseRequest(r, -1, errnum);
      });
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
    segment.Size = static_cast<uint64_t>(computeTwoPhaseTargetReadSize(
        item.Start, item.End, item.Requests.size(), /*remote_only=*/true,
        settings.two_phase_sieve_bytes, settings.two_phase_max_batch_bytes));
    segment.PathKey = item.Requests.front()->PathKey;
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
  if (settings.sched_once_per_handle) {
    // No planner round trip: every segment continues on the opened-handle
    // owner path, which is what the client does with a non-rebalanced plan
    // anyway. Nothing to record; `planner_batches=` stays at 0 as evidence.
    io_trace("MPIIOBackend::processTwoPhaseGroup planner skipped batch_id=%llu "
             "(sched_once_per_handle)\n",
             static_cast<unsigned long long>(batch_request.BatchId));
  } else if (ompfile::mpp::schedBatchRequest(batch_request, batch_segments,
                                             batch_plan, batch_entries)) {
    counters.two_phase_planner_batch_count.fetch_add(1,
                                                     std::memory_order_relaxed);
    counters.two_phase_planner_segment_count.fetch_add(
        batch_entries.size(), std::memory_order_relaxed);
    if ((batch_plan.PlanFlags & ompfile::OMPFILE_BATCH_PLAN_SCALAR_FALLBACK) !=
        0) {
      counters.two_phase_planner_scalar_fallback_count.fetch_add(
          1, std::memory_order_relaxed);
    }

    if (batch_entries.size() != coalesced.size()) {
      counters.two_phase_planner_error_count.fetch_add(
          1, std::memory_order_relaxed);
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
          counters.two_phase_planner_affinity_count.fetch_add(
              1, std::memory_order_relaxed);
        if ((entry.PlanFlags & ompfile::OMPFILE_BATCH_PLAN_REBALANCED) != 0)
          counters.two_phase_planner_rebalanced_count.fetch_add(
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
    counters.two_phase_planner_error_count.fetch_add(1,
                                                     std::memory_order_relaxed);
    io_trace("MPIIOBackend::processTwoPhaseGroup planner failed batch_id=%llu\n",
             static_cast<unsigned long long>(batch_request.BatchId));
  }

  counters.two_phase_coalesced_read_count.fetch_add(coalesced.size(),
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

    const size_t read_size = computeTwoPhaseTargetReadSize(
        item.Start, item.End, item.Requests.size(), /*remote_only=*/true,
        settings.two_phase_sieve_bytes, settings.two_phase_max_batch_bytes);
    counters.two_phase_coalesced_bytes_total.fetch_add(
        read_size, std::memory_order_relaxed);
    std::vector<char> buffer(read_size);
    size_t bytes_read = 0;

    int rc = -1;
    bool remote_rebalanced_ok = false;
    int read_errno = 0;
    const int file_id = item.Requests.front()->FileHandle;
    if (item.PlannerRebalanced && item.PlannedAggregatorRank >= 0) {
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
        counters.two_phase_planner_rebalanced_conflict_count.fetch_add(
            1, std::memory_order_relaxed);
        counters.two_phase_planner_rebalanced_skipped_count.fetch_add(
            1, std::memory_order_relaxed);
        io_log("Rebalanced segment fallback file=%d target_rank=%d offset=%ld "
               "size=%zu reason=%s errno=%d\n",
               file_id, item.PlannedAggregatorRank, item.Start, read_size,
               fallback_reason, read_errno);
      }

      if (remote_rebalanced_ok) {
        counters.two_phase_planner_rebalanced_applied_count.fetch_add(
            1, std::memory_order_relaxed);
        noteAppliedRebalancedReadForFile(file_id);
      } else if (eligible_for_rebalance) {
        read_errno = errno != 0 ? errno : EIO;
        counters.two_phase_planner_rebalanced_skipped_count.fetch_add(
            1, std::memory_order_relaxed);
        io_log("Rebalanced segment fallback file=%d target_rank=%d offset=%ld "
               "size=%zu reason=remote-read-fail errno=%d\n",
               file_id, item.PlannedAggregatorRank, item.Start, read_size,
               read_errno);
      }
    }

    if (rc != 0) {
      rc = readAtFallbackWithBytes(file_id, item.Start, buffer.data(), read_size,
                                   bytes_read);
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

uint64_t MppBackend::getTwoPhaseGroupKey(
    const TwoPhaseReadRequest &request) const {
  return request.PathKey;
}

void MppBackend::completeTwoPhaseRequest(TwoPhaseReadRequest &request,
                                         int status, int errnum) {
  two_phase_batcher.complete(request, status, errnum, [&] {
    io_trace(
        "MPIIOBackend::completeTwoPhaseRequest req=%llu file=%d offset=%ld "
        "size=%zu status=%d errno=%d\n",
        static_cast<unsigned long long>(request.DebugRequestId),
        request.FileHandle, request.Offset, request.Size, status, errnum);
  });
}

bool MppBackend::isTwoPhaseActive() const {
  return settings.two_phase_enabled;
}
