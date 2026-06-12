#include "mpi_io_backend.h"
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
  if (!mpp_remote_only &&
      tryServeTwoPhaseReadCache(request_key, request_offset, data, size)) {
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
  request.Hint = context.Hint;

  const auto enqueue_ts = std::chrono::steady_clock::now();
  std::vector<TwoPhaseReadRequest *> batch;
  std::unique_lock<std::mutex> lock(two_phase_mutex);
  const size_t queue_size_before = two_phase_queue.size();
  updateAtomicMax(two_phase_queue_max_depth,
                  static_cast<uint64_t>(queue_size_before + 1));
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
      two_phase_leader_turn_count.fetch_add(1, std::memory_order_relaxed);
      io_trace("MPIIOBackend::readAtTwoPhase leader req=%llu queue_size=%zu "
               "window_us=%llu\n",
               static_cast<unsigned long long>(request.DebugRequestId),
               two_phase_queue.size(),
               static_cast<unsigned long long>(two_phase_window_us));

      if (two_phase_window_us > 0) {
        const auto window_wait_begin = std::chrono::steady_clock::now();
        two_phase_queue_cv.wait_for(
            lock, std::chrono::microseconds(two_phase_window_us));
        const auto window_wait_end = std::chrono::steady_clock::now();
        const auto window_wait_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                window_wait_end - window_wait_begin)
                .count();
        if (window_wait_us > 0) {
          two_phase_window_wait_us_total.fetch_add(
              static_cast<uint64_t>(window_wait_us),
              std::memory_order_relaxed);
        }
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
      const auto batch_exec_begin = std::chrono::steady_clock::now();
      processTwoPhaseBatch(batch);
      const auto batch_exec_end = std::chrono::steady_clock::now();
      const auto batch_exec_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              batch_exec_end - batch_exec_begin)
              .count();
      if (batch_exec_us > 0) {
        two_phase_batch_exec_us_total.fetch_add(
            static_cast<uint64_t>(batch_exec_us), std::memory_order_relaxed);
      }
      batch.clear();
      lock.lock();

      two_phase_batch_in_progress = false;
      two_phase_queue_cv.notify_all();
    } else {
      two_phase_follower_wait_count.fetch_add(1, std::memory_order_relaxed);
      io_trace("MPIIOBackend::readAtTwoPhase follower wait req=%llu "
               "queue_size=%zu\n",
               static_cast<unsigned long long>(request.DebugRequestId),
               two_phase_queue.size());
      const auto follower_wait_begin = std::chrono::steady_clock::now();
      two_phase_queue_cv.wait(lock, [this, &request] {
        return request.Done || !two_phase_batch_in_progress;
      });
      const auto follower_wait_end = std::chrono::steady_clock::now();
      const auto follower_wait_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              follower_wait_end - follower_wait_begin)
              .count();
      if (follower_wait_us > 0) {
        two_phase_follower_wait_us_total.fetch_add(
            static_cast<uint64_t>(follower_wait_us),
            std::memory_order_relaxed);
      }
    }
  }

  const auto done_ts = std::chrono::steady_clock::now();
  const auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           done_ts - enqueue_ts)
                           .count();
  if (wait_us > 0) {
    two_phase_request_wait_us_total.fetch_add(static_cast<uint64_t>(wait_us),
                                              std::memory_order_relaxed);
  }
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

void MPIIOBackend::processTwoPhaseBatch(
    std::vector<TwoPhaseReadRequest *> &batch) {
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

void MPIIOBackend::processTwoPhaseGroup(
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
    int PlannedAggregatorRank = -1;
    bool PlannerRebalanced = false;
    bool PlannerForceFallback = false;
    int PlannerFallbackErrno = 0;
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
    segment.Size = static_cast<uint64_t>(computeForecastTargetReadSize(
        item.Requests.front()->Hint, item.Start, item.End,
        item.Requests.size(), mpp_remote_only, two_phase_sieve_bytes,
        two_phase_max_batch_bytes));
    segment.PathKey = item.Requests.front()->HasPathKey
                          ? item.Requests.front()->PathKey
                          : static_cast<uint64_t>(0);
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

    const size_t read_size = computeForecastTargetReadSize(
        item.Requests.front()->Hint, item.Start, item.End,
        item.Requests.size(), mpp_remote_only, two_phase_sieve_bytes,
        two_phase_max_batch_bytes);
    two_phase_coalesced_bytes_total.fetch_add(read_size,
                                              std::memory_order_relaxed);
    std::vector<char> buffer(read_size);
    size_t bytes_read = 0;

    int rc = -1;
    bool remote_rebalanced_ok = false;
    int read_errno = 0;
    const int file_id = item.Requests.front()->FileHandle;
    int source_affinity_rank = -1;
    bool force_no_stage_fallback = false;
    const bool source_affinity_candidate = isSourceStageAffinityEligible(
        item.Requests.front()->Hint, item.Requests.front()->HasPathKey,
        mpp_remote_only);
    if (source_affinity_candidate) {
      stage_affinity_source_candidate_count.fetch_add(
          1, std::memory_order_relaxed);
      stage_affinity_source_candidate_bytes.fetch_add(
          read_size, std::memory_order_relaxed);
    }
    const bool source_affinity_safe = canAttemptSourceStageAffinityRead(
        item.Requests.front()->Hint, file_id, item.Start, read_size,
        item.Requests.front()->HasPathKey, mpp_remote_only,
        item.PlannerForceFallback);
    if (source_affinity_safe) {
      if (sourceStageAffinityPreReadInvalidateForFile(file_id)) {
        bool ok = readAtSourceStageAffinityRankWithBytes(
            file_id, item.Requests.front()->PathKey,
            item.Requests.front()->Hint, item.Start, buffer.data(), read_size,
            bytes_read, source_affinity_rank);
        if (ok) {
          rc = 0;
          stage_affinity_source_applied_count.fetch_add(
              1, std::memory_order_relaxed);
          stage_affinity_source_applied_bytes.fetch_add(
              read_size, std::memory_order_relaxed);
        }
      } else {
        force_no_stage_fallback = true;
      }
    }
    if (source_affinity_candidate && rc != 0) {
      stage_affinity_source_fallback_count.fetch_add(
          1, std::memory_order_relaxed);
    }

    if (shouldAttemptPlannerRebalancedRead(
            rc, force_no_stage_fallback, item.PlannerRebalanced,
            item.PlannedAggregatorRank)) {
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
        two_phase_planner_rebalanced_conflict_count.fetch_add(
            1, std::memory_order_relaxed);
        two_phase_planner_rebalanced_skipped_count.fetch_add(
            1, std::memory_order_relaxed);
        io_log("Rebalanced segment fallback file=%d target_rank=%d offset=%ld "
               "size=%zu reason=%s errno=%d\n",
               file_id, item.PlannedAggregatorRank, item.Start, read_size,
               fallback_reason, read_errno);
      }

      if (remote_rebalanced_ok) {
        two_phase_planner_rebalanced_applied_count.fetch_add(
            1, std::memory_order_relaxed);
        noteAppliedRebalancedReadForFile(file_id);
      } else if (eligible_for_rebalance) {
        read_errno = errno != 0 ? errno : EIO;
        two_phase_planner_rebalanced_skipped_count.fetch_add(
            1, std::memory_order_relaxed);
        io_log("Rebalanced segment fallback file=%d target_rank=%d offset=%ld "
               "size=%zu reason=remote-read-fail errno=%d\n",
               file_id, item.PlannedAggregatorRank, item.Start, read_size,
               read_errno);
      }
    }

    if (rc != 0) {
      rc = readAtFallbackWithBytes(file_id, item.Start, buffer.data(), read_size,
                                   bytes_read, &item.Requests.front()->Hint,
                                   force_no_stage_fallback);
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

    if (!mpp_remote_only) {
      const uint64_t cache_key = getTwoPhaseGroupKey(*item.Requests.front());
      updateTwoPhaseReadCache(cache_key, item.Start, buffer);
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

uint64_t MPIIOBackend::getTwoPhaseGroupKey(
    const TwoPhaseReadRequest &request) const {
  if (request.HasPathKey)
    return request.PathKey;
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

bool MPIIOBackend::isTwoPhaseActive() const {
  return two_phase_enabled && mpp_open_enabled && mpp_io_enabled;
}
