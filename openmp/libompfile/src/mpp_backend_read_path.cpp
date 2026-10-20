#include "debug_log.h"
#include "ompfile_config.h"
#include "mpp_backend.h"
#include "mpp_shim.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <chrono>
#include <mutex>
#include <optional>
#include <thread>

namespace {

bool scalarPlannedReadRebalanceEnabledForStage(
    const ompfile::config::ClientConfig &C) {
  return C.Shared.stageReadthrough() && C.Shared.stageWriteBack() &&
         C.Shared.StageFreshnessGuard;
}

bool dirtyOwnerBatchEnabled(const ompfile::config::ClientConfig &C) {
  return C.DirtyOwnerBatch;
}

unsigned dirtyOwnerBatchMaxSegments(const ompfile::config::ClientConfig &C) {
  return C.DirtyOwnerBatchMaxSegs;
}

unsigned dirtyOwnerBatchWindowUs(const ompfile::config::ClientConfig &C) {
  return C.DirtyOwnerBatchWindowUs;
}

ompfile::route::FreshnessView queryFreshnessForRank(uint64_t path_key,
                                                    int target_rank) {
  ompfile::route::FreshnessView Result{};
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

// The dirty-owner pread batch on ompfile::batch::Batcher (LeaderUntilEmpty):
// a Fixed window of LIBOMPFILE_OPT_DIRTY_OWNER_BATCH_WINDOW_US before a
// tenure's first drain only; each drain takes the queue front plus the later
// members with the same file, remote handle, source rank and expected
// version, LIBOMPFILE_OPT_DIRTY_OWNER_BATCH_MAX_SEGS at most.
struct MppBackend::DirtyOwnerBatchSite
    : ompfile::batch::SiteDefaults<MppBackend::DirtyOwnerBatchRequest> {
  MppBackend &Backend;
  explicit DirtyOwnerBatchSite(MppBackend &B) : Backend(B) {}
  ompfile::batch::ClosePolicy closePolicy() {
    return {ompfile::batch::CloseRule::Fixed,
            dirtyOwnerBatchWindowUs(Backend.config_), 0};
  }
  size_t maxMembers() { return dirtyOwnerBatchMaxSegments(Backend.config_); }
  bool sameBatch(const DirtyOwnerBatchRequest &Seed,
                 const DirtyOwnerBatchRequest &Candidate) {
    return Candidate.FileId == Seed.FileId &&
           Candidate.RemoteHandle == Seed.RemoteHandle &&
           Candidate.SourceRank == Seed.SourceRank &&
           Candidate.ExpectedVersion == Seed.ExpectedVersion;
  }
  void execute(std::vector<DirtyOwnerBatchRequest *> &Batch) {
    Backend.executeDirtyOwnerBatch(Batch);
  }
};

void MppBackend::executeDirtyOwnerBatch(
    std::vector<DirtyOwnerBatchRequest *> &Batch) {
  bool BatchOk = false;
  if (Batch.size() > 1) {
    std::vector<ompfile::mpp::DirtyOwnerPreadBatchSegment> Segments(
        Batch.size());
    size_t TotalBytes = 0;
    for (size_t I = 0; I < Batch.size(); ++I) {
      Segments[I].offset = Batch[I]->Offset;
      Segments[I].size = Batch[I]->Size;
      Segments[I].expected_version = Batch[I]->ExpectedVersion;
      Segments[I].buffer = Batch[I]->Buffer;
      TotalBytes += Batch[I]->Size;
    }
    {
      const auto lock = instrumentedMppCallLock();
      errno = 0;
      BatchOk = ompfile::mpp::dirtyOwnerPreadBatchEx(
          Batch.front()->RemoteHandle, Batch.front()->SourceRank,
          Segments.data(), Segments.size());
    }
    if (BatchOk) {
      counters.dirty_owner_batch_count.fetch_add(1, std::memory_order_relaxed);
      counters.dirty_owner_remote_read_event_count.fetch_add(
          1, std::memory_order_relaxed);
      counters.dirty_owner_remote_read_bytes_total.fetch_add(
          TotalBytes, std::memory_order_relaxed);
      counters.dirty_owner_batch_segment_count.fetch_add(
          Batch.size(), std::memory_order_relaxed);
      counters.dirty_owner_batch_saved_events.fetch_add(
          Batch.size() - 1, std::memory_order_relaxed);
      counters.dirty_owner_batch_bytes.fetch_add(TotalBytes,
                                                 std::memory_order_relaxed);
      for (size_t I = 0; I < Batch.size(); ++I) {
        Batch[I]->BytesRead = Segments[I].bytes_read;
        Batch[I]->Status = 0;
        Batch[I]->Errno = 0;
      }
      io_log("Phase 1 dirty-worker batch staged read: request_id=%llu "
             "file=%d source_rank=%d version=%llu segments=%zu bytes=%zu\n",
             static_cast<unsigned long long>(Batch.front()->DebugRequestId),
             Batch.front()->FileId, Batch.front()->SourceRank,
             static_cast<unsigned long long>(Batch.front()->ExpectedVersion),
             Batch.size(), TotalBytes);
    } else {
      counters.dirty_owner_batch_failure_count.fetch_add(
          1, std::memory_order_relaxed);
    }
  }

  if (Batch.size() <= 1 || !BatchOk) {
    for (DirtyOwnerBatchRequest *Entry : Batch) {
      size_t ScalarBytes = 0;
      bool ScalarOk = false;
      int ScalarErrno = 0;
      {
        const auto lock = instrumentedMppCallLock();
        errno = 0;
        ScalarOk = ompfile::mpp::dirtyOwnerPreadEx(
            Entry->RemoteHandle, Entry->SourceRank, Entry->ExpectedVersion,
            Entry->Offset, Entry->Buffer, Entry->Size, ScalarBytes);
        ScalarErrno = errno;
      }
      if (ScalarOk) {
        counters.dirty_owner_single_read_count.fetch_add(
            1, std::memory_order_relaxed);
        counters.dirty_owner_remote_read_event_count.fetch_add(
            1, std::memory_order_relaxed);
        counters.dirty_owner_remote_read_bytes_total.fetch_add(
            ScalarBytes, std::memory_order_relaxed);
      }
      Entry->BytesRead = ScalarBytes;
      Entry->Status = ScalarOk ? 0 : -1;
      Entry->Errno = ScalarOk ? 0 : (ScalarErrno != 0 ? ScalarErrno : EIO);
    }
  }
}

bool MppBackend::dirtyOwnerPreadMaybeBatch(
    int file_id, int remote_handle, int source_rank, uint64_t expected_version,
    long offset, void *data, size_t size, size_t &bytes_read, int &read_errno,
    uint64_t debug_request_id) {
  bytes_read = 0;
  read_errno = 0;
  if (!dirtyOwnerBatchEnabled(config_)) {
    const auto lock = instrumentedMppCallLock();
    errno = 0;
    const bool Ok = ompfile::mpp::dirtyOwnerPreadEx(
        remote_handle, source_rank, expected_version, offset, data, size,
        bytes_read);
    read_errno = errno;
    return Ok;
  }

  counters.dirty_owner_batch_candidate_count.fetch_add(
      1, std::memory_order_relaxed);
  DirtyOwnerBatchRequest Request{};
  Request.FileId = file_id;
  Request.RemoteHandle = remote_handle;
  Request.SourceRank = source_rank;
  Request.ExpectedVersion = expected_version;
  Request.Offset = offset;
  Request.Size = size;
  Request.Buffer = data;
  Request.DebugRequestId = debug_request_id;

  DirtyOwnerBatchSite Site(*this);
  (void)dirty_owner_batcher.submit(Request, Site);
  bytes_read = Request.BytesRead;
  read_errno = Request.Errno;
  errno = Request.Errno;
  return Request.Status == 0;
}

int MppBackend::readAt(int file_id, long offset, void *data, size_t size) {
  ompfile::OmpFileReadRequestContext context{};
  context.FileHandle = file_id;
  context.Offset = static_cast<int64_t>(offset);
  context.Size = static_cast<uint64_t>(size);
  return readAtWithContext(context, data, size);
}

ompfile::route::ReadRouteConfig MppBackend::snapshotReadRouteConfig() const {
  ompfile::route::ReadRouteConfig route_config{};
  route_config.TwoPhaseActive = isTwoPhaseActive();
  route_config.StageRebalance =
      scalarPlannedReadRebalanceEnabledForStage(config_);
  route_config.ScalarOracle =
      route_config.StageRebalance && config_.DirtyOwnerForwarding.Value;
  route_config.DirtyOwnerForwarding = settings.dirty_owner_forwarding_enabled;
  return route_config;
}

int MppBackend::readAtWithContext(
    const ompfile::OmpFileReadRequestContext &context, void *data,
    size_t size) {
  if (settings.strict_mpp_init_failed)
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
  counters.pread_request_count.fetch_add(1, std::memory_order_relaxed);
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

  const ompfile::route::ReadRouteConfig route_config =
      snapshotReadRouteConfig();
  ompfile::route::ReadDecision decision =
      ompfile::route::resolveReadRoute(context, route_config);
  if (decision.Route == ompfile::route::ReadRoute::TwoPhase)
    return readAtTwoPhase(context, data, size);
  if (decision.LogTwoPhaseBypass) {
    io_log("Two-phase read bypassed for write-back coherence oracle: "
           "request_id=%llu file=%d offset=%lld size=%zu\n",
           static_cast<unsigned long long>(context.RequestId), file_id,
           static_cast<long long>(offset), size);
  }

  if (decision.Route == ompfile::route::ReadRoute::Coherent) {
    if (const std::optional<int> rc =
            readCoherent(context, route_config, data, size))
      return *rc;
    decision.Route = ompfile::route::ReadRoute::PlannedOwner;
  }
  if (decision.Route == ompfile::route::ReadRoute::PlannedOwner) {
    if (const std::optional<int> rc = readPlannedOwner(context, data, size))
      return *rc;
  }
  return readFallbackTail(context, data, size);
}

ompfile::route::CoherentObservation MppBackend::observeCoherentRead(
    const ompfile::OmpFileReadRequestContext &context) {
  ompfile::route::CoherentObservation obs{};
  obs.FreshnessKey =
      ompfile::keys::tileKeyForHint(context.PathKey, context.Hint);
  obs.Initial =
      queryFreshnessForRank(obs.FreshnessKey, context.Plan.AggregatorRank);
  {
    const auto lock = instrumentedHandleLock();
    auto write_it =
        write_epochs.file_write_epoch_history.find(context.FileHandle);
    obs.HasLocalWriteHistory =
        write_it != write_epochs.file_write_epoch_history.end() &&
        !write_it->second.empty();
  }
  return obs;
}

ompfile::route::RebalanceVerdict MppBackend::observeRebalanceVerdict(
    const ompfile::OmpFileReadRequestContext &context, size_t size) {
  ompfile::route::RebalanceVerdict verdict{};
  verdict.MetadataSafe = canApplyRebalancedRead(
      context.Hint, context.FileHandle, static_cast<long>(context.Offset),
      size, verdict.Reason, verdict.ReasonErrno);
  return verdict;
}

std::optional<int> MppBackend::readCoherent(
    const ompfile::OmpFileReadRequestContext &context,
    const ompfile::route::ReadRouteConfig &route_config, void *data,
    size_t size) {
  ompfile::route::ReadRouteState state{};
  state.Coherent = observeCoherentRead(context);
  ompfile::route::ReadDecision decision =
      ompfile::route::resolveCoherentReadRoute(
          state.Coherent, context.Plan.AggregatorRank,
          route_config.DirtyOwnerForwarding);
  if (decision.Route == ompfile::route::ReadRoute::WritebackOwner) {
    if (const std::optional<int> rc =
            readWritebackOracle(context, state, decision, data, size))
      return rc;
  } else if (decision.Route == ompfile::route::ReadRoute::RebalanceCheck) {
    state.Rebalance = observeRebalanceVerdict(context, size);
    decision = ompfile::route::resolveRebalanceRoute(
        state.Coherent, state.Rebalance, context.Plan.AggregatorRank,
        route_config.DirtyOwnerForwarding);
  }
  return executeCoherentRoute(context, state, decision, data, size);
}

std::optional<int> MppBackend::readWritebackOracle(
    const ompfile::OmpFileReadRequestContext &context,
    ompfile::route::ReadRouteState &state,
    ompfile::route::ReadDecision &decision, void *data, size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  const ompfile::route::FreshnessView &InitialFreshnessRoute =
      state.Coherent.Initial;
  const bool has_local_write_history = state.Coherent.HasLocalWriteHistory;
  const bool file_has_writeback_epoch =
      ompfile::route::fileHasWritebackEpoch(state.Coherent);
  const ompfile::route::DistributeConfig distribute =
      snapshotDistributeConfig();
  ompfile::route::DirtySourcePlan source = ompfile::route::resolveDirtySource(
      state.Coherent, context.Hint, distribute);
  if (source.NeedsHistoryScan) {
    const long read_end = ompfile::route::readEndOrOffset(offset, size);
    // The rank count comes from the distribute snapshot taken above; it is
    // read once, outside handle_mutex.
    const unsigned distribute_ranks = distribute.Ranks;
    int range_owner_rank = -1;
    {
      const auto lock = instrumentedHandleLock();
      auto write_it = write_epochs.file_write_epoch_history.find(file_id);
      if (distribute_ranks > 1 &&
          write_it != write_epochs.file_write_epoch_history.end() &&
          read_end > offset)
        range_owner_rank = ompfile::route::latestOverlappingWriteOwnerRank(
            write_it->second, offset, read_end, distribute_ranks);
    }
    ompfile::route::applyHistoryScan(source, range_owner_rank);
  }
  const int DirtySourceRank = source.SourceRank;
  const uint64_t DirtyExpectedVersion = source.ExpectedVersion;
  // Dirty-owner forwarding is independent from the optional clean-PFS
  // rebalance policy. Route cross-rank dirty data through the owner;
  // version and extent checks in the query/read protocol stay fail-closed.
  if (context.Plan.AggregatorRank != DirtySourceRank) {
    io_log("Phase 1 dirty-worker oracle candidate: request_id=%llu "
           "file=%d aggregator_rank=%d source_rank=%d version=%llu "
           "local_write_history=%d file_writeback_epoch=%d\n",
           static_cast<unsigned long long>(context.RequestId), file_id,
           context.Plan.AggregatorRank, DirtySourceRank,
           static_cast<unsigned long long>(
               InitialFreshnessRoute.SelectedVersion),
           static_cast<int>(has_local_write_history),
           static_cast<int>(file_has_writeback_epoch));
    int remote_handle = -1;
    if (getOrCreateRemoteReadHandleForRank(file_id, DirtySourceRank,
                                           remote_handle) &&
        remote_handle >= 0) {
      counters.dirty_owner_forward_attempt_count.fetch_add(
          1, std::memory_order_relaxed);
      int oracle_state = -1;
      int oracle_errno = 0;
      bool oracle_ok = false;
      {
        const auto lock = instrumentedMppCallLock();
        errno = 0;
        oracle_ok = ompfile::mpp::dirtyOwnerQueryEx(
            remote_handle, DirtySourceRank, DirtyExpectedVersion, offset,
            size, oracle_state);
        oracle_errno = errno;
      }
      const ompfile::route::DirtyOracleOutcome outcome =
          ompfile::route::classifyDirtyOracle(oracle_ok, oracle_state);
      if (outcome == ompfile::route::DirtyOracleOutcome::Dirty) {
        if (const std::optional<int> rc = readWritebackDirtyForward(
                context, state, ompfile::route::dirtyForwardDecision(source),
                remote_handle, data, size))
          return rc;
      } else if (outcome == ompfile::route::DirtyOracleOutcome::Clean) {
        state.DirtySourceKnownClean = true;
        counters.dirty_owner_forward_failure_count.fetch_add(
            1, std::memory_order_relaxed);
        io_log("Phase 1 dirty-clean pfs rebalanced disabled: "
               "request_id=%llu file=%d aggregator_rank=%d "
               "source_rank=%d offset=%lld size=%zu tile=%llu "
               "version=%llu; falling back for clean-oracle range.\n",
               static_cast<unsigned long long>(context.RequestId),
               file_id, context.Plan.AggregatorRank, DirtySourceRank,
               static_cast<long long>(offset), size,
               static_cast<unsigned long long>(context.Hint.TileId),
               static_cast<unsigned long long>(
                   InitialFreshnessRoute.SelectedVersion));
      } else {
        state.DirtySourceUnknown = true;
        counters.dirty_owner_forward_failure_count.fetch_add(
            1, std::memory_order_relaxed);
        io_log("Phase 1 dirty-worker oracle unavailable: request_id=%llu "
               "file=%d aggregator_rank=%d source_rank=%d offset=%lld "
               "size=%zu tile=%llu version=%llu state=%d errno=%d; "
               "falling back.\n",
               static_cast<unsigned long long>(context.RequestId),
               file_id, context.Plan.AggregatorRank, DirtySourceRank,
               static_cast<long long>(offset), size,
               static_cast<unsigned long long>(context.Hint.TileId),
               static_cast<unsigned long long>(
                   InitialFreshnessRoute.SelectedVersion),
               oracle_state, oracle_errno);
      }
    } else {
      state.DirtySourceUnknown = true;
      io_log("Phase 1 dirty-worker oracle skipped: request_id=%llu "
             "file=%d aggregator_rank=%d source_rank=%d reason=no-handle\n",
             static_cast<unsigned long long>(context.RequestId), file_id,
             context.Plan.AggregatorRank, DirtySourceRank);
    }
  }
  decision = ompfile::route::resolveWritebackOwnerTail(
      source, context.Plan.AggregatorRank, distribute,
      state.DirtySourceKnownClean, state.DirtySourceUnknown);
  return std::nullopt;
}

std::optional<int> MppBackend::readWritebackDirtyForward(
    const ompfile::OmpFileReadRequestContext &context,
    ompfile::route::ReadRouteState &state,
    const ompfile::route::ReadDecision &decision, int remote_handle,
    void *data, size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  const ompfile::route::FreshnessView &InitialFreshnessRoute =
      state.Coherent.Initial;
  const int DirtySourceRank = decision.SourceRank;
  const uint64_t DirtyExpectedVersion = decision.ExpectedVersion;
  size_t dirty_source_bytes_read = 0;
  int dirty_source_read_errno = 0;
  const bool dirty_source_read_ok = dirtyOwnerPreadMaybeBatch(
      file_id, remote_handle, DirtySourceRank,
      DirtyExpectedVersion, offset, data, size,
      dirty_source_bytes_read, dirty_source_read_errno,
      context.RequestId);
  if (dirty_source_read_ok) {
    counters.dirty_owner_forward_success_count.fetch_add(
        1, std::memory_order_relaxed);
    counters.dirty_owner_forward_bytes_total.fetch_add(
        dirty_source_bytes_read, std::memory_order_relaxed);
    counters.remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
    counters.remote_pread_bytes_total.fetch_add(size,
                                                std::memory_order_relaxed);
    noteAppliedRebalancedReadForFile(file_id);
    io_log("Phase 1 dirty-worker staged read: request_id=%llu "
           "file=%d aggregator_rank=%d source_rank=%d "
           "version=%llu bytes=%zu\n",
           static_cast<unsigned long long>(context.RequestId),
           file_id, context.Plan.AggregatorRank, DirtySourceRank,
           static_cast<unsigned long long>(
               InitialFreshnessRoute.SelectedVersion),
           dirty_source_bytes_read);
    return 0;
  }
  state.DirtySourceUnknown = true;
  errno = dirty_source_read_errno != 0 ? dirty_source_read_errno
                                       : EIO;
  io_log("Phase 1 dirty-worker staged read failed: request_id=%llu "
         "file=%d aggregator_rank=%d source_rank=%d offset=%lld "
         "size=%zu tile=%llu errno=%d; falling back.\n",
         static_cast<unsigned long long>(context.RequestId), file_id,
         context.Plan.AggregatorRank, DirtySourceRank,
         static_cast<long long>(offset), size,
         static_cast<unsigned long long>(context.Hint.TileId),
         errno);
  return std::nullopt;
}

std::optional<int> MppBackend::executeCoherentRoute(
    const ompfile::OmpFileReadRequestContext &context,
    const ompfile::route::ReadRouteState &state,
    const ompfile::route::ReadDecision &decision, void *data, size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  const ompfile::route::FreshnessView &InitialFreshnessRoute =
      state.Coherent.Initial;
  const bool has_local_write_history = state.Coherent.HasLocalWriteHistory;
  const bool file_has_writeback_epoch =
      ompfile::route::fileHasWritebackEpoch(state.Coherent);
  const int DirtySourceRank = decision.SourceRank;
  const uint64_t DirtyExpectedVersion = decision.ExpectedVersion;
  const bool dirty_source_known_clean = state.DirtySourceKnownClean;
  const bool dirty_source_unknown = state.DirtySourceUnknown;
  const char *rebalance_reason = state.Rebalance.Reason;
  const int rebalance_errno = state.Rebalance.ReasonErrno;
  switch (decision.Route) {
  case ompfile::route::ReadRoute::GlobalOraclePfs: {
    size_t pfs_bytes_read = 0;
    if (readAtRemoteRankNoStageWithBytes(file_id,
                                         decision.TargetRank,
                                         offset, data, size,
                                         pfs_bytes_read)) {
      noteAppliedRebalancedReadForFile(file_id);
      io_log("Phase 1 global-oracle pfs read: request_id=%llu file=%d "
             "aggregator_rank=%d offset=%lld size=%zu tile=%llu "
             "version=%llu bytes=%zu\n",
             static_cast<unsigned long long>(context.RequestId), file_id,
             context.Plan.AggregatorRank, static_cast<long long>(offset),
             size, static_cast<unsigned long long>(context.Hint.TileId),
             static_cast<unsigned long long>(
                 InitialFreshnessRoute.SelectedVersion),
             pfs_bytes_read);
      return 0;
    }
    io_log("Phase 1 global-oracle pfs read failed: request_id=%llu "
           "file=%d aggregator_rank=%d errno=%d; fail-closed.\n",
           static_cast<unsigned long long>(context.RequestId), file_id,
           context.Plan.AggregatorRank, errno);
    errno = errno != 0 ? errno : EIO;
    return -1;
  }
  case ompfile::route::ReadRoute::WritebackLocalDirty: {
    int local_remote_handle = -1;
    if (!getOrCreateRemoteReadHandleForRank(file_id, DirtySourceRank,
                                            local_remote_handle) ||
        local_remote_handle < 0) {
      io_log("Phase 1 dirty-worker local staged read missing handle: "
             "request_id=%llu file=%d aggregator_rank=%d source_rank=%d; "
             "fail-closed.\n",
             static_cast<unsigned long long>(context.RequestId), file_id,
             context.Plan.AggregatorRank, DirtySourceRank);
      errno = ENOKEY;
      return -1;
    }
    size_t local_dirty_bytes_read = 0;
    bool local_dirty_read_ok = false;
    int local_dirty_read_errno = 0;
    {
      const auto lock = instrumentedMppCallLock();
      errno = 0;
      local_dirty_read_ok = ompfile::mpp::dirtyOwnerPreadEx(
          local_remote_handle, DirtySourceRank, DirtyExpectedVersion,
          offset, data, size, local_dirty_bytes_read);
      local_dirty_read_errno = errno;
    }
    if (local_dirty_read_ok) {
      counters.dirty_owner_forward_success_count.fetch_add(
          1, std::memory_order_relaxed);
      counters.dirty_owner_forward_bytes_total.fetch_add(
          local_dirty_bytes_read, std::memory_order_relaxed);
      counters.remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
      counters.remote_pread_bytes_total.fetch_add(size,
                                                  std::memory_order_relaxed);
      noteAppliedRebalancedReadForFile(file_id);
      io_log("Phase 1 dirty-worker local staged read: request_id=%llu "
             "file=%d aggregator_rank=%d source_rank=%d version=%llu "
             "bytes=%zu\n",
             static_cast<unsigned long long>(context.RequestId), file_id,
             context.Plan.AggregatorRank, DirtySourceRank,
             static_cast<unsigned long long>(
                 InitialFreshnessRoute.SelectedVersion),
             local_dirty_bytes_read);
      return 0;
    }
    errno = local_dirty_read_errno != 0 ? local_dirty_read_errno : EIO;
    io_log("Phase 1 dirty-worker local staged read failed: "
           "request_id=%llu file=%d aggregator_rank=%d source_rank=%d "
           "version=%llu errno=%d; fail-closed.\n",
           static_cast<unsigned long long>(context.RequestId), file_id,
           context.Plan.AggregatorRank, DirtySourceRank,
           static_cast<unsigned long long>(DirtyExpectedVersion), errno);
    return -1;
  }
  case ompfile::route::ReadRoute::WritebackRefuseSourceClean: {
    io_log("Phase 1 dirty-worker source-clean pfs disabled: "
           "request_id=%llu file=%d aggregator_rank=%d source_rank=%d "
           "offset=%lld size=%zu tile=%llu version=%llu; fail-closed.\n",
           static_cast<unsigned long long>(context.RequestId), file_id,
           context.Plan.AggregatorRank, DirtySourceRank,
           static_cast<long long>(offset), size,
           static_cast<unsigned long long>(context.Hint.TileId),
           static_cast<unsigned long long>(
               InitialFreshnessRoute.SelectedVersion));
    errno = ESTALE;
    return -1;
  }
  case ompfile::route::ReadRoute::WritebackRefuseDistributed: {
    io_log("Phase 1 dirty-worker distributed fallback refused: "
           "request_id=%llu file=%d aggregator_rank=%d source_rank=%d "
           "offset=%lld size=%zu tile=%llu version=%llu clean=%d "
           "unknown=%d; fail-closed.\n",
           static_cast<unsigned long long>(context.RequestId), file_id,
           context.Plan.AggregatorRank, DirtySourceRank,
           static_cast<long long>(offset), size,
           static_cast<unsigned long long>(context.Hint.TileId),
           static_cast<unsigned long long>(
               InitialFreshnessRoute.SelectedVersion),
           static_cast<int>(dirty_source_known_clean),
           static_cast<int>(dirty_source_unknown));
    errno = ESTALE;
    return -1;
  }
  case ompfile::route::ReadRoute::WritebackConservativeOwner: {
    const int ConservativeReadSourceRank = decision.TargetRank;
    size_t source_bytes_read = 0;
    if (readAtRemoteRankWithBytes(file_id, ConservativeReadSourceRank,
                                  offset, data, size, source_bytes_read)) {
      noteAppliedRebalancedReadForFile(file_id);
      io_log("Phase 1 write-back owner-staged read: request_id=%llu "
             "file=%d aggregator_rank=%d source_rank=%d version=%llu "
             "bytes=%zu local_write_history=%d file_writeback_epoch=%d\n",
             static_cast<unsigned long long>(context.RequestId), file_id,
             context.Plan.AggregatorRank, ConservativeReadSourceRank,
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
  case ompfile::route::ReadRoute::PlannedRebalanced: {
    size_t bytes_read = 0;
    if (readAtRemoteRankWithBytes(file_id, decision.TargetRank,
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
    if (settings.dirty_owner_forwarding_enabled) {
      errno = errno != 0 ? errno : EIO;
      return -1;
    }
    return std::nullopt;
  }
  case ompfile::route::ReadRoute::DirtyOwnerRetry: {
    return readDirtyOwnerRetry(context, state, data, size);
  }
  case ompfile::route::ReadRoute::PfsCurrentSource: {
    size_t source_bytes_read = 0;
    if (readAtRemoteRankWithBytes(file_id, decision.TargetRank,
                                  offset, data, size, source_bytes_read)) {
      noteAppliedRebalancedReadForFile(file_id);
      io_log("Phase 1 planned read pfs-current source read: "
             "request_id=%llu file=%d aggregator_rank=%d source_rank=%d "
             "version=%llu bytes=%zu\n",
             static_cast<unsigned long long>(context.RequestId), file_id,
             context.Plan.AggregatorRank, decision.TargetRank,
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
  }
  case ompfile::route::ReadRoute::RebalanceBlocked: {
    io_log("Phase 1 planned read rebalanced route blocked: request_id=%llu "
           "file=%d aggregator_rank=%d reason=%s errno=%d; %s.\n",
           static_cast<unsigned long long>(context.RequestId), file_id,
           context.Plan.AggregatorRank, rebalance_reason, rebalance_errno,
           settings.dirty_owner_forwarding_enabled
               ? "fail-closed in write-back dirty-owner mode"
               : "falling back to owner handle");
    if (settings.dirty_owner_forwarding_enabled) {
      errno = rebalance_errno != 0 ? rebalance_errno : EIO;
      return -1;
    }
    return std::nullopt;
  }
  default:
    break;
  }
  assert(false && "executeCoherentRoute needs a terminal coherent route");
  return std::nullopt;
}

std::optional<int> MppBackend::readDirtyOwnerRetry(
    const ompfile::OmpFileReadRequestContext &context,
    const ompfile::route::ReadRouteState &state, void *data, size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  int remote_handle = lookupRemoteHandle(file_id);
  if (remote_handle < 0)
    return std::nullopt;
  int last_forward_errno = 0;
  bool attempted_forward = false;
  bool resolved_to_pfs = false;
  // An if-chain, not a switch: each break below must leave the loop.
  for (unsigned attempt = 0;
       attempt <= ompfile::route::kMaxDirtyOwnerForwardRetries; ++attempt) {
    const ompfile::route::FreshnessView Route = queryFreshnessForRank(
        context.PathKey, context.Plan.AggregatorRank);
    const ompfile::route::RetryStep step =
        ompfile::route::classifyRetryAttempt(
            Route, context.Plan.AggregatorRank, attempted_forward, attempt,
            state.Coherent.HasLocalWriteHistory);
    if (step == ompfile::route::RetryStep::SleepContinue) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    if (step == ompfile::route::RetryStep::Break)
      break;
    if (step == ompfile::route::RetryStep::ResolvedToPfs) {
      resolved_to_pfs = true;
      break;
    }
    if (step == ompfile::route::RetryStep::PfsCurrentSourceRead) {
      size_t source_bytes_read = 0;
      if (readAtRemoteRankWithBytes(
              file_id, ompfile::route::kConservativeWritebackOwnerRank,
              offset, data, size, source_bytes_read)) {
        noteAppliedRebalancedReadForFile(file_id);
        io_log("Phase 1 dirty-owner pfs-current source read: "
               "request_id=%llu file=%d aggregator_rank=%d "
               "source_rank=%d version=%llu bytes=%zu attempt=%u\n",
               static_cast<unsigned long long>(context.RequestId),
               file_id, context.Plan.AggregatorRank,
               ompfile::route::kConservativeWritebackOwnerRank,
               static_cast<unsigned long long>(Route.SelectedVersion),
               source_bytes_read, attempt);
        return 0;
      }
      last_forward_errno = errno != 0 ? errno : EIO;
      break;
    }
    if (step == ompfile::route::RetryStep::LocalRead) {
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

    // RetryStep::Forward
    attempted_forward = true;
    counters.dirty_owner_forward_attempt_count.fetch_add(
        1, std::memory_order_relaxed);
    size_t bytes_read = 0;
    bool forward_ok = false;
    {
      const auto lock = instrumentedMppCallLock();
      errno = 0;
      forward_ok = ompfile::mpp::dirtyOwnerPreadEx(
          remote_handle, Route.SourceRank, Route.SelectedVersion,
          offset, data, size, bytes_read);
      last_forward_errno = errno;
    }
    if (forward_ok) {
      counters.dirty_owner_forward_success_count.fetch_add(
          1, std::memory_order_relaxed);
      counters.dirty_owner_forward_bytes_total.fetch_add(
          bytes_read, std::memory_order_relaxed);
      counters.remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
      counters.remote_pread_bytes_total.fetch_add(size,
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

    counters.dirty_owner_forward_failure_count.fetch_add(
        1, std::memory_order_relaxed);
    const ompfile::route::ForwardFailure failure =
        ompfile::route::classifyForwardFailure(last_forward_errno, attempt);
    io_log("Phase 1 dirty-owner forward failed: request_id=%llu "
           "file=%d aggregator_rank=%d source_rank=%d version=%llu "
           "errno=%d attempt=%u%s\n",
           static_cast<unsigned long long>(context.RequestId), file_id,
           context.Plan.AggregatorRank, Route.SourceRank,
           static_cast<unsigned long long>(Route.SelectedVersion),
           last_forward_errno, attempt, failure.Action);
    if (failure.Next ==
        ompfile::route::ForwardFailureNext::CleanRangePlannedRead) {
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
    if (failure.Next == ompfile::route::ForwardFailureNext::Break)
      break;
    if (failure.SleepBeforeRetry)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (attempted_forward && !resolved_to_pfs) {
    errno = last_forward_errno != 0 ? last_forward_errno : EIO;
    return -1;
  }
  return std::nullopt;
}

std::optional<int> MppBackend::readPlannedOwner(
    const ompfile::OmpFileReadRequestContext &context, void *data,
    size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  int remote_handle = lookupRemoteHandle(file_id);
  if (remote_handle >= 0) {
    counters.remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
    counters.remote_pread_bytes_total.fetch_add(size,
                                                std::memory_order_relaxed);
    io_log("Phase 1 planned read route: request_id=%llu file=%d "
           "path_key=%llu aggregator_rank=%d\n",
           static_cast<unsigned long long>(context.RequestId), file_id,
           static_cast<unsigned long long>(context.PathKey),
           context.Plan.AggregatorRank);
    bool pread_ok = false;
    size_t bytes_read = 0;
    {
      const auto lock = instrumentedMppCallLock();
      pread_ok =
          ompfile::mpp::preadEx(remote_handle, offset, data, size, bytes_read);
    }
    if (!pread_ok) {
      io_log("MPP pread failed for file %d\n", file_id);
      return -1;
    }
    size_t short_bytes = 0;
    if (!finishRemoteRead(data, size, bytes_read, short_bytes))
      return -1;
    if (short_bytes > 0)
      io_trace("MPIIOBackend::readAtWithContext short-read req_id=%llu "
               "file=%d requested=%zu read=%zu short=%zu\n",
               static_cast<unsigned long long>(context.RequestId), file_id,
               size, bytes_read, short_bytes);

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
  return std::nullopt;
}

int MppBackend::readFallbackTail(
    const ompfile::OmpFileReadRequestContext &context, void *data,
    size_t size) {
  const int file_id = context.FileHandle;
  const long offset = static_cast<long>(context.Offset);
  if (settings.two_phase_enabled)
    counters.two_phase_fallback_count.fetch_add(1, std::memory_order_relaxed);

  io_trace("MPIIOBackend::readAtWithContext fallback req_id=%llu file=%d "
           "offset=%ld size=%zu\n",
           static_cast<unsigned long long>(context.RequestId), file_id, offset,
           size);
  size_t bytes_read = 0;
  return readAtFallbackWithBytes(file_id, offset, data, size, bytes_read);
}

int MppBackend::readAtFallbackWithBytes(int file_id, long offset, void *data,
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

  int remote_handle = lookupRemoteHandle(file_id);
  if (remote_handle < 0) {
    io_log("Error: Missing remote handle for file %d\n", file_id);
    traceHandleState("readAtFallback.mpp_io.missing_remote", file_id,
                     remote_handle);
    errno = EBADF;
    return -1;
  }
  counters.remote_pread_event_count.fetch_add(1, std::memory_order_relaxed);
  counters.remote_pread_bytes_total.fetch_add(size, std::memory_order_relaxed);
  bool pread_ok = false;
  {
    const auto lock = instrumentedMppCallLock();
    pread_ok =
        ompfile::mpp::preadEx(remote_handle, offset, data, size, bytes_read);
  }
  if (!pread_ok) {
    io_log("MPP pread failed for file %d\n", file_id);
    traceHandleState("readAtFallback.mpp_io.pread_fail", file_id,
                     remote_handle);
    return -1;
  }
  size_t short_bytes = 0;
  if (!finishRemoteRead(data, size, bytes_read, short_bytes))
    return -1;

  io_log("MPP read at offset completed.\n");
  return 0;
}
