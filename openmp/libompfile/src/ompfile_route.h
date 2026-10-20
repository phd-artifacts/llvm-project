#ifndef OMPFILE_ROUTE_H
#define OMPFILE_ROUTE_H

// Route resolution for libompfile's read and write paths: the decide half of
// readAtWithContext / writeAtWithContext. The execute half (MPP calls,
// the handle and MPP-call locks, retries and sleeps, counters, logging,
// errno) stays in the backend's read-path and write-path sources. A resolver
// reads only its arguments: the request context, a knob snapshot the executor
// took, and what the executor has already observed (freshness replies, local
// write history, the dirty-owner oracle answer). Where a route needs an MPP
// answer the two alternate: decide, query, decide. A task-graph plan is meant
// to pre-fill the decisions returned here (docs/architecture.md, "Client read
// and write routes").

#include "ompfile_sched.h"

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ompfile {
namespace route {

// Rank that serves a write-back read with no better-known source.
inline constexpr int kConservativeWritebackOwnerRank = 0;
// The dirty-owner retry loop runs attempts 0..kMaxDirtyOwnerForwardRetries.
inline constexpr unsigned kMaxDirtyOwnerForwardRetries = 8;

// Distributed write-back routing, sampled by snapshotDistributeConfig().
struct DistributeConfig {
  bool Active = false; // SharedConfig::WritebackDistributeActive
  unsigned Ranks = 0;  // SharedConfig::WritebackDistributeRanks when Active
};

// Read-path knobs, sampled once per request (snapshotReadRouteConfig()).
struct ReadRouteConfig {
  bool TwoPhaseActive = false;       // isTwoPhaseActive()
  bool StageRebalance = false;       // readthrough + write-back, guard on
  bool ScalarOracle = false;         // StageRebalance && DIRTY_OWNER_FORWARDING
  bool DirtyOwnerForwarding = false; // dirty_owner_forwarding_enabled
};

// One freshness query as the client sees it; Valid is false when the query
// failed or was skipped (path key 0 or rank < 0).
struct FreshnessView {
  OmpFileFreshnessDecision Decision = OmpFileFreshnessDecision::WAIT_OR_FAIL;
  int SourceRank = -1;
  uint64_t SelectedVersion = 0;
  bool Valid = false;
};

enum class ReadRoute : uint8_t {
  // resolveReadRoute
  TwoPhase,     // readAtTwoPhase
  Coherent,     // staged: freshness query, then resolveCoherentReadRoute
  PlannedOwner, // the opened owner handle; falls to Fallback when unmapped
  Fallback,     // readAtFallbackWithBytes; counts fallbacks= under two-phase
  // resolveCoherentReadRoute
  GlobalOraclePfs, // no-stage read on TargetRank (READ_PFS, dirty-owner fwd)
  WritebackOwner,  // dirty source, history scan, oracle, then a Writeback*
  RebalanceCheck,  // canApplyRebalancedRead verdict, then resolveRebalanceRoute
  // dirtyForwardDecision / resolveWritebackOwnerTail
  WritebackDirtyForward,      // dirty-owner read from SourceRank
  WritebackLocalDirty,        // dirty-owner read on SourceRank == aggregator
  WritebackRefuseSourceClean, // ESTALE
  WritebackRefuseDistributed, // ESTALE
  WritebackConservativeOwner, // staged read on TargetRank (rank 0)
  // resolveRebalanceRoute
  PlannedRebalanced, // staged read on TargetRank (the aggregator)
  DirtyOwnerRetry,   // freshness retry loop with dirty-owner forwards
  PfsCurrentSource,  // staged read on TargetRank (rank 0)
  RebalanceBlocked,  // fails closed under dirty-owner forwarding, else owner
};

struct ReadDecision {
  ReadRoute Route = ReadRoute::Fallback;
  int TargetRank = -1;            // rank whose proxy serves a remote route
  int SourceRank = -1;            // dirty-owner source (Writeback* routes)
  uint64_t ExpectedVersion = 0;   // dirty-owner expected version
  bool LogTwoPhaseBypass = false; // resolveReadRoute only
};

struct CoherentObservation {
  FreshnessView Initial{};   // query of FreshnessKey for the aggregator rank
  uint64_t FreshnessKey = 0; // keys::tileKeyForHint(PathKey, Hint)
  bool HasLocalWriteHistory = false;
};

struct RebalanceVerdict {
  bool MetadataSafe = false; // canApplyRebalancedRead's return value
  const char *Reason = "none";
  int ReasonErrno = 0;
};

struct DirtySourcePlan {
  int SourceRank = kConservativeWritebackOwnerRank;
  uint64_t ExpectedVersion = 0;
  bool FreshnessHasSource = false;
  bool HasRangeWritebackOwner = false;
  bool NeedsHistoryScan = false;
};

// What the executor of one read has observed; the routes' logs read it.
struct ReadRouteState {
  CoherentObservation Coherent{};
  RebalanceVerdict Rebalance{};
  bool DirtySourceKnownClean = false;
  bool DirtySourceUnknown = false;
};

enum class DirtyOracleOutcome : uint8_t { Dirty, Clean, Unavailable };
enum class RetryStep : uint8_t {
  SleepContinue,
  Break,
  PfsCurrentSourceRead,
  ResolvedToPfs,
  LocalRead,
  Forward,
};
enum class ForwardFailureNext : uint8_t { CleanRangePlannedRead, Break, Retry };

struct ForwardFailure {
  const char *Action = "; fail-closed"; // suffix of the forward-failed log
  ForwardFailureNext Next = ForwardFailureNext::Break;
  bool SleepBeforeRetry = false;
};

enum class WriteRoute : uint8_t {
  Batched,         // writeAtBatched
  DistributedRank, // writeAtRemoteRankWithBytes; OwnerHandle on failure
  OwnerHandle,     // the opened owner handle
};

struct WriteRouteConfig {
  bool WriteBatchActive = false; // isWriteBatchActive()
  DistributeConfig Distribute{}; // sampled only when !WriteBatchActive
};

struct WriteDecision {
  WriteRoute Route = WriteRoute::OwnerHandle;
  int TargetRank = -1; // DistributedRank: the write's and commit's rank
};

inline bool rangesOverlap(long a_start, long a_end, long b_start, long b_end) {
  if (a_end <= a_start || b_end <= b_start)
    return false;
  return a_start < b_end && b_start < a_end;
}

inline int distributedRankForKey(uint64_t key, unsigned ranks) {
  if (ranks <= 1)
    return -1;
  return static_cast<int>(key % ranks);
}

// End of [offset, offset + size), or offset itself when that overflows long.
inline long readEndOrOffset(long offset, size_t size) {
  return (size <= static_cast<size_t>(std::numeric_limits<long>::max()) &&
          offset <= std::numeric_limits<long>::max() - static_cast<long>(size))
             ? offset + static_cast<long>(size)
             : offset;
}

inline bool hasUsablePlan(const OmpFileReadRequestContext &context) {
  if ((context.ContextFlags & OMPFILE_READ_CTX_HAS_PLAN) == 0)
    return false;
  if (context.Plan.Status != 0)
    return false;
  return true;
}

inline bool fileHasWritebackEpoch(const CoherentObservation &obs) {
  return obs.Initial.Valid && obs.Initial.SelectedVersion != 0;
}

inline ReadDecision resolveReadRoute(const OmpFileReadRequestContext &context,
                                     const ReadRouteConfig &config) {
  ReadDecision decision{};
  const bool has_plan = hasUsablePlan(context);
  const bool require_scalar_oracle = config.ScalarOracle && has_plan;
  if (config.TwoPhaseActive && !require_scalar_oracle) {
    decision.Route = ReadRoute::TwoPhase;
    return decision;
  }
  decision.LogTwoPhaseBypass = config.TwoPhaseActive && require_scalar_oracle;
  if (!has_plan) {
    decision.Route = ReadRoute::Fallback;
    return decision;
  }
  decision.Route = (context.Plan.AggregatorRank >= 0 && config.StageRebalance)
                       ? ReadRoute::Coherent
                       : ReadRoute::PlannedOwner;
  return decision;
}

inline ReadDecision resolveCoherentReadRoute(const CoherentObservation &obs,
                                             int aggregator_rank,
                                             bool dirty_owner_forwarding) {
  ReadDecision decision{};
  if (dirty_owner_forwarding && obs.Initial.Valid &&
      obs.Initial.Decision == OmpFileFreshnessDecision::READ_PFS) {
    decision.Route = ReadRoute::GlobalOraclePfs;
    decision.TargetRank = aggregator_rank;
    return decision;
  }
  if (dirty_owner_forwarding &&
      (obs.HasLocalWriteHistory || fileHasWritebackEpoch(obs))) {
    decision.Route = ReadRoute::WritebackOwner;
    return decision;
  }
  decision.Route = ReadRoute::RebalanceCheck;
  return decision;
}

// Freshness commits report the rank that actually completed a write. This is
// authoritative when write batching falls back to a file owner, whereas the
// tile hash is only a prospective destination. Use the hash solely before a
// committed source is available.
inline DirtySourcePlan resolveDirtySource(const CoherentObservation &obs,
                                          const OmpFileIOHint &hint,
                                          const DistributeConfig &distribute) {
  const FreshnessView &freshness = obs.Initial;
  DirtySourcePlan plan{};
  plan.FreshnessHasSource =
      freshness.Valid &&
      freshness.Decision == OmpFileFreshnessDecision::COPY_FROM_RANK &&
      freshness.SourceRank >= 0;
  plan.SourceRank = plan.FreshnessHasSource ? freshness.SourceRank
                                            : kConservativeWritebackOwnerRank;
  if (distribute.Active && !plan.FreshnessHasSource &&
      (hint.HintFlags & OMPFILE_IO_HINT_HAS_TILE) != 0) {
    const int tile_owner_rank =
        distributedRankForKey(hint.TileId, distribute.Ranks);
    if (tile_owner_rank >= 0) {
      plan.SourceRank = tile_owner_rank;
      plan.HasRangeWritebackOwner = true;
    }
  }
  plan.NeedsHistoryScan = distribute.Active && !plan.HasRangeWritebackOwner &&
                          !plan.FreshnessHasSource;
  plan.ExpectedVersion = (freshness.Valid && freshness.SelectedVersion != 0)
                             ? freshness.SelectedVersion
                             : 0;
  return plan;
}

// Rank of the latest-sequence write overlapping [offset, read_end), -1 if
// none. The caller holds the backend's handle lock. EntryRange elements need
// Start, End, Sequence, HasTile and TileId (the backend's WriteEpochEntry).
template <typename EntryRange>
inline int latestOverlappingWriteOwnerRank(const EntryRange &entries,
                                           long offset, long read_end,
                                           unsigned distribute_ranks) {
  assert(distribute_ranks > 1 && "history scan needs more than one rank");
  uint64_t latest_sequence = 0;
  int range_owner_rank = -1;
  for (const auto &entry : entries) {
    if (entry.Sequence < latest_sequence)
      continue;
    if (!rangesOverlap(offset, read_end, entry.Start, entry.End))
      continue;
    // Without a tile hint, key on the write's start offset alone (not its own
    // byte width): two writes covering the same physical region but with
    // differing widths must still route to the same rank, which dividing by
    // (End-Start) breaks.
    const uint64_t key =
        entry.HasTile ? entry.TileId : static_cast<uint64_t>(entry.Start);
    latest_sequence = entry.Sequence;
    range_owner_rank = static_cast<int>(key % distribute_ranks);
  }
  return range_owner_rank;
}

inline void applyHistoryScan(DirtySourcePlan &plan, int range_owner_rank) {
  assert(plan.NeedsHistoryScan && "history scan result without a scan");
  if (range_owner_rank >= 0) {
    plan.SourceRank = range_owner_rank;
    plan.HasRangeWritebackOwner = true;
  } else {
    plan.SourceRank = kConservativeWritebackOwnerRank;
  }
}

inline DirtyOracleOutcome classifyDirtyOracle(bool query_ok, int oracle_state) {
  if (query_ok && oracle_state == 1)
    return DirtyOracleOutcome::Dirty;
  if (query_ok && oracle_state == 0)
    return DirtyOracleOutcome::Clean;
  return DirtyOracleOutcome::Unavailable;
}

inline ReadDecision dirtyForwardDecision(const DirtySourcePlan &plan) {
  ReadDecision decision{};
  decision.Route = ReadRoute::WritebackDirtyForward;
  decision.TargetRank = plan.SourceRank;
  decision.SourceRank = plan.SourceRank;
  decision.ExpectedVersion = plan.ExpectedVersion;
  return decision;
}

inline ReadDecision
resolveWritebackOwnerTail(const DirtySourcePlan &plan, int aggregator_rank,
                          const DistributeConfig &distribute, bool known_clean,
                          bool unknown) {
  ReadDecision decision{};
  decision.SourceRank = plan.SourceRank;
  decision.ExpectedVersion = plan.ExpectedVersion;
  if (distribute.Active && plan.SourceRank == aggregator_rank) {
    decision.Route = ReadRoute::WritebackLocalDirty;
    decision.TargetRank = plan.SourceRank;
    return decision;
  }
  if (distribute.Active && known_clean && !unknown) {
    decision.Route = ReadRoute::WritebackRefuseSourceClean;
    return decision;
  }
  if (distribute.Active) {
    decision.Route = ReadRoute::WritebackRefuseDistributed;
    return decision;
  }
  decision.Route = ReadRoute::WritebackConservativeOwner;
  decision.TargetRank = kConservativeWritebackOwnerRank;
  return decision;
}

// The rebalance branch of the old readAtWithContext, term for term. Some
// terms cannot be true for online state (docs/architecture.md); they stay so
// a supplied observation decides the same way.
inline ReadDecision resolveRebalanceRoute(const CoherentObservation &obs,
                                          const RebalanceVerdict &verdict,
                                          int aggregator_rank,
                                          bool dirty_owner_forwarding) {
  assert(verdict.Reason && "canApplyRebalancedRead always sets a reason");
  const FreshnessView &f = obs.Initial;
  const bool file_has_writeback_epoch = fileHasWritebackEpoch(obs);
  const bool freshness_needs_dirty_owner =
      f.Valid && f.Decision == OmpFileFreshnessDecision::COPY_FROM_RANK &&
      f.SourceRank >= 0 && f.SourceRank != aggregator_rank;
  const bool freshness_allows_rebalance =
      f.Valid && (f.Decision == OmpFileFreshnessDecision::READ_PFS ||
                  f.Decision == OmpFileFreshnessDecision::USE_LOCAL ||
                  (f.Decision == OmpFileFreshnessDecision::COPY_FROM_RANK &&
                   f.SourceRank == aggregator_rank));
  const bool planned_rank_current =
      f.Valid && (f.Decision == OmpFileFreshnessDecision::USE_LOCAL ||
                  (f.Decision == OmpFileFreshnessDecision::COPY_FROM_RANK &&
                   f.SourceRank == aggregator_rank));
  const bool metadata_blocked_only_by_write_history =
      std::strcmp(verdict.Reason, "unsafe-write-history") == 0;
  const bool pfs_current_after_writeback =
      dirty_owner_forwarding && file_has_writeback_epoch && f.Valid &&
      f.Decision == OmpFileFreshnessDecision::READ_PFS;
  const bool rebalanced_safe =
      freshness_allows_rebalance && !pfs_current_after_writeback &&
      (verdict.MetadataSafe ||
       (dirty_owner_forwarding && file_has_writeback_epoch &&
        planned_rank_current && metadata_blocked_only_by_write_history));
  ReadDecision decision{};
  if (rebalanced_safe) {
    decision.Route = ReadRoute::PlannedRebalanced;
    decision.TargetRank = aggregator_rank;
  } else if (dirty_owner_forwarding &&
             (freshness_needs_dirty_owner ||
              metadata_blocked_only_by_write_history)) {
    decision.Route = ReadRoute::DirtyOwnerRetry;
  } else if (pfs_current_after_writeback) {
    decision.Route = ReadRoute::PfsCurrentSource;
    decision.TargetRank = kConservativeWritebackOwnerRank;
  } else {
    decision.Route = ReadRoute::RebalanceBlocked;
  }
  return decision;
}

inline RetryStep classifyRetryAttempt(const FreshnessView &view,
                                      int aggregator_rank,
                                      bool attempted_forward, unsigned attempt,
                                      bool has_local_write_history) {
  const bool may_retry =
      attempted_forward && attempt < kMaxDirtyOwnerForwardRetries;
  if (!view.Valid)
    return may_retry ? RetryStep::SleepContinue : RetryStep::Break;
  if (view.Decision == OmpFileFreshnessDecision::READ_PFS)
    return has_local_write_history ? RetryStep::PfsCurrentSourceRead
                                   : RetryStep::ResolvedToPfs;
  if (view.Decision == OmpFileFreshnessDecision::USE_LOCAL ||
      (view.Decision == OmpFileFreshnessDecision::COPY_FROM_RANK &&
       view.SourceRank == aggregator_rank))
    return RetryStep::LocalRead;
  if (view.Decision != OmpFileFreshnessDecision::COPY_FROM_RANK ||
      view.SourceRank < 0)
    return may_retry ? RetryStep::SleepContinue : RetryStep::Break;
  return RetryStep::Forward;
}

inline ForwardFailure classifyForwardFailure(int forward_errno,
                                             unsigned attempt) {
  ForwardFailure failure{};
  if (forward_errno == ESTALE && attempt < kMaxDirtyOwnerForwardRetries)
    failure.Action = "; retrying freshness";
  else if (forward_errno == ENODATA)
    failure.Action = "; routing source-rank staged read";
  if (forward_errno == ENODATA) {
    failure.Next = ForwardFailureNext::CleanRangePlannedRead;
  } else if (forward_errno != ESTALE) {
    failure.Next = ForwardFailureNext::Break;
  } else {
    failure.Next = ForwardFailureNext::Retry;
    failure.SleepBeforeRetry = attempt < kMaxDirtyOwnerForwardRetries;
  }
  return failure;
}

// The write path's old chooseDistributedWriteRank, minus the environment.
inline int distributedWriteRank(const OmpFileWriteRequestContext &context,
                                size_t size, unsigned ranks) {
  if (ranks <= 1)
    return -1;
  uint64_t key = 0;
  if ((context.Hint.HintFlags & OMPFILE_IO_HINT_HAS_TILE) != 0)
    key = context.Hint.TileId;
  else if (context.Offset >= 0 && size > 0)
    key = static_cast<uint64_t>(context.Offset / static_cast<int64_t>(size));
  else if (context.Offset >= 0)
    key = static_cast<uint64_t>(context.Offset);
  else
    return -1;
  return static_cast<int>(key % ranks);
}

inline WriteDecision resolveWriteRoute(const OmpFileWriteRequestContext &context,
                                       size_t size,
                                       const WriteRouteConfig &config) {
  WriteDecision decision{};
  if (config.WriteBatchActive) {
    decision.Route = WriteRoute::Batched;
    return decision;
  }
  const int rank = config.Distribute.Active
                       ? distributedWriteRank(context, size,
                                              config.Distribute.Ranks)
                       : -1;
  if (rank >= 0) {
    decision.Route = WriteRoute::DistributedRank;
    decision.TargetRank = rank;
  }
  return decision;
}

} // namespace route
} // namespace ompfile

#endif // OMPFILE_ROUTE_H
