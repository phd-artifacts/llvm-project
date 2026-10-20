// The pure route resolvers in src/ompfile_route.h: each case pins one
// resolver's decision table, term for term with the code it was split from.

#include "ompfile_route.h"
#include "ompfile_test.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using ompfile::OmpFileFreshnessDecision;
using namespace ompfile::route;

ompfile::OmpFileReadRequestContext routeTestPlanned(int Agg, int Status,
                                                    bool HasPlan = true) {
  ompfile::OmpFileReadRequestContext C{};
  if (HasPlan)
    C.ContextFlags |= ompfile::OMPFILE_READ_CTX_HAS_PLAN;
  C.Plan.Status = Status;
  C.Plan.AggregatorRank = Agg;
  return C;
}

FreshnessView routeTestView(OmpFileFreshnessDecision D, int Src, uint64_t Ver,
                            bool Valid = true) {
  FreshnessView V;
  V.Decision = D;
  V.SourceRank = Src;
  V.SelectedVersion = Ver;
  V.Valid = Valid;
  return V;
}

ReadRouteConfig routeTestConfig(bool TwoPhase, bool Stage, bool Oracle) {
  ReadRouteConfig C;
  C.TwoPhaseActive = TwoPhase;
  C.StageRebalance = Stage;
  C.ScalarOracle = Oracle;
  return C;
}

CoherentObservation routeTestObs(FreshnessView V, bool History) {
  CoherentObservation O;
  O.Initial = V;
  O.HasLocalWriteHistory = History;
  return O;
}

RebalanceVerdict routeTestVerdict(bool Safe, const char *Reason) {
  RebalanceVerdict R;
  R.MetadataSafe = Safe;
  R.Reason = Reason;
  return R;
}

DistributeConfig routeTestDistribute(bool Active, unsigned Ranks) {
  DistributeConfig D;
  D.Active = Active;
  D.Ranks = Ranks;
  return D;
}

struct HistoryEntry {
  long Start = 0;
  long End = 0;
  uint64_t TileId = 0;
  uint64_t Sequence = 0;
  bool HasTile = false;
};

const auto kUseLocal = OmpFileFreshnessDecision::USE_LOCAL;
const auto kReadPfs = OmpFileFreshnessDecision::READ_PFS;
const auto kWait = OmpFileFreshnessDecision::WAIT_OR_FAIL;
const auto kCopy = OmpFileFreshnessDecision::COPY_FROM_RANK;

} // namespace

OMPFILE_TEST(route_entry_two_phase) {
  const auto NoPlan = routeTestPlanned(1, 0, /*HasPlan=*/false);
  const auto Plan = routeTestPlanned(1, 0);
  OMPFILE_CHECK(resolveReadRoute(NoPlan, routeTestConfig(true, false, false)).Route == ReadRoute::TwoPhase);
  const ReadDecision Bypass = resolveReadRoute(Plan, routeTestConfig(true, true, true));
  OMPFILE_CHECK(Bypass.Route == ReadRoute::Coherent && Bypass.LogTwoPhaseBypass);
  OMPFILE_CHECK(resolveReadRoute(Plan, routeTestConfig(true, true, false)).Route == ReadRoute::TwoPhase);
  OMPFILE_CHECK(resolveReadRoute(routeTestPlanned(1, -1), routeTestConfig(true, true, true)).Route == ReadRoute::TwoPhase);
}

OMPFILE_TEST(route_entry_planned_fallback) {
  const ReadDecision C = resolveReadRoute(routeTestPlanned(2, 0), routeTestConfig(false, true, false));
  OMPFILE_CHECK(C.Route == ReadRoute::Coherent && !C.LogTwoPhaseBypass);
  OMPFILE_CHECK(resolveReadRoute(routeTestPlanned(-1, 0), routeTestConfig(false, true, false)).Route == ReadRoute::PlannedOwner);
  OMPFILE_CHECK(resolveReadRoute(routeTestPlanned(2, 0), routeTestConfig(false, false, false)).Route == ReadRoute::PlannedOwner);
  OMPFILE_CHECK(resolveReadRoute(routeTestPlanned(2, 0, false), routeTestConfig(false, true, false)).Route == ReadRoute::Fallback);
  OMPFILE_CHECK(resolveReadRoute(routeTestPlanned(2, 1), routeTestConfig(false, true, false)).Route == ReadRoute::Fallback);
}

OMPFILE_TEST(route_coherent) {
  const ReadDecision G = resolveCoherentReadRoute(routeTestObs(routeTestView(kReadPfs, -1, 0), false), 3, true);
  OMPFILE_CHECK(G.Route == ReadRoute::GlobalOraclePfs && G.TargetRank == 3);
  OMPFILE_CHECK(resolveCoherentReadRoute(routeTestObs(routeTestView(kReadPfs, -1, 0), true), 3, false).Route == ReadRoute::RebalanceCheck);
  OMPFILE_CHECK(resolveCoherentReadRoute(routeTestObs(routeTestView(kWait, -1, 0, false), true), 3, true).Route == ReadRoute::WritebackOwner);
  OMPFILE_CHECK(resolveCoherentReadRoute(routeTestObs(routeTestView(kUseLocal, -1, 5), false), 3, true).Route == ReadRoute::WritebackOwner);
  OMPFILE_CHECK(resolveCoherentReadRoute(routeTestObs(routeTestView(kUseLocal, -1, 0), false), 3, true).Route == ReadRoute::RebalanceCheck);
}

OMPFILE_TEST(route_dirty_source) {
  ompfile::OmpFileIOHint Tile4{};
  Tile4.HintFlags = ompfile::OMPFILE_IO_HINT_HAS_TILE;
  Tile4.TileId = 4;
  ompfile::OmpFileIOHint Tile7 = Tile4;
  Tile7.TileId = 7;
  const ompfile::OmpFileIOHint NoTile{};
  const DirtySourcePlan A = resolveDirtySource(routeTestObs(routeTestView(kCopy, 2, 7), false), Tile4, routeTestDistribute(true, 3));
  OMPFILE_CHECK(A.SourceRank == 2 && A.FreshnessHasSource && !A.NeedsHistoryScan && A.ExpectedVersion == 7u);
  const DirtySourcePlan B = resolveDirtySource(routeTestObs(routeTestView(kWait, -1, 0, false), false), Tile7, routeTestDistribute(true, 3));
  OMPFILE_CHECK(B.SourceRank == 1 && B.HasRangeWritebackOwner && !B.NeedsHistoryScan);
  DirtySourcePlan S = resolveDirtySource(routeTestObs(routeTestView(kWait, -1, 0, false), false), NoTile, routeTestDistribute(true, 3));
  OMPFILE_CHECK(S.NeedsHistoryScan && S.SourceRank == 0);
  const DirtySourcePlan Off = resolveDirtySource(routeTestObs(routeTestView(kWait, -1, 0, false), false), NoTile, routeTestDistribute(false, 0));
  OMPFILE_CHECK(Off.SourceRank == 0 && !Off.NeedsHistoryScan);
  OMPFILE_CHECK(!resolveDirtySource(routeTestObs(routeTestView(kCopy, -1, 0), false), NoTile, routeTestDistribute(false, 0)).FreshnessHasSource);
  DirtySourcePlan S2 = S;
  applyHistoryScan(S, 2);
  OMPFILE_CHECK(S.SourceRank == 2 && S.HasRangeWritebackOwner);
  applyHistoryScan(S2, -1);
  OMPFILE_CHECK(S2.SourceRank == 0 && !S2.HasRangeWritebackOwner);
}

OMPFILE_TEST(route_history_scan) {
  std::vector<HistoryEntry> E(3);
  E[0].Start = 0; E[0].End = 100; E[0].TileId = 5; E[0].Sequence = 1; E[0].HasTile = true;
  E[1].Start = 200; E[1].End = 300; E[1].Sequence = 2;
  E[2].Start = 50; E[2].End = 150; E[2].TileId = 9; E[2].Sequence = 3; E[2].HasTile = true;
  OMPFILE_CHECK_EQ(latestOverlappingWriteOwnerRank(E, 60, 70, 3), 0);
  OMPFILE_CHECK_EQ(latestOverlappingWriteOwnerRank(E, 210, 220, 3), 2);
  OMPFILE_CHECK_EQ(latestOverlappingWriteOwnerRank(E, 400, 500, 3), -1);
  std::vector<HistoryEntry> F(2);
  F[0].Start = 0; F[0].End = 100; F[0].TileId = 4; F[0].Sequence = 5; F[0].HasTile = true;
  F[1].Start = 0; F[1].End = 100; F[1].TileId = 5; F[1].Sequence = 2; F[1].HasTile = true;
  OMPFILE_CHECK_EQ(latestOverlappingWriteOwnerRank(F, 10, 20, 3), 1);
  OMPFILE_CHECK_EQ(readEndOrOffset(10, 5), 15L);
  OMPFILE_CHECK_EQ(readEndOrOffset(LONG_MAX - 1, 5), LONG_MAX - 1);
  OMPFILE_CHECK_EQ(readEndOrOffset(0, SIZE_MAX), 0L);
  OMPFILE_CHECK(rangesOverlap(0, 10, 5, 15));
  OMPFILE_CHECK(!rangesOverlap(0, 10, 10, 20));
  OMPFILE_CHECK(!rangesOverlap(5, 5, 0, 10));
  OMPFILE_CHECK(!rangesOverlap(0, 10, 3, 3));
}

OMPFILE_TEST(route_writeback_tail) {
  DirtySourcePlan P;
  P.SourceRank = 2;
  P.ExpectedVersion = 9;
  const ReadDecision L = resolveWritebackOwnerTail(P, 2, routeTestDistribute(true, 3), false, false);
  OMPFILE_CHECK(L.Route == ReadRoute::WritebackLocalDirty && L.TargetRank == 2 && L.SourceRank == 2 && L.ExpectedVersion == 9u);
  OMPFILE_CHECK(resolveWritebackOwnerTail(P, 1, routeTestDistribute(true, 3), true, false).Route == ReadRoute::WritebackRefuseSourceClean);
  OMPFILE_CHECK(resolveWritebackOwnerTail(P, 1, routeTestDistribute(true, 3), true, true).Route == ReadRoute::WritebackRefuseDistributed);
  OMPFILE_CHECK(resolveWritebackOwnerTail(P, 1, routeTestDistribute(true, 3), false, false).Route == ReadRoute::WritebackRefuseDistributed);
  const ReadDecision C = resolveWritebackOwnerTail(P, 2, routeTestDistribute(false, 0), false, false);
  OMPFILE_CHECK(C.Route == ReadRoute::WritebackConservativeOwner && C.TargetRank == 0);
  const ReadDecision D = dirtyForwardDecision(P);
  OMPFILE_CHECK(D.Route == ReadRoute::WritebackDirtyForward && D.SourceRank == 2 && D.TargetRank == 2 && D.ExpectedVersion == 9u);
  OMPFILE_CHECK(classifyDirtyOracle(true, 1) == DirtyOracleOutcome::Dirty);
  OMPFILE_CHECK(classifyDirtyOracle(true, 0) == DirtyOracleOutcome::Clean);
  OMPFILE_CHECK(classifyDirtyOracle(true, 2) == DirtyOracleOutcome::Unavailable);
  OMPFILE_CHECK(classifyDirtyOracle(false, 1) == DirtyOracleOutcome::Unavailable);
}

OMPFILE_TEST(route_rebalance) {
  auto R = [](FreshnessView V, bool Safe, const char *Reason, bool Dof) {
    return resolveRebalanceRoute(routeTestObs(V, false), routeTestVerdict(Safe, Reason), 1, Dof);
  };
  const ReadDecision A = R(routeTestView(kUseLocal, -1, 0), true, "none", false);
  OMPFILE_CHECK(A.Route == ReadRoute::PlannedRebalanced && A.TargetRank == 1);
  OMPFILE_CHECK(R(routeTestView(kCopy, 2, 0), false, "remote-only-writable-file", true).Route == ReadRoute::DirtyOwnerRetry);
  OMPFILE_CHECK(R(routeTestView(kCopy, 2, 0), false, "remote-only-writable-file", false).Route == ReadRoute::RebalanceBlocked);
  OMPFILE_CHECK(R(routeTestView(kWait, -1, 0, false), false, "unsafe-write-history", true).Route == ReadRoute::DirtyOwnerRetry);
  OMPFILE_CHECK(R(routeTestView(kWait, -1, 0, false), true, "none", false).Route == ReadRoute::RebalanceBlocked);
  OMPFILE_CHECK(R(routeTestView(kReadPfs, -1, 0), true, "none", false).Route == ReadRoute::PlannedRebalanced);
  OMPFILE_CHECK(R(routeTestView(kUseLocal, -1, 4), false, "unsafe-write-history", true).Route == ReadRoute::PlannedRebalanced);
  const ReadDecision P = R(routeTestView(kReadPfs, -1, 4), true, "none", true);
  OMPFILE_CHECK(P.Route == ReadRoute::PfsCurrentSource && P.TargetRank == 0);
  OMPFILE_CHECK(R(routeTestView(kCopy, 1, 0), true, "none", false).Route == ReadRoute::PlannedRebalanced);
}

OMPFILE_TEST(route_retry_attempt) {
  const FreshnessView Invalid = routeTestView(kWait, -1, 0, false);
  OMPFILE_CHECK(classifyRetryAttempt(Invalid, 1, false, 0, false) == RetryStep::Break);
  OMPFILE_CHECK(classifyRetryAttempt(Invalid, 1, true, 3, false) == RetryStep::SleepContinue);
  OMPFILE_CHECK(classifyRetryAttempt(Invalid, 1, true, 8, false) == RetryStep::Break);
  OMPFILE_CHECK(classifyRetryAttempt(routeTestView(kReadPfs, -1, 0), 1, false, 0, true) == RetryStep::PfsCurrentSourceRead);
  OMPFILE_CHECK(classifyRetryAttempt(routeTestView(kReadPfs, -1, 0), 1, false, 0, false) == RetryStep::ResolvedToPfs);
  OMPFILE_CHECK(classifyRetryAttempt(routeTestView(kUseLocal, -1, 0), 1, false, 0, false) == RetryStep::LocalRead);
  OMPFILE_CHECK(classifyRetryAttempt(routeTestView(kCopy, 1, 0), 1, false, 0, false) == RetryStep::LocalRead);
  OMPFILE_CHECK(classifyRetryAttempt(routeTestView(kWait, -1, 0), 1, true, 0, false) == RetryStep::SleepContinue);
  OMPFILE_CHECK(classifyRetryAttempt(routeTestView(kWait, -1, 0), 1, false, 0, false) == RetryStep::Break);
  OMPFILE_CHECK(classifyRetryAttempt(routeTestView(kCopy, -1, 0), 1, false, 0, false) == RetryStep::Break);
  OMPFILE_CHECK(classifyRetryAttempt(routeTestView(kCopy, 2, 0), 1, false, 0, false) == RetryStep::Forward);
}

OMPFILE_TEST(route_forward_failure) {
  const ForwardFailure A = classifyForwardFailure(ESTALE, 0);
  OMPFILE_CHECK(A.Next == ForwardFailureNext::Retry && A.SleepBeforeRetry && std::strcmp(A.Action, "; retrying freshness") == 0);
  const ForwardFailure B = classifyForwardFailure(ESTALE, 8);
  OMPFILE_CHECK(B.Next == ForwardFailureNext::Retry && !B.SleepBeforeRetry && std::strcmp(B.Action, "; fail-closed") == 0);
  for (const unsigned Attempt : {0u, 8u}) {
    const ForwardFailure N = classifyForwardFailure(ENODATA, Attempt);
    OMPFILE_CHECK(N.Next == ForwardFailureNext::CleanRangePlannedRead && std::strcmp(N.Action, "; routing source-rank staged read") == 0);
  }
  for (const int Err : {EIO, 0}) {
    const ForwardFailure E = classifyForwardFailure(Err, 0);
    OMPFILE_CHECK(E.Next == ForwardFailureNext::Break && std::strcmp(E.Action, "; fail-closed") == 0);
  }
}

OMPFILE_TEST(route_write) {
  ompfile::OmpFileWriteRequestContext Tile{};
  Tile.Hint.HintFlags = ompfile::OMPFILE_IO_HINT_HAS_TILE;
  Tile.Hint.TileId = 7;
  WriteRouteConfig Batched;
  Batched.WriteBatchActive = true;
  OMPFILE_CHECK(resolveWriteRoute(Tile, 64, Batched).Route == WriteRoute::Batched);
  WriteRouteConfig D3;
  D3.Distribute = routeTestDistribute(true, 3);
  const WriteDecision T = resolveWriteRoute(Tile, 64, D3);
  OMPFILE_CHECK(T.Route == WriteRoute::DistributedRank && T.TargetRank == 1);
  ompfile::OmpFileWriteRequestContext Off{};
  Off.Offset = 4096;
  OMPFILE_CHECK_EQ(resolveWriteRoute(Off, 1024, D3).TargetRank, 1);
  Off.Offset = 10;
  OMPFILE_CHECK_EQ(resolveWriteRoute(Off, 0, D3).TargetRank, 1);
  Off.Offset = -1;
  const WriteDecision Neg = resolveWriteRoute(Off, 8, D3);
  OMPFILE_CHECK(Neg.Route == WriteRoute::OwnerHandle && Neg.TargetRank == -1);
  WriteRouteConfig D1;
  D1.Distribute = routeTestDistribute(true, 1);
  OMPFILE_CHECK(resolveWriteRoute(Tile, 64, D1).Route == WriteRoute::OwnerHandle);
  WriteRouteConfig Inactive;
  Inactive.Distribute = routeTestDistribute(false, 3);
  OMPFILE_CHECK(resolveWriteRoute(Tile, 64, Inactive).Route == WriteRoute::OwnerHandle);
}

OMPFILE_TEST(route_distributed_rank_for_key) {
  OMPFILE_CHECK_EQ(distributedRankForKey(5, 3), 2);
  OMPFILE_CHECK_EQ(distributedRankForKey(5, 1), -1);
  OMPFILE_CHECK_EQ(distributedRankForKey(5, 0), -1);
}

OMPFILE_TEST(route_has_usable_plan) {
  OMPFILE_CHECK(!hasUsablePlan(routeTestPlanned(0, 0, false)));
  OMPFILE_CHECK(hasUsablePlan(routeTestPlanned(0, 0)));
  OMPFILE_CHECK(!hasUsablePlan(routeTestPlanned(0, -5)));
  OMPFILE_CHECK(!fileHasWritebackEpoch(routeTestObs(routeTestView(kUseLocal, -1, 3, false), false)));
  OMPFILE_CHECK(fileHasWritebackEpoch(routeTestObs(routeTestView(kUseLocal, -1, 3), false)));
}
