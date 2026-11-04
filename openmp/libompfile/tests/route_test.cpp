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
  const DirtySourcePlan A = resolveDirtySource(routeTestObs(routeTestView(kCopy, 2, 7), false));
  OMPFILE_CHECK(A.SourceRank == 2 && A.FreshnessHasSource && A.ExpectedVersion == 7u);
  // No committed source: the conservative owner, as with the removed
  // distributed write-back off (no tile-hash guess).
  const DirtySourcePlan B = resolveDirtySource(routeTestObs(routeTestView(kWait, -1, 0, false), false));
  OMPFILE_CHECK(B.SourceRank == 0 && !B.FreshnessHasSource && B.ExpectedVersion == 0u);
  const DirtySourcePlan C = resolveDirtySource(routeTestObs(routeTestView(kCopy, -1, 0), false));
  OMPFILE_CHECK(C.SourceRank == 0 && !C.FreshnessHasSource);
  const DirtySourcePlan D = resolveDirtySource(routeTestObs(routeTestView(kUseLocal, -1, 4), false));
  OMPFILE_CHECK(D.SourceRank == 0 && !D.FreshnessHasSource && D.ExpectedVersion == 4u);
}

OMPFILE_TEST(route_ranges_overlap) {
  OMPFILE_CHECK(rangesOverlap(0, 10, 5, 15));
  OMPFILE_CHECK(!rangesOverlap(0, 10, 10, 20));
  OMPFILE_CHECK(!rangesOverlap(5, 5, 0, 10));
  OMPFILE_CHECK(!rangesOverlap(0, 10, 3, 3));
}

OMPFILE_TEST(route_writeback_tail) {
  DirtySourcePlan P;
  P.SourceRank = 2;
  P.ExpectedVersion = 9;
  P.FreshnessHasSource = true;
  const ReadDecision L = resolveWritebackOwnerTail(P, 2);
  OMPFILE_CHECK(L.Route == ReadRoute::WritebackLocalDirty && L.TargetRank == 2 && L.SourceRank == 2 && L.ExpectedVersion == 9u);
  const ReadDecision C = resolveWritebackOwnerTail(P, 1);
  OMPFILE_CHECK(C.Route == ReadRoute::WritebackConservativeOwner && C.TargetRank == 0 && C.SourceRank == 2);
  DirtySourcePlan Guess;
  Guess.SourceRank = 0;
  const ReadDecision G = resolveWritebackOwnerTail(Guess, 0);
  OMPFILE_CHECK(G.Route == ReadRoute::WritebackConservativeOwner && G.TargetRank == 0);
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
  WriteRouteConfig Batched;
  Batched.WriteBatchActive = true;
  OMPFILE_CHECK(resolveWriteRoute(Batched).Route == WriteRoute::Batched);
  // Write batching off: the owner handle, whatever the hint (the hash-routed
  // DistributedRank route went with LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_*).
  const WriteDecision O = resolveWriteRoute(WriteRouteConfig{});
  OMPFILE_CHECK(O.Route == WriteRoute::OwnerHandle && O.TargetRank == -1);
  // PlacedRank is never resolved, only built by the test placement hook.
  OMPFILE_CHECK(resolveWriteRoute(Batched).Route != WriteRoute::PlacedRank);
  OMPFILE_CHECK(O.Route != WriteRoute::PlacedRank);
}

OMPFILE_TEST(route_has_usable_plan) {
  OMPFILE_CHECK(!hasUsablePlan(routeTestPlanned(0, 0, false)));
  OMPFILE_CHECK(hasUsablePlan(routeTestPlanned(0, 0)));
  OMPFILE_CHECK(!hasUsablePlan(routeTestPlanned(0, -5)));
  OMPFILE_CHECK(!fileHasWritebackEpoch(routeTestObs(routeTestView(kUseLocal, -1, 3, false), false)));
  OMPFILE_CHECK(fileHasWritebackEpoch(routeTestObs(routeTestView(kUseLocal, -1, 3), false)));
}
