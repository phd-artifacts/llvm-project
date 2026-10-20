// The environment-knob readers in ompfile_env.h, driven through setenv so the
// acceptance rules the client and the proxy now share are pinned.

#include "ompfile_env.h"
#include "ompfile_config.h"
#include "ompfile_test.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace {

struct ScopedEnv {
  const char *Name;
  ScopedEnv(const char *N, const char *Value) : Name(N) {
    if (Value)
      setenv(N, Value, 1);
    else
      unsetenv(N);
  }
  ~ScopedEnv() { unsetenv(Name); }
};

} // namespace

using namespace ompfile::env;

OMPFILE_TEST(flag_accepts_only_one) {
  { ScopedEnv E("OMPFILE_TEST_FLAG", "1"); OMPFILE_CHECK(flag("OMPFILE_TEST_FLAG")); }
  { ScopedEnv E("OMPFILE_TEST_FLAG", "0"); OMPFILE_CHECK(!flag("OMPFILE_TEST_FLAG")); }
  { ScopedEnv E("OMPFILE_TEST_FLAG", "true"); OMPFILE_CHECK(!flag("OMPFILE_TEST_FLAG")); }
  { ScopedEnv E("OMPFILE_TEST_FLAG", "11"); OMPFILE_CHECK(!flag("OMPFILE_TEST_FLAG")); }
  { ScopedEnv E("OMPFILE_TEST_FLAG", nullptr); OMPFILE_CHECK(!flag("OMPFILE_TEST_FLAG")); }
}

OMPFILE_TEST(bool_or_reports_malformed) {
  bool Malformed = true;
  { ScopedEnv E("OMPFILE_TEST_BOOL", nullptr); OMPFILE_CHECK(boolOr("OMPFILE_TEST_BOOL", true, &Malformed)); OMPFILE_CHECK(!Malformed); }
  { ScopedEnv E("OMPFILE_TEST_BOOL", "0"); OMPFILE_CHECK(!boolOr("OMPFILE_TEST_BOOL", true, &Malformed)); OMPFILE_CHECK(!Malformed); }
  { ScopedEnv E("OMPFILE_TEST_BOOL", "1"); OMPFILE_CHECK(boolOr("OMPFILE_TEST_BOOL", false, &Malformed)); OMPFILE_CHECK(!Malformed); }
  { ScopedEnv E("OMPFILE_TEST_BOOL", "yes"); OMPFILE_CHECK(!boolOr("OMPFILE_TEST_BOOL", false, &Malformed)); OMPFILE_CHECK(Malformed); }
  { ScopedEnv E("OMPFILE_TEST_BOOL", ""); OMPFILE_CHECK(boolOr("OMPFILE_TEST_BOOL", true, &Malformed)); OMPFILE_CHECK(Malformed); }
  { ScopedEnv E("OMPFILE_TEST_BOOL", "yes"); OMPFILE_CHECK(boolOr("OMPFILE_TEST_BOOL", true)); }
}

OMPFILE_TEST(equals_is_exact) {
  ScopedEnv E("OMPFILE_TEST_EQ", "write-back");
  OMPFILE_CHECK(equals("OMPFILE_TEST_EQ", "write-back"));
  OMPFILE_CHECK(!equals("OMPFILE_TEST_EQ", "writeback"));
  OMPFILE_CHECK(!equals("OMPFILE_TEST_EQ_UNSET", "write-back"));
}

OMPFILE_TEST(positive_or_parses_strict_decimals) {
  { ScopedEnv E("OMPFILE_TEST_POS", "12"); OMPFILE_CHECK_EQ(positiveOr("OMPFILE_TEST_POS"), 12u); }
  { ScopedEnv E("OMPFILE_TEST_POS", "0"); OMPFILE_CHECK_EQ(positiveOr("OMPFILE_TEST_POS", 7), 7u); }
  { ScopedEnv E("OMPFILE_TEST_POS", "-1"); OMPFILE_CHECK_EQ(positiveOr("OMPFILE_TEST_POS"), 0u); }
  { ScopedEnv E("OMPFILE_TEST_POS", "12x"); OMPFILE_CHECK_EQ(positiveOr("OMPFILE_TEST_POS"), 0u); }
  { ScopedEnv E("OMPFILE_TEST_POS", "4294967296"); OMPFILE_CHECK_EQ(positiveOr("OMPFILE_TEST_POS"), 0u); }
  { ScopedEnv E("OMPFILE_TEST_POS", ""); OMPFILE_CHECK_EQ(positiveOr("OMPFILE_TEST_POS", 3), 3u); }
  { ScopedEnv E("OMPFILE_TEST_POS", nullptr); OMPFILE_CHECK_EQ(positiveOr("OMPFILE_TEST_POS"), 0u); }
}

OMPFILE_TEST(uint64_double_and_string_defaults) {
  { ScopedEnv E("OMPFILE_TEST_U64", "18446744073709551615"); OMPFILE_CHECK_EQ(uint64Or("OMPFILE_TEST_U64", 1), 18446744073709551615ull); }
  { ScopedEnv E("OMPFILE_TEST_U64", "12 "); OMPFILE_CHECK_EQ(uint64Or("OMPFILE_TEST_U64", 1), 1u); }
  { ScopedEnv E("OMPFILE_TEST_U64", nullptr); OMPFILE_CHECK_EQ(uint64Or("OMPFILE_TEST_U64", 9), 9u); }
  { ScopedEnv E("OMPFILE_TEST_DBL", "1.5"); OMPFILE_CHECK_EQ(doubleOr("OMPFILE_TEST_DBL", 0.0), 1.5); }
  { ScopedEnv E("OMPFILE_TEST_DBL", "abc"); OMPFILE_CHECK_EQ(doubleOr("OMPFILE_TEST_DBL", 2.5), 2.5); }
  { ScopedEnv E("OMPFILE_TEST_STR", "value"); OMPFILE_CHECK_EQ(stringOr("OMPFILE_TEST_STR", "d"), std::string("value")); }
  { ScopedEnv E("OMPFILE_TEST_STR", ""); OMPFILE_CHECK_EQ(stringOr("OMPFILE_TEST_STR", "d"), std::string("d")); }
  { ScopedEnv E("OMPFILE_TEST_STR", nullptr); OMPFILE_CHECK(stringOr("OMPFILE_TEST_STR", nullptr).empty()); }
}

OMPFILE_TEST(stage_write_mode_is_normalized_the_same_on_both_sides) {
  OMPFILE_CHECK_EQ(normalizeStageWriteMode("write-back"), std::string("write-back"));
  OMPFILE_CHECK_EQ(normalizeStageWriteMode("writeback"), std::string("write-back"));
  OMPFILE_CHECK_EQ(normalizeStageWriteMode("off"), std::string("off"));
  OMPFILE_CHECK_EQ(normalizeStageWriteMode("disabled"), std::string("off"));
  OMPFILE_CHECK_EQ(normalizeStageWriteMode("write-through"), std::string("write-through"));
  OMPFILE_CHECK_EQ(normalizeStageWriteMode("garbage"), std::string("write-through"));
  OMPFILE_CHECK_EQ(normalizeStageWriteMode(""), std::string("write-through"));
  OMPFILE_CHECK(isWriteBackMode("writeback"));
  OMPFILE_CHECK(isWriteBackMode("write-back"));
  OMPFILE_CHECK(!isWriteBackMode("write-through"));
  OMPFILE_CHECK(!isWriteBackMode(nullptr));
}

OMPFILE_TEST(mode_knobs_accept_only_remote_only_mpp) {
  {
    ScopedEnv B("LIBOMPFILE_BACKEND", nullptr);
    ScopedEnv O("LIBOMPFILE_MPP_OPEN", nullptr);
    ScopedEnv I("LIBOMPFILE_MPP_IO", nullptr);
    OMPFILE_CHECK(rejectedModeKnobs().empty());
  }
  {
    ScopedEnv B("LIBOMPFILE_BACKEND", "MPI");
    ScopedEnv O("LIBOMPFILE_MPP_OPEN", "1");
    ScopedEnv I("LIBOMPFILE_MPP_IO", "1");
    OMPFILE_CHECK(rejectedModeKnobs().empty());
  }
  {
    ScopedEnv B("LIBOMPFILE_BACKEND", "");
    ScopedEnv O("LIBOMPFILE_MPP_OPEN", "");
    ScopedEnv I("LIBOMPFILE_MPP_IO", nullptr);
    OMPFILE_CHECK(rejectedModeKnobs().empty());
  }
  {
    ScopedEnv B("LIBOMPFILE_BACKEND", "POSIX");
    ScopedEnv O("LIBOMPFILE_MPP_OPEN", "0");
    ScopedEnv I("LIBOMPFILE_MPP_IO", "1");
    const std::string R = rejectedModeKnobs();
    OMPFILE_CHECK(R.find("LIBOMPFILE_BACKEND='POSIX'") != std::string::npos);
    OMPFILE_CHECK(R.find("LIBOMPFILE_MPP_OPEN='0'") != std::string::npos);
    OMPFILE_CHECK(R.find("LIBOMPFILE_MPP_IO") == std::string::npos);
  }
  {
    ScopedEnv B("LIBOMPFILE_BACKEND", "IO_URING");
    ScopedEnv O("LIBOMPFILE_MPP_OPEN", nullptr);
    ScopedEnv I("LIBOMPFILE_MPP_IO", "0");
    OMPFILE_CHECK(!rejectedModeKnobs().empty());
  }
}

OMPFILE_TEST(removed_knobs_set_lists_only_present_names) {
  ScopedEnv A("LIBOMPFILE_ALLOWFALLBACK", "1");
  ScopedEnv C("LIBOMPFILE_MPI_COMM_SELF", nullptr);
  ScopedEnv T("LIBOMPFILE_ASYNC_TRACE", "");
  const std::string R = removedKnobsSet();
  OMPFILE_CHECK(R.find("LIBOMPFILE_ALLOWFALLBACK") != std::string::npos);
  OMPFILE_CHECK(R.find("LIBOMPFILE_MPI_COMM_SELF") == std::string::npos);
  OMPFILE_CHECK(R.find("LIBOMPFILE_ASYNC_TRACE") == std::string::npos);
}

OMPFILE_TEST(scheduler_defaults_to_headnode) {
  bool Unknown = true;
  { ScopedEnv E("LIBOMPFILE_SCHEDULER", nullptr); OMPFILE_CHECK(schedulerIsHeadnode(&Unknown)); OMPFILE_CHECK(!Unknown); }
  { ScopedEnv E("LIBOMPFILE_SCHEDULER", ""); OMPFILE_CHECK(schedulerIsHeadnode(&Unknown)); OMPFILE_CHECK(!Unknown); }
  { ScopedEnv E("LIBOMPFILE_SCHEDULER", "HEADNODE"); OMPFILE_CHECK(schedulerIsHeadnode(&Unknown)); OMPFILE_CHECK(!Unknown); }
  { ScopedEnv E("LIBOMPFILE_SCHEDULER", "headnode"); OMPFILE_CHECK(schedulerIsHeadnode(&Unknown)); OMPFILE_CHECK(!Unknown); }
  { ScopedEnv E("LIBOMPFILE_SCHEDULER", "LOCAL"); OMPFILE_CHECK(!schedulerIsHeadnode(&Unknown)); OMPFILE_CHECK(!Unknown); }
  { ScopedEnv E("LIBOMPFILE_SCHEDULER", "Local"); OMPFILE_CHECK(!schedulerIsHeadnode(&Unknown)); OMPFILE_CHECK(!Unknown); }
  { ScopedEnv E("LIBOMPFILE_SCHEDULER", "HEADNODEX"); OMPFILE_CHECK(schedulerIsHeadnode(&Unknown)); OMPFILE_CHECK(Unknown); }
  { ScopedEnv E("LIBOMPFILE_SCHEDULER", "LOC"); OMPFILE_CHECK(schedulerIsHeadnode()); }
}

OMPFILE_TEST(writeback_distribute_needs_writes_and_ranks) {
  const char *W = "LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_WRITES";
  const char *R = "LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_RANKS";
  bool Bad = true;
  { ScopedEnv EW(W, nullptr); ScopedEnv ER(R, "4"); OMPFILE_CHECK(!writebackDistributeActive(&Bad)); OMPFILE_CHECK(!Bad); OMPFILE_CHECK_EQ(writebackDistributeRankCount(), 4u); }
  { ScopedEnv EW(W, "1"); ScopedEnv ER(R, "4"); OMPFILE_CHECK(writebackDistributeActive(&Bad)); OMPFILE_CHECK(!Bad); }
  { ScopedEnv EW(W, "1"); ScopedEnv ER(R, "1"); OMPFILE_CHECK(!writebackDistributeActive(&Bad)); OMPFILE_CHECK(Bad); }
  { ScopedEnv EW(W, "1"); ScopedEnv ER(R, nullptr); OMPFILE_CHECK(!writebackDistributeActive(&Bad)); OMPFILE_CHECK(Bad); OMPFILE_CHECK_EQ(writebackDistributeRankCount(), 0u); }
  { ScopedEnv EW(W, "true"); ScopedEnv ER(R, "4"); OMPFILE_CHECK(!writebackDistributeActive(&Bad)); OMPFILE_CHECK(!Bad); }
}

namespace {
using ompfile::config::BatchPolicy;
using ompfile::config::ClientConfig;
using ompfile::config::HeadnodeConfig;
using ompfile::config::OpenMode;
using ompfile::config::ProxyConfig;

ClientConfig clientNow() { return ClientConfig::fromEnvironment(); }
ProxyConfig proxyNow() { return ProxyConfig::fromEnvironment(); }
HeadnodeConfig headnodeNow() { return HeadnodeConfig::fromEnvironment(); }
} // namespace

OMPFILE_TEST(raw_distinguishes_unset_from_empty) {
  std::string V = "x";
  { ScopedEnv E("OMPFILE_TEST_RAW", nullptr); OMPFILE_CHECK(!raw("OMPFILE_TEST_RAW", &V)); OMPFILE_CHECK(V.empty()); }
  { ScopedEnv E("OMPFILE_TEST_RAW", ""); OMPFILE_CHECK(raw("OMPFILE_TEST_RAW", &V)); OMPFILE_CHECK(V.empty()); }
  { ScopedEnv E("OMPFILE_TEST_RAW", "abc"); OMPFILE_CHECK(raw("OMPFILE_TEST_RAW", &V)); OMPFILE_CHECK(V == "abc"); }
}

OMPFILE_TEST(parse_uint64_strict_keeps_strtoull_leniency) {
  uint64_t V = 0;
  OMPFILE_CHECK(parseUint64Strict("12", &V) && V == 12u);
  OMPFILE_CHECK(parseUint64Strict(" 7", &V) && V == 7u);
  OMPFILE_CHECK(parseUint64Strict("-1", &V) && V == std::numeric_limits<uint64_t>::max());
  OMPFILE_CHECK(!parseUint64Strict("", &V));
  OMPFILE_CHECK(!parseUint64Strict("12x", &V));
  OMPFILE_CHECK(!parseUint64Strict("99999999999999999999", &V));
}

OMPFILE_TEST(set_and_not_zero_rule) {
  { ScopedEnv E("OMPFILE_TEST_SNZ", nullptr); OMPFILE_CHECK(setAndNotZeroOr("OMPFILE_TEST_SNZ", true)); OMPFILE_CHECK(!setAndNotZeroOr("OMPFILE_TEST_SNZ", false)); }
  { ScopedEnv E("OMPFILE_TEST_SNZ", ""); OMPFILE_CHECK(!setAndNotZeroOr("OMPFILE_TEST_SNZ", true)); }
  { ScopedEnv E("OMPFILE_TEST_SNZ", "0"); OMPFILE_CHECK(!setAndNotZeroOr("OMPFILE_TEST_SNZ", true)); }
  { ScopedEnv E("OMPFILE_TEST_SNZ", "false"); OMPFILE_CHECK(setAndNotZeroOr("OMPFILE_TEST_SNZ", false)); }
  { ScopedEnv E("OMPFILE_TEST_SNZ", "1"); OMPFILE_CHECK(setAndNotZeroOr("OMPFILE_TEST_SNZ", false)); }
}

OMPFILE_TEST(on_unless_zero_rule) {
  { ScopedEnv E("OMPFILE_TEST_OUZ", nullptr); OMPFILE_CHECK(onUnlessZero("OMPFILE_TEST_OUZ")); }
  { ScopedEnv E("OMPFILE_TEST_OUZ", ""); OMPFILE_CHECK(onUnlessZero("OMPFILE_TEST_OUZ")); }
  { ScopedEnv E("OMPFILE_TEST_OUZ", "0"); OMPFILE_CHECK(!onUnlessZero("OMPFILE_TEST_OUZ")); }
  { ScopedEnv E("OMPFILE_TEST_OUZ", "00"); OMPFILE_CHECK(onUnlessZero("OMPFILE_TEST_OUZ")); }
}

OMPFILE_TEST(positive_long_prefix_rule) {
  const char *N = "OMPFILE_TEST_PLP";
  { ScopedEnv E(N, nullptr); OMPFILE_CHECK_EQ(positiveLongPrefixOr(N, 2), 2L); }
  { ScopedEnv E(N, ""); OMPFILE_CHECK_EQ(positiveLongPrefixOr(N, 2), 2L); }
  { ScopedEnv E(N, "0"); OMPFILE_CHECK_EQ(positiveLongPrefixOr(N, 2), 2L); }
  { ScopedEnv E(N, "-3"); OMPFILE_CHECK_EQ(positiveLongPrefixOr(N, 2), 2L); }
  { ScopedEnv E(N, "abc"); OMPFILE_CHECK_EQ(positiveLongPrefixOr(N, 2), 2L); }
  { ScopedEnv E(N, "16"); OMPFILE_CHECK_EQ(positiveLongPrefixOr(N, 2), 16L); }
  { ScopedEnv E(N, "4abc"); OMPFILE_CHECK_EQ(positiveLongPrefixOr(N, 2), 4L); }
  { ScopedEnv E(N, " 5"); OMPFILE_CHECK_EQ(positiveLongPrefixOr(N, 2), 5L); }
}

OMPFILE_TEST(client_bool_knob_is_boolor_with_raw) {
  const char *N = "LIBOMPFILE_OPT_STATS";
  { ScopedEnv E(N, "1"); const ClientConfig C = clientNow(); OMPFILE_CHECK(C.OptStats.Value && !C.OptStats.Malformed); }
  { ScopedEnv E(N, "0"); const ClientConfig C = clientNow(); OMPFILE_CHECK(!C.OptStats.Value && !C.OptStats.Malformed); }
  { ScopedEnv E(N, "yes"); const ClientConfig C = clientNow(); OMPFILE_CHECK(!C.OptStats.Value && C.OptStats.Malformed && C.OptStats.Raw == "yes"); }
  { ScopedEnv E(N, ""); const ClientConfig C = clientNow(); OMPFILE_CHECK(!C.OptStats.Value && C.OptStats.Malformed); }
  { ScopedEnv E(N, nullptr); const ClientConfig C = clientNow(); OMPFILE_CHECK(!C.OptStats.Value && !C.OptStats.Malformed); }
}

OMPFILE_TEST(client_u64_knob_rule) {
  const char *N = "LIBOMPFILE_OPT_TWO_PHASE_WINDOW_US";
  { ScopedEnv E(N, nullptr); OMPFILE_CHECK(!clientNow().TwoPhaseWindowUs.Set); }
  { ScopedEnv E(N, ""); const ClientConfig C = clientNow(); OMPFILE_CHECK(C.TwoPhaseWindowUs.Set && !C.TwoPhaseWindowUs.Valid); OMPFILE_CHECK_EQ(uint64Or(N, 9), 9u); }
  { ScopedEnv E(N, "250"); const ClientConfig C = clientNow(); OMPFILE_CHECK(C.TwoPhaseWindowUs.Valid && C.TwoPhaseWindowUs.Value == 250u); }
  { ScopedEnv E(N, "250us"); const ClientConfig C = clientNow(); OMPFILE_CHECK(C.TwoPhaseWindowUs.Set && !C.TwoPhaseWindowUs.Valid && C.TwoPhaseWindowUs.Raw == "250us"); }
}

OMPFILE_TEST(client_policy_knob_rule) {
  const char *Names[] = {"LIBOMPFILE_OPT_TWO_PHASE", "LIBOMPFILE_OPT_WRITE_BATCH"};
  for (const char *N : Names) {
    const bool IsTwoPhase = N == Names[0];
    auto policy = [&](const char *Value) {
      ScopedEnv E(N, Value);
      const ClientConfig C = clientNow();
      return IsTwoPhase ? C.TwoPhase : C.WriteBatch;
    };
    OMPFILE_CHECK(policy(nullptr).Policy == BatchPolicy::Auto);
    OMPFILE_CHECK(policy("").Policy == BatchPolicy::Auto);
    for (const char *V : {"1", "enabled", "ENABLED"}) OMPFILE_CHECK(policy(V).Policy == BatchPolicy::Enabled);
    for (const char *V : {"0", "disabled", "DISABLED"}) OMPFILE_CHECK(policy(V).Policy == BatchPolicy::Disabled);
    for (const char *V : {"auto", "AUTO"}) OMPFILE_CHECK(policy(V).Policy == BatchPolicy::Auto && !policy(V).Malformed);
    const ompfile::config::PolicyKnob Bad = policy("Enabled");
    OMPFILE_CHECK(Bad.Policy == BatchPolicy::Disabled && Bad.Malformed && Bad.Raw == "Enabled");
  }
  // Each knob is parsed on its own.
  ScopedEnv E1("LIBOMPFILE_OPT_TWO_PHASE", "0");
  ScopedEnv E2("LIBOMPFILE_OPT_WRITE_BATCH", "1");
  const ClientConfig C = clientNow();
  OMPFILE_CHECK(C.TwoPhase.Policy == BatchPolicy::Disabled && C.WriteBatch.Policy == BatchPolicy::Enabled);
}

OMPFILE_TEST(client_open_mode_rule) {
  const char *N = "LIBOMPFILE_OPEN_MODE";
  for (const char *V : {static_cast<const char *>(nullptr), "", "readwrite"}) {
    ScopedEnv E(N, V); const ClientConfig C = clientNow();
    OMPFILE_CHECK(C.Open == OpenMode::ReadWrite && !C.OpenModeMalformed);
  }
  for (const char *V : {"readonly", "read-only", "read"}) {
    ScopedEnv E(N, V); OMPFILE_CHECK(clientNow().Open == OpenMode::ReadOnly);
  }
  ScopedEnv E(N, "RO"); const ClientConfig C = clientNow();
  OMPFILE_CHECK(C.Open == OpenMode::ReadWrite && C.OpenModeMalformed && C.OpenModeRaw == "RO");
}

OMPFILE_TEST(client_stage_predicates) {
  for (const char *V : {static_cast<const char *>(nullptr), "", "off"}) {
    ScopedEnv E("LIBOMPFILE_STAGE_MODE", V); OMPFILE_CHECK(clientNow().Shared.stageOff());
  }
  { ScopedEnv E("LIBOMPFILE_STAGE_MODE", "readthrough"); const ClientConfig C = clientNow(); OMPFILE_CHECK(C.Shared.stageReadthrough() && !C.Shared.stageOff()); }
  { ScopedEnv E("LIBOMPFILE_STAGE_MODE", "readthroughx"); const ClientConfig C = clientNow(); OMPFILE_CHECK(!C.Shared.stageReadthrough() && !C.Shared.stageOff()); }
  { ScopedEnv E("LIBOMPFILE_STAGE_WRITE_MODE", "writeback"); OMPFILE_CHECK(clientNow().Shared.stageWriteBack()); }
  { ScopedEnv E("LIBOMPFILE_STAGE_WRITE_MODE", nullptr); OMPFILE_CHECK(!clientNow().Shared.stageWriteBack()); }
  { ScopedEnv E("LIBOMPFILE_STAGE_FRESHNESS_GUARD", "0"); OMPFILE_CHECK(!clientNow().Shared.StageFreshnessGuard); }
  for (const char *V : {static_cast<const char *>(nullptr), "", "false"}) {
    ScopedEnv E("LIBOMPFILE_STAGE_FRESHNESS_GUARD", V); OMPFILE_CHECK(clientNow().Shared.StageFreshnessGuard);
  }
}

OMPFILE_TEST(client_tile_freshness_write_mode) {
  const char *T = "LIBOMPFILE_TILE_FRESHNESS_WRITE_MODE";
  const char *S = "LIBOMPFILE_STAGE_WRITE_MODE";
  { ScopedEnv E1(T, "write-back"); ScopedEnv E2(S, nullptr); OMPFILE_CHECK(!clientNow().TileFreshnessWriteThrough); }
  { ScopedEnv E1(T, "write-through"); ScopedEnv E2(S, "write-back"); OMPFILE_CHECK(clientNow().TileFreshnessWriteThrough); }
  { ScopedEnv E1(T, nullptr); ScopedEnv E2(S, "write-back"); OMPFILE_CHECK(!clientNow().TileFreshnessWriteThrough); }
  { ScopedEnv E1(T, "garbage"); ScopedEnv E2(S, nullptr); OMPFILE_CHECK(clientNow().TileFreshnessWriteThrough); }
}

OMPFILE_TEST(client_dirty_owner_and_async_knobs) {
  { ScopedEnv E("LIBOMPFILE_OPT_DIRTY_OWNER_BATCH", "1"); OMPFILE_CHECK(clientNow().DirtyOwnerBatch); }
  { ScopedEnv E("LIBOMPFILE_OPT_DIRTY_OWNER_BATCH", "true"); OMPFILE_CHECK(!clientNow().DirtyOwnerBatch); }
  const char *Segs = "LIBOMPFILE_OPT_DIRTY_OWNER_BATCH_MAX_SEGS";
  { ScopedEnv E(Segs, nullptr); OMPFILE_CHECK_EQ(clientNow().DirtyOwnerBatchMaxSegs, 16u); }
  { ScopedEnv E(Segs, "0"); OMPFILE_CHECK_EQ(clientNow().DirtyOwnerBatchMaxSegs, 16u); }
  { ScopedEnv E(Segs, "5"); OMPFILE_CHECK_EQ(clientNow().DirtyOwnerBatchMaxSegs, 5u); }
  { ScopedEnv E(Segs, "5x"); OMPFILE_CHECK_EQ(clientNow().DirtyOwnerBatchMaxSegs, 16u); }
  { ScopedEnv E("LIBOMPFILE_OPT_DIRTY_OWNER_BATCH_WINDOW_US", "100"); OMPFILE_CHECK_EQ(clientNow().DirtyOwnerBatchWindowUs, 100u); }
  { ScopedEnv E("LIBOMPFILE_OPT_DIRTY_OWNER_BATCH_WINDOW_US", "x"); OMPFILE_CHECK_EQ(clientNow().DirtyOwnerBatchWindowUs, 0u); }
  { ScopedEnv E("LIBOMPFILE_ASYNC_READ_WAIT", "range"); OMPFILE_CHECK(clientNow().AsyncReadWaitRange); }
  { ScopedEnv E("LIBOMPFILE_ASYNC_READ_WAIT", "Range"); OMPFILE_CHECK(!clientNow().AsyncReadWaitRange); }
  { ScopedEnv E("LIBOMPFILE_ASYNC_QUEUE_DEPTH", "4abc"); OMPFILE_CHECK_EQ(clientNow().AsyncQueueDepth, size_t{4}); }
  { ScopedEnv E("LIBOMPFILE_ASYNC_QUEUE_DEPTH", nullptr); OMPFILE_CHECK_EQ(clientNow().AsyncQueueDepth, size_t{2}); }
  { ScopedEnv E("LIBOMPFILE_OPT_LOCAL_DISJOINT_EXPRESS", "1"); OMPFILE_CHECK(clientNow().LocalDisjointExpress); }
  { ScopedEnv E("LIBOMPFILE_IO_TOKENS", "4abc"); const ClientConfig C = clientNow(); OMPFILE_CHECK(C.IoTokensSet && C.IoTokensRaw == "4abc"); }
  { ScopedEnv E("LIBOMPFILE_IO_TOKENS", nullptr); OMPFILE_CHECK(!clientNow().IoTokensSet); }
  { ScopedEnv E("LIBOMPFILE_MPP_PING", "1"); OMPFILE_CHECK(clientNow().MppPing); }
  { ScopedEnv E("LIBOMPFILE_MPP_PING", "true"); OMPFILE_CHECK(!clientNow().MppPing); }
}

OMPFILE_TEST(client_distribute_raw_ranks) {
  const char *W = "LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_WRITES";
  const char *R = "LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_RANKS";
  { ScopedEnv E1(W, "1"); ScopedEnv E2(R, ""); const ClientConfig C = clientNow();
    OMPFILE_CHECK(C.Shared.WritebackDistributeMisconfigured && C.WritebackDistributeRanksSet && C.WritebackDistributeRanksRaw.empty()); }
  { ScopedEnv E1(W, "1"); ScopedEnv E2(R, nullptr); OMPFILE_CHECK(!clientNow().WritebackDistributeRanksSet); }
  { ScopedEnv E1(W, "1"); ScopedEnv E2(R, "3"); const ClientConfig C = clientNow();
    OMPFILE_CHECK(C.Shared.WritebackDistributeActive && C.Shared.WritebackDistributeRanks == 3u); }
}

OMPFILE_TEST(headnode_config_rules) {
  const char *F = "LIBOMPFILE_FRESHNESS_TRACE";
  for (const char *V : {static_cast<const char *>(nullptr), "", "0"}) { ScopedEnv E(F, V); OMPFILE_CHECK(!headnodeNow().FreshnessTrace); }
  for (const char *V : {"false", "1"}) { ScopedEnv E(F, V); OMPFILE_CHECK(headnodeNow().FreshnessTrace); }
  { ScopedEnv E("LIBOMPFILE_MPP_OPEN_SPREAD", "2"); OMPFILE_CHECK(headnodeNow().SpreadSamePathOpens); }
  for (const char *V : {"true", ""}) { ScopedEnv E("LIBOMPFILE_MPP_OPEN_SPREAD", V); OMPFILE_CHECK(!headnodeNow().SpreadSamePathOpens); }
  { ScopedEnv E("LIBOMPFILE_SCHED_STATS_REPORT_EVERY", "0"); OMPFILE_CHECK_EQ(headnodeNow().BatchStatsReportEvery, 1u); }
  { ScopedEnv E("LIBOMPFILE_SCHED_STATS_REPORT_EVERY", nullptr); OMPFILE_CHECK_EQ(headnodeNow().BatchStatsReportEvery, 128u); }
  { ScopedEnv E("LIBOMPFILE_SCHED_MAX_AFFINITY_LOAD_SKEW", nullptr); OMPFILE_CHECK_EQ(headnodeNow().MaxAffinityLoadSkew, std::numeric_limits<uint64_t>::max()); }
  { ScopedEnv E("LIBOMPFILE_SCHED_MAX_AFFINITY_LOAD_SKEW", "3"); OMPFILE_CHECK_EQ(headnodeNow().MaxAffinityLoadSkew, 3u); }
  { ScopedEnv E("LIBOMPFILE_SCHED_USE_PLANNED_LOAD_FOR_SKEW", "1"); OMPFILE_CHECK(headnodeNow().UsePlannedLoadForSkew); }
}

OMPFILE_TEST(proxy_config_rules) {
  const char *T = "LIBOMPFILE_STAGE_TILE_BYTES";
  const char *Nb = "OMPFILE_CHOLESKY_NB";
  { ScopedEnv E1(T, "4096"); ScopedEnv E2(Nb, nullptr); OMPFILE_CHECK_EQ(proxyNow().StageTileBytes, 4096u); }
  { ScopedEnv E1(T, nullptr); ScopedEnv E2(Nb, "2"); OMPFILE_CHECK_EQ(proxyNow().StageTileBytes, 16u); }
  { ScopedEnv E1(T, "0"); ScopedEnv E2(Nb, "3"); OMPFILE_CHECK_EQ(proxyNow().StageTileBytes, 36u); }
  { ScopedEnv E1(T, nullptr); ScopedEnv E2(Nb, nullptr); OMPFILE_CHECK_EQ(proxyNow().StageTileBytes, 0u); }
  { ScopedEnv E("LIBOMPFILE_OPEN_FAILURE_DIAG", "0"); OMPFILE_CHECK(!proxyNow().OpenFailureDiag); }
  { ScopedEnv E("LIBOMPFILE_OPEN_FAILURE_DIAG", nullptr); OMPFILE_CHECK(proxyNow().OpenFailureDiag); }
  { ScopedEnv E("LIBOMPFILE_OPT_COHERENT_READ_REFRESH", nullptr); OMPFILE_CHECK(proxyNow().CoherentReadRefresh); }
  { ScopedEnv E("LIBOMPFILE_OPT_LOCAL_DISJOINT_STRIPE_MIN_BYTES", nullptr); OMPFILE_CHECK_EQ(proxyNow().LocalDisjointStripeMinBytes, 16777216u); }
  { ScopedEnv E("LIBOMPFILE_STAGE_WRITE_MODE", "writeback"); OMPFILE_CHECK(proxyNow().Shared.StageWriteMode == "write-back"); }
}
