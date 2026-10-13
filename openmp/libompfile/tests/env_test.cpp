// The environment-knob readers in ompfile_env.h, driven through setenv so the
// acceptance rules the client and the proxy now share are pinned.

#include "ompfile_env.h"
#include "ompfile_test.h"

#include <cstdlib>
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
