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
