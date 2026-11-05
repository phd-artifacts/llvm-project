#include "ompfile_test.h"

#include <cstdio>

int main() {
  int Failed = 0;
  for (const ompfile_test::Case &C : ompfile_test::registry()) {
    const int Before = ompfile_test::failures();
    C.Fn();
    const bool Ok = ompfile_test::failures() == Before;
    std::fprintf(stderr, "[ompfile-unit] %s: %s\n", C.Name, Ok ? "PASS" : "FAIL");
    if (!Ok)
      ++Failed;
  }
  std::fprintf(stderr, "[ompfile-unit] summary status=%s cases=%zu failed=%d checks_failed=%d\n",
               Failed == 0 ? "PASS" : "FAIL", ompfile_test::registry().size(),
               Failed, ompfile_test::failures());
  return Failed == 0 ? 0 : 1;
}
