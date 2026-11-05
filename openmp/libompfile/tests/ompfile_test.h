#ifndef OMPFILE_TEST_H
#define OMPFILE_TEST_H

// A 40-line test harness: no gtest, because llvm_gtest is only built for the
// EXCLUDE_FROM_ALL unittest targets and this binary must be part of every
// runtime build. OMPFILE_TEST(name) registers a case; OMPFILE_CHECK(cond)
// records a failure with file and line and keeps going.

#include <cstdio>
#include <vector>

namespace ompfile_test {

struct Case {
  const char *Name;
  void (*Fn)();
};

inline std::vector<Case> &registry() {
  static std::vector<Case> Cases;
  return Cases;
}

inline int &failures() {
  static int Failures = 0;
  return Failures;
}

struct Registrar {
  Registrar(const char *Name, void (*Fn)()) { registry().push_back({Name, Fn}); }
};

} // namespace ompfile_test

#define OMPFILE_TEST(name)                                                     \
  static void ompfile_test_##name();                                           \
  static ompfile_test::Registrar ompfile_test_registrar_##name(                \
      #name, &ompfile_test_##name);                                            \
  static void ompfile_test_##name()

#define OMPFILE_CHECK(cond)                                                    \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
      ++ompfile_test::failures();                                              \
    }                                                                          \
  } while (0)

#define OMPFILE_CHECK_EQ(a, b) OMPFILE_CHECK((a) == (b))

#endif // OMPFILE_TEST_H
