#pragma once

#include <cstdio>
#include <cstdarg>

// -DOMP_IO_DEBUG should enable the logging
#ifdef OMP_IO_DEBUG
constexpr bool kOmpIoDebug = true;
#else
constexpr bool kOmpIoDebug = false;
#endif

inline void io_log(const char *fmt, ...) {
  if (!kOmpIoDebug) {
    return; // will be optimized out by the compiler if not enabled
  }
  va_list args;
  va_start(args, fmt);
  fprintf(stderr, "[omp-io] ");
  vfprintf(stderr, fmt, args);
  va_end(args);
}
