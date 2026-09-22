#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <dlfcn.h>
#include <functional>
#include <thread>
#include <unistd.h>

// -DOMP_IO_DEBUG should enable the logging
#ifdef OMP_IO_DEBUG
constexpr bool kOmpIoDebug = true;
#else
constexpr bool kOmpIoDebug = false;
#endif

inline bool io_trace_enabled() {
  if (!kOmpIoDebug)
    return false;
  static const bool enabled = []() {
    const char *env = std::getenv("LIBOMPFILE_DEBUG_TRACE");
    return env && env[0] == '1' && env[1] == '\0';
  }();
  return enabled;
}

inline unsigned long io_tid_hash() {
  return static_cast<unsigned long>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

inline uint64_t io_next_trace_seq() {
  static std::atomic<uint64_t> seq{1};
  return seq.fetch_add(1, std::memory_order_relaxed);
}

// LIBOMPFILE_DEBUG=0 silences io_log at run time. Unset keeps every line
// the lanes have always seen (the compile-time OMP_IO_DEBUG default is on).
inline bool io_log_enabled() {
  static const bool enabled = []() {
    const char *env = std::getenv("LIBOMPFILE_DEBUG");
    return !(env && env[0] == '0' && env[1] == '\0');
  }();
  return enabled;
}

inline void io_log(const char *fmt, ...) {
  if (!kOmpIoDebug || !io_log_enabled()) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  fprintf(stderr, "[omp-io] ");
  vfprintf(stderr, fmt, args);
  va_end(args);
}

// Never gated: the stats lines the lanes parse ("Async IO stats",
// "Two-phase stats") go through here with the same "[omp-io] " prefix, so
// LIBOMPFILE_DEBUG=0 cannot silence a test contract.
inline void io_report(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  fprintf(stderr, "[omp-io] ");
  vfprintf(stderr, fmt, args);
  va_end(args);
}

inline void io_trace(const char *fmt, ...) {
  if (!io_trace_enabled())
    return;

  va_list args;
  va_start(args, fmt);
  fprintf(stderr, "[omp-io][trace][seq=%llu][pid=%d][tid=%lu] ",
          static_cast<unsigned long long>(io_next_trace_seq()), getpid(),
          io_tid_hash());
  vfprintf(stderr, fmt, args);
  va_end(args);
}

inline void io_trace_symbol_owner(const char *label, const void *symbol) {
  if (!io_trace_enabled())
    return;

  if (!symbol) {
    io_trace("symbol owner label=%s symbol=(null)\n",
             label ? label : "(unknown)");
    return;
  }

  Dl_info info{};
  if (dladdr(symbol, &info) != 0 && info.dli_fname) {
    io_trace("symbol owner label=%s symbol=%p object=%s base=%p\n",
             label ? label : "(unknown)", symbol, info.dli_fname,
             info.dli_fbase);
    return;
  }

  io_trace("symbol owner label=%s symbol=%p object=(unknown)\n",
           label ? label : "(unknown)", symbol);
}
