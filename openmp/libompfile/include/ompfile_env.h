#ifndef OMPFILE_ENV_H
#define OMPFILE_ENV_H

// One set of environment-knob readers for the client library (libompfile)
// and the proxy (ProxyDevice.cpp). Both sides used to carry their own copies
// with different acceptance rules; the one documented difference that
// mattered (the proxy accepting "writeback" for LIBOMPFILE_STAGE_WRITE_MODE
// while the client required "write-back") lives here once, in
// normalizeStageWriteMode. Knob semantics are documented in
// docs/runtime-configuration.md.

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace ompfile {
namespace env {

// Exactly "1" enables. Unset, empty, "0", "true", "yes" all read as false.
inline bool flag(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] == '1' && value[1] == '\0';
}

// "1" -> true, "0" -> false, unset -> the default. Anything else (including
// an empty value) is malformed: the default is returned and *malformed is
// set, so a caller that wants to log it can; the proxy stays silent.
inline bool boolOr(const char *name, bool default_value,
                   bool *malformed = nullptr) {
  if (malformed)
    *malformed = false;
  const char *value = std::getenv(name);
  if (!value)
    return default_value;
  if (value[0] == '1' && value[1] == '\0')
    return true;
  if (value[0] == '0' && value[1] == '\0')
    return false;
  if (malformed)
    *malformed = true;
  return default_value;
}

inline bool equals(const char *name, const char *expected) {
  const char *value = std::getenv(name);
  return value && std::strcmp(value, expected) == 0;
}

// Strictly positive decimal that fits an unsigned; the default (0 unless
// given) for unset, empty, garbage, zero, a trailing suffix or overflow.
inline unsigned positiveOr(const char *name, unsigned default_value = 0) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return default_value;
  char *end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno != 0 || end == value || (end && *end != '\0') || parsed == 0 ||
      parsed > static_cast<unsigned long>(std::numeric_limits<unsigned>::max()))
    return default_value;
  return static_cast<unsigned>(parsed);
}

// Whole decimal; the default for unset, empty, garbage or a trailing suffix.
inline uint64_t uint64Or(const char *name, uint64_t default_value) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return default_value;
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || (end && *end != '\0'))
    return default_value;
  return static_cast<uint64_t>(parsed);
}

inline double doubleOr(const char *name, double default_value) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return default_value;
  char *end = nullptr;
  errno = 0;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || (end && *end != '\0'))
    return default_value;
  return parsed;
}

inline std::string stringOr(const char *name, const char *default_value) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return default_value ? std::string(default_value) : std::string();
  return value;
}

// Stage write policy, normalized to "write-through", "write-back" or "off".
// Unknown values fall back to write-through so a misspelled knob never
// silently disables staging correctness.
inline std::string normalizeStageWriteMode(const std::string &raw) {
  if (raw == "write-back" || raw == "writeback")
    return "write-back";
  if (raw == "off" || raw == "disabled")
    return "off";
  return "write-through";
}

// True when a stage write-mode value (typically getenv of
// LIBOMPFILE_STAGE_WRITE_MODE) selects write-back.
inline bool isWriteBackMode(const char *value) {
  return value && normalizeStageWriteMode(value) == "write-back";
}

// ASCII case-insensitive string equality; both pointers must be non-null.
inline bool equalsIgnoreAsciiCase(const char *a, const char *b) {
  assert(a && b && "equalsIgnoreAsciiCase needs two strings");
  for (;; ++a, ++b) {
    char ca = *a;
    char cb = *b;
    if (ca >= 'a' && ca <= 'z')
      ca = static_cast<char>(ca - 'a' + 'A');
    if (cb >= 'a' && cb <= 'z')
      cb = static_cast<char>(cb - 'a' + 'A');
    if (ca != cb)
      return false;
    if (ca == '\0')
      return true;
  }
}

// LIBOMPFILE_SCHEDULER, the one reader for libompfile and the proxy. HEADNODE
// is the default (Oct 2026): remote-only MPP is the only mode and an
// origin-side open under LOCAL fails. Unset, empty or "HEADNODE" -> true;
// "LOCAL" -> false (still selectable explicitly); both compared ignoring ASCII
// case. Any other value reads as the default (true) and sets *unknown, so a
// caller that wants to log it can.
inline bool schedulerIsHeadnode(bool *unknown = nullptr) {
  if (unknown)
    *unknown = false;
  const char *value = std::getenv("LIBOMPFILE_SCHEDULER");
  if (!value || value[0] == '\0')
    return true;
  if (equalsIgnoreAsciiCase(value, "HEADNODE"))
    return true;
  if (equalsIgnoreAsciiCase(value, "LOCAL"))
    return false;
  if (unknown)
    *unknown = true;
  return true;
}

// LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_RANKS as a strictly positive integer,
// else 0. The one rank count for distributed write-back routing: libompfile's
// read and write paths and the proxy all read it here, and the application's
// Cholesky worker-node / target-region knobs are not read.
inline unsigned writebackDistributeRankCount() {
  return positiveOr("LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_RANKS");
}

// Distributed write-back routing, the one predicate for libompfile and the
// proxy: LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_WRITES=1 and a rank count above
// 1. WRITES=1 without such a count is a configuration error: routing stays
// off and *misconfigured is set, so libompfile's backend init can print its
// one error line; every other caller stays silent. Not cached: it reads the
// environment on every call.
inline bool writebackDistributeActive(bool *misconfigured = nullptr) {
  if (misconfigured)
    *misconfigured = false;
  if (!flag("LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_WRITES"))
    return false;
  if (writebackDistributeRankCount() > 1)
    return true;
  if (misconfigured)
    *misconfigured = true;
  return false;
}

// Remote-only MPP is libompfile's only mode (Oct 2026). These knobs used to
// select another backend or mode; they are read only so that a request for a
// removed mode fails loudly. Unset, empty, or the accepted value passes.
struct ModeKnob {
  const char *Name;
  const char *Accepted;
};
inline constexpr ModeKnob kModeKnobs[] = {
    {"LIBOMPFILE_BACKEND", "MPI"},
    {"LIBOMPFILE_MPP_OPEN", "1"},
    {"LIBOMPFILE_MPP_IO", "1"},
};

// NAME='value' for every mode knob set to anything else, space-joined; empty
// when the environment is compatible.
inline std::string rejectedModeKnobs() {
  std::string out;
  for (const ModeKnob &K : kModeKnobs) {
    const char *V = std::getenv(K.Name);
    if (!V || V[0] == '\0' || std::strcmp(V, K.Accepted) == 0)
      continue;
    if (!out.empty())
      out += " ";
    out += K.Name;
    out += "='";
    out += V;
    out += "'";
  }
  return out;
}

// Knobs removed by the Oct 2026 cleanup: THE one list, mirrored by
// docs/runtime-configuration.md, "Removed knobs (Oct 2026)". libompfile
// prints one `removed knob ignored: ...` io_report line at init naming every
// entry set to a non-empty value; setting one changes nothing else.
inline constexpr const char *kRemovedKnobs[] = {
    "LIBOMPFILE_MPI_COMM_SELF",
    "LIBOMPFILE_ALLOWFALLBACK",
    "LIBOMPFILE_MPP_SPLIT_PHASE_WRITES",
    "LIBOMPFILE_MPP_ASYNC_PWRITE_SEND",
    "LIBOMPFILE_ASYNC_WRITE_WORKERS",
    "LIBOMPFILE_IO_FORECAST",
    "LIBOMPFILE_IO_STAGE_AFFINITY",
    "LIBOMPFILE_OPT_DIRTY_OWNER_READAHEAD_BYTES",
    "LIBOMPFILE_OPT_STAGE_GLOBAL_INVALIDATION",
    "LIBOMPFILE_OPT_DIRTY_CLEAN_PFS_REBALANCE",
    "LIBOMPFILE_OPT_DIRTY_CLEAN_PFS_REBALANCE_ALLOW_CLEAN_PFS",
    "LIBOMPFILE_OPT_REMOTE_DIRTY_OWNER_PFS_REBALANCE",
    "LIBOMPFILE_ASYNC_TRACE",
    "LIBOMPFILE_ASYNC_DISABLE",
};

// Names from kRemovedKnobs set to a non-empty value, joined by ", "; empty
// when none is.
inline std::string removedKnobsSet() {
  std::string out;
  for (const char *N : kRemovedKnobs) {
    const char *V = std::getenv(N);
    if (!V || V[0] == '\0')
      continue;
    if (!out.empty())
      out += ", ";
    out += N;
  }
  return out;
}

} // namespace env
} // namespace ompfile

#endif // OMPFILE_ENV_H
