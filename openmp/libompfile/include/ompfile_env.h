#ifndef OMPFILE_ENV_H
#define OMPFILE_ENV_H

// One set of environment-knob readers for the client library (libompfile)
// and the proxy (ProxyDevice.cpp). Both sides used to carry their own copies
// with different acceptance rules; the one documented difference that
// mattered (the proxy accepting "writeback" for LIBOMPFILE_STAGE_WRITE_MODE
// while the client required "write-back") lives here once, in
// normalizeStageWriteMode. Knob semantics are documented in
// docs/runtime-configuration.md.

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

} // namespace env
} // namespace ompfile

#endif // OMPFILE_ENV_H
