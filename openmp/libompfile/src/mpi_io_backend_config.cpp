#include "mpi_io_backend.h"
#include "debug_log.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>

bool MPIIOBackend::parseBoolEnv(const char *name, bool default_value) {
  const char *env = std::getenv(name);
  if (!env)
    return default_value;

  if (env[0] == '1' && env[1] == '\0')
    return true;
  if (env[0] == '0' && env[1] == '\0')
    return false;

  io_log("Invalid boolean value for %s='%s'; using default=%d\n", name, env,
         static_cast<int>(default_value));
  return default_value;
}

MPIIOBackend::TwoPhasePolicy
MPIIOBackend::parseTwoPhasePolicy(const char *env_value) {
  if (!env_value || env_value[0] == '\0')
    return TwoPhasePolicy::Auto;

  if (std::strcmp(env_value, "1") == 0)
    return TwoPhasePolicy::Enabled;
  if (std::strcmp(env_value, "0") == 0)
    return TwoPhasePolicy::Disabled;
  if (std::strcmp(env_value, "enabled") == 0 ||
      std::strcmp(env_value, "ENABLED") == 0) {
    return TwoPhasePolicy::Enabled;
  }
  if (std::strcmp(env_value, "disabled") == 0 ||
      std::strcmp(env_value, "DISABLED") == 0) {
    return TwoPhasePolicy::Disabled;
  }
  if (std::strcmp(env_value, "auto") == 0 ||
      std::strcmp(env_value, "AUTO") == 0) {
    return TwoPhasePolicy::Auto;
  }

  io_log("Invalid LIBOMPFILE_OPT_TWO_PHASE='%s'; using disabled policy.\n",
         env_value);
  return TwoPhasePolicy::Disabled;
}

MPIIOBackend::ForecastMode
MPIIOBackend::parseForecastMode(const char *env_value) {
  if (!env_value || env_value[0] == '\0' || std::strcmp(env_value, "off") == 0 ||
      std::strcmp(env_value, "OFF") == 0 || std::strcmp(env_value, "0") == 0) {
    return ForecastMode::Off;
  }

  if (std::strcmp(env_value, "cholesky-oracle") == 0)
    return ForecastMode::CholeskyOracle;

  io_log("Invalid LIBOMPFILE_IO_FORECAST='%s'; using off mode.\n", env_value);
  return ForecastMode::Off;
}

MPIIOBackend::StageAffinityMode
MPIIOBackend::parseStageAffinityMode(const char *env_value) {
  if (!env_value || env_value[0] == '\0' || std::strcmp(env_value, "off") == 0)
    return StageAffinityMode::Off;
  if (std::strcmp(env_value, "cholesky-oracle") == 0)
    return StageAffinityMode::CholeskyOracle;
  io_log("Invalid LIBOMPFILE_IO_STAGE_AFFINITY='%s'; using off mode.\n",
         env_value);
  return StageAffinityMode::Off;
}

const char *
MPIIOBackend::twoPhasePolicyToString(TwoPhasePolicy policy) {
  switch (policy) {
  case TwoPhasePolicy::Disabled:
    return "disabled";
  case TwoPhasePolicy::Enabled:
    return "enabled";
  case TwoPhasePolicy::Auto:
    return "auto";
  }
  return "unknown";
}

const char *MPIIOBackend::forecastModeToString(ForecastMode mode) {
  switch (mode) {
  case ForecastMode::Off:
    return "off";
  case ForecastMode::CholeskyOracle:
    return "cholesky-oracle";
  }
  return "unknown";
}

const char *MPIIOBackend::stageAffinityModeToString(StageAffinityMode mode) {
  switch (mode) {
  case StageAffinityMode::Off:
    return "off";
  case StageAffinityMode::CholeskyOracle:
    return "cholesky-oracle";
  }
  return "off";
}

uint64_t MPIIOBackend::parseUint64Env(const char *name, uint64_t default_value) {
  const char *env = std::getenv(name);
  if (!env)
    return default_value;

  errno = 0;
  char *end_ptr = nullptr;
  const unsigned long long value = std::strtoull(env, &end_ptr, 10);
  if (errno != 0 || end_ptr == env || (end_ptr && *end_ptr != '\0')) {
    io_log("Invalid integer value for %s='%s'; using default=%llu\n", name, env,
           static_cast<unsigned long long>(default_value));
    return default_value;
  }

  return static_cast<uint64_t>(value);
}
