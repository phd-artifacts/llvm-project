#include "mpp_backend.h"
#include "debug_log.h"
#include "ompfile_config.h"

bool MppBackend::resolveBoolKnob(const ompfile::config::BoolKnob &knob) {
  if (knob.Malformed)
    io_log("Invalid boolean value for %s='%s'; using default=%d\n", knob.Name,
           knob.Raw.c_str(), static_cast<int>(knob.Default));
  return knob.Value;
}

uint64_t MppBackend::resolveU64Knob(const ompfile::config::U64Knob &knob,
                                    uint64_t default_value) {
  if (!knob.Set)
    return default_value;
  if (!knob.Valid) {
    io_log("Invalid integer value for %s='%s'; using default=%llu\n", knob.Name,
           knob.Raw.c_str(), static_cast<unsigned long long>(default_value));
    return default_value;
  }
  return knob.Value;
}

MppBackend::TwoPhasePolicy
MppBackend::resolvePolicyKnob(const ompfile::config::PolicyKnob &knob) {
  if (knob.Malformed)
    io_log("Invalid LIBOMPFILE_OPT_TWO_PHASE='%s'; using disabled policy.\n",
           knob.Raw.c_str());
  return knob.Policy;
}

const char *
MppBackend::twoPhasePolicyToString(TwoPhasePolicy policy) {
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

ompfile::route::DistributeConfig
MppBackend::snapshotDistributeConfig() const {
  ompfile::route::DistributeConfig distribute{};
  distribute.Active = config_.Shared.WritebackDistributeActive;
  if (distribute.Active)
    distribute.Ranks = config_.Shared.WritebackDistributeRanks;
  return distribute;
}
