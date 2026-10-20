#ifndef OMPFILE_CONFIG_H
#define OMPFILE_CONFIG_H

// Configuration snapshots: every LIBOMPFILE_* knob a consumer acts on, parsed
// once by the rules in ompfile_env.h.
//  - ClientConfig: when libompfile's client context is created (the first
//    omp_file_* call in a process); copied into the backend.
//  - ProxyConfig: when a ProxyDevice is constructed.
//  - HeadnodeConfig: in OmpFileHeadnodeManager::initialize().
// Changing the environment afterwards has no effect. Parsing is silent: a
// consumer that logs a malformed value does so itself, from Raw/Malformed, at
// the point it always did. Header-only and exception-free, because the proxy
// is built with -fno-exceptions. Knob semantics:
// docs/runtime-configuration.md.

#include "ompfile_env.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace ompfile {
namespace config {

enum class BatchPolicy { Disabled, Enabled, Auto };
enum class OpenMode { ReadWrite, ReadOnly };

// env::boolOr, plus what a consumer needs to log a malformed value.
struct BoolKnob {
  const char *Name = "";
  bool Default = false;
  bool Value = false;
  bool Malformed = false;
  std::string Raw; // the raw value, filled only when Malformed

  static BoolKnob read(const char *KnobName, bool DefaultValue) {
    BoolKnob K;
    K.Name = KnobName;
    K.Default = DefaultValue;
    K.Value = env::boolOr(KnobName, DefaultValue, &K.Malformed);
    if (K.Malformed)
      (void)env::raw(KnobName, &K.Raw);
    return K;
  }
};

// env::parseUint64Strict. Set mirrors raw() returning true, so an empty value
// is Set and not Valid.
struct U64Knob {
  const char *Name = "";
  bool Set = false;
  bool Valid = false;
  uint64_t Value = 0;
  std::string Raw;

  static U64Knob read(const char *KnobName) {
    U64Knob K;
    K.Name = KnobName;
    K.Set = env::raw(KnobName, &K.Raw);
    if (K.Set)
      K.Valid = env::parseUint64Strict(K.Raw.c_str(), &K.Value);
    return K;
  }
};

// LIBOMPFILE_OPT_TWO_PHASE / LIBOMPFILE_OPT_WRITE_BATCH: unset or empty ->
// Auto; 1, enabled, ENABLED -> Enabled; 0, disabled, DISABLED -> Disabled;
// auto, AUTO -> Auto; anything else -> Disabled with Malformed set.
struct PolicyKnob {
  BatchPolicy Policy = BatchPolicy::Auto;
  bool Malformed = false;
  std::string Raw;

  static PolicyKnob read(const char *KnobName) {
    PolicyKnob K;
    if (!env::raw(KnobName, &K.Raw) || K.Raw.empty())
      return K;
    const std::string &V = K.Raw;
    if (V == "1" || V == "enabled" || V == "ENABLED") {
      K.Policy = BatchPolicy::Enabled;
    } else if (V == "0" || V == "disabled" || V == "DISABLED") {
      K.Policy = BatchPolicy::Disabled;
    } else if (V == "auto" || V == "AUTO") {
      K.Policy = BatchPolicy::Auto;
    } else {
      K.Policy = BatchPolicy::Disabled;
      K.Malformed = true;
    }
    return K;
  }
};

// Knobs libompfile and the proxy both read, by the same rule on both sides.
struct SharedConfig {
  bool SchedulerHeadnode = true; // env::schedulerIsHeadnode
  bool SchedulerUnknown = false;
  std::string SchedulerRaw;                     // raw LIBOMPFILE_SCHEDULER
  std::string StageMode = "off";                // stringOr(.., "off")
  std::string StageWriteMode = "write-through"; // normalized
  bool StageFreshnessGuard = true;              // boolOr(.., true)
  unsigned WritebackDistributeRanks = 0;        // writebackDistributeRankCount
  bool WritebackDistributeActive = false;
  bool WritebackDistributeMisconfigured = false;

  bool stageOff() const { return StageMode == "off"; }
  bool stageReadthrough() const { return StageMode == "readthrough"; }
  bool stageWriteBack() const { return StageWriteMode == "write-back"; }

  static SharedConfig fromEnvironment() {
    SharedConfig C;
    C.SchedulerHeadnode = env::schedulerIsHeadnode(&C.SchedulerUnknown);
    (void)env::raw("LIBOMPFILE_SCHEDULER", &C.SchedulerRaw);
    C.StageMode = env::stringOr("LIBOMPFILE_STAGE_MODE", "off");
    C.StageWriteMode = env::normalizeStageWriteMode(
        env::stringOr("LIBOMPFILE_STAGE_WRITE_MODE", "write-through"));
    C.StageFreshnessGuard =
        env::boolOr("LIBOMPFILE_STAGE_FRESHNESS_GUARD", true);
    C.WritebackDistributeRanks = env::writebackDistributeRankCount();
    C.WritebackDistributeActive =
        env::writebackDistributeActive(&C.WritebackDistributeMisconfigured);
    return C;
  }
};

// libompfile's knobs: built once by OmpFileClientContext, copied into the
// backend.
struct ClientConfig {
  SharedConfig Shared;
  std::string RejectedModeKnobs; // env::rejectedModeKnobs()
  std::string RemovedKnobsSet;   // env::removedKnobsSet()
  // Also gates the Async IO stats line. Accepts only "1"; the proxy's
  // LLVM BoolEnvar reader of the same knob also accepts true/yes/on.
  BoolKnob OptStats;
  BoolKnob MppCallLockFree;
  OpenMode Open = OpenMode::ReadWrite; // LIBOMPFILE_OPEN_MODE
  bool OpenModeMalformed = false;
  std::string OpenModeRaw;
  BoolKnob WritableReadRebalance;
  BoolKnob DirtyOwnerForwarding;
  bool WritebackDistributeRanksSet = false; // raw(.._DISTRIBUTE_RANKS)
  std::string WritebackDistributeRanksRaw;
  bool MppPing = false; // flag(LIBOMPFILE_MPP_PING)
  PolicyKnob TwoPhase;
  U64Knob TwoPhaseWindowUs;
  U64Knob TwoPhaseMaxBatchBytes;
  U64Knob TwoPhaseSieveBytes;
  PolicyKnob WriteBatch;
  U64Knob WriteBatchWindowUs;
  BoolKnob WriteBatchWindowAdaptive;
  BoolKnob TwoPhaseWindowAdaptive;
  BoolKnob SchedOncePerHandle;
  U64Knob WriteBatchMaxBatchBytes;
  // LIBOMPFILE_TILE_FRESHNESS_WRITE_MODE: exactly write-back -> false,
  // exactly write-through -> true, otherwise the stage write mode decides.
  bool TileFreshnessWriteThrough = true;
  bool DirtyOwnerBatch = false;          // flag(..._DIRTY_OWNER_BATCH)
  unsigned DirtyOwnerBatchMaxSegs = 16;  // positiveOr, else 16, at least 1
  unsigned DirtyOwnerBatchWindowUs = 0;  // positiveOr, else 0
  // LIBOMPFILE_ASYNC_READ_WAIT=range: a read (or a synchronous write) waits
  // only for the queued async writes that could cover its own byte range of
  // the same file, instead of draining the whole engine. Off (drain) by
  // default; see the AsyncWriteEngine::waitForRange comment for the contract.
  bool AsyncReadWaitRange = false;
  bool LocalDisjointExpress = false; // flag(..._LOCAL_DISJOINT_EXPRESS)
  // LIBOMPFILE_IO_TOKENS, raw: std::stoi stays in the client (exceptions).
  bool IoTokensSet = false;
  std::string IoTokensRaw;
  size_t AsyncQueueDepth = 2; // positiveLongPrefixOr(.., 2)

  static ClientConfig fromEnvironment() {
    ClientConfig C;
    C.Shared = SharedConfig::fromEnvironment();
    C.RejectedModeKnobs = env::rejectedModeKnobs();
    C.RemovedKnobsSet = env::removedKnobsSet();
    C.OptStats = BoolKnob::read("LIBOMPFILE_OPT_STATS", false);
    C.MppCallLockFree =
        BoolKnob::read("LIBOMPFILE_OPT_MPP_CALL_LOCK_FREE", false);
    {
      std::string Mode;
      const bool Set = env::raw("LIBOMPFILE_OPEN_MODE", &Mode);
      if (!Set || Mode.empty() || Mode == "readwrite") {
        C.Open = OpenMode::ReadWrite;
      } else if (Mode == "readonly" || Mode == "read-only" ||
                 Mode == "read") {
        C.Open = OpenMode::ReadOnly;
      } else {
        C.Open = OpenMode::ReadWrite;
        C.OpenModeMalformed = true;
        C.OpenModeRaw = Mode;
      }
    }
    C.WritableReadRebalance =
        BoolKnob::read("LIBOMPFILE_OPT_WRITABLE_READ_REBALANCE", false);
    C.DirtyOwnerForwarding =
        BoolKnob::read("LIBOMPFILE_OPT_DIRTY_OWNER_FORWARDING", false);
    C.WritebackDistributeRanksSet =
        env::raw("LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_RANKS",
                 &C.WritebackDistributeRanksRaw);
    C.MppPing = env::flag("LIBOMPFILE_MPP_PING");
    C.TwoPhase = PolicyKnob::read("LIBOMPFILE_OPT_TWO_PHASE");
    C.TwoPhaseWindowUs = U64Knob::read("LIBOMPFILE_OPT_TWO_PHASE_WINDOW_US");
    C.TwoPhaseMaxBatchBytes =
        U64Knob::read("LIBOMPFILE_OPT_TWO_PHASE_MAX_BATCH_BYTES");
    C.TwoPhaseSieveBytes = U64Knob::read("LIBOMPFILE_OPT_TWO_PHASE_SIEVE_BYTES");
    C.WriteBatch = PolicyKnob::read("LIBOMPFILE_OPT_WRITE_BATCH");
    C.WriteBatchWindowUs =
        U64Knob::read("LIBOMPFILE_OPT_WRITE_BATCH_WINDOW_US");
    C.WriteBatchWindowAdaptive =
        BoolKnob::read("LIBOMPFILE_OPT_WRITE_BATCH_WINDOW_ADAPTIVE", false);
    C.TwoPhaseWindowAdaptive =
        BoolKnob::read("LIBOMPFILE_OPT_TWO_PHASE_WINDOW_ADAPTIVE", false);
    C.SchedOncePerHandle =
        BoolKnob::read("LIBOMPFILE_OPT_SCHED_ONCE_PER_HANDLE", false);
    C.WriteBatchMaxBatchBytes =
        U64Knob::read("LIBOMPFILE_OPT_WRITE_BATCH_MAX_BATCH_BYTES");
    {
      std::string Mode;
      const bool Set = env::raw("LIBOMPFILE_TILE_FRESHNESS_WRITE_MODE", &Mode);
      if (Set && Mode == "write-back")
        C.TileFreshnessWriteThrough = false;
      else if (Set && Mode == "write-through")
        C.TileFreshnessWriteThrough = true;
      else
        C.TileFreshnessWriteThrough = !C.Shared.stageWriteBack();
    }
    C.DirtyOwnerBatch = env::flag("LIBOMPFILE_OPT_DIRTY_OWNER_BATCH");
    {
      const unsigned Segs =
          env::positiveOr("LIBOMPFILE_OPT_DIRTY_OWNER_BATCH_MAX_SEGS");
      C.DirtyOwnerBatchMaxSegs = Segs ? std::max(1u, Segs) : 16u;
    }
    C.DirtyOwnerBatchWindowUs =
        env::positiveOr("LIBOMPFILE_OPT_DIRTY_OWNER_BATCH_WINDOW_US");
    C.AsyncReadWaitRange = env::equals("LIBOMPFILE_ASYNC_READ_WAIT", "range");
    C.LocalDisjointExpress = env::flag("LIBOMPFILE_OPT_LOCAL_DISJOINT_EXPRESS");
    C.IoTokensSet = env::raw("LIBOMPFILE_IO_TOKENS", &C.IoTokensRaw);
    C.AsyncQueueDepth = static_cast<size_t>(
        env::positiveLongPrefixOr("LIBOMPFILE_ASYNC_QUEUE_DEPTH", 2));
    return C;
  }
};

// The proxy's knobs: built once in the ProxyDevice constructor. The five LLVM
// BoolEnvar knobs (OPEN_CACHE, OPEN_CACHE_KEEP_OPEN, OPT_STATS,
// OPEN_CREATE_ON_WRITE, MPP_FORCE_BLOCKING_PWRITE) stay LLVM Envars: their
// rule is LLVM's and this header must not depend on LLVM.
struct ProxyConfig {
  SharedConfig Shared;
  bool LocalDisjointWrite = false;
  bool LocalDisjointRead = false;
  bool CoherentReadRefresh = true;
  uint64_t LocalDisjointWriteCombine = 0;
  uint64_t LocalDisjointReadahead = 0;
  uint64_t LocalDisjointWriteStreams = 1;
  uint64_t LocalDisjointStripeMinBytes = 16u << 20;
  uint64_t LocalDisjointReadStreams = 1;
  std::string StageSyncPolicy = "cache";
  std::string WritethroughFsyncPolicy = "each";
  std::string StagePopulateMode = "windowed";
  std::string StageRootPolicy = "topology";
  std::string TopologyFile;
  std::string StageDirName = "libompfile-stage";
  std::string StorageEnvironment = "unspecified";
  uint64_t StageMinFreeBytes = 0;
  uint64_t StageWindowBytes = 4ULL << 20;
  double StageWindowScale = 1.0;
  uint64_t StageDirtyWatermarkBytes = 0;
  bool StageFlushCoalesce = false;
  // LIBOMPFILE_STAGE_TILE_BYTES, else the OMPFILE_CHOLESKY_NB fallback
  // (NB^2 * sizeof(float), 0 on overflow); docs/known-issues.md.
  uint64_t StageTileBytes = 0;
  bool OpenFailureDiag = true; // onUnlessZero(LIBOMPFILE_OPEN_FAILURE_DIAG)

  static uint64_t stageTileBytesFromEnvironment() {
    if (const uint64_t Bytes = env::uint64Or("LIBOMPFILE_STAGE_TILE_BYTES", 0))
      return Bytes;
    const uint64_t Nb = env::uint64Or("OMPFILE_CHOLESKY_NB", 0);
    if (Nb == 0 || Nb > (std::numeric_limits<uint64_t>::max() / Nb) ||
        (Nb * Nb) > (std::numeric_limits<uint64_t>::max() / sizeof(float)))
      return 0;
    return Nb * Nb * sizeof(float);
  }

  static ProxyConfig fromEnvironment() {
    ProxyConfig P;
    P.Shared = SharedConfig::fromEnvironment();
    P.LocalDisjointWrite =
        env::boolOr("LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE", false);
    P.LocalDisjointRead =
        env::boolOr("LIBOMPFILE_OPT_LOCAL_DISJOINT_READ", false);
    P.CoherentReadRefresh =
        env::boolOr("LIBOMPFILE_OPT_COHERENT_READ_REFRESH", true);
    P.LocalDisjointWriteCombine =
        env::uint64Or("LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE_COMBINE", 0);
    P.LocalDisjointReadahead =
        env::uint64Or("LIBOMPFILE_OPT_LOCAL_DISJOINT_READAHEAD", 0);
    P.LocalDisjointWriteStreams =
        env::uint64Or("LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE_STREAMS", 1);
    P.LocalDisjointStripeMinBytes = env::uint64Or(
        "LIBOMPFILE_OPT_LOCAL_DISJOINT_STRIPE_MIN_BYTES", 16u << 20);
    P.LocalDisjointReadStreams =
        env::uint64Or("LIBOMPFILE_OPT_LOCAL_DISJOINT_READ_STREAMS", 1);
    P.StageSyncPolicy = env::stringOr("LIBOMPFILE_STAGE_SYNC_POLICY", "cache");
    P.WritethroughFsyncPolicy =
        env::stringOr("LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY", "each");
    P.StagePopulateMode =
        env::stringOr("LIBOMPFILE_STAGE_POPULATE_MODE", "windowed");
    P.StageRootPolicy =
        env::stringOr("LIBOMPFILE_STAGE_ROOT_POLICY", "topology");
    P.TopologyFile = env::stringOr("LIBOMPFILE_TOPOLOGY_FILE", "");
    P.StageDirName =
        env::stringOr("LIBOMPFILE_STAGE_DIR_NAME", "libompfile-stage");
    P.StorageEnvironment =
        env::stringOr("LIBOMPFILE_STORAGE_ENVIRONMENT", "unspecified");
    P.StageMinFreeBytes = env::uint64Or("LIBOMPFILE_STAGE_MIN_FREE_BYTES", 0);
    P.StageWindowBytes =
        env::uint64Or("LIBOMPFILE_STAGE_WINDOW_BYTES", 4ULL << 20);
    P.StageWindowScale = env::doubleOr("LIBOMPFILE_STAGE_WINDOW_SCALE", 1.0);
    P.StageDirtyWatermarkBytes =
        env::uint64Or("LIBOMPFILE_STAGE_DIRTY_WATERMARK_BYTES", 0);
    P.StageFlushCoalesce =
        env::boolOr("LIBOMPFILE_STAGE_FLUSH_COALESCE", false);
    P.StageTileBytes = stageTileBytesFromEnvironment();
    P.OpenFailureDiag = env::onUnlessZero("LIBOMPFILE_OPEN_FAILURE_DIAG");
    return P;
  }
};

// The headnode scheduler's knobs: built once per
// OmpFileHeadnodeManager::initialize().
struct HeadnodeConfig {
  uint64_t MaxAffinityLoadSkew = std::numeric_limits<uint64_t>::max();
  uint64_t BatchStatsReportEvery = 128;
  bool UsePlannedLoadForSkew = false;
  bool SpreadSamePathOpens = false;
  bool FreshnessTrace = false;
  bool SchedTrace = false;

  static HeadnodeConfig fromEnvironment() {
    HeadnodeConfig H;
    H.MaxAffinityLoadSkew =
        env::uint64Or("LIBOMPFILE_SCHED_MAX_AFFINITY_LOAD_SKEW",
                      std::numeric_limits<uint64_t>::max());
    H.BatchStatsReportEvery = std::max<uint64_t>(
        1, env::uint64Or("LIBOMPFILE_SCHED_STATS_REPORT_EVERY", 128));
    H.UsePlannedLoadForSkew =
        env::uint64Or("LIBOMPFILE_SCHED_USE_PLANNED_LOAD_FOR_SKEW", 0) != 0;
    H.SpreadSamePathOpens = env::uint64Or("LIBOMPFILE_MPP_OPEN_SPREAD", 0) != 0;
    H.FreshnessTrace = env::setAndNotZeroOr("LIBOMPFILE_FRESHNESS_TRACE", false);
    H.SchedTrace = env::flag("LIBOMPFILE_DEBUG_TRACE");
    return H;
  }
};

} // namespace config
} // namespace ompfile

#endif // OMPFILE_CONFIG_H
