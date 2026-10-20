//===-- ProxyOmpFileConfig.cpp - Proxy ompfile configuration ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ProxyOmpFile construction, initialize() and deinitialize(): the proxy
// knobs, the stage-root topology resolution and its preflight, and the
// proxy trace helper. The trace sequence counter lives only in this TU.
//
//===----------------------------------------------------------------------===//

#include "ProxyOmpFile.h"

namespace {
bool ompfileProxyTraceEnabled() {
  static const bool Enabled = ompfile::env::flag("LIBOMPFILE_DEBUG_TRACE");
  return Enabled;
}

uint64_t nextProxyTraceSeq() {
  static std::atomic<uint64_t> Seq{1};
  return Seq.fetch_add(1, std::memory_order_relaxed);
}

unsigned long proxyThreadIdHash() {
  return static_cast<unsigned long>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

std::string trimWhitespace(std::string Value) {
  const size_t Begin = Value.find_first_not_of(" \t\r\n");
  if (Begin == std::string::npos)
    return {};
  const size_t End = Value.find_last_not_of(" \t\r\n");
  return Value.substr(Begin, End - Begin + 1);
}

std::string shortHostname(std::string Host) {
  Host = trimWhitespace(std::move(Host));
  const size_t DotPos = Host.find('.');
  if (DotPos != std::string::npos)
    Host.resize(DotPos);
  return Host;
}

std::string getLocalShortHostname() {
  char Buffer[HOST_NAME_MAX + 1] = {};
  if (gethostname(Buffer, sizeof(Buffer) - 1) != 0)
    return "unknown";
  return shortHostname(Buffer);
}

std::string defaultStageRunStem() {
  const char *JobId = std::getenv("SLURM_JOB_ID");
  if (JobId && JobId[0] != '\0')
    return std::string("job-") + JobId;
  return "run-local";
}

bool hostsMatch(const std::string &Expected, const std::string &Observed) {
  return fnmatch(shortHostname(Expected).c_str(), shortHostname(Observed).c_str(),
                 0) == 0;
}

struct ResolvedTopologyConfig {
  std::string StageRoot;
  std::string StageClass;
  std::string SharedStoragePath;
  std::string SharedStorageClass;
  std::string StageDecisionReason;
  uint64_t EntryCount = 0;
  std::string Error;
};

std::string expandTopologyVariables(const std::string &Value) {
  std::string Expanded;
  Expanded.reserve(Value.size());
  for (size_t I = 0; I < Value.size(); ++I) {
    if (Value[I] != '$') {
      Expanded.push_back(Value[I]);
      continue;
    }
    std::string Name;
    size_t Next = I + 1;
    if (Next < Value.size() && Value[Next] == '{') {
      const size_t End = Value.find('}', Next + 1);
      if (End == std::string::npos) {
        Expanded.push_back(Value[I]);
        continue;
      }
      Name = Value.substr(Next + 1, End - Next - 1);
      I = End;
    } else {
      while (Next < Value.size() &&
             (std::isalnum(static_cast<unsigned char>(Value[Next])) ||
              Value[Next] == '_')) {
        Name.push_back(Value[Next]);
        ++Next;
      }
      if (Name.empty()) {
        Expanded.push_back(Value[I]);
        continue;
      }
      I = Next - 1;
    }
    const char *Env = std::getenv(Name.c_str());
    if (Env)
      Expanded += Env;
  }
  return Expanded;
}

std::vector<std::string> splitCandidateSpec(const std::string &Value) {
  std::vector<std::string> Candidates;
  std::stringstream Stream(Value);
  std::string Item;
  while (std::getline(Stream, Item, ':')) {
    Item = trimWhitespace(expandTopologyVariables(Item));
    if (!Item.empty())
      Candidates.push_back(Item);
  }
  return Candidates;
}

std::string trimTrailingSlashes(std::string Value) {
  while (Value.size() > 1 && Value.back() == '/')
    Value.pop_back();
  return Value;
}

std::string shellEscapeSingleQuoted(const std::string &Value) {
  std::string Escaped;
  Escaped.reserve(Value.size() + 8);
  Escaped.push_back('\'');
  for (char Ch : Value) {
    if (Ch == '\'')
      Escaped += "'\\''";
    else
      Escaped.push_back(Ch);
  }
  Escaped.push_back('\'');
  return Escaped;
}

std::string runFirstLineCommand(const std::string &Command) {
  std::string Result;
  FILE *Pipe = popen(Command.c_str(), "r");
  if (!Pipe)
    return Result;
  char Buffer[4096] = {};
  if (fgets(Buffer, sizeof(Buffer), Pipe))
    Result = trimWhitespace(Buffer);
  (void)pclose(Pipe);
  return Result;
}

std::string findmntValueForPath(const std::string &Path, const char *Column) {
  if (Path.empty())
    return {};
  return runFirstLineCommand("findmnt -rn -T " + shellEscapeSingleQuoted(Path) +
                             " -o " + Column + " 2>/dev/null");
}

bool fsTypeIsMemoryBacked(const std::string &FsType) {
  return FsType == "tmpfs" || FsType == "ramfs";
}

bool fsTypeIsShared(const std::string &FsType, const std::string &Source) {
  if (FsType.rfind("nfs", 0) == 0)
    return true;
  return FsType == "lustre" || FsType == "gpfs" || FsType == "wekafs" ||
         FsType == "virtiofs" || FsType == "beegfs" || FsType == "ceph" ||
         FsType == "cephfs" || FsType == "glusterfs" ||
         FsType == "orangefs" || FsType == "pvfs2" || FsType == "panfs" ||
         Source.rfind("//", 0) == 0;
}

bool sourceLooksLocalBlockDevice(const std::string &Source) {
  return Source.rfind("/dev/", 0) == 0;
}

std::string inferLocalStorageClass(const std::string &Source,
                                   const std::string &MountTarget) {
  const std::string LowerSource = [&]() {
    std::string Copy = Source;
    std::transform(Copy.begin(), Copy.end(), Copy.begin(),
                   [](unsigned char Ch) { return std::tolower(Ch); });
    return Copy;
  }();
  const std::string LowerTarget = [&]() {
    std::string Copy = MountTarget;
    std::transform(Copy.begin(), Copy.end(), Copy.begin(),
                   [](unsigned char Ch) { return std::tolower(Ch); });
    return Copy;
  }();
  if (LowerSource.find("nvme") != std::string::npos ||
      LowerTarget.find("nvme") != std::string::npos)
    return "local-nvme";
  if (LowerSource.find("/sd") != std::string::npos ||
      LowerSource.find("/hd") != std::string::npos ||
      LowerSource.find("hdd") != std::string::npos ||
      LowerSource.find("sata") != std::string::npos ||
      LowerTarget.find("hdd") != std::string::npos ||
      LowerTarget.find("ssd") != std::string::npos)
    return "local-hdd";
  return "local-posix";
}

std::string inferSharedStorageClass(const std::string &FsType) {
  if (FsType.rfind("nfs", 0) == 0)
    return "nfs";
  if (!FsType.empty())
    return FsType;
  return "parallel_fs";
}

bool ensureDirectoryTreePath(const std::string &Path, int &ErrnoOut) {
  ErrnoOut = 0;
  if (Path.empty()) {
    ErrnoOut = EINVAL;
    return false;
  }

  size_t Pos = Path[0] == '/' ? 1 : 0;
  while (Pos <= Path.size()) {
    Pos = Path.find('/', Pos);
    const std::string Prefix =
        Path.substr(0, Pos == std::string::npos ? Path.size() : Pos);
    if (!Prefix.empty()) {
      if (::mkdir(Prefix.c_str(), 0775) != 0 && errno != EEXIST) {
        ErrnoOut = errno;
        return false;
      }
    }
    if (Pos == std::string::npos)
      break;
    ++Pos;
  }
  return true;
}

bool touchProbeFile(const std::string &Directory, int &ErrnoOut) {
  ErrnoOut = 0;
  const std::string ProbePath = Directory + "/probe-" +
                                std::to_string(::getpid()) + ".tmp";
  const char ProbeContents[] = "libompfile-stage-probe\n";
  int Fd = ::open(ProbePath.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0664);
  if (Fd < 0) {
    ErrnoOut = errno;
    return false;
  }
  const ssize_t Written =
      ::write(Fd, ProbeContents, sizeof(ProbeContents) - 1);
  if (Written != static_cast<ssize_t>(sizeof(ProbeContents) - 1)) {
    ErrnoOut = (Written < 0) ? errno : EIO;
    (void)::close(Fd);
    (void)::unlink(ProbePath.c_str());
    return false;
  }
  if (::fsync(Fd) != 0) {
    ErrnoOut = errno;
    (void)::close(Fd);
    (void)::unlink(ProbePath.c_str());
    return false;
  }
  if (::lseek(Fd, 0, SEEK_SET) < 0) {
    ErrnoOut = errno;
    (void)::close(Fd);
    (void)::unlink(ProbePath.c_str());
    return false;
  }
  char Buffer[sizeof(ProbeContents)] = {};
  const ssize_t Read = ::read(Fd, Buffer, sizeof(ProbeContents) - 1);
  const bool Ok =
      Read == static_cast<ssize_t>(sizeof(ProbeContents) - 1) &&
      std::strncmp(Buffer, ProbeContents, sizeof(ProbeContents) - 1) == 0;
  if (!Ok)
    ErrnoOut = (Read < 0) ? errno : EIO;
  (void)::close(Fd);
  (void)::unlink(ProbePath.c_str());
  return Ok;
}

std::string buildStageRootForCandidate(const std::string &CandidateRoot,
                                       const std::string &UserName,
                                       const std::string &StageDirName,
                                       const std::string &StageRunStem) {
  std::string Root = trimTrailingSlashes(CandidateRoot);
  return Root + "/" + UserName + "/" + StageDirName + "/" + StageRunStem;
}

struct TopologyResolveOptions {
  std::string LocalHost;
  std::string UserName;
  std::string StageDirName;
  std::string StageRunStem;
};

bool resolveSharedCandidate(const std::string &Candidate,
                            std::string &SelectedPath,
                            std::string &SelectedClass,
                            std::string &ErrorOut) {
  const std::string Expanded = expandTopologyVariables(Candidate);
  struct stat PathStat {};
  if (::stat(Expanded.c_str(), &PathStat) != 0) {
    ErrorOut = "shared-missing";
    return false;
  }
  const std::string FsType = findmntValueForPath(Expanded, "FSTYPE");
  const std::string Source = findmntValueForPath(Expanded, "SOURCE");
  if (fsTypeIsMemoryBacked(FsType)) {
    ErrorOut = "shared-memory-backed";
    return false;
  }
  if (!fsTypeIsShared(FsType, Source)) {
    ErrorOut = "shared-not-shared";
    return false;
  }
  if (::access(Expanded.c_str(), R_OK | X_OK) != 0) {
    ErrorOut = "shared-access-failed";
    return false;
  }
  SelectedPath = Expanded;
  SelectedClass = inferSharedStorageClass(FsType);
  ErrorOut.clear();
  return true;
}

bool resolveLocalHintCandidate(const std::string &Candidate,
                               const TopologyResolveOptions &Options,
                               std::string &SelectedStageRoot,
                               std::string &SelectedStageClass,
                               std::string &ErrorOut) {
  const std::string Expanded = expandTopologyVariables(Candidate);
  struct stat PathStat {};
  if (::stat(Expanded.c_str(), &PathStat) != 0) {
    ErrorOut = "local-candidate-missing";
    return false;
  }
  const std::string MountTarget = findmntValueForPath(Expanded, "TARGET");
  const std::string FsType = findmntValueForPath(Expanded, "FSTYPE");
  const std::string Source = findmntValueForPath(Expanded, "SOURCE");
  if (fsTypeIsMemoryBacked(FsType)) {
    ErrorOut = "local-memory-backed";
    return false;
  }
  if (fsTypeIsShared(FsType, Source)) {
    ErrorOut = "local-shared-fs";
    return false;
  }
  if (!sourceLooksLocalBlockDevice(Source)) {
    ErrorOut = "local-not-host-device";
    return false;
  }
  const std::string StageRoot = buildStageRootForCandidate(
      Expanded, Options.UserName, Options.StageDirName, Options.StageRunStem);
  int ErrnoOut = 0;
  if (!ensureDirectoryTreePath(StageRoot, ErrnoOut)) {
    ErrorOut = "local-mkdir-failed";
    return false;
  }
  if (!touchProbeFile(StageRoot, ErrnoOut)) {
    ErrorOut = "local-touch-failed";
    return false;
  }
  SelectedStageRoot = StageRoot;
  SelectedStageClass = inferLocalStorageClass(Source, MountTarget);
  ErrorOut.clear();
  return true;
}

bool resolveTopologyConfig(const std::string &TopologyFile,
                           const TopologyResolveOptions &Options,
                           ResolvedTopologyConfig &Config) {
  Config = {};
  if (TopologyFile.empty()) {
    Config.Error = "topology-unset";
    return false;
  }

  std::ifstream Input(TopologyFile);
  if (!Input) {
    Config.Error = "open-failed";
    return false;
  }

  std::string Line;
  bool GlobalRuleSeen = false;
  bool LocalRuleMatched = false;
  while (std::getline(Input, Line)) {
    Line = trimWhitespace(std::move(Line));
    if (Line.empty() || Line[0] == '#')
      continue;

    std::istringstream Stream(Line);
    std::string Tag;
    if (!(Stream >> Tag))
      continue;

    if (Tag == "global" || Tag == "global-hint") {
      ++Config.EntryCount;
      if (GlobalRuleSeen)
        continue;
      GlobalRuleSeen = true;
      if (Tag == "global") {
        std::string SharedPath;
        std::string SharedClass;
        if (!(Stream >> SharedPath)) {
          Config.Error = "global-path-missing";
          return false;
        }
        SharedPath = expandTopologyVariables(SharedPath);
        if (!resolveSharedCandidate(SharedPath, Config.SharedStoragePath,
                                    Config.SharedStorageClass, Config.Error))
          return false;
        continue;
      }

      std::string CandidateSpec;
      if (!(Stream >> CandidateSpec)) {
        Config.Error = "global-hint-missing";
        return false;
      }
      bool ResolvedShared = false;
      for (const std::string &Candidate : splitCandidateSpec(CandidateSpec)) {
        if (resolveSharedCandidate(Candidate, Config.SharedStoragePath,
                                   Config.SharedStorageClass, Config.Error)) {
          ResolvedShared = true;
          break;
        }
      }
      if (!ResolvedShared && Config.Error.empty())
        Config.Error = "global-unresolved";
      if (!ResolvedShared)
        return false;
      continue;
    }

    if (Tag == "host-none" || Tag == "host" || Tag == "host-hint") {
      std::string HostPattern;
      if (!(Stream >> HostPattern))
        continue;
      ++Config.EntryCount;
      if (LocalRuleMatched || !hostsMatch(HostPattern, Options.LocalHost))
        continue;
      LocalRuleMatched = true;

      if (Tag == "host-none") {
        Config.StageDecisionReason = "host-none";
        Config.Error.clear();
        continue;
      }

      if (Tag == "host") {
        std::string Path;
        std::string StageClass;
        if (!(Stream >> Path)) {
          Config.Error = "host-path-missing";
          return false;
        }
        Stream >> StageClass;
        Path = expandTopologyVariables(Path);
        int ErrnoOut = 0;
        if (!ensureDirectoryTreePath(Path, ErrnoOut)) {
          Config.Error = "host-stage-root-unusable";
          return false;
        }
        if (!touchProbeFile(Path, ErrnoOut)) {
          Config.Error = "host-stage-root-touch-failed";
          return false;
        }
        Config.StageRoot = Path;
        Config.StageClass = StageClass.empty() ? "local-posix" : StageClass;
        Config.StageDecisionReason = "resolved-stage-root";
        Config.Error.clear();
        continue;
      }

      std::string CandidateSpec;
      if (!(Stream >> CandidateSpec)) {
        Config.Error = "host-hint-missing";
        return false;
      }
      bool ResolvedLocal = false;
      for (const std::string &Candidate : splitCandidateSpec(CandidateSpec)) {
        if (resolveLocalHintCandidate(Candidate, Options, Config.StageRoot,
                                      Config.StageClass, Config.Error)) {
          Config.StageDecisionReason = "hinted-stage-root";
          ResolvedLocal = true;
          break;
        }
      }
      if (!ResolvedLocal && Config.Error.empty())
        Config.Error = "host-hint-unresolved";
      if (!ResolvedLocal)
        return false;
      continue;
    }
  }

  if (!GlobalRuleSeen) {
    Config.Error = Config.EntryCount == 0 ? "empty-topology" : "shared-missing";
    return false;
  }

  if (!LocalRuleMatched) {
    Config.Error = Config.EntryCount == 0 ? "empty-topology" : "host-missing";
    return false;
  }

  return !Config.StageRoot.empty() || Config.StageDecisionReason == "host-none";
}
} // namespace

[[noreturn]] void ProxyOmpFile::fatalStageConfig(const char *Reason) {
  fprintf(stderr,
          "MPIProxyDevice --> fatal stage config rank=%d reason=%s "
          "stage_mode=%s topology_file=%s local_host=%s stage_root=%s "
          "stage_class=%s shared_storage_path=%s shared_storage_class=%s "
          "stage_decision=%s load_error=%s\n",
          EventSystem.LocalRank, Reason ? Reason : "(unknown)",
          OmpFileStageMode.c_str(),
          OmpFileTopologyFile.empty() ? "(unset)"
                                      : OmpFileTopologyFile.c_str(),
          OmpFileStageLocalHost.c_str(),
          OmpFileStageRoot.empty() ? "(unset)" : OmpFileStageRoot.c_str(),
          OmpFileSelectedStageClass.empty() ? "(unset)"
                                            : OmpFileSelectedStageClass.c_str(),
          OmpFileSharedStoragePath.empty()
              ? "(unset)"
              : OmpFileSharedStoragePath.c_str(),
          OmpFileSharedStorageClass.empty()
              ? "(unset)"
              : OmpFileSharedStorageClass.c_str(),
          OmpFileStageDecisionReason.empty()
              ? "(unset)"
              : OmpFileStageDecisionReason.c_str(),
          OmpFileTopologyLoadError.empty()
              ? "(none)"
              : OmpFileTopologyLoadError.c_str());
  std::fflush(stderr);
  std::exit(EXIT_FAILURE);
}

bool ProxyOmpFile::validateStageConfigFast() {
  if (OmpFileStageMode == "off")
    return true;
  if (OmpFileStageRootPolicy != "topology")
    return true;
  if (OmpFileTopologyFile.empty()) {
    OmpFileTopologyLoadError = "topology-required";
    return false;
  }
  if (!OmpFileTopologyLoaded)
    return false;
  if (OmpFileStageRoot.empty())
    return OmpFileStageDecisionReason == "host-none";
  int ErrnoOut = 0;
  if (!ensureDirectoryTree(OmpFileStageRoot, ErrnoOut)) {
    OmpFileTopologyLoadError = "stage-root-unusable";
    return false;
  }
  if (!hasStageFreeSpace(ErrnoOut)) {
    OmpFileTopologyLoadError = "stage-root-low-space";
    return false;
  }
  if (!touchProbeFile(OmpFileStageRoot, ErrnoOut)) {
    OmpFileTopologyLoadError = "stage-root-touch-failed";
    return false;
  }
  return true;
}

void ProxyOmpFile::traceOmpFile(const char *Fmt, ...) const {
  if (!ompfileProxyTraceEnabled())
    return;

  va_list Args;
  va_start(Args, Fmt);
  fprintf(stderr,
          "[ompfile-proxy][trace][seq=%llu][pid=%d][tid=%lu][rank=%d] ",
          static_cast<unsigned long long>(nextProxyTraceSeq()), getpid(),
          proxyThreadIdHash(), EventSystem.LocalRank);
  vfprintf(stderr, Fmt, Args);
  va_end(Args);
}

ProxyOmpFile::ProxyOmpFile(EventSystemTy &ES)
    : EventSystem(ES),
      OmpFileOpenCacheEnable("LIBOMPFILE_OPT_OPEN_CACHE", false),
      OmpFileOpenCacheKeepOpen("LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN", true),
      OmpFileOptStats("LIBOMPFILE_OPT_STATS", false),
      OmpFileOpenCreateOnWrite("LIBOMPFILE_OPEN_CREATE_ON_WRITE", true),
      OmpFileForceBlockingPwrite("LIBOMPFILE_MPP_FORCE_BLOCKING_PWRITE",
                                 false),
      OmpFileOpenEioRetries("OMPTARGET_OMPFILE_OPEN_EIO_RETRIES", 16),
      OmpFileOpenEioBackoffUs("OMPTARGET_OMPFILE_OPEN_EIO_BACKOFF_US",
                               200000),
      OmpFileMPIFragmentSize("OMPTARGET_MPI_FRAGMENT_SIZE", 100e6),
      OmpFileConfig(ompfile::config::ProxyConfig::fromEnvironment()),
      OmpFileLocalDisjointWrite(OmpFileConfig.LocalDisjointWrite),
      OmpFileLocalDisjointRead(OmpFileConfig.LocalDisjointRead),
      OmpFileCoherentReadRefresh(OmpFileConfig.CoherentReadRefresh),
      OmpFileLocalDisjointWriteCombine(
          OmpFileConfig.LocalDisjointWriteCombine),
      OmpFileLocalDisjointReadahead(OmpFileConfig.LocalDisjointReadahead),
      OmpFileLocalDisjointWriteStreams(
          OmpFileConfig.LocalDisjointWriteStreams),
      OmpFileLocalDisjointStripeMinBytes(
          OmpFileConfig.LocalDisjointStripeMinBytes),
      OmpFileLocalDisjointReadStreams(OmpFileConfig.LocalDisjointReadStreams),
      OmpFileHeadnodeScheduler(OmpFileConfig.Shared.SchedulerHeadnode),
      OmpFileStageMode(OmpFileConfig.Shared.StageMode),
      OmpFileStageWriteMode(OmpFileConfig.Shared.StageWriteMode),
      OmpFileStageSyncPolicy(OmpFileConfig.StageSyncPolicy),
      OmpFileWritethroughFsyncPolicy(OmpFileConfig.WritethroughFsyncPolicy),
      OmpFileStagePopulateMode(OmpFileConfig.StagePopulateMode),
      OmpFileStageRootPolicy(OmpFileConfig.StageRootPolicy),
      OmpFileTopologyFile(OmpFileConfig.TopologyFile),
      OmpFileStageDirName(OmpFileConfig.StageDirName),
      OmpFileStageRunStem(defaultStageRunStem()),
      OmpFileStorageEnvironment(OmpFileConfig.StorageEnvironment),
      OmpFileStageMinFreeBytes(OmpFileConfig.StageMinFreeBytes),
      OmpFileStageWindowBytes(OmpFileConfig.StageWindowBytes),
      OmpFileStageWindowScale(OmpFileConfig.StageWindowScale),
      OmpFileStageDirtyWatermarkBytes(OmpFileConfig.StageDirtyWatermarkBytes),
      OmpFileStageFreshnessGuard(OmpFileConfig.Shared.StageFreshnessGuard),
      // Stays off by default. Coalescing concurrent watermark flushes is
      // a large win only where the source filesystem's fdatasync is
      // expensive: on sorgan's NFS it is a strict improvement at every
      // measured point (119, 125: 1.05x at P=2 with one handler thread,
      // up to 1.94x at P=8 with four, source fsyncs 12-35 -> 9 for 72
      // tiles), while on AMD's wekafs, where those commits are nearly
      // free, it measures 0.94x-1.01x (405094). A default that helps one
      // cluster and costs the other is a per-lane choice, not a default.
      OmpFileStageFlushCoalesce(OmpFileConfig.StageFlushCoalesce),
      OmpFileStageLocalHost(getLocalShortHostname()) {}

void ProxyOmpFile::initialize() {
  if (EventSystem.LocalRank == getHeadnodeRank()) {
    OmpFileHeadnodeManager::instance().initialize(
        EventSystem.numWorkerRanks(), getHeadnodeRank());
    DP("Initialized global OMPFile headnode manager on rank %d\n",
       EventSystem.LocalRank);
  }
  ResolvedTopologyConfig TopologyConfig;
  OmpFileTopologyLoaded = resolveTopologyConfig(OmpFileTopologyFile,
                                                TopologyResolveOptions{
                                                    OmpFileStageLocalHost,
                                                    envStringOrDefault("USER", "unknown"),
                                                    OmpFileStageDirName,
                                                    OmpFileStageRunStem,
                                                },
                                                TopologyConfig);
  OmpFileStageRoot = TopologyConfig.StageRoot;
  OmpFileSelectedStageClass = TopologyConfig.StageClass;
  OmpFileSharedStoragePath = TopologyConfig.SharedStoragePath;
  OmpFileSharedStorageClass = TopologyConfig.SharedStorageClass;
  OmpFileStageDecisionReason = TopologyConfig.StageDecisionReason;
  OmpFileTopologyEntries = TopologyConfig.EntryCount;
  OmpFileTopologyLoadError = TopologyConfig.Error;
  // Apply the adaptive stage-window scale. The base window is the floor;
  // scale multiplies it and is clamped to >= 1.0 so a stray value can
  // never shrink the window below the configured base. Effective window
  // is what every stage-populate/eviction decision must consult.
  double Scale = OmpFileStageWindowScale;
  if (!(Scale >= 1.0))
    Scale = 1.0;
  OmpFileStageEffectiveWindowBytes =
      static_cast<uint64_t>(static_cast<double>(OmpFileStageWindowBytes) *
                            Scale);
  if (OmpFileOptStats || OmpFileOpenCacheEnable) {
    fprintf(stderr,
            "MPIProxyDevice --> OMPFile cache config rank=%d enabled=%d "
            "keep_open=%d stats=%d blocking_pwrite=%d fragment_size=%lld "
            "coherent_read_refresh=%d\n",
            EventSystem.LocalRank, (int)OmpFileOpenCacheEnable.get(),
            (int)OmpFileOpenCacheKeepOpen.get(), (int)OmpFileOptStats.get(),
            (int)OmpFileForceBlockingPwrite.get(),
            static_cast<long long>(OmpFileMPIFragmentSize.get()),
            (int)OmpFileCoherentReadRefresh);
  }
  if (OmpFileOptStats || OmpFileOpenCacheEnable ||
      OmpFileStageMode != "off" || !OmpFileTopologyFile.empty()) {
    fprintf(stderr,
            "MPIProxyDevice --> OMPFile stage config rank=%d "
            "stage_mode=%s stage_root_policy=%s topology_file_set=%d "
            "topology_file=%s topology_loaded=%d topology_entries=%llu "
            "stage_root=%s stage_class=%s shared_storage_path=%s "
            "shared_storage_class=%s storage_environment=%s "
            "stage_sync_policy=%s stage_write_mode=%s "
            "stage_freshness_guard=%d stage_flush_coalesce=%d "
            "stage_populate_mode=%s "
            "stage_window_bytes=%llu stage_window_scale=%.3f "
            "stage_effective_window_bytes=%llu "
            "stage_dirty_watermark_bytes=%llu "
            "stage_min_free_bytes=%llu "
            "local_host=%s stage_decision=%s load_error=%s\n",
            EventSystem.LocalRank, OmpFileStageMode.c_str(),
            OmpFileStageRootPolicy.c_str(),
            static_cast<int>(!OmpFileTopologyFile.empty()),
            OmpFileTopologyFile.empty() ? "(unset)"
                                        : OmpFileTopologyFile.c_str(),
            static_cast<int>(OmpFileTopologyLoaded),
            static_cast<unsigned long long>(OmpFileTopologyEntries),
            OmpFileStageRoot.empty() ? "(unset)" : OmpFileStageRoot.c_str(),
            OmpFileSelectedStageClass.empty()
                ? "(unset)"
                : OmpFileSelectedStageClass.c_str(),
             OmpFileSharedStoragePath.empty()
                 ? "(unset)"
                 : OmpFileSharedStoragePath.c_str(),
             OmpFileSharedStorageClass.empty()
                 ? "(unset)"
                 : OmpFileSharedStorageClass.c_str(),
             OmpFileStorageEnvironment.c_str(),
             OmpFileStageSyncPolicy.c_str(),
             OmpFileStageWriteMode.c_str(),
             static_cast<int>(OmpFileStageFreshnessGuard),
             static_cast<int>(OmpFileStageFlushCoalesce),
             OmpFileStagePopulateMode.c_str(),
             static_cast<unsigned long long>(OmpFileStageWindowBytes),
             OmpFileStageWindowScale,
             static_cast<unsigned long long>(OmpFileStageEffectiveWindowBytes),
             static_cast<unsigned long long>(OmpFileStageDirtyWatermarkBytes),
             static_cast<unsigned long long>(OmpFileStageMinFreeBytes),
             OmpFileStageLocalHost.c_str(),
             OmpFileStageDecisionReason.empty()
                 ? "(unset)"
                 : OmpFileStageDecisionReason.c_str(),
             OmpFileTopologyLoadError.empty()
                 ? "(none)"
                 : OmpFileTopologyLoadError.c_str());
  }
  if (!validateStageConfigFast())
    fatalStageConfig("stage-preflight-failed");
  if (OmpFileStageMode == "off") {
    fprintf(stderr,
            "MPIProxyDevice --> staging disabled on host %s reason=stage_mode_off\n",
            OmpFileStageLocalHost.c_str());
  } else if (OmpFileStageDecisionReason == "host-none" ||
             OmpFileStageRoot.empty()) {
    fprintf(stderr,
            "MPIProxyDevice --> staging disabled on host %s reason=%s\n",
            OmpFileStageLocalHost.c_str(),
            OmpFileStageDecisionReason.empty()
                ? "stage_root_unset"
                : OmpFileStageDecisionReason.c_str());
  } else {
    fprintf(stderr,
            "MPIProxyDevice --> touching local disk on host %s selected=%s class=%s result=ok\n",
            OmpFileStageLocalHost.c_str(), OmpFileStageRoot.c_str(),
            OmpFileSelectedStageClass.empty()
                ? "(unset)"
                : OmpFileSelectedStageClass.c_str());
  }
  setActiveProxyOmpFile(this);
}

void ProxyOmpFile::deinitialize() {
  drainOmpFileOpenCache();
  drainOmpFileStageCache();
  closeLocalWriteFds();
  closeLocalReadFds();
  reportOmpFileStats("proxy-dtor");
  clearActiveProxyOmpFile(this);
}
