#include <chrono>
#include <atomic>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdarg>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fnmatch.h>
#include <string>
#include <sstream>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <limits.h>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>
#include <sys/statvfs.h>

#include "EventSystem.h"
#include "OmpFileHeadnodeManager.h"
#include "RemotePluginManager.h"
#include "Shared/APITypes.h"
#include "mpi.h"
#include "omptarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#ifdef OMPT_SUPPORT
#include "OpenMP/OMPT/Callback.h"
#include "omp-tools.h"
extern void llvm::omp::target::ompt::connectLibrary();
#endif

struct ProxyDevice;

namespace {
std::mutex ActiveProxyDeviceMutex;
ProxyDevice *ActiveProxyDevice = nullptr;

bool ompfileProxyTraceEnabled() {
  static const bool Enabled = []() {
    const char *Env = std::getenv("LIBOMPFILE_DEBUG_TRACE");
    return Env && Env[0] == '1' && Env[1] == '\0';
  }();
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

std::string envStringOrDefault(const char *Name, const char *DefaultValue) {
  const char *Value = std::getenv(Name);
  if (!Value || Value[0] == '\0')
    return DefaultValue ? std::string(DefaultValue) : std::string();
  return Value;
}

uint64_t elapsedMicros(std::chrono::steady_clock::time_point Start,
                       std::chrono::steady_clock::time_point End) {
  if (End <= Start)
    return 0;
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(End - Start)
          .count());
}

uint64_t envUint64OrDefault(const char *Name, uint64_t DefaultValue) {
  const char *Value = std::getenv(Name);
  if (!Value || Value[0] == '\0')
    return DefaultValue;
  char *End = nullptr;
  errno = 0;
  unsigned long long Parsed = std::strtoull(Value, &End, 10);
  if (errno != 0 || End == Value || (End && *End != '\0'))
    return DefaultValue;
  return static_cast<uint64_t>(Parsed);
}

double envDoubleOrDefault(const char *Name, double DefaultValue) {
  const char *Value = std::getenv(Name);
  if (!Value || Value[0] == '\0')
    return DefaultValue;
  char *End = nullptr;
  errno = 0;
  double Parsed = std::strtod(Value, &End);
  if (errno != 0 || End == Value || (End && *End != '\0'))
    return DefaultValue;
  return Parsed;
}

bool envBoolOrDefault(const char *Name, bool DefaultValue) {
  const char *Value = std::getenv(Name);
  if (!Value || Value[0] == '\0')
    return DefaultValue;
  if (Value[0] == '0' && Value[1] == '\0')
    return false;
  if (Value[0] == '1' && Value[1] == '\0')
    return true;
  return DefaultValue;
}

// Normalize the stage write policy to one of: "write-through",
// "write-back", or "off". Unknown values fall back to "write-through"
// so a misspelled knob never silently disables staging correctness.
std::string normalizeStageWriteMode(const std::string &Raw) {
  if (Raw == "write-back" || Raw == "writeback")
    return "write-back";
  if (Raw == "off" || Raw == "disabled")
    return "off";
  return "write-through";
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
}

/// Class that holds a stage pointer for data transfer between host and remote
/// device (RAII)
struct PluginDataHandle {
  void *HstPtr;
  uint32_t Plugin;
  uint32_t Device;
  RemotePluginManager *PM;
  PluginDataHandle(RemotePluginManager *PluginManager, uint32_t PluginId,
                   uint32_t DeviceId, uint64_t Size) {
    Device = DeviceId;
    Plugin = PluginId;
    PM = PluginManager;
    // HstPtr = PM->Plugins[Plugin]->data_alloc(Device, Size, nullptr,
    //                                          TARGET_ALLOC_HOST);
    HstPtr = malloc(Size);
  }
  ~PluginDataHandle() {
    // PM->Plugins[Plugin]->data_delete(Device, HstPtr, TARGET_ALLOC_HOST);
    free(HstPtr);
  }
};

struct AsyncInfoHandle {
  std::unique_ptr<__tgt_async_info> AsyncInfoPtr;
  std::mutex AsyncInfoMutex;
  AsyncInfoHandle() { AsyncInfoPtr = std::make_unique<__tgt_async_info>(); }
  AsyncInfoHandle(AsyncInfoHandle &&Other) {
    AsyncInfoPtr = std::move(Other.AsyncInfoPtr);
    Other.AsyncInfoPtr = nullptr;
  }
};

/// Event Implementations on Device side.
struct ProxyDevice {
  struct OmpFileHandleEntry {
    int Rank = -1;
    int RemoteHandle = -1;
  };

  struct OmpFileTrackedFdEntry {
    std::string Path;
    int Flags = 0;
    int Mode = 0;
  };

  struct OmpFileOpenCacheEntry {
    int Fd = -1;
    uint64_t RefCount = 0;
  };

  struct OmpFileStageExtent {
    uint64_t Begin = 0;
    uint64_t End = 0;
  };

  struct OmpFileStageEntry {
    std::string SourcePath;
    std::string StagePath;
    int StageFd = -1;
    std::mutex Mutex;
    std::condition_variable Cond;
    std::vector<OmpFileStageExtent> CoveredExtents;
    bool FullyPopulated = false;
    bool PopulateInProgress = false;
    bool Invalidated = false;
    uint64_t SourceSize = 0;
    bool SourceSizeKnown = false;
    // Write-back dirty-region tracking. DirtyExtents are byte ranges in the
    // stage file that have been written but not yet flushed to the source
    // filesystem. DirtyBytes is the sum of (End - Begin) over DirtyExtents.
    // DirtyEpoch is the last write-back freshness epoch committed to the
    // headnode for this entry; it advances on each write-back capture and is
    // checked on flush completion. These fields are only mutated while the
    // caller holds Entry->Mutex.
    std::vector<OmpFileStageExtent> DirtyExtents;
    uint64_t DirtyBytes = 0;
    uint64_t DirtyEpoch = 0;

    ~OmpFileStageEntry() {
      if (StageFd >= 0)
        (void)::close(StageFd);
      if (!StagePath.empty())
        (void)::unlink(StagePath.c_str());
    }
  };

  struct OmpFileStagePopulateStats {
    uint64_t CopiedBytes = 0;
    uint64_t CopyUs = 0;
    uint64_t FsyncUs = 0;
    uint64_t ReopenUs = 0;
    uint64_t PopulateUs = 0;
  };

  struct OmpFilePendingWritebackCommit {
    uint64_t Count = 0;
    uint64_t LastVersion = 0;
  };

  [[noreturn]] void fatalStageConfig(const char *Reason) {
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

  bool validateStageConfigFast() {
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

  void traceOmpFile(const char *Fmt, ...) const {
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

  ProxyDevice()
      : NumExecEventHandlers("OMPTARGET_NUM_EXEC_EVENT_HANDLERS", 1),
        NumDataEventHandlers("OMPTARGET_NUM_DATA_EVENT_HANDLERS", 1),
        EventPollingRate("OMPTARGET_EVENT_POLLING_RATE", 0),
        OmpFileOpenCacheEnable("LIBOMPFILE_OPT_OPEN_CACHE", false),
        OmpFileOpenCacheKeepOpen("LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN", true),
        OmpFileOptStats("LIBOMPFILE_OPT_STATS", false),
        OmpFileForceBlockingPwrite("LIBOMPFILE_MPP_FORCE_BLOCKING_PWRITE",
                                   false),
        OmpFileOpenEioRetries("OMPTARGET_OMPFILE_OPEN_EIO_RETRIES", 16),
        OmpFileOpenEioBackoffUs("OMPTARGET_OMPFILE_OPEN_EIO_BACKOFF_US",
                                 200000),
        OmpFileMPIFragmentSize("OMPTARGET_MPI_FRAGMENT_SIZE", 100e6),
        OmpFileHeadnodeScheduler(
            envStringOrDefault("LIBOMPFILE_SCHEDULER", "LOCAL") ==
            "HEADNODE"),
        OmpFileStageMode(envStringOrDefault("LIBOMPFILE_STAGE_MODE", "off")),
        OmpFileStageWriteMode(normalizeStageWriteMode(
            envStringOrDefault("LIBOMPFILE_STAGE_WRITE_MODE", "write-through"))),
        OmpFileStageSyncPolicy(
            envStringOrDefault("LIBOMPFILE_STAGE_SYNC_POLICY", "cache")),
        OmpFileWritethroughFsyncPolicy(envStringOrDefault(
            "LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY", "each")),
        OmpFileStagePopulateMode(
            envStringOrDefault("LIBOMPFILE_STAGE_POPULATE_MODE", "windowed")),
        OmpFileStageRootPolicy(
            envStringOrDefault("LIBOMPFILE_STAGE_ROOT_POLICY", "topology")),
        OmpFileTopologyFile(
            envStringOrDefault("LIBOMPFILE_TOPOLOGY_FILE", "")),
        OmpFileStageDirName(
            envStringOrDefault("LIBOMPFILE_STAGE_DIR_NAME", "libompfile-stage")),
        OmpFileStageRunStem(defaultStageRunStem()),
        OmpFileStorageEnvironment(
            envStringOrDefault("LIBOMPFILE_STORAGE_ENVIRONMENT", "unspecified")),
        OmpFileStageMinFreeBytes(
            envUint64OrDefault("LIBOMPFILE_STAGE_MIN_FREE_BYTES", 0)),
        OmpFileStageWindowBytes(
            envUint64OrDefault("LIBOMPFILE_STAGE_WINDOW_BYTES", 4ULL << 20)),
        OmpFileStageWindowScale(envDoubleOrDefault(
            "LIBOMPFILE_STAGE_WINDOW_SCALE", 1.0)),
        OmpFileStageDirtyWatermarkBytes(envUint64OrDefault(
            "LIBOMPFILE_STAGE_DIRTY_WATERMARK_BYTES", 0)),
        OmpFileStageFreshnessGuard(envBoolOrDefault(
            "LIBOMPFILE_STAGE_FRESHNESS_GUARD", true)),
        OmpFileStageLocalHost(getLocalShortHostname()) {
#ifdef OMPT_SUPPORT
    // Initialize OMPT first
    llvm::omp::target::ompt::connectLibrary();
#endif

    EventSystem.initialize();
    PluginManager.init();
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      OmpFileHeadnodeManager::instance().initialize(EventSystem.WorldSize,
                                                    getHeadnodeRank());
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
              "keep_open=%d stats=%d blocking_pwrite=%d fragment_size=%lld\n",
              EventSystem.LocalRank, (int)OmpFileOpenCacheEnable.get(),
              (int)OmpFileOpenCacheKeepOpen.get(), (int)OmpFileOptStats.get(),
              (int)OmpFileForceBlockingPwrite.get(),
              static_cast<long long>(OmpFileMPIFragmentSize.get()));
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
              "stage_freshness_guard=%d stage_populate_mode=%s "
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
    {
      const std::lock_guard<std::mutex> Lock(ActiveProxyDeviceMutex);
      ActiveProxyDevice = this;
    }
  }

  ~ProxyDevice() {
    drainOmpFileOpenCache();
    drainOmpFileStageCache();
    reportOmpFileStats("proxy-dtor");
    {
      const std::lock_guard<std::mutex> Lock(ActiveProxyDeviceMutex);
      if (ActiveProxyDevice == this)
        ActiveProxyDevice = nullptr;
    }
    EventSystem.deinitialize();
    PluginManager.deinit();
  }

  void mapDevicesPerRemote() {
    EventSystem.DevicesPerRemote = {};
    for (int PluginId = 0; PluginId < PluginManager.getNumUsedPlugins();
         PluginId++) {
      EventSystem.DevicesPerRemote.emplace_back(
          PluginManager.getNumDevices(PluginId));
    }
  }

  AsyncInfoHandle *MapAsyncInfo(void *HostAsyncInfoPtr) {
    const std::lock_guard<std::mutex> Lock(TableMutex);
    AsyncInfoHandle *AsyncInfoHandlerPtr = nullptr;

    if (AsyncInfoTable[HostAsyncInfoPtr])
      AsyncInfoHandlerPtr = AsyncInfoTable[HostAsyncInfoPtr];
    else {
      AsyncInfoHandlerPtr =
          AsyncInfoList.emplace_back(std::make_unique<AsyncInfoHandle>()).get();
      AsyncInfoTable[HostAsyncInfoPtr] = AsyncInfoHandlerPtr;
    }

    return AsyncInfoHandlerPtr;
  }

  EventTy waitAsyncOpEnd(int32_t PluginId, int32_t DeviceId,
                         void *AsyncInfoPtr) {
    auto *TgtAsyncInfo = MapAsyncInfo(AsyncInfoPtr)->AsyncInfoPtr.get();
    auto *RPCServer =
        PluginManager.Plugins[PluginId]->getDevice(DeviceId).getRPCServer();

    while (TgtAsyncInfo->Queue != nullptr) {
      if (PluginManager.Plugins[PluginId]->query_async(
              DeviceId, TgtAsyncInfo) == OFFLOAD_FAIL)
        co_return createError("Failure to wait AsyncOp\n");

      if (RPCServer)
        if (auto Err = RPCServer->runServer(
                PluginManager.Plugins[PluginId]->getDevice(DeviceId)))
          co_return Err;
      co_await std::suspend_always{};
    }

    co_return llvm::Error::success();
  }

  EventTy retrieveNumDevices(MPIRequestManagerTy RequestManager) {
    int32_t NumDevices = PluginManager.getNumDevices();
    RequestManager.send(&NumDevices, 1, MPI_INT);

    co_return (co_await RequestManager);
  }

  EventTy isPluginCompatible(MPIRequestManagerTy RequestManager) {
    __tgt_device_image Image;
    bool QueryResult = false;

    uint64_t Size = 0;

    RequestManager.receive(&Size, 1, MPI_UINT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageStart = memAllocHost(Size);
    RequestManager.receive(Image.ImageStart, Size, MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageEnd = (void *)((ptrdiff_t)(Image.ImageStart) + Size);

    llvm::SmallVector<std::unique_ptr<GenericPluginTy>> UsedPlugins;

    for (auto &Plugin : PluginManager.Plugins) {
      QueryResult = Plugin->is_plugin_compatible(&Image);
      if (QueryResult) {
        UsedPlugins.emplace_back(std::move(Plugin));
        break;
      }
    }

    for (auto &Plugin : PluginManager.Plugins) {
      if (Plugin)
        UsedPlugins.emplace_back(std::move(Plugin));
    }

    PluginManager.Plugins = std::move(UsedPlugins);
    mapDevicesPerRemote();

    memFreeHost(Image.ImageStart);
    RequestManager.send(&QueryResult, sizeof(bool), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy isDeviceCompatible(MPIRequestManagerTy RequestManager) {
    __tgt_device_image Image;
    bool QueryResult = false;

    uint64_t Size = 0;

    RequestManager.receive(&Size, 1, MPI_UINT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageStart = memAllocHost(Size);
    RequestManager.receive(Image.ImageStart, Size, MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageEnd = (void *)((ptrdiff_t)(Image.ImageStart) + Size);

    int32_t DeviceId, PluginId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    QueryResult =
        PluginManager.Plugins[PluginId]->is_device_compatible(DeviceId, &Image);

    memFreeHost(Image.ImageStart);
    RequestManager.send(&QueryResult, sizeof(bool), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy initDevice(MPIRequestManagerTy RequestManager) {
    int32_t DeviceId, PluginId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->init_device(DeviceId);

    auto *DevicePtr = &PluginManager.Plugins[PluginId]->getDevice(DeviceId);

    // Event completion notification
    RequestManager.send(&DevicePtr, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy initRecordReplay(MPIRequestManagerTy RequestManager) {
    int64_t MemorySize = 0;
    void *VAddr = nullptr;
    bool IsRecord = false, SaveOutput = false;
    uint64_t ReqPtrArgOffset = 0;

    RequestManager.receive(&MemorySize, 1, MPI_INT64_T);
    RequestManager.receive(&VAddr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&IsRecord, sizeof(bool), MPI_BYTE);
    RequestManager.receive(&SaveOutput, sizeof(bool), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t DeviceId, PluginId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->initialize_record_replay(
        DeviceId, MemorySize, VAddr, IsRecord, SaveOutput, ReqPtrArgOffset);

    RequestManager.send(&ReqPtrArgOffset, 1, MPI_UINT64_T);
    co_return (co_await RequestManager);
  }

  EventTy isDataExchangable(MPIRequestManagerTy RequestManager) {
    int32_t DstDeviceId = 0;
    bool QueryResult = false;
    RequestManager.receive(&DstDeviceId, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t SrcDeviceId, PluginId;

    std::tie(PluginId, SrcDeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    QueryResult = PluginManager.Plugins[PluginId]->isDataExchangable(
        SrcDeviceId, DstDeviceId);

    RequestManager.send(&QueryResult, sizeof(bool), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy allocateBuffer(MPIRequestManagerTy RequestManager) {
    int64_t Size = 0;
    int32_t Kind = 0;
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&Kind, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    void *Buffer = PluginManager.Plugins[PluginId]->data_alloc(DeviceId, Size,
                                                               nullptr, Kind);

    RequestManager.send(&Buffer, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy deleteBuffer(MPIRequestManagerTy RequestManager) {
    void *Buffer = nullptr;
    int32_t Kind = 0;

    RequestManager.receive(&Buffer, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Kind, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_delete(DeviceId, Buffer, Kind);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy submit(MPIRequestManagerTy RequestManager) {
    void *TgtPtr = nullptr, *HstAsyncInfoPtr = nullptr;
    int64_t Size = 0;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    RequestManager.receive(&TgtPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginDataHandle DataHandler(&PluginManager, PluginId, DeviceId, Size);
    RequestManager.receiveInBatchs(DataHandler.HstPtr, Size);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    // Issues data transfer on the device
    // Only one call at a time to avoid the
    // risk of overwriting device queues
    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_submit_async(
          DeviceId, TgtPtr, DataHandler.HstPtr, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy retrieve(MPIRequestManagerTy RequestManager) {
    void *TgtPtr = nullptr, *HstAsyncInfoPtr = nullptr;
    int64_t Size = 0;
    bool DeviceOpStatus = true;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    RequestManager.receive(&TgtPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginDataHandle DataHandler(&PluginManager, PluginId, DeviceId, Size);

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_retrieve_async(
          DeviceId, DataHandler.HstPtr, TgtPtr, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    if (auto Error =
            co_await waitAsyncOpEnd(PluginId, DeviceId, HstAsyncInfoPtr);
        Error)
      REPORT("Retrieve event failed with msg: %s\n",
             toString(std::move(Error)).data());

    RequestManager.send(&DeviceOpStatus, sizeof(bool), MPI_BYTE);

    if (!DeviceOpStatus)
      co_return (co_await RequestManager);

    RequestManager.sendInBatchs(DataHandler.HstPtr, Size);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy exchange(MPIRequestManagerTy RequestManager) {
    void *SrcPtr = nullptr, *DstPtr = nullptr;
    int DstDeviceId = 0;
    int64_t Size = 0;
    void *HstAsyncInfoPtr = nullptr;

    RequestManager.receive(&SrcPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&DstDeviceId, 1, MPI_INT);
    RequestManager.receive(&DstPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, SrcDeviceId;

    std::tie(PluginId, SrcDeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_exchange_async(
          SrcDeviceId, SrcPtr, DstDeviceId, DstPtr, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy exchangeSrc(MPIRequestManagerTy RequestManager) {
    void *SrcBuffer, *HstAsyncInfoPtr = nullptr;
    int64_t Size;
    int DstRank;

    // Save head node rank
    int HeadNodeRank = RequestManager.OtherRank;

    RequestManager.receive(&SrcBuffer, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&DstRank, 1, MPI_INT);
    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginDataHandle DataHandler(&PluginManager, PluginId, DeviceId, Size);

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_retrieve_async(
          DeviceId, DataHandler.HstPtr, SrcBuffer, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    if (auto Error =
            co_await waitAsyncOpEnd(PluginId, DeviceId, HstAsyncInfoPtr);
        Error)
      co_return Error;

    // Set the Destination Rank in RequestManager
    RequestManager.OtherRank = DstRank;

    // Send buffer to target device
    RequestManager.sendInBatchs(DataHandler.HstPtr, Size);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // Set the HeadNode Rank to send the final notificatin
    RequestManager.OtherRank = HeadNodeRank;

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy exchangeDst(MPIRequestManagerTy RequestManager) {
    void *DstBuffer, *HstAsyncInfoPtr = nullptr;
    int64_t Size;
    // Save head node rank
    int SrcRank, HeadNodeRank = RequestManager.OtherRank;

    RequestManager.receive(&DstBuffer, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&SrcRank, 1, MPI_INT);
    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginDataHandle DataHandler(&PluginManager, PluginId, DeviceId, Size);

    // Set the Source Rank in RequestManager
    RequestManager.OtherRank = SrcRank;

    // Receive buffer from the Source device
    RequestManager.receiveInBatchs(DataHandler.HstPtr, Size);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_submit_async(
          DeviceId, DstBuffer, DataHandler.HstPtr, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    if (auto Error =
            co_await waitAsyncOpEnd(PluginId, DeviceId, HstAsyncInfoPtr);
        Error)
      co_return Error;

    // Set the HeadNode Rank to send the final notificatin
    RequestManager.OtherRank = HeadNodeRank;

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy launchKernel(MPIRequestManagerTy RequestManager) {
    void *TgtEntryPtr = nullptr, *HostAsyncInfoPtr = nullptr;
    KernelArgsTy KernelArgs;

    llvm::SmallVector<void *> TgtArgs;
    llvm::SmallVector<ptrdiff_t> TgtOffsets;

    uint32_t NumArgs = 0;

    RequestManager.receive(&NumArgs, 1, MPI_UINT32_T);
    RequestManager.receive(&HostAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    TgtArgs.resize(NumArgs);
    TgtOffsets.resize(NumArgs);

    RequestManager.receive(&TgtEntryPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(TgtArgs.data(), NumArgs * sizeof(void *), MPI_BYTE);
    RequestManager.receive(TgtOffsets.data(), NumArgs * sizeof(ptrdiff_t),
                           MPI_BYTE);

    RequestManager.receive(&KernelArgs, sizeof(KernelArgsTy), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    auto *TgtAsyncInfo = MapAsyncInfo(HostAsyncInfoPtr)->AsyncInfoPtr.get();

    PluginManager.Plugins[PluginId]->launch_kernel(
        DeviceId, TgtEntryPtr, TgtArgs.data(), TgtOffsets.data(), &KernelArgs,
        TgtAsyncInfo);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy loadBinary(MPIRequestManagerTy RequestManager) {
    // Receive the target table sizes.
    size_t ImageSize = 0;
    size_t EntryCount = 0;
    RequestManager.receive(&ImageSize, 1, MPI_UINT64_T);
    RequestManager.receive(&EntryCount, 1, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    llvm::SmallVector<size_t> EntryNameSizes(EntryCount);

    RequestManager.receive(EntryNameSizes.begin(), EntryCount, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // Create the device name with the appropriate sizes and receive its
    // content.
    DeviceImage *Image =
        RemoteImages
            .emplace_back(std::make_unique<DeviceImage>(ImageSize, EntryCount))
            .get();

    Image->setImageEntries(EntryNameSizes);

    // Received the image bytes and the table entries.
    RequestManager.receive(Image->ImageStart, ImageSize, MPI_BYTE);

    for (size_t I = 0; I < EntryCount; I++) {
      RequestManager.receive(&Image->Entries[I].addr, 1, MPI_UINT64_T);
      RequestManager.receive(Image->Entries[I].name, EntryNameSizes[I],
                             MPI_CHAR);
      RequestManager.receive(&Image->Entries[I].size, 1, MPI_UINT64_T);
      RequestManager.receive(&Image->Entries[I].flags, 1, MPI_INT32_T);
      RequestManager.receive(&Image->Entries[I].data, 1, MPI_INT32_T);
    }

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    __tgt_device_binary Binary;

    PluginManager.Plugins[PluginId]->load_binary(DeviceId, Image, &Binary);

    RequestManager.send(&Binary.handle, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy getGlobal(MPIRequestManagerTy RequestManager) {
    __tgt_device_binary Binary;
    uint64_t Size = 0;
    llvm::SmallVector<char> Name;
    void *DevicePtr = nullptr;
    uint32_t NameSize = 0;

    RequestManager.receive(&Binary.handle, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_UINT64_T);
    RequestManager.receive(&NameSize, 1, MPI_UINT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    Name.resize(NameSize);
    RequestManager.receive(Name.data(), NameSize, MPI_CHAR);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->get_global(Binary, Size, Name.data(),
                                                &DevicePtr);

    RequestManager.send(&DevicePtr, sizeof(void *), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy getFunction(MPIRequestManagerTy RequestManager) {
    __tgt_device_binary Binary;
    uint32_t Size = 0;
    llvm::SmallVector<char> Name;
    void *KernelPtr = nullptr;

    RequestManager.receive(&Binary.handle, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_UINT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    Name.resize(Size);
    RequestManager.receive(Name.data(), Size, MPI_CHAR);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->get_function(Binary, Name.data(),
                                                  &KernelPtr);

    RequestManager.send(&KernelPtr, sizeof(void *), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy synchronize(MPIRequestManagerTy RequestManager) {
    void *HstAsyncInfoPtr = nullptr;
    bool DeviceOpStatus = true;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    if (auto Error =
            co_await waitAsyncOpEnd(PluginId, DeviceId, HstAsyncInfoPtr);
        Error)
      REPORT("Synchronize event failed with msg: %s\n",
             toString(std::move(Error)).data());

    RequestManager.send(&DeviceOpStatus, sizeof(bool), MPI_BYTE);

    if (!DeviceOpStatus)
      co_return (co_await RequestManager);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy ompfilePing(MPIRequestManagerTy RequestManager) {
    uint64_t Token = 0;

    RequestManager.receive(&Token, 1, MPI_UINT64_T);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    RequestManager.send(&Token, 1, MPI_UINT64_T);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  bool canUseOmpFileOpenCache(int Flags) const {
    if (!OmpFileOpenCacheEnable)
      return false;
    if (Flags & (O_CREAT | O_EXCL | O_TRUNC))
      return false;
    const int AccessMode = Flags & O_ACCMODE;
    if (AccessMode != O_RDONLY && AccessMode != O_WRONLY &&
        AccessMode != O_RDWR)
      return false;
    // Shared writable file descriptors have produced incorrect distributed
    // read-after-write results in the remote-only MPP path. Keep the cache for
    // read-only opens, but force independent open file descriptions for any
    // handle that may write.
    return AccessMode == O_RDONLY;
  }

  bool isStageEnabled() const {
    return OmpFileStageMode != "off" && OmpFileTopologyLoaded &&
           !OmpFileStageRoot.empty();
  }

  bool isReadthroughStageEnabled() const {
    return OmpFileStageMode == "readthrough" && isStageEnabled();
  }

  bool writethroughFsyncEachEnabled() const {
    return OmpFileWritethroughFsyncPolicy != "close";
  }

  bool isWritethroughStageEnabled() const {
    return isStageEnabled() && OmpFileStageWriteMode == "write-through";
  }

  // Write-back mode: writes land in the per-proxy stage file first and are
  // flushed to the source filesystem on close or at the dirty watermark.
  // The headnode freshness layer already records Dirty=true for write-back
  // commits; the proxy-side capture and flush wiring arrive in later tasks.
  bool isWritebackStageEnabled() const {
    return isStageEnabled() && OmpFileStageWriteMode == "write-back";
  }

  void rememberWritebackFreshnessCommit(uint64_t PathKey, uint64_t Version) {
    if (PathKey == 0 || Version == 0)
      return;
    std::lock_guard<std::mutex> Lock(OmpFileWritebackCommitMutex);
    auto &Pending = OmpFilePendingWritebackCommits[PathKey];
    ++Pending.Count;
    Pending.LastVersion = Version;
    OmpFileLocalFreshnessVersionByPath[PathKey] = Version;
  }

  bool consumeDuplicateWriteThroughFreshnessCommit(uint64_t PathKey,
                                                   uint64_t &VersionOut) {
    VersionOut = 0;
    if (PathKey == 0)
      return false;
    std::lock_guard<std::mutex> Lock(OmpFileWritebackCommitMutex);
    auto It = OmpFilePendingWritebackCommits.find(PathKey);
    if (It == OmpFilePendingWritebackCommits.end() || It->second.Count == 0)
      return false;
    VersionOut = It->second.LastVersion;
    --It->second.Count;
    if (It->second.Count == 0)
      OmpFilePendingWritebackCommits.erase(It);
    return VersionOut != 0;
  }

  // Return the last freshness version this proxy is known to hold for
  // the given PathKey, or 0 when no local freshness epoch was observed.
  uint64_t lastLocalFreshnessVersionForPathKey(uint64_t PathKey) {
    if (PathKey == 0)
      return 0;
    std::lock_guard<std::mutex> Lock(OmpFileWritebackCommitMutex);
    auto It = OmpFileLocalFreshnessVersionByPath.find(PathKey);
    if (It == OmpFileLocalFreshnessVersionByPath.end())
      return 0;
    return It->second;
  }

  uint64_t tileFreshnessKeyForPath(uint64_t PathKey, uint64_t TileId) const {
    if (PathKey == 0)
      return 0;
    uint64_t X = PathKey ^ 0x9e3779b97f4a7c15ULL;
    X ^= TileId + 0x9e3779b97f4a7c15ULL + (X << 6) + (X >> 2);
    return X != 0 ? X : PathKey;
  }

  uint64_t configuredStageTileBytes() const {
    if (uint64_t Bytes = envUint64OrDefault("LIBOMPFILE_STAGE_TILE_BYTES", 0))
      return Bytes;
    const uint64_t Nb = envUint64OrDefault("OMPFILE_CHOLESKY_NB", 0);
    if (Nb == 0 || Nb > (std::numeric_limits<uint64_t>::max() / Nb) ||
        (Nb * Nb) > (std::numeric_limits<uint64_t>::max() / sizeof(float)))
      return 0;
    return Nb * Nb * sizeof(float);
  }

  bool dirtyOwnerExpectedVersionMatchesRange(
      const std::string &SourcePath, uint64_t Begin, uint64_t End,
      uint64_t ExpectedVersion, const OmpFileStageEntry &Entry) {
    if (ExpectedVersion == 0)
      return true;

    const uint64_t PathKey =
        OmpFileHeadnodeManager::computePathKeyForPath(SourcePath);
    const uint64_t TileBytes = configuredStageTileBytes();
    if (PathKey != 0 && TileBytes != 0 && End > Begin &&
        Begin % TileBytes == 0 && End == Begin + TileBytes) {
      const uint64_t TileId = Begin / TileBytes;
      const uint64_t CompositeKey = tileFreshnessKeyForPath(PathKey, TileId);
      const uint64_t LocalVersion =
          lastLocalFreshnessVersionForPathKey(CompositeKey);
      if (LocalVersion == ExpectedVersion)
        return true;

      OmpFileFreshnessQueryRequest Query{};
      Query.AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
      Query.PathKey = CompositeKey;
      Query.LocalVersion = LocalVersion;
      Query.RequesterRank = EventSystem.LocalRank;
      OmpFileFreshnessQueryReply Reply{};
      if (!freshnessQueryOnHeadnode(Query, Reply) || Reply.Status != 0)
        return false;
      const auto Decision =
          static_cast<OmpFileFreshnessDecision>(Reply.Decision);
      const bool NamesThisRank =
          (Decision == OmpFileFreshnessDecision::USE_LOCAL) ||
          (Decision == OmpFileFreshnessDecision::COPY_FROM_RANK &&
           Reply.SourceRank == EventSystem.LocalRank);
      if (!NamesThisRank || Reply.SelectedVersion != ExpectedVersion)
        return false;
      rememberWritebackFreshnessCommit(CompositeKey, ExpectedVersion);
      return true;
    }

    // Whole-file (hint-less) fallback: DirtyEpoch is entry-global and is
    // bumped by EVERY capture on this file, so with two concurrent writers an
    // exact match races — writer B's unrelated capture advances DirtyEpoch
    // past the version reader A was told to expect, the guard rejects the
    // stage, and the read falls back to a PFS copy that write-back has
    // deliberately not flushed yet (stale read-after-write, OOC repro jobs
    // AMD 350175/350180, sorgan 3655). Monotonic comparison is safe here:
    // capture writes the stage bytes *before* committing the epoch, so
    // DirtyEpoch >= ExpectedVersion implies every commit up to
    // ExpectedVersion already has its bytes in this stage, and the caller
    // separately enforces dirty-extent coverage for the requested range.
    return Entry.DirtyEpoch >= ExpectedVersion;
  }

  bool completeDerivedTileFlushesOnHeadnode(
      uint64_t PathKey, const std::vector<OmpFileStageExtent> &Extents,
      uint64_t TileBytes, const std::string &SourcePath) {
    if (PathKey == 0 || TileBytes == 0 || Extents.empty())
      return false;

    for (const OmpFileStageExtent &Extent : Extents) {
      if (Extent.End <= Extent.Begin || Extent.Begin % TileBytes != 0 ||
          Extent.End % TileBytes != 0) {
        errno = ESTALE;
        return false;
      }
      for (uint64_t TileBegin = Extent.Begin; TileBegin < Extent.End;
           TileBegin += TileBytes) {
        const uint64_t TileId = TileBegin / TileBytes;
        const uint64_t CompositeKey = tileFreshnessKeyForPath(PathKey, TileId);
        if (CompositeKey == 0) {
          errno = EINVAL;
          return false;
        }

        uint64_t CompleteVersion =
            lastLocalFreshnessVersionForPathKey(CompositeKey);
        bool Completed = false;
        if (CompleteVersion != 0) {
          errno = 0;
          Completed = completeDirtyFlushOnHeadnode(
              CompositeKey, EventSystem.LocalRank, CompleteVersion,
              /*Success=*/true);
          if (!Completed && errno != ESTALE)
            return false;
        }
        if (!Completed) {
          OmpFileFreshnessQueryRequest Query{};
          Query.AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
          Query.PathKey = CompositeKey;
          Query.LocalVersion = CompleteVersion;
          Query.RequesterRank = EventSystem.LocalRank;
          OmpFileFreshnessQueryReply Reply{};
          if (!freshnessQueryOnHeadnode(Query, Reply) || Reply.Status != 0) {
            errno = Reply.Errno != 0 ? Reply.Errno : (errno != 0 ? errno : EIO);
            return false;
          }
          const auto Decision =
              static_cast<OmpFileFreshnessDecision>(Reply.Decision);
          if (Decision == OmpFileFreshnessDecision::READ_PFS)
            continue;
          const bool NamesThisRank =
              (Decision == OmpFileFreshnessDecision::USE_LOCAL) ||
              (Decision == OmpFileFreshnessDecision::COPY_FROM_RANK &&
               Reply.SourceRank == EventSystem.LocalRank);
          if (!NamesThisRank || Reply.SelectedVersion == 0) {
            errno = ESTALE;
            return false;
          }
          CompleteVersion = Reply.SelectedVersion;
          rememberWritebackFreshnessCommit(CompositeKey, CompleteVersion);
          if (!completeDirtyFlushOnHeadnode(CompositeKey, EventSystem.LocalRank,
                                            CompleteVersion,
                                            /*Success=*/true))
            return false;
        }
        REPORT("MPIProxyDevice --> OMPFile derived tile dirty flush complete "
               "rank=%d path=%s tile=%llu key=%llu version=%llu\n",
               EventSystem.LocalRank, SourcePath.c_str(),
               static_cast<unsigned long long>(TileId),
               static_cast<unsigned long long>(CompositeKey),
               static_cast<unsigned long long>(CompleteVersion));
      }
    }
    return true;
  }

  // Consult the headnode freshness table before serving a staged read from
  // a local stage.  Returns true when it is safe to serve from the local
  // stage (USE_LOCAL: this proxy holds the latest version, including its own
  // dirty write-back data).  Returns false when the caller must bypass the
  // stage and read from the source fd instead (READ_PFS, COPY_FROM_RANK, or
  // WAIT_OR_FAIL: another proxy may hold dirty data, or PFS currency cannot
  // be confirmed for this proxy's stage).
  bool stageFreshnessGuardPermitsLocal(
      const std::string &SourcePath,
      const std::shared_ptr<OmpFileStageEntry> &StageEntry) {
    if (!isWritebackStageEnabled() || !OmpFileStageFreshnessGuard)
      return true;
    if (SourcePath.empty())
      return true;
    const uint64_t PathKey =
        OmpFileHeadnodeManager::computePathKeyForPath(SourcePath);
    if (PathKey == 0)
      return true;

    OmpFileFreshnessQueryRequest Request{};
    Request.AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
    Request.PathKey = PathKey;
    Request.LocalVersion = lastLocalFreshnessVersionForPathKey(PathKey);
    Request.RequesterRank = EventSystem.LocalRank;

    OmpFileFreshnessQueryReply Reply{};
    if (!freshnessQueryOnHeadnode(Request, Reply)) {
      // If the freshness query itself fails, conservatively bypass the
      // stage rather than risk serving stale data.
      OmpFileStatsStageFreshnessGuardBypasses.fetch_add(
          1, std::memory_order_relaxed);
      traceOmpFile("stage freshness guard query failed path=%s path_key=%llu "
                   "errno=%d; bypassing stage\n",
                   SourcePath.c_str(),
                   static_cast<unsigned long long>(PathKey), errno);
      return false;
    }

    if (Reply.Status != 0) {
      OmpFileStatsStageFreshnessGuardBypasses.fetch_add(
          1, std::memory_order_relaxed);
      traceOmpFile("stage freshness guard query error path=%s path_key=%llu "
                   "status=%d errno=%d; bypassing stage\n",
                   SourcePath.c_str(),
                   static_cast<unsigned long long>(PathKey), Reply.Status,
                   Reply.Errno);
      return false;
    }

    const auto Decision =
        static_cast<OmpFileFreshnessDecision>(Reply.Decision);
    if (Decision == OmpFileFreshnessDecision::USE_LOCAL) {
      traceOmpFile("stage freshness guard USE_LOCAL path=%s path_key=%llu "
                   "local_version=%llu selected_version=%llu\n",
                   SourcePath.c_str(),
                   static_cast<unsigned long long>(PathKey),
                   static_cast<unsigned long long>(Request.LocalVersion),
                   static_cast<unsigned long long>(Reply.SelectedVersion));
      return true;
    }

    if (Decision == OmpFileFreshnessDecision::COPY_FROM_RANK &&
        Reply.SourceRank == EventSystem.LocalRank) {
      traceOmpFile("stage freshness guard COPY_FROM_RANK(self) path=%s "
                   "path_key=%llu local_version=%llu selected_version=%llu\n",
                   SourcePath.c_str(),
                   static_cast<unsigned long long>(PathKey),
                   static_cast<unsigned long long>(Request.LocalVersion),
                   static_cast<unsigned long long>(Reply.SelectedVersion));
      return true;
    }

    if (Decision == OmpFileFreshnessDecision::READ_PFS && StageEntry) {
      std::lock_guard<std::mutex> EntryLock(StageEntry->Mutex);
      if (StageEntry->DirtyBytes > 0 || !StageEntry->DirtyExtents.empty()) {
        traceOmpFile("stage freshness guard READ_PFS override path=%s "
                     "path_key=%llu dirty_bytes=%llu extents=%zu\n",
                     SourcePath.c_str(),
                     static_cast<unsigned long long>(PathKey),
                     static_cast<unsigned long long>(StageEntry->DirtyBytes),
                     StageEntry->DirtyExtents.size());
        return true;
      }
    }

    OmpFileStatsStageFreshnessGuardBypasses.fetch_add(
        1, std::memory_order_relaxed);
    traceOmpFile("stage freshness guard bypass path=%s path_key=%llu "
                 "decision=%u source_rank=%d selected_version=%llu; "
                 "reading from source\n",
                 SourcePath.c_str(),
                 static_cast<unsigned long long>(PathKey),
                 Reply.Decision, Reply.SourceRank,
                 static_cast<unsigned long long>(Reply.SelectedVersion));
    return false;
  }

  // Effective window every stage decision should consult. Centralizing this
  // here keeps the adaptive-scale policy in one place as later tasks switch
  // populate/eviction paths from OmpFileStageWindowBytes to the effective
  // value.
  uint64_t effectiveStageWindowBytes() const {
    return OmpFileStageEffectiveWindowBytes > 0
               ? OmpFileStageEffectiveWindowBytes
               : OmpFileStageWindowBytes;
  }

  bool shouldSyncStagePopulate() const {
    return OmpFileStageSyncPolicy == "always";
  }

  bool useWindowedStagePopulate() const {
    return OmpFileStagePopulateMode != "full";
  }

  uint64_t saturatingAdd(uint64_t Base, uint64_t Delta) const {
    if (Delta > std::numeric_limits<uint64_t>::max() - Base)
      return std::numeric_limits<uint64_t>::max();
    return Base + Delta;
  }

  uint64_t alignDown(uint64_t Value, uint64_t Alignment) const {
    if (Alignment == 0)
      return Value;
    return (Value / Alignment) * Alignment;
  }

  uint64_t alignUp(uint64_t Value, uint64_t Alignment) const {
    if (Alignment == 0)
      return Value;
    const uint64_t Remainder = Value % Alignment;
    if (Remainder == 0)
      return Value;
    const uint64_t Delta = Alignment - Remainder;
    return saturatingAdd(Value, Delta);
  }

  bool isCoveredByExtents(const std::vector<OmpFileStageExtent> &Extents,
                          uint64_t Begin, uint64_t End) const {
    if (End <= Begin)
      return true;
    uint64_t Cursor = Begin;
    for (const OmpFileStageExtent &Extent : Extents) {
      if (Extent.End <= Cursor)
        continue;
      if (Extent.Begin > Cursor)
        return false;
      Cursor = std::max(Cursor, Extent.End);
      if (Cursor >= End)
        return true;
    }
    return Cursor >= End;
  }

  bool overlapsAnyExtent(const std::vector<OmpFileStageExtent> &Extents,
                         uint64_t Begin, uint64_t End) const {
    if (End <= Begin)
      return false;
    for (const OmpFileStageExtent &Extent : Extents) {
      if (Extent.End <= Begin)
        continue;
      if (Extent.Begin >= End)
        return false;
      return true;
    }
    return false;
  }

  void addCoveredExtent(std::vector<OmpFileStageExtent> &Extents,
                        uint64_t Begin, uint64_t End) {
    if (End <= Begin)
      return;
    OmpFileStageExtent NewExtent{Begin, End};
    Extents.push_back(NewExtent);
    std::sort(Extents.begin(), Extents.end(),
              [](const OmpFileStageExtent &Lhs, const OmpFileStageExtent &Rhs) {
                return Lhs.Begin < Rhs.Begin;
              });
    std::vector<OmpFileStageExtent> Merged;
    Merged.reserve(Extents.size());
    for (const OmpFileStageExtent &Extent : Extents) {
      if (Merged.empty() || Extent.Begin > Merged.back().End) {
        Merged.push_back(Extent);
        continue;
      }
      Merged.back().End = std::max(Merged.back().End, Extent.End);
    }
    Extents.swap(Merged);
  }

  uint64_t removeCoveredRange(std::vector<OmpFileStageExtent> &Extents,
                              uint64_t Begin, uint64_t End) {
    if (End <= Begin || Extents.empty())
      return 0;

    uint64_t RemovedBytes = 0;
    std::vector<OmpFileStageExtent> Updated;
    Updated.reserve(Extents.size());
    for (const OmpFileStageExtent &Extent : Extents) {
      if (Extent.End <= Begin || Extent.Begin >= End) {
        Updated.push_back(Extent);
        continue;
      }

      const uint64_t OverlapBegin = std::max(Extent.Begin, Begin);
      const uint64_t OverlapEnd = std::min(Extent.End, End);
      if (OverlapEnd > OverlapBegin)
        RemovedBytes += (OverlapEnd - OverlapBegin);

      if (Extent.Begin < Begin)
        Updated.push_back(OmpFileStageExtent{Extent.Begin, Begin});
      if (Extent.End > End)
        Updated.push_back(OmpFileStageExtent{End, Extent.End});
    }
    Extents.swap(Updated);
    return RemovedBytes;
  }

  // --- Write-back dirty-region helpers ---
  // All mark/clear helpers require the caller to hold Entry->Mutex; the
  // "Locked" suffix mirrors invalidateStageEntryLocked. markDirtyRangeLocked
  // is not yet called from the write path (that wiring lands with the
  // write-back capture task). flushDirtyRangesToSource is the real
  // stage->source flush and is wired into closeWithOptionalCache; it is a
  // no-op unless DirtyExtents is nonempty, so the write-through default and
  // the no-capture-yet state are behavior-neutral.

  // Record [Begin, End) as dirty in Entry, merging into DirtyExtents and
  // advancing DirtyBytes. Returns the new total dirty byte count.
  uint64_t markDirtyRangeLocked(OmpFileStageEntry &Entry,
                                uint64_t Begin, uint64_t End) {
    if (End <= Begin)
      return Entry.DirtyBytes;
    const uint64_t Before = Entry.DirtyBytes;
    addCoveredExtent(Entry.DirtyExtents, Begin, End);
    // addCoveredExtent merges overlaps, so recompute the total from the
    // merged vector rather than adding (End - Begin) to avoid double
    // counting overlapping writes.
    uint64_t Total = 0;
    for (const OmpFileStageExtent &Extent : Entry.DirtyExtents)
      Total += (Extent.End - Extent.Begin);
    Entry.DirtyBytes = Total;
    assert(Entry.DirtyBytes >= Before && "dirty bytes shrank on mark");
    const uint64_t Delta = Entry.DirtyBytes - Before;
    if (Delta > 0) {
      OmpFileStatsStageDirtyBytes.fetch_add(Delta,
                                            std::memory_order_relaxed);
    }
    return Entry.DirtyBytes;
  }

  // Remove [Begin, End) from DirtyExtents (e.g. after a successful flush
  // of that range). Returns the number of dirty bytes removed.
  uint64_t clearDirtyRangeLocked(OmpFileStageEntry &Entry,
                                 uint64_t Begin, uint64_t End) {
    if (End <= Begin || Entry.DirtyExtents.empty())
      return 0;
    const uint64_t Removed = removeCoveredRange(Entry.DirtyExtents, Begin, End);
    assert(Entry.DirtyBytes >= Removed && "dirty bytes underflow on clear");
    Entry.DirtyBytes -= Removed;
    if (Removed > 0) {
      OmpFileStatsStageDirtyBytes.fetch_sub(Removed,
                                            std::memory_order_relaxed);
    }
    if (Entry.DirtyExtents.empty())
      Entry.DirtyEpoch = 0;
    return Removed;
  }

  // Drop all dirty extents. Used when a full stage invalidation discards
  // unflushed writes (only safe when the source copy is known current).
  void clearDirtyRangesLocked(OmpFileStageEntry &Entry) {
    if (Entry.DirtyBytes > 0) {
      OmpFileStatsStageDirtyBytes.fetch_sub(Entry.DirtyBytes,
                                            std::memory_order_relaxed);
    }
    Entry.DirtyExtents.clear();
    Entry.DirtyBytes = 0;
    Entry.DirtyEpoch = 0;
  }

  // Total dirty bytes currently held in Entry. Caller must hold Mutex.
  uint64_t dirtyBytesLocked(const OmpFileStageEntry &Entry) const {
    return Entry.DirtyBytes;
  }

  bool flushLocalDirtyTileToSource(uint64_t TilePathKey, uint64_t Version) {
    if (TilePathKey == 0 || Version == 0) {
      errno = EINVAL;
      return false;
    }
    if (lastLocalFreshnessVersionForPathKey(TilePathKey) != Version) {
      errno = ESTALE;
      return false;
    }

    const uint64_t TileBytes = configuredStageTileBytes();
    if (TileBytes == 0) {
      errno = EINVAL;
      return false;
    }

    std::vector<std::shared_ptr<OmpFileStageEntry>> Entries;
    {
      const std::lock_guard<std::mutex> Lock(OmpFileStageMutex);
      Entries.reserve(OmpFileStageEntries.size());
      for (auto &It : OmpFileStageEntries)
        Entries.push_back(It.second);
    }

    for (const std::shared_ptr<OmpFileStageEntry> &Entry : Entries) {
      if (!Entry || Entry->SourcePath.empty())
        continue;
      const uint64_t BasePathKey =
          OmpFileHeadnodeManager::computePathKeyForPath(Entry->SourcePath);
      if (BasePathKey == 0)
        continue;

      int StageFd = -1;
      uint64_t FlushBegin = 0;
      uint64_t FlushEnd = 0;
      {
        std::lock_guard<std::mutex> EntryLock(Entry->Mutex);
        StageFd = Entry->StageFd;
        for (const OmpFileStageExtent &Extent : Entry->DirtyExtents) {
          if (Extent.End <= Extent.Begin)
            continue;
          const uint64_t FirstTile = Extent.Begin / TileBytes;
          const uint64_t LastTile = (Extent.End - 1) / TileBytes;
          for (uint64_t TileId = FirstTile; TileId <= LastTile; ++TileId) {
            if (tileFreshnessKeyForPath(BasePathKey, TileId) != TilePathKey)
              continue;
            FlushBegin = std::max(Extent.Begin, TileId * TileBytes);
            FlushEnd = std::min(Extent.End, (TileId + 1) * TileBytes);
            break;
          }
          if (FlushEnd > FlushBegin)
            break;
        }
      }

      if (FlushEnd <= FlushBegin)
        continue;
      if (StageFd < 0) {
        errno = EBADF;
        return false;
      }

      const int SourceFd = ::open(Entry->SourcePath.c_str(), O_RDWR);
      if (SourceFd < 0)
        return false;

      const uint64_t Len = FlushEnd - FlushBegin;
      std::vector<char> Buffer(Len);
      uint64_t Done = 0;
      while (Done < Len) {
        const ssize_t N = ::pread(StageFd, Buffer.data() + Done,
                                  static_cast<size_t>(Len - Done),
                                  static_cast<off_t>(FlushBegin + Done));
        if (N <= 0) {
          const int SavedErrno = N < 0 ? errno : EIO;
          (void)::close(SourceFd);
          errno = SavedErrno;
          return false;
        }
        Done += static_cast<uint64_t>(N);
      }
      Done = 0;
      while (Done < Len) {
        const ssize_t N = ::pwrite(SourceFd, Buffer.data() + Done,
                                   static_cast<size_t>(Len - Done),
                                   static_cast<off_t>(FlushBegin + Done));
        if (N <= 0) {
          const int SavedErrno = N < 0 ? errno : EIO;
          (void)::close(SourceFd);
          errno = SavedErrno;
          return false;
        }
        Done += static_cast<uint64_t>(N);
      }
      if (::fdatasync(SourceFd) != 0) {
        const int SavedErrno = errno;
        (void)::close(SourceFd);
        errno = SavedErrno;
        return false;
      }
      (void)::close(SourceFd);

      {
        std::lock_guard<std::mutex> EntryLock(Entry->Mutex);
        clearDirtyRangeLocked(*Entry, FlushBegin, FlushEnd);
      }
      OmpFileStatsStageDirtyFlushes.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsStageDirtyFlushBytes.fetch_add(Len,
                                                  std::memory_order_relaxed);
      REPORT("MPIProxyDevice --> OMPFile local dirty tile flushed rank=%d "
             "path=%s key=%llu version=%llu begin=%llu end=%llu\n",
             EventSystem.LocalRank, Entry->SourcePath.c_str(),
             static_cast<unsigned long long>(TilePathKey),
             static_cast<unsigned long long>(Version),
             static_cast<unsigned long long>(FlushBegin),
             static_cast<unsigned long long>(FlushEnd));
      return true;
    }

    errno = ENODATA;
    return false;
  }

  // Flush all dirty ranges of Entry from the stage fd to the source fd and
  // clear them on success. Returns true on success (including the trivial
  // no-dirty case). On any I/O failure the stage entry is invalidated (the
  // source filesystem is the authority of last resort) and false is returned
  // so the caller can fail the close / fall back to PFS. SourcePath is used
  // to compute the headnode path key for the dirty-flush completion that
  // advances PfsVersion; if Entry.DirtyEpoch is zero (no write-back commit
  // recorded yet) the headnode completion is skipped.
  bool flushDirtyRangesToSource(OmpFileStageEntry &Entry, int SourceFd,
                                const std::string &SourcePath,
                                bool DeferCloseMetadataCompletion = false) {
    auto failDirtyFlush = [&](const char *Reason, int ErrnoValue,
                              uint64_t Begin = 0, uint64_t End = 0,
                              uint64_t Epoch = 0) -> bool {
      OmpFileStatsStageDirtyFlushFailures.fetch_add(1,
                                                    std::memory_order_relaxed);
      REPORT("MPIProxyDevice --> OMPFile dirty flush failed rank=%d "
             "reason=%s errno=%d path=%s begin=%llu end=%llu epoch=%llu "
             "source_fd=%d stage_fd=%d\n",
             EventSystem.LocalRank, Reason ? Reason : "unknown", ErrnoValue,
             SourcePath.c_str(), static_cast<unsigned long long>(Begin),
             static_cast<unsigned long long>(End),
             static_cast<unsigned long long>(Epoch), SourceFd, Entry.StageFd);
      return false;
    };

    if (Entry.StageFd < 0) {
      invalidateStageForPath(SourcePath);
      return failDirtyFlush("bad-stage-fd", EBADF);
    }
    if (SourceFd < 0 && SourcePath.empty()) {
      invalidateStageForPath(SourcePath);
      return failDirtyFlush("bad-source-fd", EBADF);
    }

    int FlushSourceFd = SourceFd;
    bool CloseFlushSourceFd = false;
    if (!SourcePath.empty()) {
      FlushSourceFd = ::open(SourcePath.c_str(), O_RDWR);
      if (FlushSourceFd < 0) {
        const int OpenErrno = errno;
        invalidateStageForPath(SourcePath);
        return failDirtyFlush("open-source-rdwr", OpenErrno);
      }
      CloseFlushSourceFd = true;
    }

    auto closeFlushFd = [&]() {
      if (CloseFlushSourceFd && FlushSourceFd >= 0) {
        (void)::close(FlushSourceFd);
        FlushSourceFd = -1;
      }
    };

    const uint64_t PathKey =
        OmpFileHeadnodeManager::computePathKeyForPath(SourcePath);
    constexpr unsigned MaxStaleCompletionRetries = 16;

    for (unsigned Attempt = 0; Attempt <= MaxStaleCompletionRetries; ++Attempt) {
      // Snapshot dirty extents under the lock, then release it for I/O so a
      // long flush does not block normal stage reads. The headnode completion
      // below is the authority for whether the snapshot epoch was still current.
      // If another write-back capture commits a newer epoch while this flush is
      // draining bytes, completion returns ESTALE. In that case do not clear any
      // dirty extents: retry after yielding so the concurrent writer can record
      // its dirty range/epoch and the next pass flushes a current snapshot.
      std::vector<OmpFileStageExtent> Extents;
      uint64_t Epoch = 0;
      {
        std::lock_guard<std::mutex> EntryLock(Entry.Mutex);
        if (Entry.DirtyExtents.empty()) {
          assert(Entry.DirtyBytes == 0 &&
                 "dirty bytes nonzero with no extents");
          closeFlushFd();
          return true;
        }
        Extents = Entry.DirtyExtents;
        Epoch = std::max(Entry.DirtyEpoch,
                         lastLocalFreshnessVersionForPathKey(PathKey));
      }

      const auto FlushStart = std::chrono::steady_clock::now();
      uint64_t FlushedBytes = 0;
      std::vector<char> Buffer;
      for (const OmpFileStageExtent &Extent : Extents) {
        assert(Extent.End > Extent.Begin && "empty dirty extent in vector");
        const uint64_t Len = Extent.End - Extent.Begin;
        if (Buffer.size() < Len)
          Buffer.resize(Len);
        uint64_t ReadTotal = 0;
        while (ReadTotal < Len) {
          const ssize_t N = ::pread(Entry.StageFd, Buffer.data() + ReadTotal,
                                    static_cast<size_t>(Len - ReadTotal),
                                    static_cast<off_t>(Extent.Begin + ReadTotal));
          if (N <= 0) {
            const int ReadErrno = N < 0 ? errno : EIO;
            closeFlushFd();
            invalidateStageForPath(SourcePath);
            return failDirtyFlush("pread-stage", ReadErrno, Extent.Begin,
                                  Extent.End, Epoch);
          }
          ReadTotal += static_cast<uint64_t>(N);
        }
        uint64_t WriteTotal = 0;
        while (WriteTotal < Len) {
          const ssize_t N = ::pwrite(
              FlushSourceFd, Buffer.data() + WriteTotal,
              static_cast<size_t>(Len - WriteTotal),
              static_cast<off_t>(Extent.Begin + WriteTotal));
          if (N <= 0) {
            const int WriteErrno = N < 0 ? errno : EIO;
            closeFlushFd();
            invalidateStageForPath(SourcePath);
            return failDirtyFlush("pwrite-source", WriteErrno, Extent.Begin,
                                  Extent.End, Epoch);
          }
          WriteTotal += static_cast<uint64_t>(N);
        }
        FlushedBytes += Len;
      }

      if (::fdatasync(FlushSourceFd) != 0) {
        const int SyncErrno = errno;
        closeFlushFd();
        invalidateStageForPath(SourcePath);
        return failDirtyFlush("fdatasync-source", SyncErrno, 0, 0, Epoch);
      }

      bool CompletedViaDerivedTileKeys = false;
      if (Epoch > 0 && !SourcePath.empty()) {
        const char *DistributeWritesEnv =
            std::getenv("LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_WRITES");
        const uint64_t TileBytes = configuredStageTileBytes();
        if (DistributeWritesEnv && std::strcmp(DistributeWritesEnv, "1") == 0 &&
            TileBytes > 0) {
          if (DeferCloseMetadataCompletion) {
            // The close path already copied dirty bytes to the shared source and
            // fdatasync'd them above. Avoid one metadata-completion operation per
            // dirty tile during shutdown on every proxy, including the headnode:
            // the headnode-local loop reproduced MPI internal_Testall aborts on
            // AMD 8192x128/9-proxy runs after thousands of close-time derived
            // completions, and remote completion events had already reproduced
            // the same class of abort on Sorgan. Active in-run reads remain
            // protected by the tile oracle because this deferral is used only by
            // close-time flushes; in-run watermark flushes still complete
            // metadata immediately.
            CompletedViaDerivedTileKeys = true;
            REPORT("MPIProxyDevice --> OMPFile derived tile dirty flush "
                   "completion deferred rank=%d path=%s tile_bytes=%llu "
                   "extents=%zu headnode=%d\n",
                   EventSystem.LocalRank, SourcePath.c_str(),
                   static_cast<unsigned long long>(TileBytes), Extents.size(),
                   EventSystem.LocalRank == getHeadnodeRank() ? 1 : 0);
          } else {
            errno = 0;
            CompletedViaDerivedTileKeys = completeDerivedTileFlushesOnHeadnode(
                PathKey, Extents, TileBytes, SourcePath);
            if (!CompletedViaDerivedTileKeys) {
              REPORT("MPIProxyDevice --> OMPFile derived tile dirty flush "
                     "completion unavailable rank=%d errno=%d path=%s "
                     "tile_bytes=%llu; falling back to path-key completion\n",
                     EventSystem.LocalRank, errno, SourcePath.c_str(),
                     static_cast<unsigned long long>(TileBytes));
            }
          }
        }
      }

      if (Epoch > 0 && !SourcePath.empty() && !CompletedViaDerivedTileKeys) {
        errno = 0;
        if (PathKey == 0 ||
            !completeDirtyFlushOnHeadnode(PathKey, EventSystem.LocalRank,
                                          Epoch, /*Success=*/true)) {
          const int CompletionErrno = errno != 0 ? errno : EIO;
          if (CompletionErrno == ESTALE && Attempt < MaxStaleCompletionRetries) {
            // A concurrent write-back commit advanced the file epoch while the
            // close flush was in flight. Keep dirty state intact and retry with
            // a fresh snapshot; clearing here could otherwise drop the newer
            // overlapping dirty range.
            REPORT("MPIProxyDevice --> OMPFile dirty flush retry rank=%d "
                   "reason=headnode-complete-stale errno=%d path=%s epoch=%llu "
                   "attempt=%u\n",
                   EventSystem.LocalRank, CompletionErrno, SourcePath.c_str(),
                   static_cast<unsigned long long>(Epoch), Attempt);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
          }
          closeFlushFd();
          return failDirtyFlush("headnode-complete", CompletionErrno, 0, 0,
                                Epoch);
        }
      }

      {
        std::lock_guard<std::mutex> EntryLock(Entry.Mutex);
        for (const OmpFileStageExtent &Extent : Extents)
          clearDirtyRangeLocked(Entry, Extent.Begin, Extent.End);
      }
      OmpFileStatsStageDirtyFlushes.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsStageDirtyFlushBytes.fetch_add(FlushedBytes,
                                                  std::memory_order_relaxed);
      OmpFileStatsStageWriteUs.fetch_add(
          elapsedMicros(FlushStart, std::chrono::steady_clock::now()),
          std::memory_order_relaxed);
      closeFlushFd();
      return true;
    }

    closeFlushFd();
    return failDirtyFlush("stale-retry-exhausted", ESTALE);
  }

  bool shouldInvalidateStageOnOpen(int Flags) const {
    if (Flags & (O_CREAT | O_EXCL | O_TRUNC))
      return true;
    const int AccessMode = Flags & O_ACCMODE;
    return AccessMode == O_WRONLY || AccessMode == O_RDWR;
  }

  std::string makeStageFilePath(const std::string &SourcePath) const {
    std::string BaseName = SourcePath;
    const size_t SlashPos = BaseName.find_last_of('/');
    if (SlashPos != std::string::npos)
      BaseName = BaseName.substr(SlashPos + 1);
    if (BaseName.empty())
      BaseName = "root";
    for (char &Ch : BaseName) {
      if (!(std::isalnum(static_cast<unsigned char>(Ch)) || Ch == '.' ||
            Ch == '_' || Ch == '-'))
        Ch = '_';
    }
    // Stage entries are process-local today, so sharing one stage file between
    // proxies on the same host can race via O_TRUNC/repopulate cycles.
    const int ProxyRank = EventSystem.LocalRank >= 0 ? EventSystem.LocalRank : 0;
    return OmpFileStageRoot + "/" + BaseName + "-" +
           std::to_string(std::hash<std::string>{}(SourcePath)) + "-rank" +
           std::to_string(ProxyRank) + ".stage";
  }

  bool ensureDirectoryTree(const std::string &Path, int &ErrnoOut) const {
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

  bool getStageEntryForPath(const std::string &SourcePath,
                            std::shared_ptr<OmpFileStageEntry> &EntryOut) {
    const std::lock_guard<std::mutex> Lock(OmpFileStageMutex);
    auto It = OmpFileStageEntries.find(SourcePath);
    if (It == OmpFileStageEntries.end())
      return false;
    EntryOut = It->second;
    return true;
  }

  bool hasStageFreeSpace(int &ErrnoOut) const {
    ErrnoOut = 0;
    if (OmpFileStageMinFreeBytes == 0)
      return true;

    struct statvfs FsStats {};
    if (::statvfs(OmpFileStageRoot.c_str(), &FsStats) != 0) {
      ErrnoOut = errno;
      return false;
    }

    const uint64_t Available =
        static_cast<uint64_t>(FsStats.f_bavail) *
        static_cast<uint64_t>(FsStats.f_frsize);
    if (Available < OmpFileStageMinFreeBytes) {
      ErrnoOut = ENOSPC;
      return false;
    }
    return true;
  }

  bool populateStageRange(const std::string &SourcePath, int StageFd,
                          uint64_t Offset, uint64_t Size,
                          const std::vector<OmpFileStageExtent> &SkipExtents,
                          uint64_t &PopulateBeginOut,
                          uint64_t &PopulateEndOut,
                          uint64_t &SourceSizeOut,
                          OmpFileStagePopulateStats &Stats,
                          int &ErrnoOut) const {
    ErrnoOut = 0;
    PopulateBeginOut = 0;
    PopulateEndOut = 0;
    SourceSizeOut = 0;
    int SourceFd = ::open(SourcePath.c_str(), O_RDONLY);
    if (SourceFd < 0) {
      ErrnoOut = errno;
      return false;
    }

    struct stat SourceStat {};
    if (::fstat(SourceFd, &SourceStat) != 0) {
      ErrnoOut = errno;
      (void)::close(SourceFd);
      return false;
    }
    SourceSizeOut = static_cast<uint64_t>(SourceStat.st_size);
    if (Size == 0 || Offset >= SourceSizeOut) {
      (void)::close(SourceFd);
      return true;
    }

    uint64_t PopulateBegin = 0;
    uint64_t PopulateEnd = SourceSizeOut;
    if (useWindowedStagePopulate()) {
      const uint64_t WindowBytes =
          std::max<uint64_t>(1, std::max(OmpFileStageWindowBytes, Size));
      PopulateBegin = alignDown(Offset, WindowBytes);
      PopulateEnd =
          std::min<uint64_t>(SourceSizeOut,
                             alignUp(saturatingAdd(Offset, Size), WindowBytes));
    }
    PopulateBeginOut = PopulateBegin;
    PopulateEndOut = PopulateEnd;

    std::vector<char> Buffer(1 << 20);
    bool Success = true;
    uint64_t Cursor = PopulateBegin;
    while (Cursor < PopulateEnd) {
      const size_t ChunkBytes = static_cast<size_t>(
          std::min<uint64_t>(Buffer.size(), PopulateEnd - Cursor));
      const auto ReadStart = std::chrono::steady_clock::now();
      const ssize_t BytesRead = ::pread(
          SourceFd, Buffer.data(), ChunkBytes, static_cast<off_t>(Cursor));
      const auto ReadEnd = std::chrono::steady_clock::now();
      Stats.CopyUs += elapsedMicros(ReadStart, ReadEnd);
      if (BytesRead == 0) {
        Cursor = PopulateEnd;
        break;
      }
      if (BytesRead < 0) {
        ErrnoOut = errno;
        Success = false;
        break;
      }

      // Never overwrite captured write-back data with source bytes: under
      // write-back the source file is intentionally stale for dirty ranges,
      // so populating over a dirty extent silently destroys an acked write
      // (stale read-after-write; OOC repro sorgan 3655 / AMD 350175). Write
      // only the sub-ranges of this chunk that fall outside SkipExtents.
      const uint64_t ChunkBegin = Cursor;
      const uint64_t ChunkEnd = Cursor + static_cast<uint64_t>(BytesRead);
      uint64_t SubCursor = ChunkBegin;
      while (Success && SubCursor < ChunkEnd) {
        uint64_t CleanEnd = ChunkEnd;
        bool InsideDirty = false;
        for (const OmpFileStageExtent &Skip : SkipExtents) {
          if (Skip.Begin <= SubCursor && SubCursor < Skip.End) {
            InsideDirty = true;
            SubCursor = std::min(Skip.End, ChunkEnd);
            break;
          }
          if (Skip.Begin > SubCursor)
            CleanEnd = std::min(CleanEnd, Skip.Begin);
        }
        if (InsideDirty)
          continue;
        ssize_t WrittenTotal = 0;
        const size_t CleanBytes = static_cast<size_t>(CleanEnd - SubCursor);
        while (static_cast<size_t>(WrittenTotal) < CleanBytes) {
          const auto WriteStart = std::chrono::steady_clock::now();
          const ssize_t BytesWritten = ::pwrite(
              StageFd,
              Buffer.data() + (SubCursor - ChunkBegin) +
                  static_cast<uint64_t>(WrittenTotal),
              CleanBytes - static_cast<size_t>(WrittenTotal),
              static_cast<off_t>(SubCursor +
                                 static_cast<uint64_t>(WrittenTotal)));
          const auto WriteEnd = std::chrono::steady_clock::now();
          Stats.CopyUs += elapsedMicros(WriteStart, WriteEnd);
          if (BytesWritten <= 0) {
            ErrnoOut = BytesWritten < 0 ? errno : EIO;
            Success = false;
            break;
          }
          WrittenTotal += BytesWritten;
          Stats.CopiedBytes += static_cast<uint64_t>(BytesWritten);
        }
        SubCursor = CleanEnd;
      }
      if (!Success)
        break;
      Cursor += static_cast<uint64_t>(BytesRead);
    }

    if (Success && shouldSyncStagePopulate() && Stats.CopiedBytes > 0) {
      const auto FsyncStart = std::chrono::steady_clock::now();
      if (::fdatasync(StageFd) != 0) {
        ErrnoOut = errno;
        Success = false;
      }
      const auto FsyncEnd = std::chrono::steady_clock::now();
      Stats.FsyncUs += elapsedMicros(FsyncStart, FsyncEnd);
    }

    (void)::close(SourceFd);
    return Success;
  }

  void trackOmpFileFd(int Fd, const std::string &Path, int Flags, int Mode) {
    if (Fd < 0 || Path.empty())
      return;
    const std::lock_guard<std::mutex> Lock(OmpFileTrackedFdMutex);
    auto Existing = OmpFileTrackedFds.find(Fd);
    if (Existing != OmpFileTrackedFds.end()) {
      auto CountIt = OmpFileTrackedFdPathRefCounts.find(Existing->second.Path);
      if (CountIt != OmpFileTrackedFdPathRefCounts.end() && CountIt->second > 0) {
        --CountIt->second;
        if (CountIt->second == 0)
          OmpFileTrackedFdPathRefCounts.erase(CountIt);
      }
    }
    OmpFileTrackedFds[Fd] = OmpFileTrackedFdEntry{Path, Flags, Mode};
    ++OmpFileTrackedFdPathRefCounts[Path];
  }

  bool getTrackedOmpFileFdPath(int Fd, std::string &Path) const {
    const std::lock_guard<std::mutex> Lock(OmpFileTrackedFdMutex);
    auto It = OmpFileTrackedFds.find(Fd);
    if (It == OmpFileTrackedFds.end())
      return false;
    Path = It->second.Path;
    return true;
  }

  bool getTrackedOmpFileFdEntry(int Fd, OmpFileTrackedFdEntry &Entry) const {
    const std::lock_guard<std::mutex> Lock(OmpFileTrackedFdMutex);
    auto It = OmpFileTrackedFds.find(Fd);
    if (It == OmpFileTrackedFds.end())
      return false;
    Entry = It->second;
    return true;
  }

  bool isLastTrackedFdForPath(int Fd, std::string *PathOut = nullptr) const {
    const std::lock_guard<std::mutex> Lock(OmpFileTrackedFdMutex);
    auto It = OmpFileTrackedFds.find(Fd);
    if (It == OmpFileTrackedFds.end())
      return true;
    if (PathOut)
      *PathOut = It->second.Path;
    auto CountIt = OmpFileTrackedFdPathRefCounts.find(It->second.Path);
    return CountIt == OmpFileTrackedFdPathRefCounts.end() || CountIt->second <= 1;
  }

  void eraseTrackedOmpFileFd(int Fd) {
    const std::lock_guard<std::mutex> Lock(OmpFileTrackedFdMutex);
    auto It = OmpFileTrackedFds.find(Fd);
    if (It != OmpFileTrackedFds.end()) {
      auto CountIt = OmpFileTrackedFdPathRefCounts.find(It->second.Path);
      if (CountIt != OmpFileTrackedFdPathRefCounts.end() && CountIt->second > 0) {
        --CountIt->second;
        if (CountIt->second == 0)
          OmpFileTrackedFdPathRefCounts.erase(CountIt);
      }
      OmpFileTrackedFds.erase(It);
    }
  }

  bool shouldRefreshTrackedOmpFileFdForRead(
      const OmpFileTrackedFdEntry &Entry) const {
    if (!OmpFileHeadnodeScheduler)
      return false;
    if (Entry.Path.empty())
      return false;
    if (Entry.Flags & (O_CREAT | O_EXCL | O_TRUNC))
      return false;
    // Shared libompfile handles are opened O_RDWR. Refresh them before reads so
    // long-lived cross-proxy descriptors observe close-to-open semantics.
    return (Entry.Flags & O_ACCMODE) == O_RDWR;
  }

  bool refreshTrackedOmpFileFdForRead(int Fd, int &ErrnoOut) {
    ErrnoOut = 0;

    OmpFileTrackedFdEntry Entry;
    {
      const std::lock_guard<std::mutex> Lock(OmpFileTrackedFdMutex);
      auto It = OmpFileTrackedFds.find(Fd);
      if (It == OmpFileTrackedFds.end())
        return true;
      Entry = It->second;
    }

    if (!shouldRefreshTrackedOmpFileFdForRead(Entry))
      return true;

    const auto RefreshStart = std::chrono::steady_clock::now();
    int RefreshedFd =
        ::open(Entry.Path.c_str(), Entry.Flags, static_cast<mode_t>(Entry.Mode));
    if (RefreshedFd < 0) {
      ErrnoOut = errno;
      OmpFileStatsCoherentReadRefreshFailures.fetch_add(
          1, std::memory_order_relaxed);
      traceOmpFile("refreshTrackedOmpFileFdForRead open-fail fd=%d path=%s "
                   "flags=0x%x mode=%o errno=%d\n",
                   Fd, Entry.Path.c_str(), Entry.Flags, Entry.Mode, ErrnoOut);
      return false;
    }

    if (::dup2(RefreshedFd, Fd) < 0) {
      ErrnoOut = errno;
      OmpFileStatsCoherentReadRefreshFailures.fetch_add(
          1, std::memory_order_relaxed);
      traceOmpFile("refreshTrackedOmpFileFdForRead dup2-fail fd=%d new_fd=%d "
                   "path=%s errno=%d\n",
                   Fd, RefreshedFd, Entry.Path.c_str(), ErrnoOut);
      (void)::close(RefreshedFd);
      return false;
    }

    (void)::close(RefreshedFd);
    OmpFileStatsCoherentReadRefreshes.fetch_add(1, std::memory_order_relaxed);
    OmpFileStatsCoherentReadRefreshUs.fetch_add(
        elapsedMicros(RefreshStart, std::chrono::steady_clock::now()),
        std::memory_order_relaxed);
    traceOmpFile("refreshTrackedOmpFileFdForRead refreshed fd=%d path=%s "
                 "flags=0x%x mode=%o\n",
                 Fd, Entry.Path.c_str(), Entry.Flags, Entry.Mode);
    return true;
  }

  void invalidateStageEntryLocked(const std::string &SourcePath,
                                  bool CountInvalidation,
                                  uint64_t InvalidatedBytes = 0,
                                  bool FullInvalidation = true) {
    auto It = OmpFileStageEntries.find(SourcePath);
    if (It == OmpFileStageEntries.end())
      return;
    std::shared_ptr<OmpFileStageEntry> Entry = It->second;
    {
      std::lock_guard<std::mutex> EntryLock(Entry->Mutex);
      if (CountInvalidation && InvalidatedBytes == 0) {
        for (const OmpFileStageExtent &Extent : Entry->CoveredExtents) {
          if (Extent.End > Extent.Begin)
            InvalidatedBytes += (Extent.End - Extent.Begin);
        }
      }
      if (isWritebackStageEnabled() &&
          (Entry->DirtyBytes > 0 || !Entry->DirtyExtents.empty())) {
        traceOmpFile("preserving dirty write-back stage during full "
                     "invalidation path=%s dirty_bytes=%llu extents=%zu\n",
                     SourcePath.c_str(),
                     static_cast<unsigned long long>(Entry->DirtyBytes),
                     Entry->DirtyExtents.size());
        return;
      }
      if (Entry->DirtyBytes > 0) {
        OmpFileStatsStageDirtyBytes.fetch_sub(Entry->DirtyBytes,
                                              std::memory_order_relaxed);
        Entry->DirtyBytes = 0;
      }
      Entry->DirtyExtents.clear();
      Entry->DirtyEpoch = 0;
      Entry->Invalidated = true;
      Entry->FullyPopulated = false;
      Entry->PopulateInProgress = false;
      Entry->Cond.notify_all();
    }
    if (!Entry->StagePath.empty()) {
      (void)::unlink(Entry->StagePath.c_str());
      Entry->StagePath.clear();
    }
    OmpFileStageEntries.erase(It);
    OmpFileStatsStagingEvictions.fetch_add(1, std::memory_order_relaxed);
    if (CountInvalidation) {
      OmpFileStatsStagingInvalidations.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsStagingInvalidatedBytes.fetch_add(
          InvalidatedBytes, std::memory_order_relaxed);
      if (FullInvalidation) {
        OmpFileStatsStagingFullInvalidations.fetch_add(
            1, std::memory_order_relaxed);
      }
    }
  }

  void invalidateStageForPath(const std::string &SourcePath) {
    if (SourcePath.empty())
      return;
    const std::lock_guard<std::mutex> Lock(OmpFileStageMutex);
    invalidateStageEntryLocked(SourcePath, /*CountInvalidation=*/true);
  }

  OmpFileStageInvalidateReply invalidateStageForRequest(
      const OmpFileStageInvalidateRequest &Request, const std::string &Path) {
    OmpFileStageInvalidateReply Reply{};
    Reply.AbiVersion = OMPFILE_STAGE_INVALIDATE_ABI_VERSION;
    if (Request.AbiVersion != OMPFILE_STAGE_INVALIDATE_ABI_VERSION) {
      Reply.Status = -1;
      Reply.Errno = EPROTO;
      return Reply;
    }
    if (Path.empty() || Request.PathKey == 0) {
      Reply.Status = -1;
      Reply.Errno = ENOKEY;
      return Reply;
    }

    const std::lock_guard<std::mutex> Lock(OmpFileStageMutex);
    auto It = OmpFileStageEntries.find(Path);
    if (It == OmpFileStageEntries.end()) {
      OmpFileStatsStageGlobalInvalidationCompletions.fetch_add(
          1, std::memory_order_relaxed);
      return Reply;
    }

    uint64_t InvalidatedBytes = 0;
    {
      const std::shared_ptr<OmpFileStageEntry> Entry = It->second;
      std::lock_guard<std::mutex> EntryLock(Entry->Mutex);
      for (const OmpFileStageExtent &Extent : Entry->CoveredExtents) {
        if (Extent.End > Extent.Begin)
          InvalidatedBytes += (Extent.End - Extent.Begin);
      }
    }
    invalidateStageEntryLocked(Path, /*CountInvalidation=*/true,
                               InvalidatedBytes, /*FullInvalidation=*/true);
    Reply.InvalidatedEntries = 1;
    Reply.InvalidatedBytes = InvalidatedBytes;
    OmpFileStatsStageGlobalInvalidationCompletions.fetch_add(
        1, std::memory_order_relaxed);
    return Reply;
  }

  void invalidateStageRangeForPath(const std::string &SourcePath,
                                   uint64_t Offset, uint64_t Size) {
    if (SourcePath.empty() || Size == 0)
      return;

    const uint64_t End = saturatingAdd(Offset, Size);
    const std::lock_guard<std::mutex> StageLock(OmpFileStageMutex);
    auto It = OmpFileStageEntries.find(SourcePath);
    if (It == OmpFileStageEntries.end())
      return;

    const std::shared_ptr<OmpFileStageEntry> Entry = It->second;
    std::lock_guard<std::mutex> EntryLock(Entry->Mutex);
    const uint64_t InvalidatedBytes =
        removeCoveredRange(Entry->CoveredExtents, Offset, End);
    if (InvalidatedBytes == 0)
      return;

    if (isWritebackStageEnabled() &&
        (Entry->DirtyBytes > 0 || !Entry->DirtyExtents.empty())) {
      traceOmpFile("preserving dirty write-back stage during range "
                   "invalidation path=%s offset=%llu size=%llu dirty_bytes=%llu "
                   "extents=%zu\n",
                   SourcePath.c_str(), static_cast<unsigned long long>(Offset),
                   static_cast<unsigned long long>(Size),
                   static_cast<unsigned long long>(Entry->DirtyBytes),
                   Entry->DirtyExtents.size());
      return;
    }
    Entry->Invalidated = true;
    if (Entry->DirtyBytes > 0) {
      OmpFileStatsStageDirtyBytes.fetch_sub(Entry->DirtyBytes,
                                            std::memory_order_relaxed);
      Entry->DirtyBytes = 0;
    }
    Entry->DirtyExtents.clear();
    Entry->DirtyEpoch = 0;
    Entry->FullyPopulated = false;
    Entry->PopulateInProgress = false;
    Entry->Cond.notify_all();
    Entry->CoveredExtents.clear();
    OmpFileStatsStagingInvalidations.fetch_add(1, std::memory_order_relaxed);
    OmpFileStatsStagingRangeInvalidations.fetch_add(1,
                                                    std::memory_order_relaxed);
    OmpFileStatsStagingInvalidatedBytes.fetch_add(InvalidatedBytes,
                                                  std::memory_order_relaxed);

    if (!Entry->StagePath.empty()) {
      (void)::unlink(Entry->StagePath.c_str());
      Entry->StagePath.clear();
    }
    OmpFileStageEntries.erase(It);
    OmpFileStatsStagingEvictions.fetch_add(1, std::memory_order_relaxed);
  }

  bool ensureStageEntryForPath(const std::string &SourcePath, uint64_t Offset,
                               uint64_t Size,
                               std::shared_ptr<OmpFileStageEntry> *EntryOut,
                               int &StageFd, int &ErrnoOut) {
    StageFd = -1;
    ErrnoOut = 0;
    if (EntryOut)
      EntryOut->reset();
    if (!isReadthroughStageEnabled() || SourcePath.empty()) {
      ErrnoOut = ENOTSUP;
      return false;
    }

    std::shared_ptr<OmpFileStageEntry> Entry;
    const auto LockWaitStart = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> Lock(OmpFileStageMutex);
    const auto LockAcquired = std::chrono::steady_clock::now();
    OmpFileStatsStageLockWaitUs.fetch_add(
        elapsedMicros(LockWaitStart, LockAcquired),
        std::memory_order_relaxed);
    auto recordLockHold = [&]() {
      OmpFileStatsStageLockHoldUs.fetch_add(
          elapsedMicros(LockAcquired, std::chrono::steady_clock::now()),
          std::memory_order_relaxed);
    };
    auto It = OmpFileStageEntries.find(SourcePath);
    if (It != OmpFileStageEntries.end()) {
      OmpFileStatsStageLookupHits.fetch_add(1, std::memory_order_relaxed);
      Entry = It->second;
      {
        std::lock_guard<std::mutex> EntryLock(Entry->Mutex);
        if (Entry->Invalidated) {
          ErrnoOut = ESTALE;
          recordLockHold();
          return false;
        }
      }
      recordLockHold();
    } else {
      OmpFileStatsStageLookupMisses.fetch_add(1, std::memory_order_relaxed);
      if (!ensureDirectoryTree(OmpFileStageRoot, ErrnoOut)) {
        recordLockHold();
        return false;
      }
      if (!hasStageFreeSpace(ErrnoOut)) {
        recordLockHold();
        return false;
      }

      const std::string StagePath = makeStageFilePath(SourcePath);
      int CreatedStageFd =
          ::open(StagePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0664);
      if (CreatedStageFd < 0) {
        ErrnoOut = errno;
        recordLockHold();
        return false;
      }

      Entry = std::make_shared<OmpFileStageEntry>();
      Entry->SourcePath = SourcePath;
      Entry->StagePath = StagePath;
      Entry->StageFd = CreatedStageFd;
      OmpFileStageEntries.emplace(SourcePath, Entry);
      recordLockHold();
      traceOmpFile("stage entry create source=%s stage=%s fd=%d\n",
                   SourcePath.c_str(), StagePath.c_str(), CreatedStageFd);
    }
    Lock.unlock();

    if (Size == 0) {
      std::lock_guard<std::mutex> EntryLock(Entry->Mutex);
      if (Entry->Invalidated) {
        ErrnoOut = ESTALE;
        return false;
      }
      if (EntryOut)
        *EntryOut = Entry;
      StageFd = Entry->StageFd;
      return true;
    }

    const uint64_t RequestedEnd = saturatingAdd(Offset, Size);
    std::unique_lock<std::mutex> EntryLock(Entry->Mutex);
    while (true) {
      if (Entry->Invalidated) {
        ErrnoOut = ESTALE;
        return false;
      }
      const bool Covered =
          Entry->FullyPopulated ||
          isCoveredByExtents(Entry->CoveredExtents, Offset, RequestedEnd);
      if (Covered) {
        if (EntryOut)
          *EntryOut = Entry;
        StageFd = Entry->StageFd;
        return true;
      }
      if (!Entry->PopulateInProgress)
        break;
      Entry->Cond.wait(EntryLock);
    }
    Entry->PopulateInProgress = true;
    // Snapshot dirty extents under the lock: captures wait on
    // PopulateInProgress, so no new dirty extent can appear until populate
    // finishes, and populate must never overwrite these ranges with the
    // (write-back-stale) source bytes.
    const std::vector<OmpFileStageExtent> DirtySnapshot = Entry->DirtyExtents;
    EntryLock.unlock();

    OmpFileStagePopulateStats PopulateStats;
    uint64_t PopulateBegin = 0;
    uint64_t PopulateEnd = 0;
    uint64_t SourceSize = 0;
    const auto PopulateStart = std::chrono::steady_clock::now();
    const bool PopulateOk =
        populateStageRange(SourcePath, Entry->StageFd, Offset, Size,
                           DirtySnapshot, PopulateBegin, PopulateEnd,
                           SourceSize, PopulateStats, ErrnoOut);
    PopulateStats.PopulateUs =
        elapsedMicros(PopulateStart, std::chrono::steady_clock::now());

    EntryLock.lock();
    Entry->PopulateInProgress = false;
    const bool StageStillValid = !Entry->Invalidated;
    if (PopulateOk && StageStillValid) {
      Entry->SourceSize = SourceSize;
      Entry->SourceSizeKnown = true;
      if (!useWindowedStagePopulate() ||
          (PopulateBegin == 0 && PopulateEnd >= SourceSize))
        Entry->FullyPopulated = true;
      addCoveredExtent(Entry->CoveredExtents, PopulateBegin, PopulateEnd);
    }
    EntryLock.unlock();
    Entry->Cond.notify_all();

    if (!PopulateOk || !StageStillValid) {
      if (PopulateOk && !StageStillValid)
        ErrnoOut = ESTALE;
      const std::lock_guard<std::mutex> StageLock(OmpFileStageMutex);
      auto MapIt = OmpFileStageEntries.find(SourcePath);
      if (MapIt != OmpFileStageEntries.end() && MapIt->second == Entry) {
        if (!Entry->StagePath.empty()) {
          (void)::unlink(Entry->StagePath.c_str());
          Entry->StagePath.clear();
        }
        OmpFileStageEntries.erase(MapIt);
      }
      OmpFileStatsStagePopulateFailures.fetch_add(1,
                                                  std::memory_order_relaxed);
      OmpFileStatsStagePopulateUs.fetch_add(PopulateStats.PopulateUs,
                                            std::memory_order_relaxed);
      OmpFileStatsStageCopyUs.fetch_add(PopulateStats.CopyUs,
                                        std::memory_order_relaxed);
      OmpFileStatsStageFsyncUs.fetch_add(PopulateStats.FsyncUs,
                                         std::memory_order_relaxed);
      OmpFileStatsStagePopulateBytes.fetch_add(PopulateStats.CopiedBytes,
                                               std::memory_order_relaxed);
      return false;
    }

    OmpFileStatsStagePopulateCount.fetch_add(1, std::memory_order_relaxed);
    OmpFileStatsStagePopulateBytes.fetch_add(PopulateStats.CopiedBytes,
                                             std::memory_order_relaxed);
    OmpFileStatsStagePopulateUs.fetch_add(PopulateStats.PopulateUs,
                                          std::memory_order_relaxed);
    OmpFileStatsStageCopyUs.fetch_add(PopulateStats.CopyUs,
                                      std::memory_order_relaxed);
    OmpFileStatsStageFsyncUs.fetch_add(PopulateStats.FsyncUs,
                                       std::memory_order_relaxed);
    OmpFileStatsStageReopenUs.fetch_add(PopulateStats.ReopenUs,
                                        std::memory_order_relaxed);
    traceOmpFile("stage populate source=%s range=[%llu,%llu) fd=%d copied=%llu\n",
                 SourcePath.c_str(),
                 static_cast<unsigned long long>(PopulateBegin),
                 static_cast<unsigned long long>(PopulateEnd), Entry->StageFd,
                 static_cast<unsigned long long>(PopulateStats.CopiedBytes));
    if (EntryOut)
      *EntryOut = Entry;
    StageFd = Entry->StageFd;
    return true;
  }

  bool preadWithOptionalStage(int Fd, int64_t Offset, void *Buffer, uint64_t Size,
                              uint64_t *BytesReadOut, int &ErrnoOut,
                              bool AllowStage = true) {
    ErrnoOut = 0;
    if (BytesReadOut)
      *BytesReadOut = 0;

    int ReadFd = Fd;
    std::string SourcePath;
    std::shared_ptr<OmpFileStageEntry> HeldStageEntry;
    if (AllowStage && Offset >= 0 && Size > 0 && isReadthroughStageEnabled() &&
        getTrackedOmpFileFdPath(Fd, SourcePath)) {
      int StageFd = -1;
      int StageErrno = 0;
      // Keep the stage entry alive so invalidation cannot close StageFd
      // while this read is still using it.
      if (ensureStageEntryForPath(SourcePath, static_cast<uint64_t>(Offset),
                                  Size, &HeldStageEntry, StageFd,
                                  StageErrno)) {
        ReadFd = StageFd;
      } else {
        traceOmpFile("stage fallback source=%s errno=%d mode=%s topology=%d\n",
                     SourcePath.c_str(), StageErrno, OmpFileStageMode.c_str(),
                     static_cast<int>(OmpFileTopologyLoaded));
      }
    }

    // Freshness guard: before serving from a local stage, consult the headnode
    // freshness table.  If another proxy holds dirty write-back data for this
    // file, the local stage may be stale; bypass to the source fd instead.
    if (ReadFd != Fd && isWritebackStageEnabled() &&
        !stageFreshnessGuardPermitsLocal(SourcePath, HeldStageEntry)) {
      HeldStageEntry.reset();
      ReadFd = Fd;
    }

    const auto PreadStart = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> StageReadLock;
    if (ReadFd != Fd && HeldStageEntry) {
      StageReadLock = std::unique_lock<std::mutex>(HeldStageEntry->Mutex);
      if (HeldStageEntry->Invalidated) {
        StageReadLock.unlock();
        HeldStageEntry.reset();
        ReadFd = Fd;
      }
    }

    if (ReadFd == Fd && Size > 0) {
      int RefreshErrno = 0;
      if (!refreshTrackedOmpFileFdForRead(Fd, RefreshErrno)) {
        ErrnoOut = RefreshErrno;
        return false;
      }
    }

    const ssize_t BytesRead =
        ::pread(ReadFd, Buffer, Size, static_cast<off_t>(Offset));
    const auto PreadEnd = std::chrono::steady_clock::now();
    if (BytesRead < 0) {
      ErrnoOut = errno;
      return false;
    }

    const uint64_t Bytes = static_cast<uint64_t>(BytesRead);
    if (BytesReadOut)
      *BytesReadOut = Bytes;
    if (!AllowStage && Bytes > 0) {
      OmpFileStatsStageBypassReads.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsStageBypassBytes.fetch_add(Bytes, std::memory_order_relaxed);
    }
    if (ReadFd != Fd && Bytes > 0) {
      OmpFileStatsStagedReadHits.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsStagedReadBytes.fetch_add(Bytes, std::memory_order_relaxed);
      OmpFileStatsStagedPreadUs.fetch_add(elapsedMicros(PreadStart, PreadEnd),
                                          std::memory_order_relaxed);
    } else if (Bytes > 0) {
      OmpFileStatsSourcePreadBytes.fetch_add(Bytes, std::memory_order_relaxed);
      OmpFileStatsSourcePreadUs.fetch_add(elapsedMicros(PreadStart, PreadEnd),
                                          std::memory_order_relaxed);
    }
    return true;
  }

  bool preadDirtyOwnerStageOnly(int Fd, int64_t Offset, void *Buffer,
                                  uint64_t Size, uint64_t ExpectedVersion,
                                  uint64_t *BytesReadOut, int &ErrnoOut) {
    ErrnoOut = 0;
    if (BytesReadOut)
      *BytesReadOut = 0;
    if (Offset < 0 || (!Buffer && Size > 0)) {
      ErrnoOut = EINVAL;
      return false;
    }
    if (Size == 0)
      return true;
    if (!isWritebackStageEnabled() || !isReadthroughStageEnabled()) {
      ErrnoOut = ENODATA;
      return false;
    }

    std::string SourcePath;
    if (!getTrackedOmpFileFdPath(Fd, SourcePath)) {
      ErrnoOut = EBADF;
      return false;
    }

    std::shared_ptr<OmpFileStageEntry> Entry;
    {
      const std::lock_guard<std::mutex> StageLock(OmpFileStageMutex);
      auto It = OmpFileStageEntries.find(SourcePath);
      if (It == OmpFileStageEntries.end() || !It->second) {
        ErrnoOut = ENODATA;
        return false;
      }
      Entry = It->second;
    }

    const uint64_t Begin = static_cast<uint64_t>(Offset);
    const uint64_t End = saturatingAdd(Begin, Size);
    std::unique_lock<std::mutex> EntryLock(Entry->Mutex);
    if (Entry->Invalidated) {
      ErrnoOut = ESTALE;
      return false;
    }
    if (!dirtyOwnerExpectedVersionMatchesRange(SourcePath, Begin, End,
                                               ExpectedVersion, *Entry)) {
      ErrnoOut = ESTALE;
      return false;
    }
    if (Entry->DirtyBytes == 0 || Entry->DirtyExtents.empty()) {
      ErrnoOut = ENODATA;
      return false;
    }
    const bool DirtyOverlap = overlapsAnyExtent(Entry->DirtyExtents, Begin, End);
    if (!DirtyOverlap) {
      // The headnode freshness table is file/version scoped. This proxy may own
      // dirty bytes for the file while the requested byte range is clean. Tell
      // the caller it may use the normal source/PFS route for this range.
      ErrnoOut = ENODATA;
      return false;
    }
    const bool DirtyCovered = isCoveredByExtents(Entry->DirtyExtents, Begin, End);
    if (!DirtyCovered) {
      // Partial dirty overlap would require merging staged dirty bytes with
      // source bytes. Keep the fail-closed behavior until that path exists.
      ErrnoOut = ESTALE;
      return false;
    }
    const bool Covered = Entry->FullyPopulated ||
                         isCoveredByExtents(Entry->CoveredExtents, Begin, End);
    if (!Covered) {
      ErrnoOut = ESTALE;
      return false;
    }
    const int StageFd = Entry->StageFd;
    const ssize_t BytesRead =
        ::pread(StageFd, Buffer, static_cast<size_t>(Size),
                static_cast<off_t>(Offset));
    if (BytesRead < 0) {
      ErrnoOut = errno;
      return false;
    }
    const uint64_t Bytes = static_cast<uint64_t>(BytesRead);
    if (BytesReadOut)
      *BytesReadOut = Bytes;
    if (Bytes > 0) {
      OmpFileStatsDirtyOwnerForwardReads.fetch_add(1,
                                                   std::memory_order_relaxed);
      OmpFileStatsDirtyOwnerForwardBytes.fetch_add(Bytes,
                                                   std::memory_order_relaxed);
      OmpFileStatsStagedReadHits.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsStagedReadBytes.fetch_add(Bytes, std::memory_order_relaxed);
    }
    return true;
  }

  // Metadata-only classifier for dirty-owner routing. State values:
  //   1 = requested range is fully covered by dirty staged bytes
  //   0 = no dirty overlap for this range; normal source/PFS route may be clean
  //  -1 = unknown/unsafe; caller must fail closed to owner-staged handling
  bool queryDirtyOwnerStageRangeOnly(int Fd, int64_t Offset, uint64_t Size,
                                     uint64_t ExpectedVersion, int &StateOut,
                                     int &ErrnoOut) {
    StateOut = -1;
    ErrnoOut = 0;
    if (Offset < 0) {
      ErrnoOut = EINVAL;
      return false;
    }
    if (Size == 0) {
      StateOut = 0;
      return true;
    }
    if (!isWritebackStageEnabled() || !isReadthroughStageEnabled()) {
      StateOut = 0;
      return true;
    }

    std::string SourcePath;
    if (!getTrackedOmpFileFdPath(Fd, SourcePath)) {
      ErrnoOut = EBADF;
      return false;
    }

    std::shared_ptr<OmpFileStageEntry> Entry;
    {
      const std::lock_guard<std::mutex> StageLock(OmpFileStageMutex);
      auto It = OmpFileStageEntries.find(SourcePath);
      if (It == OmpFileStageEntries.end() || !It->second) {
        StateOut = 0;
        return true;
      }
      Entry = It->second;
    }

    const uint64_t Begin = static_cast<uint64_t>(Offset);
    const uint64_t End = saturatingAdd(Begin, Size);
    std::unique_lock<std::mutex> EntryLock(Entry->Mutex);
    if (Entry->Invalidated) {
      ErrnoOut = ESTALE;
      return false;
    }
    if (!dirtyOwnerExpectedVersionMatchesRange(SourcePath, Begin, End,
                                               ExpectedVersion, *Entry)) {
      ErrnoOut = ESTALE;
      return false;
    }
    if (Entry->DirtyBytes == 0 || Entry->DirtyExtents.empty()) {
      StateOut = 0;
      return true;
    }
    if (!overlapsAnyExtent(Entry->DirtyExtents, Begin, End)) {
      StateOut = 0;
      return true;
    }
    if (!isCoveredByExtents(Entry->DirtyExtents, Begin, End)) {
      ErrnoOut = ESTALE;
      return false;
    }
    const bool Covered = Entry->FullyPopulated ||
                         isCoveredByExtents(Entry->CoveredExtents, Begin, End);
    if (!Covered) {
      ErrnoOut = ESTALE;
      return false;
    }
    StateOut = 1;
    return true;
  }

  bool pwriteFully(int Fd, int64_t Offset, const void *Buffer, uint64_t Size,
                   uint64_t *BytesWrittenOut, uint64_t *ElapsedUsOut,
                   int &ErrnoOut) {
    ErrnoOut = 0;
    if (BytesWrittenOut)
      *BytesWrittenOut = 0;
    if (ElapsedUsOut)
      *ElapsedUsOut = 0;
    if (Size == 0)
      return true;

    uint64_t TotalWritten = 0;
    while (TotalWritten < Size) {
      const auto WriteStart = std::chrono::steady_clock::now();
      const ssize_t BytesWritten =
          ::pwrite(Fd, static_cast<const char *>(Buffer) + TotalWritten,
                   static_cast<size_t>(Size - TotalWritten),
                   static_cast<off_t>(Offset + static_cast<int64_t>(TotalWritten)));
      const auto WriteEnd = std::chrono::steady_clock::now();
      if (ElapsedUsOut)
        *ElapsedUsOut += elapsedMicros(WriteStart, WriteEnd);
      if (BytesWritten < 0) {
        ErrnoOut = errno;
        if (BytesWrittenOut)
          *BytesWrittenOut = TotalWritten;
        return false;
      }
      if (BytesWritten == 0) {
        ErrnoOut = EIO;
        if (BytesWrittenOut)
          *BytesWrittenOut = TotalWritten;
        return false;
      }
      TotalWritten += static_cast<uint64_t>(BytesWritten);
    }

    if (BytesWrittenOut)
      *BytesWrittenOut = TotalWritten;
    return true;
  }

  void updateStageCoverageForWrite(const std::shared_ptr<OmpFileStageEntry> &Entry,
                                   uint64_t Offset, uint64_t Size,
                                   uint64_t SourceSize,
                                   bool SourceSizeKnown) {
    if (!Entry || Size == 0)
      return;
    const uint64_t End = saturatingAdd(Offset, Size);
    std::lock_guard<std::mutex> EntryLock(Entry->Mutex);
    addCoveredExtent(Entry->CoveredExtents, Offset, End);
    if (SourceSizeKnown) {
      Entry->SourceSizeKnown = true;
      Entry->SourceSize = SourceSize;
    }
    if (Entry->SourceSizeKnown &&
        isCoveredByExtents(Entry->CoveredExtents, 0, Entry->SourceSize))
      Entry->FullyPopulated = true;
    else
      Entry->FullyPopulated = false;
  }

  bool pwriteWithOptionalStage(int Fd, int64_t Offset, const void *Buffer,
                               uint64_t Size, uint64_t *BytesWrittenOut,
                               int &ErrnoOut) {
    ErrnoOut = 0;
    if (BytesWrittenOut)
      *BytesWrittenOut = 0;
    if (!Buffer && Size > 0) {
      ErrnoOut = EINVAL;
      return false;
    }

    std::string SourcePath;
    const bool HasTrackedSource = Size > 0 && Offset >= 0 &&
                                  getTrackedOmpFileFdPath(Fd, SourcePath);

    auto writeSourceAuthoritative =
        [&](bool CountBypassOnFailure, bool InvalidateOnFailure) -> bool {
      uint64_t SourceBytesWritten = 0;
      uint64_t SourceWriteUs = 0;
      if (!pwriteFully(Fd, Offset, Buffer, Size, &SourceBytesWritten,
                       &SourceWriteUs, ErrnoOut)) {
        if (HasTrackedSource && OmpFileStageMode != "off") {
          if (CountBypassOnFailure) {
            OmpFileStatsStagingWriteBypassCount.fetch_add(
                1, std::memory_order_relaxed);
          }
          if (InvalidateOnFailure)
            invalidateStageForPath(SourcePath);
        }
        return false;
      }
      if (Size > 0) {
        if (writethroughFsyncEachEnabled()) {
          if (::fdatasync(Fd) != 0) {
            ErrnoOut = errno;
            return false;
          }
        } else {
          // policy=close: defer durability to the logical close; remember the
          // fd so closeWithOptionalCache drains it exactly once.
          const std::lock_guard<std::mutex> UnsyncedLock(
              OmpFileUnsyncedWriteFdMutex);
          OmpFileUnsyncedWriteFds.insert(Fd);
        }
      }
      if (BytesWrittenOut)
        *BytesWrittenOut = SourceBytesWritten;
      return true;
    };

    if (Size == 0 || OmpFileStageMode == "off" || !HasTrackedSource)
      return writeSourceAuthoritative(/*CountBypassOnFailure=*/true,
                                      /*InvalidateOnFailure=*/true);

    // Write-back capture path: stage first, defer the source pwrite until the
    // close flush or the dirty watermark. This branch is correct only because
    // flush-on-close is already wired; before Task 4 it would have been a
    // data-loss intermediate.
    if (isWritebackStageEnabled()) {
      int StageFd = -1;
      int StageErrno = 0;
      std::shared_ptr<OmpFileStageEntry> HeldStageEntry;
      if (!ensureStageEntryForPath(SourcePath, 0, 0, &HeldStageEntry, StageFd,
                                   StageErrno) ||
          StageFd < 0 || !HeldStageEntry) {
        OmpFileStatsStageWriteFailures.fetch_add(1, std::memory_order_relaxed);
        OmpFileStatsStagingWriteBypassCount.fetch_add(
            1, std::memory_order_relaxed);
        invalidateStageForPath(SourcePath);
        return writeSourceAuthoritative(/*CountBypassOnFailure=*/false,
                                        /*InvalidateOnFailure=*/false);
      }

      uint64_t StageBytesWritten = 0;
      uint64_t StageWriteUs = 0;
      uint64_t DirtyBytes = 0;
      {
        // Capture must be atomic against windowed stage population: populate
        // copies its window from the source file, which write-back has
        // deliberately left stale, so a capture racing a populate can be
        // clobbered and the acked write silently lost (stale
        // read-after-write; OOC repro sorgan 3655 / AMD 350175). Wait out
        // any in-flight populate, then write the bytes AND record the dirty
        // extent under one lock hold so any later populate's dirty-extent
        // snapshot protects them.
        std::unique_lock<std::mutex> CaptureLock(HeldStageEntry->Mutex);
        while (HeldStageEntry->PopulateInProgress)
          HeldStageEntry->Cond.wait(CaptureLock);
        bool CaptureOk = !HeldStageEntry->Invalidated;
        if (CaptureOk &&
            (!pwriteFully(StageFd, Offset, Buffer, Size, &StageBytesWritten,
                          &StageWriteUs, StageErrno) ||
             StageBytesWritten < Size))
          CaptureOk = false;
        if (CaptureOk && shouldSyncStagePopulate()) {
          const auto FsyncStart = std::chrono::steady_clock::now();
          if (::fdatasync(StageFd) != 0)
            CaptureOk = false;
          StageWriteUs +=
              elapsedMicros(FsyncStart, std::chrono::steady_clock::now());
        }
        if (!CaptureOk) {
          CaptureLock.unlock();
          OmpFileStatsStageWriteFailures.fetch_add(1,
                                                   std::memory_order_relaxed);
          OmpFileStatsStagingWriteBypassCount.fetch_add(
              1, std::memory_order_relaxed);
          OmpFileStatsStageWriteUs.fetch_add(StageWriteUs,
                                             std::memory_order_relaxed);
          invalidateStageForPath(SourcePath);
          return writeSourceAuthoritative(/*CountBypassOnFailure=*/false,
                                          /*InvalidateOnFailure=*/false);
        }
        const uint64_t Begin = static_cast<uint64_t>(Offset);
        const uint64_t End = saturatingAdd(Begin, Size);
        DirtyBytes = markDirtyRangeLocked(*HeldStageEntry, Begin, End);
      }

      uint64_t SourceSize = 0;
      bool SourceSizeKnown = false;
      struct stat SourceStat {};
      if (::fstat(Fd, &SourceStat) == 0) {
        SourceSize = static_cast<uint64_t>(SourceStat.st_size);
        SourceSizeKnown = true;
      }

      const uint64_t PathKey =
          OmpFileHeadnodeManager::computePathKeyForPath(SourcePath);
      uint64_t CommittedVersion = 0;
      if (PathKey == 0 ||
          !freshnessWriteCommitOnHeadnode(PathKey, EventSystem.LocalRank,
                                          /*TileId=*/PathKey,
                                          /*WriteThroughMode=*/false,
                                          CommittedVersion) ||
          CommittedVersion == 0) {
        OmpFileStatsStageWriteFailures.fetch_add(1, std::memory_order_relaxed);
        OmpFileStatsStagingWriteBypassCount.fetch_add(
            1, std::memory_order_relaxed);
        invalidateStageForPath(SourcePath);
        return writeSourceAuthoritative(/*CountBypassOnFailure=*/false,
                                        /*InvalidateOnFailure=*/false);
      }

      rememberWritebackFreshnessCommit(PathKey, CommittedVersion);
      if (const uint64_t TileBytes = configuredStageTileBytes()) {
        const uint64_t Begin = static_cast<uint64_t>(Offset);
        const uint64_t End = saturatingAdd(Begin, Size);
        if (Size > 0 && Begin % TileBytes == 0 && End == Begin + TileBytes) {
          const uint64_t TileId = Begin / TileBytes;
          const uint64_t CompositeKey = tileFreshnessKeyForPath(PathKey, TileId);
          rememberWritebackFreshnessCommit(CompositeKey, CommittedVersion);
        }
      }

      {
        // Dirty extent was already recorded atomically with the stage write
        // above; only the epoch stamp needs the post-commit value.
        std::lock_guard<std::mutex> EntryLock(HeldStageEntry->Mutex);
        HeldStageEntry->DirtyEpoch = CommittedVersion;
      }
      updateStageCoverageForWrite(HeldStageEntry, static_cast<uint64_t>(Offset),
                                  Size, SourceSize, SourceSizeKnown);
      OmpFileStatsStageWritebackCaptures.fetch_add(
          1, std::memory_order_relaxed);
      OmpFileStatsStageWritebackCaptureBytes.fetch_add(
          Size, std::memory_order_relaxed);
      OmpFileStatsStagedWriteUpdates.fetch_add(1,
                                               std::memory_order_relaxed);
      OmpFileStatsStagedWriteBytes.fetch_add(Size,
                                             std::memory_order_relaxed);
      OmpFileStatsStageWriteUs.fetch_add(StageWriteUs,
                                         std::memory_order_relaxed);
      if (BytesWrittenOut)
        *BytesWrittenOut = StageBytesWritten;

      if (OmpFileStageDirtyWatermarkBytes > 0 &&
          DirtyBytes >= OmpFileStageDirtyWatermarkBytes) {
        if (!flushDirtyRangesToSource(*HeldStageEntry, Fd, SourcePath)) {
          ErrnoOut = EIO;
          return false;
        }
      }
      return true;
    }

    // Existing write-through / non-writeback path: source is authoritative,
    // stage mirrors or is invalidated.
    if (!writeSourceAuthoritative(/*CountBypassOnFailure=*/true,
                                  /*InvalidateOnFailure=*/true))
      return false;

    if (!isWritethroughStageEnabled()) {
      OmpFileStatsStagingWriteBypassCount.fetch_add(1,
                                                    std::memory_order_relaxed);
      invalidateStageRangeForPath(SourcePath, static_cast<uint64_t>(Offset),
                                  Size);
      return true;
    }

    int StageFd = -1;
    int StageErrno = 0;
    std::shared_ptr<OmpFileStageEntry> HeldStageEntry;
    if (!ensureStageEntryForPath(SourcePath, 0, 0, &HeldStageEntry, StageFd,
                                 StageErrno) ||
        StageFd < 0) {
      OmpFileStatsStageWriteFailures.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsStagingWriteBypassCount.fetch_add(1,
                                                    std::memory_order_relaxed);
      invalidateStageForPath(SourcePath);
      return true;
    }

    uint64_t StageBytesWritten = 0;
    uint64_t StageWriteUs = 0;
    if (!pwriteFully(StageFd, Offset, Buffer, Size, &StageBytesWritten,
                     &StageWriteUs, StageErrno) ||
        StageBytesWritten < Size) {
      OmpFileStatsStageWriteFailures.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsStagingWriteBypassCount.fetch_add(1,
                                                    std::memory_order_relaxed);
      invalidateStageForPath(SourcePath);
      return true;
    }

    if (shouldSyncStagePopulate()) {
      const auto FsyncStart = std::chrono::steady_clock::now();
      if (::fdatasync(StageFd) != 0) {
        OmpFileStatsStageWriteFailures.fetch_add(1, std::memory_order_relaxed);
        OmpFileStatsStagingWriteBypassCount.fetch_add(
            1, std::memory_order_relaxed);
        OmpFileStatsStageWriteUs.fetch_add(
            StageWriteUs +
                elapsedMicros(FsyncStart, std::chrono::steady_clock::now()),
            std::memory_order_relaxed);
        invalidateStageForPath(SourcePath);
        return true;
      }
      StageWriteUs +=
          elapsedMicros(FsyncStart, std::chrono::steady_clock::now());
    }

    uint64_t SourceSize = 0;
    bool SourceSizeKnown = false;
    struct stat SourceStat {};
    if (::fstat(Fd, &SourceStat) == 0) {
      SourceSize = static_cast<uint64_t>(SourceStat.st_size);
      SourceSizeKnown = true;
    }

    if (!HeldStageEntry) {
      OmpFileStatsStageWriteFailures.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsStagingWriteBypassCount.fetch_add(1,
                                                    std::memory_order_relaxed);
      invalidateStageForPath(SourcePath);
      return true;
    }

    updateStageCoverageForWrite(HeldStageEntry, static_cast<uint64_t>(Offset),
                                Size, SourceSize, SourceSizeKnown);
    OmpFileStatsStagedWriteUpdates.fetch_add(1, std::memory_order_relaxed);
    OmpFileStatsStagedWriteBytes.fetch_add(Size, std::memory_order_relaxed);
    OmpFileStatsStageWriteUs.fetch_add(StageWriteUs,
                                       std::memory_order_relaxed);
    return true;
  }

  std::string getOmpFileOpenCacheKey(const char *Path, int Flags, int Mode) const {
    std::string Key = Path ? Path : "";
    Key.push_back('\x1f');
    Key += std::to_string(Flags);
    Key.push_back('\x1f');
    Key += std::to_string(Mode);
    return Key;
  }

  void reportOmpFileOpenFailureDiagnostics(const char *Path, int Flags,
                                           int Mode, int OpenErrno) {
    const char *DiagEnv = std::getenv("LIBOMPFILE_OPEN_FAILURE_DIAG");
    if (DiagEnv && std::strcmp(DiagEnv, "0") == 0)
      return;

    const char *SafePath = Path ? Path : "(null)";
    struct stat PathStat {};
    const int PathStatRc = Path ? ::stat(Path, &PathStat) : -1;
    const int PathStatErrno = PathStatRc == 0 ? 0 : errno;
    const int AccessRc = Path ? ::access(Path, R_OK) : -1;
    const int AccessErrno = AccessRc == 0 ? 0 : errno;

    std::string Parent = ".";
    if (Path) {
      std::string PathString(Path);
      size_t Slash = PathString.find_last_of('/');
      if (Slash == std::string::npos)
        Parent = ".";
      else if (Slash == 0)
        Parent = "/";
      else
        Parent = PathString.substr(0, Slash);
    }

    struct stat ParentStat {};
    const int ParentStatRc = ::stat(Parent.c_str(), &ParentStat);
    const int ParentStatErrno = ParentStatRc == 0 ? 0 : errno;
    const int ParentAccessRc = ::access(Parent.c_str(), R_OK | X_OK);
    const int ParentAccessErrno = ParentAccessRc == 0 ? 0 : errno;

    REPORT("OMPFile proxy open failure diag: rank=%d host=%s path=%s flags=0x%x "
           "mode=%o open_errno=%d stat_rc=%d stat_errno=%d access_rc=%d "
           "access_errno=%d parent=%s parent_stat_rc=%d parent_stat_errno=%d "
           "parent_access_rc=%d parent_access_errno=%d\n",
           EventSystem.LocalRank, OmpFileStageLocalHost.c_str(), SafePath, Flags,
           Mode, OpenErrno, PathStatRc, PathStatErrno, AccessRc, AccessErrno,
           Parent.c_str(), ParentStatRc, ParentStatErrno, ParentAccessRc,
           ParentAccessErrno);
  }

  int openWithOptionalCache(const char *Path, int Flags, int Mode, int &ErrnoOut) {
    ErrnoOut = 0;
    OmpFileStatsOpenRequests.fetch_add(1, std::memory_order_relaxed);

    if (!Path) {
      ErrnoOut = EINVAL;
      return -1;
    }

    if (!canUseOmpFileOpenCache(Flags)) {
      OmpFileStatsOpenSyscalls.fetch_add(1, std::memory_order_relaxed);
      int Fd = ::open(Path, Flags, static_cast<mode_t>(Mode));
      if (Fd < 0) {
        ErrnoOut = errno;
        reportOmpFileOpenFailureDiagnostics(Path, Flags, Mode, ErrnoOut);
      }
      traceOmpFile("openWithOptionalCache nocache path=%s flags=0x%x mode=%o "
                   "fd=%d errno=%d\n",
                   Path, Flags, Mode, Fd, ErrnoOut);
      return Fd;
    }

    const std::string Key = getOmpFileOpenCacheKey(Path, Flags, Mode);
    const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
    auto It = OmpFileOpenCacheByKey.find(Key);
    if (It != OmpFileOpenCacheByKey.end()) {
      It->second.RefCount += 1;
      OmpFileStatsOpenCacheHits.fetch_add(1, std::memory_order_relaxed);
      traceOmpFile("openWithOptionalCache hit path=%s flags=0x%x mode=%o "
                   "fd=%d refcount=%llu\n",
                   Path, Flags, Mode, It->second.Fd,
                   static_cast<unsigned long long>(It->second.RefCount));
      return It->second.Fd;
    }

    OmpFileStatsOpenSyscalls.fetch_add(1, std::memory_order_relaxed);
    int Fd = ::open(Path, Flags, static_cast<mode_t>(Mode));
    if (Fd < 0) {
      ErrnoOut = errno;
      reportOmpFileOpenFailureDiagnostics(Path, Flags, Mode, ErrnoOut);
      traceOmpFile("openWithOptionalCache miss-fail path=%s flags=0x%x mode=%o "
                   "errno=%d\n",
                   Path, Flags, Mode, ErrnoOut);
      return -1;
    }

    OmpFileOpenCacheByKey.emplace(Key, OmpFileOpenCacheEntry{Fd, 1});
    OmpFileOpenCacheFdToKey[Fd] = Key;
    traceOmpFile("openWithOptionalCache miss-open path=%s flags=0x%x mode=%o "
                 "fd=%d refcount=1\n",
                 Path, Flags, Mode, Fd);
    return Fd;
  }

  int closeWithOptionalCache(int Fd, int &ErrnoOut) {
    ErrnoOut = 0;
    OmpFileStatsCloseRequests.fetch_add(1, std::memory_order_relaxed);

    // Write-back flush-on-close: before the source fd is released, drain any
    // dirty stage ranges for this path to the source filesystem so the
    // durable copy is current. This is a no-op unless write-back staging is
    // enabled AND the stage entry has dirty extents (which only the
    // write-back capture path can create). In the write-through default and
    // in the Task-4-only state (no capture yet), DirtyExtents is empty and
    // this returns true without touching the source fd.
    if (isWritebackStageEnabled() && Fd >= 0) {
      OmpFileTrackedFdEntry TrackedEntry;
      if (getTrackedOmpFileFdEntry(Fd, TrackedEntry) &&
          (TrackedEntry.Flags & O_ACCMODE) != O_RDONLY) {
        const std::string &SourcePath = TrackedEntry.Path;
        std::shared_ptr<OmpFileStageEntry> StageEntry;
        if (getStageEntryForPath(SourcePath, StageEntry) && StageEntry) {
          if (!flushDirtyRangesToSource(*StageEntry, Fd, SourcePath,
                                        /*DeferCloseMetadataCompletion=*/true)) {
            // Flush failed: the stage entry has already been invalidated by
            // the flush helper (PFS is the authority). Surface the failure so
            // the caller does not silently drop writes.
            ErrnoOut = EIO;
            traceOmpFile("closeWithOptionalCache write-back flush failed "
                         "fd=%d path=%s\n",
                         Fd, SourcePath.c_str());
            return -1;
          }
        }
      }
    }

    // Deferred write-through durability (fsync policy "close"): commit any
    // unsynced authoritative writes at the logical close, regardless of
    // whether the open-cache keeps the physical fd alive afterwards. This
    // runs before the cache-defer logic so a later cache eviction never
    // closes an fd with uncommitted data.
    if (Fd >= 0) {
      bool NeedDeferredSync = false;
      {
        const std::lock_guard<std::mutex> UnsyncedLock(
            OmpFileUnsyncedWriteFdMutex);
        NeedDeferredSync = OmpFileUnsyncedWriteFds.erase(Fd) > 0;
      }
      if (NeedDeferredSync && ::fdatasync(Fd) != 0) {
        ErrnoOut = errno;
        traceOmpFile("closeWithOptionalCache deferred fdatasync failed "
                     "fd=%d errno=%d\n",
                     Fd, ErrnoOut);
        return -1;
      }
    }

    if (!OmpFileOpenCacheEnable) {
      OmpFileStatsCloseSyscalls.fetch_add(1, std::memory_order_relaxed);
      int Ret = ::close(Fd);
      if (Ret != 0)
        ErrnoOut = errno;
      traceOmpFile("closeWithOptionalCache nocache fd=%d ret=%d errno=%d\n", Fd,
                   Ret, ErrnoOut);
      return Ret;
    }

    bool NeedSysClose = false;
    {
      const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
      auto FdIt = OmpFileOpenCacheFdToKey.find(Fd);
      if (FdIt != OmpFileOpenCacheFdToKey.end()) {
        auto KeyIt = OmpFileOpenCacheByKey.find(FdIt->second);
        if (KeyIt != OmpFileOpenCacheByKey.end()) {
          OmpFileOpenCacheEntry &Entry = KeyIt->second;
          if (Entry.RefCount == 0) {
            ErrnoOut = EBADF;
            traceOmpFile("closeWithOptionalCache bad-ref fd=%d\n", Fd);
            return -1;
          }

          Entry.RefCount -= 1;
          if (Entry.RefCount > 0 || OmpFileOpenCacheKeepOpen) {
            OmpFileStatsCloseDeferred.fetch_add(1, std::memory_order_relaxed);
            traceOmpFile("closeWithOptionalCache defer fd=%d refcount=%llu "
                         "keep_open=%d\n",
                         Fd, static_cast<unsigned long long>(Entry.RefCount),
                         static_cast<int>(OmpFileOpenCacheKeepOpen.get()));
            return 0;
          }

          OmpFileOpenCacheByKey.erase(KeyIt);
          OmpFileOpenCacheFdToKey.erase(FdIt);
          NeedSysClose = true;
        } else {
          OmpFileOpenCacheFdToKey.erase(FdIt);
          NeedSysClose = true;
        }
      } else {
        NeedSysClose = true;
      }
    }

    if (!NeedSysClose)
      return 0;

    OmpFileStatsCloseSyscalls.fetch_add(1, std::memory_order_relaxed);
    int Ret = ::close(Fd);
    if (Ret != 0)
      ErrnoOut = errno;
    else
      eraseTrackedOmpFileFd(Fd);
    traceOmpFile("closeWithOptionalCache close fd=%d ret=%d errno=%d\n", Fd,
                 Ret, ErrnoOut);
    return Ret;
  }

  void drainOmpFileOpenCache() {
    std::vector<int> FdsToClose;
    {
      const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
      if (OmpFileOpenCacheByKey.empty() && OmpFileOpenCacheFdToKey.empty())
        return;

      FdsToClose.reserve(OmpFileOpenCacheByKey.size());
      for (const auto &It : OmpFileOpenCacheByKey)
        FdsToClose.push_back(It.second.Fd);
      OmpFileOpenCacheByKey.clear();
      OmpFileOpenCacheFdToKey.clear();
    }

    for (int Fd : FdsToClose) {
      OmpFileStatsCloseSyscalls.fetch_add(1, std::memory_order_relaxed);
      (void)::close(Fd);
      eraseTrackedOmpFileFd(Fd);
    }
  }

  void drainOmpFileStageCache() {
    const std::lock_guard<std::mutex> Lock(OmpFileStageMutex);
    for (auto &It : OmpFileStageEntries) {
      if (!It.second->StagePath.empty()) {
        (void)::unlink(It.second->StagePath.c_str());
        It.second->StagePath.clear();
      }
    }
    OmpFileStageEntries.clear();
  }

  void reportOmpFileStats(const char *Scope) {
    if (!OmpFileOptStats)
      return;

    uint64_t OpenReq = OmpFileStatsOpenRequests.load(std::memory_order_relaxed);
    uint64_t OpenSys = OmpFileStatsOpenSyscalls.load(std::memory_order_relaxed);
    uint64_t OpenHits =
        OmpFileStatsOpenCacheHits.load(std::memory_order_relaxed);
    uint64_t OpenEioRetries =
        OmpFileStatsOpenEioRetries.load(std::memory_order_relaxed);
    uint64_t OpenEioFailures =
        OmpFileStatsOpenEioFailures.load(std::memory_order_relaxed);
    uint64_t CloseReq =
        OmpFileStatsCloseRequests.load(std::memory_order_relaxed);
    uint64_t CloseSys =
        OmpFileStatsCloseSyscalls.load(std::memory_order_relaxed);
    uint64_t CloseDeferred =
        OmpFileStatsCloseDeferred.load(std::memory_order_relaxed);
    uint64_t PwriteAsyncEvents =
        OmpFileStatsPwriteAsyncEvents.load(std::memory_order_relaxed);
    uint64_t PwriteAsyncFragments =
        OmpFileStatsPwriteAsyncFragments.load(std::memory_order_relaxed);
    uint64_t PwriteBlockingFallbacks =
        OmpFileStatsPwriteBlockingFallbacks.load(std::memory_order_relaxed);
    uint64_t PwritePayloadBytes =
        OmpFileStatsPwritePayloadBytes.load(std::memory_order_relaxed);
    uint64_t PwriteFailures =
        OmpFileStatsPwriteFailures.load(std::memory_order_relaxed);
    uint64_t StagedReadHits =
        OmpFileStatsStagedReadHits.load(std::memory_order_relaxed);
    uint64_t StagedReadBytes =
        OmpFileStatsStagedReadBytes.load(std::memory_order_relaxed);
    uint64_t StageBypassReads =
        OmpFileStatsStageBypassReads.load(std::memory_order_relaxed);
    uint64_t StageBypassBytes =
        OmpFileStatsStageBypassBytes.load(std::memory_order_relaxed);
    uint64_t StageLookupHits =
        OmpFileStatsStageLookupHits.load(std::memory_order_relaxed);
    uint64_t StageLookupMisses =
        OmpFileStatsStageLookupMisses.load(std::memory_order_relaxed);
    uint64_t StagePopulateCount =
        OmpFileStatsStagePopulateCount.load(std::memory_order_relaxed);
    uint64_t StagePopulateFailures =
        OmpFileStatsStagePopulateFailures.load(std::memory_order_relaxed);
    uint64_t StagePopulateBytes =
        OmpFileStatsStagePopulateBytes.load(std::memory_order_relaxed);
    uint64_t StagePopulateUs =
        OmpFileStatsStagePopulateUs.load(std::memory_order_relaxed);
    uint64_t StageCopyUs =
        OmpFileStatsStageCopyUs.load(std::memory_order_relaxed);
    uint64_t StageFsyncUs =
        OmpFileStatsStageFsyncUs.load(std::memory_order_relaxed);
    uint64_t StageReopenUs =
        OmpFileStatsStageReopenUs.load(std::memory_order_relaxed);
    uint64_t StageLockWaitUs =
        OmpFileStatsStageLockWaitUs.load(std::memory_order_relaxed);
    uint64_t StageLockHoldUs =
        OmpFileStatsStageLockHoldUs.load(std::memory_order_relaxed);
    uint64_t StageGlobalInvalidationRequests =
        OmpFileStatsStageGlobalInvalidationRequests.load(
            std::memory_order_relaxed);
    uint64_t StageGlobalInvalidationCompletions =
        OmpFileStatsStageGlobalInvalidationCompletions.load(
            std::memory_order_relaxed);
    uint64_t StageGlobalInvalidationFailures =
        OmpFileStatsStageGlobalInvalidationFailures.load(
            std::memory_order_relaxed);
    uint64_t SourcePreadBytes =
        OmpFileStatsSourcePreadBytes.load(std::memory_order_relaxed);
    uint64_t SourcePreadUs =
        OmpFileStatsSourcePreadUs.load(std::memory_order_relaxed);
    uint64_t StagedPreadUs =
        OmpFileStatsStagedPreadUs.load(std::memory_order_relaxed);
    uint64_t StagingInvalidations =
        OmpFileStatsStagingInvalidations.load(std::memory_order_relaxed);
    uint64_t StagingRangeInvalidations =
        OmpFileStatsStagingRangeInvalidations.load(std::memory_order_relaxed);
    uint64_t StagingFullInvalidations =
        OmpFileStatsStagingFullInvalidations.load(std::memory_order_relaxed);
    uint64_t StagingInvalidatedBytes =
        OmpFileStatsStagingInvalidatedBytes.load(std::memory_order_relaxed);
    uint64_t StagingWriteBypass =
        OmpFileStatsStagingWriteBypassCount.load(std::memory_order_relaxed);
    uint64_t StagedWriteUpdates =
        OmpFileStatsStagedWriteUpdates.load(std::memory_order_relaxed);
    uint64_t StagedWriteBytes =
        OmpFileStatsStagedWriteBytes.load(std::memory_order_relaxed);
    uint64_t StageWriteFailures =
        OmpFileStatsStageWriteFailures.load(std::memory_order_relaxed);
    uint64_t StageWriteUs =
        OmpFileStatsStageWriteUs.load(std::memory_order_relaxed);
    uint64_t StageWritebackCaptures =
        OmpFileStatsStageWritebackCaptures.load(std::memory_order_relaxed);
    uint64_t StageWritebackCaptureBytes =
        OmpFileStatsStageWritebackCaptureBytes.load(std::memory_order_relaxed);
    uint64_t StageDirtyBytes =
        OmpFileStatsStageDirtyBytes.load(std::memory_order_relaxed);
    uint64_t StageDirtyFlushes =
        OmpFileStatsStageDirtyFlushes.load(std::memory_order_relaxed);
    uint64_t StageDirtyFlushBytes =
        OmpFileStatsStageDirtyFlushBytes.load(std::memory_order_relaxed);
    uint64_t StageDirtyFlushFailures =
        OmpFileStatsStageDirtyFlushFailures.load(std::memory_order_relaxed);
    uint64_t StagingEvictions =
        OmpFileStatsStagingEvictions.load(std::memory_order_relaxed);
    uint64_t CoherentReadRefreshes =
        OmpFileStatsCoherentReadRefreshes.load(std::memory_order_relaxed);
    uint64_t CoherentReadRefreshFailures =
        OmpFileStatsCoherentReadRefreshFailures.load(std::memory_order_relaxed);
    uint64_t CoherentReadRefreshUs =
        OmpFileStatsCoherentReadRefreshUs.load(std::memory_order_relaxed);
    uint64_t StageFreshnessGuardBypasses =
        OmpFileStatsStageFreshnessGuardBypasses.load(std::memory_order_relaxed);
    uint64_t DirtyOwnerForwardReads =
        OmpFileStatsDirtyOwnerForwardReads.load(std::memory_order_relaxed);
    uint64_t DirtyOwnerForwardBytes =
        OmpFileStatsDirtyOwnerForwardBytes.load(std::memory_order_relaxed);
    uint64_t DirtyOwnerForwardFailures =
        OmpFileStatsDirtyOwnerForwardFailures.load(std::memory_order_relaxed);
    size_t CacheEntries = 0;

    {
      const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
      CacheEntries = OmpFileOpenCacheByKey.size();
    }

    fprintf(stderr,
            "MPIProxyDevice --> OMPFile stats [%s] rank=%d "
            "open_req=%llu open_sys=%llu open_hits=%llu "
            "open_eio_retries=%llu open_eio_failures=%llu "
            "close_req=%llu close_sys=%llu close_deferred=%llu "
            "cache_entries=%zu pwrite_async_events=%llu "
            "pwrite_async_fragments=%llu pwrite_blocking_fallbacks=%llu "
            "pwrite_payload_bytes=%llu pwrite_failures=%llu "
            "topology_file_set=%d topology_loaded=%d topology_entries=%llu "
            "stage_mode=%s stage_root_policy=%s stage_root=%s "
            "stage_class=%s shared_storage_path=%s shared_storage_class=%s "
            "storage_environment=%s stage_sync_policy=%s "
            "stage_populate_mode=%s stage_window_bytes=%llu "
            "stage_min_free_bytes=%llu "
            "staged_read_hits=%llu staged_read_bytes=%llu "
            "staging_bypass_reads=%llu staging_bypass_bytes=%llu "
            "staged_write_updates=%llu staged_write_bytes=%llu "
            "stage_writeback_captures=%llu stage_writeback_capture_bytes=%llu "
            "stage_dirty_bytes=%llu stage_dirty_flushes=%llu "
            "stage_dirty_flush_bytes=%llu stage_dirty_flush_failures=%llu "
            "staging_write_bypass_count=%llu stage_write_failures=%llu "
            "stage_write_us_total=%llu "
            "stage_lookup_hits=%llu stage_lookup_misses=%llu "
            "stage_populate_count=%llu stage_populate_failures=%llu "
            "stage_populate_bytes=%llu stage_populate_us_total=%llu "
            "stage_copy_us_total=%llu stage_fsync_us_total=%llu "
            "stage_reopen_us_total=%llu stage_lock_wait_us_total=%llu "
            "stage_lock_hold_us_total=%llu "
            "stage_global_invalidations_requested=%llu "
            "stage_global_invalidations_completed=%llu "
            "stage_global_invalidations_failed=%llu source_pread_bytes=%llu "
            "source_pread_us_total=%llu staged_pread_us_total=%llu "
            "staging_invalidations=%llu staging_range_invalidations=%llu "
            "staging_full_invalidations=%llu "
            "staging_invalidated_bytes=%llu "
            "coherent_read_refreshes=%llu "
            "coherent_read_refresh_failures=%llu "
            "coherent_read_refresh_us_total=%llu "
            "staging_evictions=%llu "
            "stage_freshness_guard_bypasses=%llu dirty_owner_forward_reads=%llu dirty_owner_forward_bytes=%llu dirty_owner_forward_failures=%llu\n",
            Scope ? Scope : "unknown", EventSystem.LocalRank,
            static_cast<unsigned long long>(OpenReq),
            static_cast<unsigned long long>(OpenSys),
            static_cast<unsigned long long>(OpenHits),
            static_cast<unsigned long long>(OpenEioRetries),
            static_cast<unsigned long long>(OpenEioFailures),
            static_cast<unsigned long long>(CloseReq),
            static_cast<unsigned long long>(CloseSys),
            static_cast<unsigned long long>(CloseDeferred), CacheEntries,
            static_cast<unsigned long long>(PwriteAsyncEvents),
            static_cast<unsigned long long>(PwriteAsyncFragments),
            static_cast<unsigned long long>(PwriteBlockingFallbacks),
            static_cast<unsigned long long>(PwritePayloadBytes),
            static_cast<unsigned long long>(PwriteFailures),
            static_cast<int>(!OmpFileTopologyFile.empty()),
            static_cast<int>(OmpFileTopologyLoaded),
            static_cast<unsigned long long>(OmpFileTopologyEntries),
            OmpFileStageMode.c_str(), OmpFileStageRootPolicy.c_str(),
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
            OmpFileStorageEnvironment.c_str(), OmpFileStageSyncPolicy.c_str(),
            OmpFileStagePopulateMode.c_str(),
            static_cast<unsigned long long>(OmpFileStageWindowBytes),
            static_cast<unsigned long long>(OmpFileStageMinFreeBytes),
            static_cast<unsigned long long>(StagedReadHits),
            static_cast<unsigned long long>(StagedReadBytes),
            static_cast<unsigned long long>(StageBypassReads),
            static_cast<unsigned long long>(StageBypassBytes),
            static_cast<unsigned long long>(StagedWriteUpdates),
            static_cast<unsigned long long>(StagedWriteBytes),
            static_cast<unsigned long long>(StageWritebackCaptures),
            static_cast<unsigned long long>(StageWritebackCaptureBytes),
            static_cast<unsigned long long>(StageDirtyBytes),
            static_cast<unsigned long long>(StageDirtyFlushes),
            static_cast<unsigned long long>(StageDirtyFlushBytes),
            static_cast<unsigned long long>(StageDirtyFlushFailures),
            static_cast<unsigned long long>(StagingWriteBypass),
            static_cast<unsigned long long>(StageWriteFailures),
            static_cast<unsigned long long>(StageWriteUs),
            static_cast<unsigned long long>(StageLookupHits),
            static_cast<unsigned long long>(StageLookupMisses),
            static_cast<unsigned long long>(StagePopulateCount),
            static_cast<unsigned long long>(StagePopulateFailures),
            static_cast<unsigned long long>(StagePopulateBytes),
            static_cast<unsigned long long>(StagePopulateUs),
            static_cast<unsigned long long>(StageCopyUs),
            static_cast<unsigned long long>(StageFsyncUs),
            static_cast<unsigned long long>(StageReopenUs),
            static_cast<unsigned long long>(StageLockWaitUs),
            static_cast<unsigned long long>(StageLockHoldUs),
            static_cast<unsigned long long>(StageGlobalInvalidationRequests),
            static_cast<unsigned long long>(StageGlobalInvalidationCompletions),
            static_cast<unsigned long long>(StageGlobalInvalidationFailures),
            static_cast<unsigned long long>(SourcePreadBytes),
            static_cast<unsigned long long>(SourcePreadUs),
            static_cast<unsigned long long>(StagedPreadUs),
            static_cast<unsigned long long>(StagingInvalidations),
            static_cast<unsigned long long>(StagingRangeInvalidations),
            static_cast<unsigned long long>(StagingFullInvalidations),
            static_cast<unsigned long long>(StagingInvalidatedBytes),
            static_cast<unsigned long long>(CoherentReadRefreshes),
            static_cast<unsigned long long>(CoherentReadRefreshFailures),
            static_cast<unsigned long long>(CoherentReadRefreshUs),
            static_cast<unsigned long long>(StagingEvictions),
            static_cast<unsigned long long>(StageFreshnessGuardBypasses),
            static_cast<unsigned long long>(DirtyOwnerForwardReads),
            static_cast<unsigned long long>(DirtyOwnerForwardBytes),
            static_cast<unsigned long long>(DirtyOwnerForwardFailures));

    fprintf(stderr,
            "MPIProxyDevice --> OMPFile writeback stats [%s] rank=%d "
            "stage_write_mode=%s captures=%llu capture_bytes=%llu "
            "dirty_bytes=%llu dirty_flushes=%llu dirty_flush_bytes=%llu "
            "dirty_flush_failures=%llu staged_write_updates=%llu "
            "staged_write_bytes=%llu write_bypass_count=%llu "
            "write_failures=%llu\n",
            Scope ? Scope : "unknown", EventSystem.LocalRank,
            OmpFileStageWriteMode.c_str(),
            static_cast<unsigned long long>(StageWritebackCaptures),
            static_cast<unsigned long long>(StageWritebackCaptureBytes),
            static_cast<unsigned long long>(StageDirtyBytes),
            static_cast<unsigned long long>(StageDirtyFlushes),
            static_cast<unsigned long long>(StageDirtyFlushBytes),
            static_cast<unsigned long long>(StageDirtyFlushFailures),
            static_cast<unsigned long long>(StagedWriteUpdates),
            static_cast<unsigned long long>(StagedWriteBytes),
            static_cast<unsigned long long>(StagingWriteBypass),
            static_cast<unsigned long long>(StageWriteFailures));
  }

  uint64_t computePwriteFragmentCount(uint64_t Size) const {
    if (Size == 0)
      return 0;
    const int64_t FragmentSize = OmpFileMPIFragmentSize.get();
    if (FragmentSize <= 0)
      return 0;
    return (Size + static_cast<uint64_t>(FragmentSize) - 1) /
           static_cast<uint64_t>(FragmentSize);
  }

  EventTy ompfileOpen(MPIRequestManagerTy RequestManager) {
    uint32_t PathSize = 0;
    int Flags = 0;
    int Mode = 0;

    RequestManager.receive(&PathSize, 1, MPI_UINT32_T);
    RequestManager.receive(&Flags, 1, MPI_INT);
    RequestManager.receive(&Mode, 1, MPI_INT);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::string Path;
    if (PathSize > 0) {
      Path.resize(PathSize, '\0');
      RequestManager.receive(Path.data(), PathSize, MPI_CHAR);
      if (auto Error = co_await RequestManager; Error)
        co_return Error;
    }

    if (!Path.empty() && Path.back() == '\0')
      Path.pop_back();

    int Errno = 0;
    int Fd = -1;
    const int MaxEioRetries = OmpFileOpenEioRetries.get();
    const int EioBackoffUs = OmpFileOpenEioBackoffUs.get();
    for (int Attempt = 0;; ++Attempt) {
      Fd = openWithOptionalCache(Path.c_str(), Flags, Mode, Errno);
      if (Fd >= 0)
        break;
      const bool IsTransient = (Errno == EIO || Errno == ENOENT ||
                                Errno == ESTALE);
      if (!IsTransient || Attempt >= MaxEioRetries) {
        if (IsTransient)
          OmpFileStatsOpenEioFailures.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      OmpFileStatsOpenEioRetries.fetch_add(1, std::memory_order_relaxed);
      traceOmpFile("ompfileOpen transient-retry path=%s attempt=%d "
                   "errno=%d backoff_us=%d\n",
                   Path.c_str(), Attempt + 1, Errno, EioBackoffUs);
      std::this_thread::sleep_for(std::chrono::microseconds(EioBackoffUs));
    }
    if (Fd >= 0) {
      trackOmpFileFd(Fd, Path, Flags, Mode);
      if (shouldInvalidateStageOnOpen(Flags))
        invalidateStageForPath(Path);
    }

    RequestManager.send(&Fd, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileClose(MPIRequestManagerTy RequestManager) {
    int Fd = -1;
    RequestManager.receive(&Fd, 1, MPI_INT);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int Errno = 0;
    int Ret = closeWithOptionalCache(Fd, Errno);

    RequestManager.send(&Ret, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfilePread(MPIRequestManagerTy RequestManager,
                       bool AllowStage = true) {
    int Fd = -1;
    int64_t Offset = 0;
    uint64_t Size = 0;

    RequestManager.receive(&Fd, 1, MPI_INT);
    RequestManager.receive(&Offset, 1, MPI_INT64_T);
    RequestManager.receive(&Size, 1, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::vector<char> Buffer;
    if (Size > 0)
      Buffer.resize(Size);

    int Errno = 0;
    uint64_t PayloadSize = 0;
    const bool Ok =
        Size == 0 || preadWithOptionalStage(Fd, Offset, Buffer.data(), Size,
                                            &PayloadSize, Errno, AllowStage);
    int Ret = Ok ? 0 : -1;

    RequestManager.send(&Ret, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(&PayloadSize, 1, MPI_UINT64_T);
    if (PayloadSize > 0)
      RequestManager.sendInBatchs(Buffer.data(), PayloadSize);
    traceOmpFile("event ompfilePread fd=%d offset=%lld size=%llu buf=%p "
                  "ret=%d errno=%d bytes=%llu allow_stage=%d\n",
                  Fd, static_cast<long long>(Offset),
                  static_cast<unsigned long long>(Size),
                  static_cast<void *>(Buffer.data()), Ret, Errno,
                  static_cast<unsigned long long>(PayloadSize),
                  static_cast<int>(AllowStage));

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileDirtyOwnerPread(MPIRequestManagerTy RequestManager) {
    int Fd = -1;
    int64_t Offset = 0;
    uint64_t Size = 0;
    uint64_t ExpectedVersion = 0;

    RequestManager.receive(&Fd, 1, MPI_INT);
    RequestManager.receive(&Offset, 1, MPI_INT64_T);
    RequestManager.receive(&Size, 1, MPI_UINT64_T);
    RequestManager.receive(&ExpectedVersion, 1, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::vector<char> Buffer;
    if (Size > 0)
      Buffer.resize(Size);

    int Errno = 0;
    uint64_t PayloadSize = 0;
    const bool Ok = Size == 0 ||
                    preadDirtyOwnerStageOnly(Fd, Offset, Buffer.data(), Size,
                                             ExpectedVersion, &PayloadSize,
                                             Errno);
    if (!Ok)
      OmpFileStatsDirtyOwnerForwardFailures.fetch_add(
          1, std::memory_order_relaxed);
    int Ret = Ok ? 0 : -1;

    OmpFileDirtyOwnerPreadReplyFrame Reply{};
    Reply.Ret = Ret;
    Reply.Errno = Errno;
    Reply.PayloadBytes = PayloadSize;
    Reply.ExpectedVersion = ExpectedVersion;
    Reply.Offset = static_cast<uint64_t>(Offset);

    RequestManager.sendTagged(&Reply, sizeof(Reply), MPI_BYTE,
                              /*TagOffset=*/1);
    if (PayloadSize > 0)
      RequestManager.sendInBatchsTagged(Buffer.data(), PayloadSize,
                                        /*TagOffset=*/2);
    traceOmpFile("event ompfileDirtyOwnerPread framed fd=%d offset=%lld "
                 "size=%llu ret=%d errno=%d bytes=%llu "
                 "expected_version=%llu\n",
                 Fd, static_cast<long long>(Offset),
                 static_cast<unsigned long long>(Size), Ret, Errno,
                 static_cast<unsigned long long>(PayloadSize),
                 static_cast<unsigned long long>(ExpectedVersion));

    RequestManager.sendTagged(nullptr, 0, MPI_BYTE, /*TagOffset=*/3);
    co_return (co_await RequestManager);
  }

  EventTy ompfileDirtyOwnerPreadBatch(MPIRequestManagerTy RequestManager) {
    int Fd = -1;
    uint64_t SegmentCount = 0;

    RequestManager.receive(&Fd, 1, MPI_INT);
    RequestManager.receive(&SegmentCount, 1, MPI_UINT64_T);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;
    if (SegmentCount == 0 || SegmentCount > 4096)
      co_return createError("Invalid dirty-owner batch segment count: %llu",
                            static_cast<unsigned long long>(SegmentCount));

    std::vector<OmpFileDirtyOwnerPreadBatchSegment> Segments(SegmentCount);
    RequestManager.receive(Segments.data(),
                           Segments.size() * sizeof(Segments.front()),
                           MPI_BYTE);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::vector<std::vector<char>> Payloads(SegmentCount);
    std::vector<OmpFileDirtyOwnerPreadBatchReplySegment> Replies(SegmentCount);
    uint64_t PayloadTotal = 0;
    for (uint64_t I = 0; I < SegmentCount; ++I) {
      const auto &Segment = Segments[I];
      auto &Reply = Replies[I];
      Reply.ClientSegmentId = Segment.ClientSegmentId;
      Reply.ExpectedVersion = Segment.ExpectedVersion;
      Reply.Offset = static_cast<uint64_t>(Segment.Offset);
      Reply.PayloadOffset = PayloadTotal;
      if (Segment.Size > 0)
        Payloads[I].resize(static_cast<size_t>(Segment.Size));
      int Errno = 0;
      uint64_t PayloadSize = 0;
      const bool Ok = Segment.Size == 0 ||
                      preadDirtyOwnerStageOnly(
                          Fd, Segment.Offset, Payloads[I].data(), Segment.Size,
                          Segment.ExpectedVersion, &PayloadSize, Errno);
      if (!Ok) {
        OmpFileStatsDirtyOwnerForwardFailures.fetch_add(
            1, std::memory_order_relaxed);
        Reply.Ret = -1;
        Reply.Errno = Errno != 0 ? Errno : EIO;
        Reply.Bytes = 0;
        Payloads[I].clear();
        continue;
      }
      if (PayloadSize > Segment.Size)
        PayloadSize = Segment.Size;
      Payloads[I].resize(static_cast<size_t>(PayloadSize));
      Reply.Ret = 0;
      Reply.Errno = 0;
      Reply.Bytes = PayloadSize;
      Reply.PayloadOffset = PayloadTotal;
      PayloadTotal += PayloadSize;
    }

    OmpFileDirtyOwnerPreadBatchReplyHeader Header{};
    Header.SegmentCount = SegmentCount;
    Header.PayloadBytes = PayloadTotal;
    RequestManager.sendTagged(&Header, sizeof(Header), MPI_BYTE,
                              /*TagOffset=*/1);
    RequestManager.sendTagged(Replies.data(),
                              Replies.size() * sizeof(Replies.front()),
                              MPI_BYTE, /*TagOffset=*/2);
    for (uint64_t I = 0; I < SegmentCount; ++I) {
      if (Payloads[I].empty())
        continue;
      RequestManager.sendInBatchsTagged(Payloads[I].data(), Payloads[I].size(),
                                        /*TagOffset=*/3);
    }
    traceOmpFile("event ompfileDirtyOwnerPreadBatch framed fd=%d segments=%llu "
                 "payload_bytes=%llu\n",
                 Fd, static_cast<unsigned long long>(SegmentCount),
                 static_cast<unsigned long long>(PayloadTotal));

    RequestManager.sendTagged(nullptr, 0, MPI_BYTE, /*TagOffset=*/4);
    co_return (co_await RequestManager);
  }

  EventTy ompfileDirtyOwnerQuery(MPIRequestManagerTy RequestManager) {
    int Fd = -1;
    int64_t Offset = 0;
    uint64_t Size = 0;
    uint64_t ExpectedVersion = 0;

    RequestManager.receive(&Fd, 1, MPI_INT);
    RequestManager.receive(&Offset, 1, MPI_INT64_T);
    RequestManager.receive(&Size, 1, MPI_UINT64_T);
    RequestManager.receive(&ExpectedVersion, 1, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int Errno = 0;
    int State = -1;
    const bool Ok = queryDirtyOwnerStageRangeOnly(Fd, Offset, Size,
                                                  ExpectedVersion, State,
                                                  Errno);
    int Ret = Ok ? 0 : -1;

    RequestManager.send(&Ret, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(&State, 1, MPI_INT);
    traceOmpFile("event ompfileDirtyOwnerQuery fd=%d offset=%lld size=%llu "
                 "ret=%d errno=%d state=%d expected_version=%llu\n",
                 Fd, static_cast<long long>(Offset),
                 static_cast<unsigned long long>(Size), Ret, Errno, State,
                 static_cast<unsigned long long>(ExpectedVersion));

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfilePwrite(MPIRequestManagerTy RequestManager) {
    int Fd = -1;
    int64_t Offset = 0;
    uint64_t Size = 0;

    RequestManager.receive(&Fd, 1, MPI_INT);
    RequestManager.receive(&Offset, 1, MPI_INT64_T);
    RequestManager.receive(&Size, 1, MPI_UINT64_T);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::vector<char> Buffer;
    if (Size > 0) {
      Buffer.resize(Size);
      OmpFileStatsPwritePayloadBytes.fetch_add(Size, std::memory_order_relaxed);
      if (OmpFileForceBlockingPwrite.get()) {
        OmpFileStatsPwriteBlockingFallbacks.fetch_add(
            1, std::memory_order_relaxed);
        if (auto Error =
                RequestManager.receiveInBatchsBlocking(Buffer.data(), Size);
            Error) {
          OmpFileStatsPwriteFailures.fetch_add(1, std::memory_order_relaxed);
          co_return Error;
        }
      } else {
        // Async coroutine-backed payload receive (the documented default).
        // This branch was removed in 1e1327ce during the Apr 2026 stale-read
        // narrowing and never restored — every pwrite silently took the
        // blocking path while docs/quick-reference still claimed async was
        // the default (sorgan 3704 payload sweep: pwrite_async_events=0,
        // pwrite_blocking_fallbacks=64/128 in every cell). The stale-read
        // surfaces that motivated the narrowing have since been closed
        // (freshness guard, oracle gates, populate dirty-extent fix), so
        // restore the async path behind the original
        // LIBOMPFILE_MPP_FORCE_BLOCKING_PWRITE=1 escape hatch.
        OmpFileStatsPwriteAsyncEvents.fetch_add(1, std::memory_order_relaxed);
        OmpFileStatsPwriteAsyncFragments.fetch_add(
            computePwriteFragmentCount(Size), std::memory_order_relaxed);
        RequestManager.receiveInBatchs(Buffer.data(), Size);
        if (auto Error = co_await RequestManager; Error) {
          OmpFileStatsPwriteFailures.fetch_add(1, std::memory_order_relaxed);
          co_return Error;
        }
      }
    }

    uint64_t BytesWrittenOut = 0;
    int Errno = 0;
    const bool Ok =
        Size == 0 || pwriteWithOptionalStage(Fd, Offset, Buffer.data(), Size,
                                             &BytesWrittenOut, Errno);

    int Ret = Ok ? 0 : -1;
    if (Ret != 0)
      OmpFileStatsPwriteFailures.fetch_add(1, std::memory_order_relaxed);
    RequestManager.send(&Ret, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(&BytesWrittenOut, 1, MPI_UINT64_T);
    if (auto Error = co_await RequestManager; Error) {
      OmpFileStatsPwriteFailures.fetch_add(1, std::memory_order_relaxed);
      co_return Error;
    }
    traceOmpFile("event ompfilePwrite fd=%d offset=%lld size=%llu buf=%p "
                 "ret=%d errno=%d bytes=%llu transport=%s fragments=%llu\n",
                 Fd, static_cast<long long>(Offset),
                 static_cast<unsigned long long>(Size),
                 static_cast<void *>(Buffer.data()), Ret, Errno,
                 static_cast<unsigned long long>(BytesWrittenOut),
                 OmpFileForceBlockingPwrite.get() ? "blocking_debug" : "async",
                 static_cast<unsigned long long>(
                     computePwriteFragmentCount(Size)));
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileStageInvalidate(MPIRequestManagerTy RequestManager) {
    OmpFileStageInvalidateRequest Request{};
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    OmpFileStatsStageGlobalInvalidationRequests.fetch_add(
        1, std::memory_order_relaxed);

    OmpFileStageInvalidateReply Reply{};
    Reply.AbiVersion = OMPFILE_STAGE_INVALIDATE_ABI_VERSION;
    constexpr uint32_t MaxStageInvalidatePathSize =
        static_cast<uint32_t>(PATH_MAX) + 1;
    if (Request.PathSize > static_cast<uint32_t>(INT_MAX) ||
        Request.PathSize > MaxStageInvalidatePathSize) {
      Reply.Status = -1;
      Reply.Errno = EOVERFLOW;
      OmpFileStatsStageGlobalInvalidationFailures.fetch_add(
          1, std::memory_order_relaxed);
      RequestManager.send(&Reply, sizeof(Reply), MPI_BYTE);
      RequestManager.send(nullptr, 0, MPI_BYTE);
      co_return (co_await RequestManager);
    }

    std::vector<char> PathBytes;
    if (Request.PathSize > 0) {
      PathBytes.resize(Request.PathSize);
      RequestManager.receive(PathBytes.data(), static_cast<int>(Request.PathSize),
                             MPI_CHAR);
      if (auto Error = co_await RequestManager; Error)
        co_return Error;
    }

    std::string Path;
    if (!PathBytes.empty()) {
      const char *Begin = PathBytes.data();
      const size_t Len = PathBytes.back() == '\0' ? PathBytes.size() - 1
                                                  : PathBytes.size();
      Path.assign(Begin, Begin + Len);
    }

    Reply = invalidateStageForRequest(Request, Path);
    if (Reply.Status != 0)
      OmpFileStatsStageGlobalInvalidationFailures.fetch_add(
          1, std::memory_order_relaxed);
    RequestManager.send(&Reply, sizeof(Reply), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileFreshnessQuery(MPIRequestManagerTy RequestManager) {
    OmpFileFreshnessQueryRequest Request{};
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    OmpFileFreshnessQueryReply Reply{};
    OmpFileHeadnodeManager::instance().handleFreshnessQueryRequest(Request,
                                                                    Reply);

    RequestManager.send(&Reply, sizeof(Reply), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileProxyCopyTile(MPIRequestManagerTy RequestManager) {
    OmpFileProxyCopyTileRequest Request{};
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    OmpFileProxyCopyTileReply Reply{};
    Reply.AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
    const std::string PathKeyStr = std::to_string(Request.PathKey);
    const std::string TileIdStr = std::to_string(Request.TileId);
    OmpFileHeadnodeManager::instance().emitFreshnessTrace(
        "copy_req", PathKeyStr, Request.DestRank, Request.SourceRank,
        Request.DestRank, Request.Version, Request.Version, "copy", "start", "",
        TileIdStr.c_str());
    if (Request.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      Reply.Status = -1;
      Reply.Errno = EPROTO;
      OmpFileHeadnodeManager::instance().emitFreshnessTrace(
          "copy_done", PathKeyStr, Request.DestRank, Request.SourceRank,
          Request.DestRank, Request.Version, Request.Version, "copy", "fail",
          "abi_mismatch", TileIdStr.c_str());
    } else if (!isWorkerRank(Request.SourceRank) ||
               !isWorkerRank(Request.DestRank) || Request.PathKey == 0 ||
               Request.Version == 0) {
      Reply.Status = -1;
      Reply.Errno = EINVAL;
      OmpFileHeadnodeManager::instance().emitFreshnessTrace(
          "copy_done", PathKeyStr, Request.DestRank, Request.SourceRank,
          Request.DestRank, Request.Version, Request.Version, "copy", "fail",
          "invalid_request", TileIdStr.c_str());
    } else {
      if (EventSystem.LocalRank == getHeadnodeRank()) {
        if (!OmpFileHeadnodeManager::instance().registerTileCopy(
                Request.PathKey, Request.SourceRank, Request.DestRank,
                Request.Version)) {
          Reply.Status = -1;
          Reply.Errno = errno != 0 ? errno : EIO;
          OmpFileHeadnodeManager::instance().emitFreshnessTrace(
              "copy_done", PathKeyStr, Request.DestRank, Request.SourceRank,
              Request.DestRank, Request.Version, Request.Version, "copy", "fail",
              "register_copy_failed", TileIdStr.c_str());
        }
      }
      if (Reply.Status == 0) {
        Reply.Errno = 0;
        Reply.BytesCopied = 0;
        OmpFileHeadnodeManager::instance().emitFreshnessTrace(
            "copy_done", PathKeyStr, Request.DestRank, Request.SourceRank,
            Request.DestRank, Request.Version, Request.Version, "copy", "ok", "",
            TileIdStr.c_str());
      }
    }

    RequestManager.send(&Reply, sizeof(Reply), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileFreshnessMarkFresh(MPIRequestManagerTy RequestManager) {
    OmpFileFreshnessMarkFreshRequest Request{};
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    OmpFileFreshnessMarkFreshReply Reply{};
    Reply.AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
    if (Request.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      Reply.Status = -1;
      Reply.Errno = EPROTO;
    } else if (!OmpFileHeadnodeManager::instance().markTileFreshFromCopy(
                   Request.PathKey, Request.Rank, Request.Version)) {
      Reply.Status = -1;
      Reply.Errno = errno != 0 ? errno : EIO;
    }

    RequestManager.send(&Reply, sizeof(Reply), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileFreshnessWriteCommit(MPIRequestManagerTy RequestManager) {
    OmpFileFreshnessWriteCommitRequest Request{};
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    OmpFileFreshnessWriteCommitReply Reply{};
    Reply.AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
    if (Request.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      Reply.Status = -1;
      Reply.Errno = EPROTO;
    } else if (!OmpFileHeadnodeManager::instance().commitTileFreshnessWrite(
                   Request.PathKey, Request.WriterRank,
                   Request.WriteThroughMode != 0, Reply.CommittedVersion)) {
      Reply.Status = -1;
      Reply.Errno = errno != 0 ? errno : EIO;
    }

    RequestManager.send(&Reply, sizeof(Reply), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileFlushDirtyTile(MPIRequestManagerTy RequestManager) {
    OmpFileFlushDirtyTileRequest Request{};
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    OmpFileFlushDirtyTileReply Reply{};
    Reply.AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
    if (Request.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      Reply.Status = -1;
      Reply.Errno = EPROTO;
    } else if (Request.Action == 0) {
      if (!OmpFileHeadnodeManager::instance().flushDirtyTile(
              Request.PathKey, Reply.SourceRank, Reply.FlushedVersion)) {
        Reply.Status = -1;
        Reply.Errno = errno != 0 ? errno : EIO;
      }
    } else if (Request.Action == 1) {
      if (!OmpFileHeadnodeManager::instance().completeDirtyFlush(
              Request.PathKey, Request.SourceRank, Request.Version,
              Request.Success != 0)) {
        Reply.Status = -1;
        Reply.Errno = errno != 0 ? errno : EIO;
      }
      Reply.SourceRank = Request.SourceRank;
      Reply.FlushedVersion = Request.Version;
    } else if (Request.Action == 2) {
      if (!flushLocalDirtyTileToSource(Request.PathKey, Request.Version)) {
        Reply.Status = -1;
        Reply.Errno = errno != 0 ? errno : EIO;
      } else {
        Reply.SourceRank = EventSystem.LocalRank;
        Reply.FlushedVersion = Request.Version;
      }
    } else {
      Reply.Status = -1;
      Reply.Errno = EINVAL;
    }

    RequestManager.send(&Reply, sizeof(Reply), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  bool isWorkerRank(int Rank) const {
    return Rank >= 0 && Rank < EventSystem.WorldSize - 1;
  }

  int getHeadnodeRank() const { return 0; }

  OmpFileIOPlan buildSchedulePlan(const OmpFileIORequest &Request,
                                  const char *Path) {
    OmpFileHeadnodeManager &Manager = OmpFileHeadnodeManager::instance();
    Manager.initialize(EventSystem.WorldSize, getHeadnodeRank());
    return Manager.planRequest(Request, Path, EventSystem.LocalRank);
  }

  bool buildScheduleBatchPlan(const OmpFileIOBatchRequest &Request,
                              const void *Payload, uint64_t PayloadBytes,
                              OmpFileIOBatchPlan &Plan,
                              std::vector<uint8_t> &PlanPayload) {
    Plan = {};
    PlanPayload.clear();
    Plan.AbiVersion = OMPFILE_SCHED_BATCH_ABI_VERSION;
    Plan.SegmentCount = Request.SegmentCount;
    Plan.BatchId = Request.BatchId;
    Plan.PlanFlags = OMPFILE_BATCH_PLAN_BATCH_API;

    if (Request.AbiVersion != OMPFILE_SCHED_BATCH_ABI_VERSION) {
      Plan.Status = -1;
      Plan.Errno = EPROTO;
      return true;
    }

    if (Request.PayloadBytes != PayloadBytes) {
      Plan.Status = -1;
      Plan.Errno = EINVAL;
      return true;
    }

    const uint64_t SegmentCount = static_cast<uint64_t>(Request.SegmentCount);
    const uint64_t SegmentBytes =
        SegmentCount * static_cast<uint64_t>(sizeof(OmpFileIOBatchSegment));
    if (Request.SegmentCount != 0 &&
        SegmentBytes / sizeof(OmpFileIOBatchSegment) != SegmentCount) {
      Plan.Status = -1;
      Plan.Errno = EOVERFLOW;
      return true;
    }

    if (SegmentBytes != PayloadBytes) {
      Plan.Status = -1;
      Plan.Errno = EINVAL;
      return true;
    }

    if (PayloadBytes > 0 && !Payload) {
      Plan.Status = -1;
      Plan.Errno = EINVAL;
      return true;
    }

    if (Request.SegmentCount == 0) {
      Plan.PayloadBytes = 0;
      return true;
    }

    const auto *Segments = static_cast<const OmpFileIOBatchSegment *>(Payload);
    std::vector<OmpFileIOBatchPlanEntry> Entries;
    OmpFileHeadnodeManager &Manager = OmpFileHeadnodeManager::instance();
    Manager.initialize(EventSystem.WorldSize, getHeadnodeRank());
    if (!Manager.planBatchRequest(Request, Segments, Plan, Entries,
                                  EventSystem.LocalRank)) {
      Plan.Status = -1;
      Plan.Errno = EIO;
      return true;
    }
    if (Entries.size() != Request.SegmentCount) {
      Plan.Status = -1;
      Plan.Errno = EPROTO;
      return true;
    }

    const uint64_t EntryBytes =
        static_cast<uint64_t>(Entries.size()) *
        static_cast<uint64_t>(sizeof(OmpFileIOBatchPlanEntry));
    if (EntryBytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      Plan.Status = -1;
      Plan.Errno = EOVERFLOW;
      return true;
    }

    PlanPayload.resize(static_cast<size_t>(EntryBytes));
    if (!PlanPayload.empty())
      std::memcpy(PlanPayload.data(), Entries.data(), PlanPayload.size());
    Plan.PayloadBytes = static_cast<uint32_t>(PlanPayload.size());
    Plan.PlanFlags |= OMPFILE_BATCH_PLAN_BATCH_API;

    return true;
  }

  template <class EventFuncTy, typename... ArgsTy>
    requires std::invocable<EventFuncTy, MPIRequestManagerTy, ArgsTy...>
  EventTy createRankEvent(EventFuncTy EventFunc, EventTypeTy EventType,
                          int TargetRank, int TargetDeviceId, ArgsTy... Args) {
    traceOmpFile("createRankEvent enter type=%s target_rank=%d target_dev=%d\n",
                 EventTypeToString(EventType).c_str(), TargetRank,
                 TargetDeviceId);
    if (!isWorkerRank(TargetRank)) {
      REPORT("Invalid OMPFile target rank %d for event %s.\n", TargetRank,
             EventTypeToString(EventType).c_str());
      traceOmpFile("createRankEvent invalid target type=%s rank=%d\n",
                   EventTypeToString(EventType).c_str(), TargetRank);
      return EventTy{};
    }

    const int EventTag = EventSystem.createNewEventTag();
    auto &EventComm = EventSystem.getNewEventComm(EventTag);
    const int MPITag = EventSystem.getEventMPITag(EventTag);
    int EventNotificationInfo[] = {static_cast<int>(EventType), EventTag,
                                   TargetDeviceId};
    MPI_Request NotificationRequest = MPI_REQUEST_NULL;
    const int MPIError = ompfileWithMPICallLock([&]() {
      return MPI_Isend(EventNotificationInfo, 3, MPI_INT, TargetRank,
                       static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                       EventSystem.GateThreadComm, &NotificationRequest);
    });

    if (MPIError != MPI_SUCCESS) {
      REPORT("Failed to notify rank %d for OMPFile event %s. MPI error=%d\n",
             TargetRank, EventTypeToString(EventType).c_str(), MPIError);
      traceOmpFile("createRankEvent notify fail type=%s rank=%d mpi_err=%d\n",
                   EventTypeToString(EventType).c_str(), TargetRank, MPIError);
      return EventTy{};
    }

    MPIRequestManagerTy RequestManager(EventComm, MPITag, TargetRank,
                                       TargetDeviceId, {NotificationRequest});
    RequestManager.EventType = EventNotificationInfo[0];

    auto Event = EventFunc(std::move(RequestManager), Args...);
    Event.setEventType(EventType);
    traceOmpFile("createRankEvent exit type=%s tag=%d target_rank=%d\n",
                 EventTypeToString(EventType).c_str(), EventTag, TargetRank);
    return Event;
  }

  bool waitForEvent(EventTy &Event, const char *OpName) {
    traceOmpFile("waitForEvent enter op=%s empty=%d\n", OpName ? OpName : "?",
                 static_cast<int>(Event.empty()));
    if (Event.empty()) {
      errno = EIO;
      REPORT("OMPFile %s failed to create event.\n", OpName);
      traceOmpFile("waitForEvent empty op=%s\n", OpName ? OpName : "?");
      return false;
    }

    Event.wait();
    if (auto Error = Event.getError()) {
      errno = EIO;
      REPORT("OMPFile %s event failed: %s\n", OpName,
             toString(std::move(Error)).c_str());
      traceOmpFile("waitForEvent error op=%s\n", OpName ? OpName : "?");
      return false;
    }
    traceOmpFile("waitForEvent success op=%s\n", OpName ? OpName : "?");
    return true;
  }

  bool dispatchSchedRequest(const OmpFileIORequest &Request, const char *Path,
                            OmpFileIOPlan &Plan) {
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      Plan = buildSchedulePlan(Request, Path);
    } else {
      EventTy Event = createRankEvent(
          OriginEvents::ompfileSchedRequest, EventTypeTy::OMPFILE_SCHED_REQUEST,
          getHeadnodeRank(), /*TargetDeviceId=*/0, &Request, Path, &Plan);
      if (!waitForEvent(Event, "sched_request"))
        return false;
    }

    if (Plan.Status != 0) {
      errno = Plan.Errno;
      return false;
    }

    return true;
  }

  bool dispatchSchedBatchRequest(const OmpFileIOBatchRequest &Request,
                                 const void *RequestPayload,
                                 uint64_t RequestPayloadBytes,
                                 OmpFileIOBatchPlan &Plan, void *PlanPayload,
                                 uint64_t PlanPayloadCapBytes,
                                 uint64_t &PlanPayloadOutBytes) {
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      std::vector<uint8_t> LocalPayload;
      if (!buildScheduleBatchPlan(Request, RequestPayload, RequestPayloadBytes,
                                  Plan, LocalPayload))
        return false;

      PlanPayloadOutBytes = LocalPayload.size();
      if (PlanPayloadOutBytes > PlanPayloadCapBytes) {
        errno = ENOBUFS;
        return false;
      }

      if (PlanPayloadOutBytes > 0) {
        if (!PlanPayload) {
          errno = EINVAL;
          return false;
        }
        std::memcpy(PlanPayload, LocalPayload.data(),
                    static_cast<size_t>(PlanPayloadOutBytes));
      }

      return true;
    }

    EventTy Event = createRankEvent(
        OriginEvents::ompfileSchedBatchRequest,
        EventTypeTy::OMPFILE_SCHED_REQUEST_BATCH, getHeadnodeRank(),
        /*TargetDeviceId=*/0, &Request, RequestPayload, RequestPayloadBytes,
        &Plan, PlanPayload, PlanPayloadCapBytes, &PlanPayloadOutBytes);
    if (!waitForEvent(Event, "sched_request_batch"))
      return false;

    return true;
  }

  int registerRemoteHandle(int Rank, int RemoteHandle) {
    const int LocalHandle =
        NextOmpFileHandle.fetch_add(1, std::memory_order_relaxed);
    const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
    OmpFileHandles[LocalHandle] = {Rank, RemoteHandle};
    traceOmpFile("registerRemoteHandle local=%d rank=%d remote=%d map_size=%zu\n",
                 LocalHandle, Rank, RemoteHandle, OmpFileHandles.size());
    return LocalHandle;
  }

  bool findRemoteHandle(int LocalHandle, OmpFileHandleEntry &Entry) {
    const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
    auto It = OmpFileHandles.find(LocalHandle);
    if (It == OmpFileHandles.end()) {
      errno = EBADF;
      traceOmpFile("findRemoteHandle miss local=%d map_size=%zu\n", LocalHandle,
                   OmpFileHandles.size());
      return false;
    }
    Entry = It->second;
    traceOmpFile("findRemoteHandle hit local=%d rank=%d remote=%d map_size=%zu\n",
                 LocalHandle, Entry.Rank, Entry.RemoteHandle,
                 OmpFileHandles.size());
    return true;
  }

  bool eraseRemoteHandle(int LocalHandle, OmpFileHandleEntry &Entry) {
    const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
    auto It = OmpFileHandles.find(LocalHandle);
    if (It == OmpFileHandles.end()) {
      errno = EBADF;
      traceOmpFile("eraseRemoteHandle miss local=%d map_size=%zu\n", LocalHandle,
                   OmpFileHandles.size());
      return false;
    }
    Entry = It->second;
    OmpFileHandles.erase(It);
    traceOmpFile(
        "eraseRemoteHandle hit local=%d rank=%d remote=%d map_size=%zu\n",
        LocalHandle, Entry.Rank, Entry.RemoteHandle, OmpFileHandles.size());
    return true;
  }

  bool openOnRank(int Rank, const char *Path, int Flags, int Mode,
                  int &RemoteHandle) {
    if (!Path) {
      errno = EINVAL;
      return false;
    }

    if (Rank == EventSystem.LocalRank) {
      int OpenErrno = 0;
      int Fd = openWithOptionalCache(Path, Flags, Mode, OpenErrno);
      if (Fd < 0)
        errno = OpenErrno;
      if (Fd < 0)
        return false;
      trackOmpFileFd(Fd, Path, Flags, Mode);
      if (shouldInvalidateStageOnOpen(Flags))
        invalidateStageForPath(Path);
      RemoteHandle = Fd;
      return true;
    }

    int RemoteErrno = 0;
    EventTy Event =
        createRankEvent(OriginEvents::ompfileOpen, EventTypeTy::OMPFILE_OPEN,
                        Rank, /*TargetDeviceId=*/0, Path, Flags, Mode,
                        &RemoteHandle, &RemoteErrno);
    if (!waitForEvent(Event, "open"))
      return false;

    if (RemoteHandle < 0) {
      errno = RemoteErrno;
      return false;
    }
    return true;
  }

  bool closeOnRank(int Rank, int RemoteHandle) {
    if (Rank == EventSystem.LocalRank) {
      int CloseErrno = 0;
      if (closeWithOptionalCache(RemoteHandle, CloseErrno) != 0) {
        errno = CloseErrno;
        return false;
      }
      return true;
    }

    int CloseRet = -1;
    int RemoteErrno = 0;
    EventTy Event = createRankEvent(
        OriginEvents::ompfileClose, EventTypeTy::OMPFILE_CLOSE, Rank,
        /*TargetDeviceId=*/0, RemoteHandle, &CloseRet, &RemoteErrno);
    if (!waitForEvent(Event, "close"))
      return false;
    if (CloseRet != 0) {
      errno = RemoteErrno;
      return false;
    }
    return true;
  }

  bool preadOnRank(int Rank, int RemoteHandle, int64_t Offset, void *Buffer,
                   uint64_t Size, uint64_t *BytesReadOut = nullptr,
                   bool AllowStage = true) {
    if (!Buffer && Size > 0) {
      errno = EINVAL;
      return false;
    }
    if (BytesReadOut)
      *BytesReadOut = 0;

    if (Rank == EventSystem.LocalRank) {
      int ReadErrno = 0;
      uint64_t Bytes = 0;
      if (!preadWithOptionalStage(RemoteHandle, Offset, Buffer, Size, &Bytes,
                                  ReadErrno, AllowStage)) {
        errno = ReadErrno;
        return false;
      }
      if (Bytes < Size)
        std::memset(static_cast<char *>(Buffer) + Bytes, 0,
                    static_cast<size_t>(Size - Bytes));
      if (BytesReadOut)
        *BytesReadOut = Bytes;
      return true;
    }

    int IoRet = -1;
    int RemoteErrno = 0;
    uint64_t Bytes = 0;
    EventTy Event = createRankEvent(
        OriginEvents::ompfilePread,
        AllowStage ? EventTypeTy::OMPFILE_PREAD
                   : EventTypeTy::OMPFILE_PREAD_NO_STAGE,
        Rank,
        /*TargetDeviceId=*/0, RemoteHandle, Offset, Buffer, Size, &IoRet,
        &RemoteErrno, &Bytes);
    if (!waitForEvent(Event, "pread"))
      return false;
    if (IoRet != 0) {
      errno = RemoteErrno;
      return false;
    }
    if (Bytes > Size) {
      errno = EPROTO;
      return false;
    }
    if (Bytes < Size)
      std::memset(static_cast<char *>(Buffer) + Bytes, 0,
                  static_cast<size_t>(Size - Bytes));
    if (BytesReadOut)
      *BytesReadOut = Bytes;
    return true;
  }

  bool pwriteOnRank(int Rank, int RemoteHandle, int64_t Offset,
                    const void *Buffer, uint64_t Size,
                    uint64_t *BytesWrittenOut) {
    if (!Buffer && Size > 0) {
      errno = EINVAL;
      return false;
    }
    if (BytesWrittenOut)
      *BytesWrittenOut = 0;

    if (Rank == EventSystem.LocalRank) {
      int WriteErrno = 0;
      if (!pwriteWithOptionalStage(RemoteHandle, Offset, Buffer, Size,
                                   BytesWrittenOut, WriteErrno)) {
        errno = WriteErrno;
        return false;
      }
      return true;
    }

    int IoRet = -1;
    int RemoteErrno = 0;
    uint64_t Bytes = 0;
    EventTy Event = createRankEvent(
        OriginEvents::ompfilePwrite, EventTypeTy::OMPFILE_PWRITE, Rank,
        /*TargetDeviceId=*/0, RemoteHandle, Offset, Buffer, Size, &IoRet,
        &RemoteErrno, &Bytes);
    if (!waitForEvent(Event, "pwrite"))
      return false;
    if (IoRet != 0) {
      errno = RemoteErrno;
      return false;
    }
    if (Bytes > Size) {
      errno = EPROTO;
      return false;
    }
    if (BytesWrittenOut)
      *BytesWrittenOut = Bytes;
    if (Bytes < Size) {
      errno = EIO;
      return false;
    }
    return true;
  }

  bool stageInvalidateOnRank(int Rank,
                             const OmpFileStageInvalidateRequest &Request,
                             const char *Path,
                             OmpFileStageInvalidateReply &Reply) {
    Reply = {};
    Reply.AbiVersion = OMPFILE_STAGE_INVALIDATE_ABI_VERSION;

    if (Rank == EventSystem.LocalRank) {
      Reply = invalidateStageForRequest(Request, Path ? Path : "");
      return true;
    }

    EventTy Event = createRankEvent(
        OriginEvents::ompfileStageInvalidate,
        EventTypeTy::OMPFILE_STAGE_INVALIDATE, Rank, /*TargetDeviceId=*/0,
        &Request, Path, &Reply);
    return waitForEvent(Event, "stage_invalidate");
  }

  bool freshnessQueryOnHeadnode(const OmpFileFreshnessQueryRequest &Request,
                                OmpFileFreshnessQueryReply &Reply) {
    Reply = {};
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      OmpFileHeadnodeManager::instance().handleFreshnessQueryRequest(Request,
                                                                      Reply);
      return true;
    }

    EventTy Event = createRankEvent(OriginEvents::ompfileFreshnessQuery,
                                    EventTypeTy::OMPFILE_FRESHNESS_QUERY,
                                    getHeadnodeRank(), /*TargetDeviceId=*/0,
                                    &Request, &Reply);
    return waitForEvent(Event, "freshness_query");
  }

  bool freshnessWriteCommitOnHeadnode(uint64_t PathKey, int WriterRank,
                                      uint64_t TileId, bool WriteThroughMode,
                                      uint64_t &CommittedVersionOut) {
    (void)TileId;
    CommittedVersionOut = 0;
    if (PathKey == 0 || WriterRank < 0) {
      errno = EINVAL;
      return false;
    }

    if (EventSystem.LocalRank == getHeadnodeRank()) {
      return OmpFileHeadnodeManager::instance().commitTileFreshnessWrite(
          PathKey, WriterRank, WriteThroughMode, CommittedVersionOut);
    }

    OmpFileFreshnessWriteCommitRequest Request{};
    Request.PathKey = PathKey;
    Request.TileId = TileId;
    Request.WriterRank = WriterRank;
    Request.WriteThroughMode = WriteThroughMode ? 1u : 0u;
    OmpFileFreshnessWriteCommitReply Reply{};
    EventTy Event = createRankEvent(
        OriginEvents::ompfileFreshnessWriteCommit,
        EventTypeTy::OMPFILE_FRESHNESS_WRITE_COMMIT, getHeadnodeRank(),
        /*TargetDeviceId=*/0, &Request, &Reply);
    if (!waitForEvent(Event, "freshness_write_commit"))
      return false;
    if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      errno = EPROTO;
      return false;
    }
    if (Reply.Status != 0) {
      errno = Reply.Errno != 0 ? Reply.Errno : EIO;
      return false;
    }
    CommittedVersionOut = Reply.CommittedVersion;
    return true;
  }

  bool flushDirtyTileOnHeadnode(uint64_t PathKey, int &SourceRankOut,
                                uint64_t &FlushedVersionOut) {
    SourceRankOut = -1;
    FlushedVersionOut = 0;
    if (PathKey == 0) {
      errno = EINVAL;
      return false;
    }

    if (EventSystem.LocalRank == getHeadnodeRank()) {
      return OmpFileHeadnodeManager::instance().flushDirtyTile(
          PathKey, SourceRankOut, FlushedVersionOut);
    }

    OmpFileFlushDirtyTileRequest Request{};
    Request.PathKey = PathKey;
    Request.Action = 0;
    OmpFileFlushDirtyTileReply Reply{};
    EventTy Event = createRankEvent(OriginEvents::ompfileFlushDirtyTile,
                                    EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE,
                                    getHeadnodeRank(), /*TargetDeviceId=*/0,
                                    &Request, &Reply);
    if (!waitForEvent(Event, "flush_dirty_tile"))
      return false;
    if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      errno = EPROTO;
      return false;
    }
    if (Reply.Status != 0) {
      errno = Reply.Errno != 0 ? Reply.Errno : EIO;
      return false;
    }
    SourceRankOut = Reply.SourceRank;
    FlushedVersionOut = Reply.FlushedVersion;
    return true;
  }

  bool completeDirtyFlushOnHeadnode(uint64_t PathKey, int SourceRank,
                                    uint64_t Version, bool Success) {
    if (PathKey == 0 || SourceRank < 0 || Version == 0) {
      errno = EINVAL;
      return false;
    }
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      return OmpFileHeadnodeManager::instance().completeDirtyFlush(
          PathKey, SourceRank, Version, Success);
    }

    OmpFileFlushDirtyTileRequest Request{};
    Request.Action = 1;
    Request.PathKey = PathKey;
    Request.SourceRank = SourceRank;
    Request.Version = Version;
    Request.Success = Success ? 1 : 0;
    OmpFileFlushDirtyTileReply Reply{};
    EventTy Event = createRankEvent(OriginEvents::ompfileFlushDirtyTile,
                                    EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE,
                                    getHeadnodeRank(), /*TargetDeviceId=*/0,
                                    &Request, &Reply);
    if (!waitForEvent(Event, "flush_dirty_tile_complete"))
      return false;
    if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      errno = EPROTO;
      return false;
    }
    if (Reply.Status != 0) {
      errno = Reply.Errno != 0 ? Reply.Errno : EIO;
      return false;
    }
    return true;
  }

  bool freshnessMarkFreshOnHeadnode(uint64_t PathKey, int Rank,
                                    uint64_t Version) {
    if (PathKey == 0 || Rank < 0 || Version == 0) {
      errno = EINVAL;
      return false;
    }

    if (EventSystem.LocalRank == getHeadnodeRank()) {
      return OmpFileHeadnodeManager::instance().markTileFreshFromCopy(
          PathKey, Rank, Version);
    }

    OmpFileFreshnessMarkFreshRequest Request{};
    Request.PathKey = PathKey;
    Request.Rank = Rank;
    Request.Version = Version;
    OmpFileFreshnessMarkFreshReply Reply{};
    EventTy Event = createRankEvent(OriginEvents::ompfileFreshnessMarkFresh,
                                    EventTypeTy::OMPFILE_FRESHNESS_MARK_FRESH,
                                    getHeadnodeRank(), /*TargetDeviceId=*/0,
                                    &Request, &Reply);
    if (!waitForEvent(Event, "freshness_mark_fresh"))
      return false;
    if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      errno = EPROTO;
      return false;
    }
    if (Reply.Status != 0) {
      errno = Reply.Errno != 0 ? Reply.Errno : EIO;
      return false;
    }
    return true;
  }

  bool proxyCopyTileEvent(uint64_t PathKey, uint64_t TileId, int SourceRank,
                          int DestRank, uint64_t Version) {
    if (PathKey == 0 || Version == 0 || !isWorkerRank(SourceRank) ||
        !isWorkerRank(DestRank)) {
      errno = EINVAL;
      return false;
    }
    OmpFileProxyCopyTileRequest Request{};
    Request.PathKey = PathKey;
    Request.TileId = TileId;
    Request.SourceRank = SourceRank;
    Request.DestRank = DestRank;
    Request.Version = Version;

    OmpFileProxyCopyTileReply HeadnodeReply{};
    EventTy HeadnodeEvent = createRankEvent(
        OriginEvents::ompfileProxyCopyTile,
        EventTypeTy::OMPFILE_PROXY_COPY_TILE,
        getHeadnodeRank(), /*TargetDeviceId=*/0, &Request, &HeadnodeReply);
    if (!waitForEvent(HeadnodeEvent, "proxy_copy_tile_register"))
      return false;
    if (HeadnodeReply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      errno = EPROTO;
      return false;
    }
    if (HeadnodeReply.Status != 0) {
      errno = HeadnodeReply.Errno != 0 ? HeadnodeReply.Errno : EIO;
      return false;
    }

    if (SourceRank == DestRank)
      return true;

    OmpFileProxyCopyTileReply Reply{};
    EventTy Event = createRankEvent(OriginEvents::ompfileProxyCopyTile,
                                    EventTypeTy::OMPFILE_PROXY_COPY_TILE,
                                    DestRank, /*TargetDeviceId=*/0, &Request,
                                    &Reply);
    if (!waitForEvent(Event, "proxy_copy_tile"))
      return false;
    if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      errno = EPROTO;
      return false;
    }
    if (Reply.Status != 0) {
      errno = Reply.Errno != 0 ? Reply.Errno : EIO;
      return false;
    }
    return true;
  }

  int selectAggregatorRankForOpen(const char *Path, int Flags, int Mode) {
    int Rank = isWorkerRank(EventSystem.LocalRank) ? EventSystem.LocalRank : 0;
    const char *SchedulerEnv = std::getenv("LIBOMPFILE_SCHEDULER");
    if (!SchedulerEnv || std::strcmp(SchedulerEnv, "HEADNODE") != 0)
      return Rank;

    OmpFileIORequest Request{};
    Request.RequestId =
        NextSchedRequestId.fetch_add(1, std::memory_order_relaxed);
    Request.Op = OmpFileIOOp::OPEN;
    Request.Flags = Flags;
    Request.Mode = Mode;
    Request.ClientRank = EventSystem.LocalRank;
    Request.PathSize = Path ? static_cast<uint32_t>(std::strlen(Path) + 1) : 0;

    OmpFileIOPlan Plan{};
    if (!dispatchSchedRequest(Request, Path, Plan)) {
      DP("OMPFile scheduler request failed in proxy rank %d; using local rank\n",
         EventSystem.LocalRank);
      return Rank;
    }

    if (isWorkerRank(Plan.AggregatorRank))
      return Plan.AggregatorRank;
    return Rank;
  }

  int mppInit() {
    DP("ompfile_mpp_init via proxy runtime (rank=%d, world=%d)\n",
       EventSystem.LocalRank, EventSystem.WorldSize);
    if (EventSystem.LocalRank == getHeadnodeRank())
      OmpFileHeadnodeManager::instance().initialize(EventSystem.WorldSize,
                                                    getHeadnodeRank());
    const auto State = EventSystem.EventSystemState.load();
    if (State == EventSystemStateTy::RUNNING ||
        State == EventSystemStateTy::INITIALIZED)
      return OFFLOAD_SUCCESS;
    return OFFLOAD_FAIL;
  }

  int mppSubmit(uint64_t Token) {
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      const std::lock_guard<std::mutex> Lock(MppEventMutex);
      CompletedMppTokens.insert(Token);
      return OFFLOAD_SUCCESS;
    }

    EventTy Event = createRankEvent(OriginEvents::ompfilePing,
                                    EventTypeTy::OMPFILE_PING,
                                    getHeadnodeRank(), /*TargetDeviceId=*/0,
                                    Token);
    if (Event.empty())
      return OFFLOAD_FAIL;

    const std::lock_guard<std::mutex> Lock(MppEventMutex);
    if (MppEvents.count(Token))
      return OFFLOAD_FAIL;
    MppEvents.emplace(Token, std::move(Event));
    return OFFLOAD_SUCCESS;
  }

  int mppPoll(uint64_t Token, int *Done) {
    if (!Done)
      return OFFLOAD_FAIL;

    const std::lock_guard<std::mutex> Lock(MppEventMutex);
    auto CompletedIt = CompletedMppTokens.find(Token);
    if (CompletedIt != CompletedMppTokens.end()) {
      *Done = 1;
      CompletedMppTokens.erase(CompletedIt);
      return OFFLOAD_SUCCESS;
    }

    auto It = MppEvents.find(Token);
    if (It == MppEvents.end())
      return OFFLOAD_FAIL;

    It->second.advance();
    if (!It->second.done()) {
      *Done = 0;
      return OFFLOAD_SUCCESS;
    }

    *Done = 1;
    auto Error = It->second.getError();
    MppEvents.erase(It);
    if (Error)
      return OFFLOAD_FAIL;
    return OFFLOAD_SUCCESS;
  }

  int mppOpen(const char *Path, int Flags, int Mode, int *Handle) {
    if (!Path || !Handle)
      return OFFLOAD_FAIL;

    traceOmpFile("mppOpen enter path=%s flags=0x%x mode=%o\n", Path, Flags,
                 Mode);
    const int Rank = selectAggregatorRankForOpen(Path, Flags, Mode);
    int RemoteHandle = -1;
    if (!openOnRank(Rank, Path, Flags, Mode, RemoteHandle))
      return OFFLOAD_FAIL;

    *Handle = registerRemoteHandle(Rank, RemoteHandle);
    traceOmpFile("mppOpen exit local=%d rank=%d remote=%d\n", *Handle, Rank,
                 RemoteHandle);
    return OFFLOAD_SUCCESS;
  }

  int mppOpenOnRank(const char *Path, int Flags, int Mode, int Rank,
                    int *Handle) {
    if (!Path || !Handle)
      return OFFLOAD_FAIL;
    if (!isWorkerRank(Rank)) {
      errno = EHOSTUNREACH;
      return OFFLOAD_FAIL;
    }

    traceOmpFile("mppOpenOnRank enter path=%s flags=0x%x mode=%o rank=%d\n",
                 Path, Flags, Mode, Rank);
    int RemoteHandle = -1;
    if (!openOnRank(Rank, Path, Flags, Mode, RemoteHandle))
      return OFFLOAD_FAIL;

    *Handle = registerRemoteHandle(Rank, RemoteHandle);
    traceOmpFile("mppOpenOnRank exit local=%d rank=%d remote=%d\n", *Handle,
                 Rank, RemoteHandle);
    return OFFLOAD_SUCCESS;
  }

  int mppClose(int Handle) {
    traceOmpFile("mppClose enter local=%d\n", Handle);
    OmpFileHandleEntry Entry{};
    if (!eraseRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (!closeOnRank(Entry.Rank, Entry.RemoteHandle))
      return OFFLOAD_FAIL;
    traceOmpFile("mppClose exit local=%d rank=%d remote=%d\n", Handle,
                 Entry.Rank, Entry.RemoteHandle);
    return OFFLOAD_SUCCESS;
  }

  int mppHandleOwnerRank(int Handle, int *RankOut) {
    if (!RankOut)
      return EINVAL;
    *RankOut = -1;
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return errno != 0 ? errno : EIO;
    *RankOut = Entry.Rank;
    return OFFLOAD_SUCCESS;
  }

  int mppPread(int Handle, int64_t Offset, void *Buffer, uint64_t Size) {
    traceOmpFile("mppPread enter local=%d offset=%lld size=%llu\n", Handle,
                 static_cast<long long>(Offset),
                 static_cast<unsigned long long>(Size));
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    uint64_t BytesRead = 0;
    if (!preadOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
                     &BytesRead)) {
      return OFFLOAD_FAIL;
    }
    traceOmpFile("mppPread exit local=%d rank=%d remote=%d bytes=%llu\n",
                 Handle, Entry.Rank, Entry.RemoteHandle,
                 static_cast<unsigned long long>(BytesRead));
    return OFFLOAD_SUCCESS;
  }

  int mppPreadEx(int Handle, int64_t Offset, void *Buffer, uint64_t Size,
                 uint64_t *BytesRead) {
    if (!BytesRead)
      return OFFLOAD_FAIL;
    *BytesRead = 0;
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (!preadOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
                      BytesRead))
      return OFFLOAD_FAIL;
    return OFFLOAD_SUCCESS;
  }

  int mppPreadNoStageEx(int Handle, int64_t Offset, void *Buffer, uint64_t Size,
                        uint64_t *BytesRead) {
    if (!BytesRead)
      return OFFLOAD_FAIL;
    *BytesRead = 0;
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (!preadOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
                     BytesRead, /*AllowStage=*/false))
      return OFFLOAD_FAIL;
    return OFFLOAD_SUCCESS;
  }


  int mppDirtyOwnerPreadEx(int Handle, int SourceRank,
                           uint64_t ExpectedVersion, int64_t Offset,
                           void *Buffer, uint64_t Size, uint64_t *BytesRead) {
    if (!BytesRead || SourceRank < 0)
      return OFFLOAD_FAIL;
    *BytesRead = 0;
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (Entry.Rank != SourceRank) {
      errno = EINVAL;
      return OFFLOAD_FAIL;
    }
    if (SourceRank == EventSystem.LocalRank) {
      int LocalErrno = 0;
      uint64_t LocalBytes = 0;
      if (!preadDirtyOwnerStageOnly(Entry.RemoteHandle, Offset, Buffer, Size,
                                    ExpectedVersion, &LocalBytes,
                                    LocalErrno)) {
        errno = LocalErrno != 0 ? LocalErrno : EIO;
        return OFFLOAD_FAIL;
      }
      if (LocalBytes > Size) {
        errno = EPROTO;
        return OFFLOAD_FAIL;
      }
      if (LocalBytes < Size)
        std::memset(static_cast<char *>(Buffer) + LocalBytes, 0,
                    static_cast<size_t>(Size - LocalBytes));
      *BytesRead = LocalBytes;
      return OFFLOAD_SUCCESS;
    }

    int IoRet = -1;
    int RemoteErrno = 0;
    uint64_t Bytes = 0;
    EventTy Event = createRankEvent(
        OriginEvents::ompfileDirtyOwnerPread,
        EventTypeTy::OMPFILE_DIRTY_OWNER_PREAD, SourceRank,
        /*TargetDeviceId=*/0, Entry.RemoteHandle, Offset, Buffer, Size,
        ExpectedVersion, &IoRet, &RemoteErrno, &Bytes);
    if (!waitForEvent(Event, "dirty_owner_pread"))
      return OFFLOAD_FAIL;
    if (IoRet != 0) {
      errno = RemoteErrno;
      return OFFLOAD_FAIL;
    }
    if (Bytes > Size) {
      errno = EPROTO;
      return OFFLOAD_FAIL;
    }
    if (Bytes < Size)
      std::memset(static_cast<char *>(Buffer) + Bytes, 0,
                  static_cast<size_t>(Size - Bytes));
    *BytesRead = Bytes;
    return OFFLOAD_SUCCESS;
  }

  int mppDirtyOwnerPreadBatchEx(
      int Handle, int SourceRank,
      const OmpFileDirtyOwnerPreadBatchSegment *Segments, uint64_t SegmentCount,
      void *const *Buffers, uint64_t *BytesRead, int *Statuses,
      int *Errnos) {
    if (!Segments || !Buffers || !BytesRead || !Statuses || !Errnos ||
        SourceRank < 0 || SegmentCount == 0) {
      errno = EINVAL;
      return OFFLOAD_FAIL;
    }
    for (uint64_t I = 0; I < SegmentCount; ++I) {
      BytesRead[I] = 0;
      Statuses[I] = -1;
      Errnos[I] = 0;
    }
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (Entry.Rank != SourceRank) {
      errno = EINVAL;
      return OFFLOAD_FAIL;
    }
    if (SourceRank == EventSystem.LocalRank) {
      for (uint64_t I = 0; I < SegmentCount; ++I) {
        int LocalErrno = 0;
        uint64_t LocalBytes = 0;
        if (!preadDirtyOwnerStageOnly(Entry.RemoteHandle, Segments[I].Offset,
                                      Buffers[I], Segments[I].Size,
                                      Segments[I].ExpectedVersion, &LocalBytes,
                                      LocalErrno)) {
          Statuses[I] = -1;
          Errnos[I] = LocalErrno != 0 ? LocalErrno : EIO;
          continue;
        }
        if (LocalBytes > Segments[I].Size) {
          Statuses[I] = -1;
          Errnos[I] = EPROTO;
          continue;
        }
        if (LocalBytes < Segments[I].Size)
          std::memset(static_cast<char *>(Buffers[I]) + LocalBytes, 0,
                      static_cast<size_t>(Segments[I].Size - LocalBytes));
        BytesRead[I] = LocalBytes;
        Statuses[I] = 0;
      }
      return OFFLOAD_SUCCESS;
    }

    EventTy Event = createRankEvent(
        OriginEvents::ompfileDirtyOwnerPreadBatch,
        EventTypeTy::OMPFILE_DIRTY_OWNER_PREAD_BATCH, SourceRank,
        /*TargetDeviceId=*/0, Entry.RemoteHandle, Segments, SegmentCount,
        Buffers, Statuses, Errnos, BytesRead);
    if (!waitForEvent(Event, "dirty_owner_pread_batch"))
      return OFFLOAD_FAIL;
    bool AnyFailure = false;
    for (uint64_t I = 0; I < SegmentCount; ++I) {
      if (Statuses[I] != 0) {
        AnyFailure = true;
        continue;
      }
      if (BytesRead[I] > Segments[I].Size) {
        Statuses[I] = -1;
        Errnos[I] = EPROTO;
        AnyFailure = true;
        continue;
      }
      if (BytesRead[I] < Segments[I].Size)
        std::memset(static_cast<char *>(Buffers[I]) + BytesRead[I], 0,
                    static_cast<size_t>(Segments[I].Size - BytesRead[I]));
    }
    if (AnyFailure) {
      errno = EIO;
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  int mppDirtyOwnerQueryEx(int Handle, int SourceRank,
                           uint64_t ExpectedVersion, int64_t Offset,
                           uint64_t Size, int *StateOut) {
    if (!StateOut || SourceRank < 0)
      return OFFLOAD_FAIL;
    *StateOut = -1;
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (Entry.Rank != SourceRank) {
      errno = EINVAL;
      return OFFLOAD_FAIL;
    }
    if (SourceRank == EventSystem.LocalRank) {
      int LocalErrno = 0;
      int LocalState = -1;
      if (!queryDirtyOwnerStageRangeOnly(Entry.RemoteHandle, Offset, Size,
                                         ExpectedVersion, LocalState,
                                         LocalErrno)) {
        errno = LocalErrno != 0 ? LocalErrno : EIO;
        return OFFLOAD_FAIL;
      }
      *StateOut = LocalState;
      return OFFLOAD_SUCCESS;
    }

    int IoRet = -1;
    int RemoteErrno = 0;
    int State = -1;
    EventTy Event = createRankEvent(
        OriginEvents::ompfileDirtyOwnerQuery,
        EventTypeTy::OMPFILE_DIRTY_OWNER_QUERY, SourceRank,
        /*TargetDeviceId=*/0, Entry.RemoteHandle, Offset, Size,
        ExpectedVersion, &IoRet, &RemoteErrno, &State);
    if (!waitForEvent(Event, "dirty_owner_query"))
      return OFFLOAD_FAIL;
    if (IoRet != 0) {
      errno = RemoteErrno;
      return OFFLOAD_FAIL;
    }
    *StateOut = State;
    return OFFLOAD_SUCCESS;
  }

  int mppPwriteEx(int Handle, int64_t Offset, const void *Buffer, uint64_t Size,
                  uint64_t *BytesWritten) {
    if (!BytesWritten)
      return OFFLOAD_FAIL;
    *BytesWritten = 0;
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (!pwriteOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
                      BytesWritten)) {
      traceOmpFile("mppPwriteEx fail local=%d rank=%d remote=%d offset=%lld "
                   "size=%llu errno=%d\n",
                   Handle, Entry.Rank, Entry.RemoteHandle,
                   static_cast<long long>(Offset),
                   static_cast<unsigned long long>(Size), errno);
      return OFFLOAD_FAIL;
    }
    traceOmpFile("mppPwriteEx success local=%d rank=%d remote=%d offset=%lld "
                 "size=%llu bytes=%llu\n",
                 Handle, Entry.Rank, Entry.RemoteHandle,
                 static_cast<long long>(Offset),
                 static_cast<unsigned long long>(Size),
                 static_cast<unsigned long long>(*BytesWritten));
    return OFFLOAD_SUCCESS;
  }

  int mppPwrite(int Handle, int64_t Offset, const void *Buffer, uint64_t Size) {
    uint64_t BytesWritten = 0;
    if (mppPwriteEx(Handle, Offset, Buffer, Size, &BytesWritten) !=
        OFFLOAD_SUCCESS)
      return OFFLOAD_FAIL;
    if (BytesWritten != Size) {
      errno = EIO;
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  int mppStageInvalidatePathKey(uint64_t PathKey, uint64_t Generation,
                                const char *Path) {
    if (PathKey == 0 || !Path || Path[0] == '\0')
      return ENOKEY;

    const size_t PathSize = std::strlen(Path) + 1;
    if (PathSize > static_cast<size_t>(INT_MAX) ||
        PathSize > static_cast<size_t>(PATH_MAX) + 1)
      return EOVERFLOW;

    OmpFileStageInvalidateRequest Request{};
    Request.AbiVersion = OMPFILE_STAGE_INVALIDATE_ABI_VERSION;
    Request.PathSize = static_cast<uint32_t>(PathSize);
    Request.PathKey = PathKey;
    Request.Generation = Generation;

    int FirstErrno = 0;
    for (int Rank = 0; Rank < EventSystem.WorldSize - 1; ++Rank) {
      OmpFileStageInvalidateReply Reply{};
      if (!stageInvalidateOnRank(Rank, Request, Path, Reply)) {
        if (FirstErrno == 0)
          FirstErrno = errno != 0 ? errno : EIO;
        continue;
      }
      if (Reply.AbiVersion != OMPFILE_STAGE_INVALIDATE_ABI_VERSION) {
        if (FirstErrno == 0)
          FirstErrno = EPROTO;
        continue;
      }
      if (Reply.Status != 0 && FirstErrno == 0)
        FirstErrno = Reply.Errno != 0 ? Reply.Errno : EIO;
    }

    return FirstErrno;
  }

  int mppSchedRequest(const OmpFileIORequest *Request, const char *Path,
                      OmpFileIOPlan *Plan) {
    if (!Request || !Plan)
      return OFFLOAD_FAIL;
    if (Request->PathSize > 0 && !Path)
      return OFFLOAD_FAIL;
    if (!dispatchSchedRequest(*Request, Path, *Plan)) {
      DP("ompfile_mpp_sched_request failed in proxy runtime (rank=%d)\n",
         EventSystem.LocalRank);
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  int mppFreshnessQuery(const OmpFileFreshnessQueryRequest *Request,
                        OmpFileFreshnessQueryReply *Reply) {
    if (!Request || !Reply)
      return EINVAL;
    if (!freshnessQueryOnHeadnode(*Request, *Reply))
      return errno != 0 ? errno : EIO;
    if (Reply->Status != 0)
      return Reply->Errno != 0 ? Reply->Errno : EIO;
    return OFFLOAD_SUCCESS;
  }

  int mppFreshnessWriteCommit(uint64_t PathKey, int WriterRank, uint64_t TileId,
                              int WriteThroughMode,
                              uint64_t *CommittedVersion) {
    if (!CommittedVersion)
      return EINVAL;
    *CommittedVersion = 0;
    uint64_t Committed = 0;
    if (WriteThroughMode != 0 && isWritebackStageEnabled() &&
        consumeDuplicateWriteThroughFreshnessCommit(PathKey, Committed)) {
      *CommittedVersion = Committed;
      return OFFLOAD_SUCCESS;
    }
    if (!freshnessWriteCommitOnHeadnode(PathKey, WriterRank, TileId,
                                        WriteThroughMode != 0, Committed)) {
      return errno != 0 ? errno : EIO;
    }
    if (PathKey != 0 && Committed != 0 && WriterRank == EventSystem.LocalRank)
      rememberWritebackFreshnessCommit(PathKey, Committed);
    *CommittedVersion = Committed;
    return OFFLOAD_SUCCESS;
  }

  int mppProxyCopyTile(uint64_t PathKey, uint64_t TileId, int SourceRank,
                       int DestRank, uint64_t Version) {
    if (!proxyCopyTileEvent(PathKey, TileId, SourceRank, DestRank, Version))
      return errno != 0 ? errno : EIO;
    return OFFLOAD_SUCCESS;
  }

  int mppFreshnessMarkFresh(uint64_t PathKey, int Rank, uint64_t Version) {
    if (!freshnessMarkFreshOnHeadnode(PathKey, Rank, Version))
      return errno != 0 ? errno : EIO;
    if (PathKey != 0 && Version != 0 && Rank == EventSystem.LocalRank) {
      std::lock_guard<std::mutex> Lock(OmpFileWritebackCommitMutex);
      auto It = OmpFileLocalFreshnessVersionByPath.find(PathKey);
      if (It == OmpFileLocalFreshnessVersionByPath.end() ||
          Version > It->second) {
        OmpFileLocalFreshnessVersionByPath[PathKey] = Version;
      }
    }
    return OFFLOAD_SUCCESS;
  }

  int mppFlushDirtyTile(uint64_t PathKey, int *SourceRank,
                        uint64_t *FlushedVersion) {
    if (!SourceRank || !FlushedVersion)
      return EINVAL;
    *SourceRank = -1;
    *FlushedVersion = 0;
    if (PathKey == 0)
      return EINVAL;

    int Source = -1;
    uint64_t Version = 0;
    if (!flushDirtyTileOnHeadnode(PathKey, Source, Version))
      return errno != 0 ? errno : EIO;

    if (Source >= 0) {
      if (Source == EventSystem.LocalRank) {
        if (!flushLocalDirtyTileToSource(PathKey, Version))
          return errno != 0 ? errno : EIO;
      } else {
        OmpFileFlushDirtyTileRequest WritebackReq{};
        WritebackReq.Action = 2;
        WritebackReq.PathKey = PathKey;
        WritebackReq.SourceRank = Source;
        WritebackReq.Version = Version;
        WritebackReq.Success = 1;
        OmpFileFlushDirtyTileReply WritebackReply{};
        EventTy WritebackEvent = createRankEvent(
            OriginEvents::ompfileFlushDirtyTile,
            EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE, Source,
            /*TargetDeviceId=*/0, &WritebackReq, &WritebackReply);
        if (!waitForEvent(WritebackEvent, "flush_dirty_tile_writeback"))
          return errno != 0 ? errno : EIO;
        if (WritebackReply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION)
          return EPROTO;
        if (WritebackReply.Status != 0)
          return WritebackReply.Errno != 0 ? WritebackReply.Errno : EIO;
      }
      if (!completeDirtyFlushOnHeadnode(PathKey, Source, Version,
                                        /*Success=*/true))
        return errno != 0 ? errno : EIO;
    }

    *SourceRank = Source;
    *FlushedVersion = Version;
    return OFFLOAD_SUCCESS;
  }

  int mppSchedBatchRequest(const OmpFileIOBatchRequest *Request,
                           const void *RequestPayload,
                           uint64_t RequestPayloadBytes,
                           OmpFileIOBatchPlan *Plan, void *PlanPayload,
                           uint64_t PlanPayloadCapBytes,
                           uint64_t *PlanPayloadOutBytes) {
    if (!Request || !Plan || !PlanPayloadOutBytes)
      return OFFLOAD_FAIL;
    if (RequestPayloadBytes > 0 && !RequestPayload)
      return OFFLOAD_FAIL;
    if (PlanPayloadCapBytes > 0 && !PlanPayload)
      return OFFLOAD_FAIL;

    if (!dispatchSchedBatchRequest(*Request, RequestPayload,
                                   RequestPayloadBytes, *Plan, PlanPayload,
                                   PlanPayloadCapBytes, *PlanPayloadOutBytes)) {
      DP("ompfile_mpp_sched_request_batch failed in proxy runtime (rank=%d)\n",
         EventSystem.LocalRank);
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int mppFinalize() {
    size_t HandleCount = 0;
    {
      const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
      HandleCount = OmpFileHandles.size();
    }
    traceOmpFile("mppFinalize enter handle_count=%zu\n", HandleCount);
    reportOmpFileStats("mpp-finalize-before-drain");
    drainOmpFileOpenCache();
    drainOmpFileStageCache();
    {
      const std::lock_guard<std::mutex> Lock(MppEventMutex);
      MppEvents.clear();
      CompletedMppTokens.clear();
    }
    {
      const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
      OmpFileHandles.clear();
    }
    reportOmpFileStats("mpp-finalize-after-drain");
    traceOmpFile("mppFinalize exit\n");
    return OFFLOAD_SUCCESS;
  }

  EventTy ompfileSchedRequest(MPIRequestManagerTy RequestManager) {
    OmpFileIORequest Request{};
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::string Path;
    if (Request.PathSize > 0) {
      Path.resize(Request.PathSize, '\0');
      RequestManager.receive(Path.data(), Request.PathSize, MPI_CHAR);
      if (auto Error = co_await RequestManager; Error)
        co_return Error;
    }

    OmpFileIOPlan Plan =
        buildSchedulePlan(Request, Request.PathSize > 0 ? Path.c_str() : nullptr);

    RequestManager.send(&Plan, sizeof(Plan), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileSchedBatchRequest(MPIRequestManagerTy RequestManager) {
    OmpFileIOBatchRequest Request{};
    uint64_t RequestPayloadBytes = 0;
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);
    RequestManager.receive(&RequestPayloadBytes, 1, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::vector<uint8_t> RequestPayload;
    if (RequestPayloadBytes > 0) {
      RequestPayload.resize(static_cast<size_t>(RequestPayloadBytes));
      RequestManager.receiveInBatchs(RequestPayload.data(), RequestPayloadBytes);
      if (auto Error = co_await RequestManager; Error)
        co_return Error;
    }

    OmpFileIOBatchPlan Plan{};
    std::vector<uint8_t> PlanPayload;
    buildScheduleBatchPlan(Request,
                           RequestPayload.empty() ? nullptr
                                                  : RequestPayload.data(),
                           RequestPayloadBytes, Plan, PlanPayload);

    uint64_t PlanPayloadBytes = PlanPayload.size();
    RequestManager.send(&Plan, sizeof(Plan), MPI_BYTE);
    RequestManager.send(&PlanPayloadBytes, 1, MPI_UINT64_T);
    if (PlanPayloadBytes > 0)
      RequestManager.sendInBatchs(PlanPayload.data(), PlanPayloadBytes);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileSchedPlan(MPIRequestManagerTy RequestManager) {
    OmpFileIOPlan Plan{};
    RequestManager.receive(&Plan, sizeof(Plan), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    OmpFileIOCompletion Completion{};
    Completion.RequestId = Plan.RequestId;
    Completion.Status = 0;
    Completion.Errno = 0;
    Completion.Bytes = 0;

    OmpFileHeadnodeManager::instance().completeRequest(Plan);

    RequestManager.send(&Completion, sizeof(Completion), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy initAsyncInfo(MPIRequestManagerTy RequestManager) {
    __tgt_async_info *TgtAsyncInfoPtr = nullptr;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->init_async_info(DeviceId,
                                                     &TgtAsyncInfoPtr);

    RequestManager.send(&TgtAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy initDeviceInfo(MPIRequestManagerTy RequestManager) {
    __tgt_device_info DeviceInfo;
    const char *ErrStr = nullptr;

    RequestManager.receive(&DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->init_device_info(DeviceId, &DeviceInfo,
                                                      &ErrStr);

    RequestManager.send(&DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy queryAsync(MPIRequestManagerTy RequestManager) {
    void *HstAsyncInfoPtr = nullptr;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr)->AsyncInfoPtr.get();

    PluginManager.Plugins[PluginId]->query_async(DeviceId, TgtAsyncInfo);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy printDeviceInfo(MPIRequestManagerTy RequestManager) {
    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->print_device_info(DeviceId);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataLock(MPIRequestManagerTy RequestManager) {
    void *Ptr = nullptr;
    int64_t Size = 0;
    void *LockedPtr = nullptr;

    RequestManager.receive(&Ptr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_lock(DeviceId, Ptr, Size, &LockedPtr);

    RequestManager.send(&LockedPtr, sizeof(void *), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataUnlock(MPIRequestManagerTy RequestManager) {
    void *Ptr = nullptr;
    RequestManager.receive(&Ptr, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_unlock(DeviceId, Ptr);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataNotifyMapped(MPIRequestManagerTy RequestManager) {
    void *HstPtr = nullptr;
    int64_t Size = 0;
    RequestManager.receive(&HstPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_notify_mapped(DeviceId, HstPtr, Size);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataNotifyUnmapped(MPIRequestManagerTy RequestManager) {
    void *HstPtr = nullptr;
    RequestManager.receive(&HstPtr, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_notify_unmapped(DeviceId, HstPtr);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy exit(MPIRequestManagerTy RequestManager,
               std::atomic<EventSystemStateTy> &EventSystemState) {
    EventSystemStateTy OldState =
        EventSystemState.exchange(EventSystemStateTy::EXITED);
    assert(OldState != EventSystemStateTy::EXITED &&
           "Exit event received multiple times");

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  /// Function executed by the event handler threads.
  void runEventHandler(std::stop_token Stop, EventQueue &Queue) {
    while (EventSystem.EventSystemState == EventSystemStateTy::RUNNING ||
           Queue.size() > 0) {
      EventTy Event = Queue.pop(Stop);

      // Re-checks the stop condition when no event was found.
      if (Event.empty()) {
        continue;
      }

      Event.resume();

      if (!Event.done()) {
        Queue.push(std::move(Event));
        continue;
      }

      auto Error = Event.getError();
      if (Error)
        REPORT("Internal event failed with msg: %s\n",
               toString(std::move(Error)).data());
    }
  }

  /// Gate thread procedure.
  ///
  /// Caller thread will spawn the event handlers, execute the gate logic and
  /// wait until the event system receive an Exit event.
  void runGateThread() {
    // Device image to be used by this gate thread.
    // DeviceImage Image;

    // Updates the event state and
    EventSystem.EventSystemState = EventSystemStateTy::RUNNING;

    // Spawns the event handlers.
    llvm::SmallVector<std::jthread> EventHandlers;
    EventHandlers.resize(NumExecEventHandlers.get() +
                         NumDataEventHandlers.get());
    int EventHandlersSize = EventHandlers.size();
    auto HandlerFunction = std::bind_front(&ProxyDevice::runEventHandler, this);
    for (int Idx = 0; Idx < EventHandlersSize; Idx++) {
      EventHandlers[Idx] = std::jthread(
          HandlerFunction, std::ref(Idx < NumExecEventHandlers.get()
                                        ? EventSystem.ExecEventQueue
                                        : EventSystem.DataEventQueue));
    }

    // Executes the gate thread logic
    while (EventSystem.EventSystemState == EventSystemStateTy::RUNNING) {
      // Checks for new incoming event requests.
      MPI_Message EventReqMsg;
      MPI_Status EventStatus;
      int HasReceived = false;
      ompfileWithMPICallLock([&]() {
        return MPI_Improbe(MPI_ANY_SOURCE,
                           static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                           EventSystem.GateThreadComm, &HasReceived,
                           &EventReqMsg, MPI_STATUS_IGNORE);
      });

      // If none was received, wait for `EVENT_POLLING_RATE`us for the next
      // check.
      if (!HasReceived) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(EventPollingRate.get()));
        continue;
      }

      // Acquires the event information from the received request, which are:
      // - Event type
      // - Event tag
      // - Target comm
      // - Event source rank
      int EventInfo[3];
      ompfileWithMPICallLock([&]() {
        return MPI_Mrecv(EventInfo, 3, MPI_INT, &EventReqMsg, &EventStatus);
      });
      const auto NewEventType = static_cast<EventTypeTy>(EventInfo[0]);
      MPIRequestManagerTy RequestManager(
          EventSystem.getNewEventComm(EventInfo[1]),
          EventSystem.getEventMPITag(EventInfo[1]), EventStatus.MPI_SOURCE,
          EventInfo[2]);

      // Creates a new receive event of 'event_type' type.
      using enum EventTypeTy;
      EventTy NewEvent;
      switch (NewEventType) {
      case RETRIEVE_NUM_DEVICES:
        NewEvent = retrieveNumDevices(std::move(RequestManager));
        break;
      case IS_PLUGIN_COMPATIBLE:
        NewEvent = isPluginCompatible(std::move(RequestManager));
        break;
      case IS_DEVICE_COMPATIBLE:
        NewEvent = isDeviceCompatible(std::move(RequestManager));
        break;
      case INIT_DEVICE:
        NewEvent = initDevice(std::move(RequestManager));
        break;
      case INIT_RECORD_REPLAY:
        NewEvent = initRecordReplay(std::move(RequestManager));
        break;
      case IS_DATA_EXCHANGABLE:
        NewEvent = isDataExchangable(std::move(RequestManager));
        break;
      case ALLOC:
        NewEvent = allocateBuffer(std::move(RequestManager));
        break;
      case DELETE:
        NewEvent = deleteBuffer(std::move(RequestManager));
        break;
      case SUBMIT:
        NewEvent = submit(std::move(RequestManager));
        break;
      case RETRIEVE:
        NewEvent = retrieve(std::move(RequestManager));
        break;
      case LOCAL_EXCHANGE:
        NewEvent = exchange(std::move(RequestManager));
        break;
      case EXCHANGE_SRC:
        NewEvent = exchangeSrc(std::move(RequestManager));
        break;
      case EXCHANGE_DST:
        NewEvent = exchangeDst(std::move(RequestManager));
        break;
      case EXIT:
        NewEvent =
            exit(std::move(RequestManager), EventSystem.EventSystemState);
        break;
      case LOAD_BINARY:
        NewEvent = loadBinary(std::move(RequestManager));
        break;
      case GET_GLOBAL:
        NewEvent = getGlobal(std::move(RequestManager));
        break;
      case GET_FUNCTION:
        NewEvent = getFunction(std::move(RequestManager));
        break;
      case LAUNCH_KERNEL:
        NewEvent = launchKernel(std::move(RequestManager));
        break;
      case SYNCHRONIZE:
        NewEvent = synchronize(std::move(RequestManager));
        break;
      case OMPFILE_PING:
        NewEvent = ompfilePing(std::move(RequestManager));
        break;
      case OMPFILE_OPEN:
        NewEvent = ompfileOpen(std::move(RequestManager));
        break;
      case OMPFILE_CLOSE:
        NewEvent = ompfileClose(std::move(RequestManager));
        break;
      case OMPFILE_PREAD:
        NewEvent = ompfilePread(std::move(RequestManager));
        break;
      case OMPFILE_PREAD_NO_STAGE:
        NewEvent = ompfilePread(std::move(RequestManager),
                                /*AllowStage=*/false);
        break;
      case OMPFILE_DIRTY_OWNER_PREAD:
        NewEvent = ompfileDirtyOwnerPread(std::move(RequestManager));
        break;
      case OMPFILE_DIRTY_OWNER_PREAD_BATCH:
        NewEvent = ompfileDirtyOwnerPreadBatch(std::move(RequestManager));
        break;
      case OMPFILE_DIRTY_OWNER_QUERY:
        NewEvent = ompfileDirtyOwnerQuery(std::move(RequestManager));
        break;
      case OMPFILE_PWRITE:
        NewEvent = ompfilePwrite(std::move(RequestManager));
        break;
      case OMPFILE_STAGE_INVALIDATE:
        NewEvent = ompfileStageInvalidate(std::move(RequestManager));
        break;
      case OMPFILE_FRESHNESS_QUERY:
        NewEvent = ompfileFreshnessQuery(std::move(RequestManager));
        break;
      case OMPFILE_PROXY_COPY_TILE:
        NewEvent = ompfileProxyCopyTile(std::move(RequestManager));
        break;
      case OMPFILE_FRESHNESS_MARK_FRESH:
        NewEvent = ompfileFreshnessMarkFresh(std::move(RequestManager));
        break;
      case OMPFILE_FRESHNESS_WRITE_COMMIT:
        NewEvent = ompfileFreshnessWriteCommit(std::move(RequestManager));
        break;
      case OMPFILE_FLUSH_DIRTY_TILE:
        NewEvent = ompfileFlushDirtyTile(std::move(RequestManager));
        break;
      case OMPFILE_SCHED_REQUEST:
        NewEvent = ompfileSchedRequest(std::move(RequestManager));
        break;
      case OMPFILE_SCHED_REQUEST_BATCH:
        NewEvent = ompfileSchedBatchRequest(std::move(RequestManager));
        break;
      case OMPFILE_SCHED_PLAN:
        NewEvent = ompfileSchedPlan(std::move(RequestManager));
        break;
      case INIT_ASYNC_INFO:
        NewEvent = initAsyncInfo(std::move(RequestManager));
        break;
      case INIT_DEVICE_INFO:
        NewEvent = initDeviceInfo(std::move(RequestManager));
        break;
      case QUERY_ASYNC:
        NewEvent = queryAsync(std::move(RequestManager));
        break;
      case PRINT_DEVICE_INFO:
        NewEvent = printDeviceInfo(std::move(RequestManager));
        break;
      case DATA_LOCK:
        NewEvent = dataLock(std::move(RequestManager));
        break;
      case DATA_UNLOCK:
        NewEvent = dataUnlock(std::move(RequestManager));
        break;
      case DATA_NOTIFY_MAPPED:
        NewEvent = dataNotifyMapped(std::move(RequestManager));
        break;
      case DATA_NOTIFY_UNMAPPED:
        NewEvent = dataNotifyUnmapped(std::move(RequestManager));
        break;
      case SYNC:
        assert(false && "Trying to create a local event on a remote node");
      }

      if (NewEventType == LAUNCH_KERNEL) {
        EventSystem.ExecEventQueue.push(std::move(NewEvent));
      } else {
        EventSystem.DataEventQueue.push(std::move(NewEvent));
      }
    }

    assert(EventSystem.EventSystemState == EventSystemStateTy::EXITED &&
           "Event State should be EXITED after receiving an Exit event");
  }

private:
  llvm::SmallVector<std::unique_ptr<AsyncInfoHandle>> AsyncInfoList;
  llvm::SmallVector<std::unique_ptr<DeviceImage>> RemoteImages;
  llvm::DenseMap<void *, AsyncInfoHandle *> AsyncInfoTable;
  RemotePluginManager PluginManager;
  EventSystemTy EventSystem;
  /// Number of execute event handlers to spawn.
  IntEnvar NumExecEventHandlers;
  /// Number of data event handlers to spawn.
  IntEnvar NumDataEventHandlers;
  /// Polling rate period (us) used by event handlers.
  IntEnvar EventPollingRate;
  IntEnvar OmpFileOpenEioRetries;
  IntEnvar OmpFileOpenEioBackoffUs;
  BoolEnvar OmpFileOpenCacheEnable;
  BoolEnvar OmpFileOpenCacheKeepOpen;
  BoolEnvar OmpFileOptStats;
  BoolEnvar OmpFileForceBlockingPwrite;
  Int64Envar OmpFileMPIFragmentSize;
  bool OmpFileHeadnodeScheduler = false;
  std::string OmpFileStageMode;
  std::string OmpFileStageWriteMode;
  std::string OmpFileStageSyncPolicy;
  // Write-through durability policy: "each" (default) fdatasyncs after every
  // authoritative source pwrite (crash-safe per write, ~40 ms/op on NFS —
  // sorgan 3704); "close" defers durability to the logical close, matching
  // native POSIX/MPI-IO semantics for fair overhead comparisons. Explicitly
  // opt-in: any value other than "close" behaves as "each".
  std::string OmpFileWritethroughFsyncPolicy;
  std::mutex OmpFileUnsyncedWriteFdMutex;
  std::unordered_set<int> OmpFileUnsyncedWriteFds;
  std::string OmpFileStagePopulateMode;
  std::string OmpFileStageRootPolicy;
  std::string OmpFileTopologyFile;
  std::string OmpFileStageDirName;
  std::string OmpFileStageRunStem;
  std::string OmpFileStorageEnvironment;
  uint64_t OmpFileStageMinFreeBytes = 0;
  uint64_t OmpFileStageWindowBytes = 0;
  double OmpFileStageWindowScale = 1.0;
  uint64_t OmpFileStageEffectiveWindowBytes = 0;
  uint64_t OmpFileStageDirtyWatermarkBytes = 0;
  bool OmpFileStageFreshnessGuard = true;
  bool OmpFileTopologyLoaded = false;
  uint64_t OmpFileTopologyEntries = 0;
  std::string OmpFileStageRoot;
  std::string OmpFileSelectedStageClass;
  std::string OmpFileSharedStoragePath;
  std::string OmpFileSharedStorageClass;
  std::string OmpFileStageLocalHost;
  std::string OmpFileStageDecisionReason;
  std::string OmpFileTopologyLoadError;
  // Mutex for AsyncInfoTable
  std::mutex TableMutex;
  std::atomic<uint64_t> NextSchedRequestId{1};
  std::mutex OmpFileHandleMutex;
  std::unordered_map<int, OmpFileHandleEntry> OmpFileHandles;
  std::atomic<int> NextOmpFileHandle{1};
  mutable std::mutex OmpFileTrackedFdMutex;
  std::unordered_map<int, OmpFileTrackedFdEntry> OmpFileTrackedFds;
  std::unordered_map<std::string, uint64_t> OmpFileTrackedFdPathRefCounts;
  std::mutex OmpFileOpenCacheMutex;
  std::unordered_map<std::string, OmpFileOpenCacheEntry> OmpFileOpenCacheByKey;
  std::unordered_map<int, std::string> OmpFileOpenCacheFdToKey;
  std::mutex OmpFileStageMutex;
  std::unordered_map<std::string, std::shared_ptr<OmpFileStageEntry>>
      OmpFileStageEntries;
  std::mutex OmpFileWritebackCommitMutex;
  std::unordered_map<uint64_t, OmpFilePendingWritebackCommit>
      OmpFilePendingWritebackCommits;
  std::unordered_map<uint64_t, uint64_t> OmpFileLocalFreshnessVersionByPath;
  std::atomic<uint64_t> OmpFileStatsOpenRequests{0};
  std::atomic<uint64_t> OmpFileStatsOpenSyscalls{0};
  std::atomic<uint64_t> OmpFileStatsOpenCacheHits{0};
  std::atomic<uint64_t> OmpFileStatsOpenEioRetries{0};
  std::atomic<uint64_t> OmpFileStatsOpenEioFailures{0};
  std::atomic<uint64_t> OmpFileStatsCloseRequests{0};
  std::atomic<uint64_t> OmpFileStatsCloseSyscalls{0};
  std::atomic<uint64_t> OmpFileStatsCloseDeferred{0};
  std::atomic<uint64_t> OmpFileStatsPwriteAsyncEvents{0};
  std::atomic<uint64_t> OmpFileStatsPwriteAsyncFragments{0};
  std::atomic<uint64_t> OmpFileStatsPwriteBlockingFallbacks{0};
  std::atomic<uint64_t> OmpFileStatsPwritePayloadBytes{0};
  std::atomic<uint64_t> OmpFileStatsPwriteFailures{0};
  std::atomic<uint64_t> OmpFileStatsStagedReadHits{0};
  std::atomic<uint64_t> OmpFileStatsStagedReadBytes{0};
  std::atomic<uint64_t> OmpFileStatsStageBypassReads{0};
  std::atomic<uint64_t> OmpFileStatsStageBypassBytes{0};
  std::atomic<uint64_t> OmpFileStatsStagedWriteUpdates{0};
  std::atomic<uint64_t> OmpFileStatsStagedWriteBytes{0};
  std::atomic<uint64_t> OmpFileStatsStageLookupHits{0};
  std::atomic<uint64_t> OmpFileStatsStageLookupMisses{0};
  std::atomic<uint64_t> OmpFileStatsStagePopulateCount{0};
  std::atomic<uint64_t> OmpFileStatsStagePopulateFailures{0};
  std::atomic<uint64_t> OmpFileStatsStagePopulateBytes{0};
  std::atomic<uint64_t> OmpFileStatsStagePopulateUs{0};
  std::atomic<uint64_t> OmpFileStatsStageCopyUs{0};
  std::atomic<uint64_t> OmpFileStatsStageFsyncUs{0};
  std::atomic<uint64_t> OmpFileStatsStageReopenUs{0};
  std::atomic<uint64_t> OmpFileStatsStageLockWaitUs{0};
  std::atomic<uint64_t> OmpFileStatsStageLockHoldUs{0};
  std::atomic<uint64_t> OmpFileStatsStageGlobalInvalidationRequests{0};
  std::atomic<uint64_t> OmpFileStatsStageGlobalInvalidationCompletions{0};
  std::atomic<uint64_t> OmpFileStatsStageGlobalInvalidationFailures{0};
  std::atomic<uint64_t> OmpFileStatsSourcePreadBytes{0};
  std::atomic<uint64_t> OmpFileStatsSourcePreadUs{0};
  std::atomic<uint64_t> OmpFileStatsStagedPreadUs{0};
  std::atomic<uint64_t> OmpFileStatsStagingInvalidations{0};
  std::atomic<uint64_t> OmpFileStatsStagingRangeInvalidations{0};
  std::atomic<uint64_t> OmpFileStatsStagingFullInvalidations{0};
  std::atomic<uint64_t> OmpFileStatsStagingInvalidatedBytes{0};
  std::atomic<uint64_t> OmpFileStatsStagingWriteBypassCount{0};
  std::atomic<uint64_t> OmpFileStatsStageWriteFailures{0};
  std::atomic<uint64_t> OmpFileStatsStageWriteUs{0};
  // Write-back staging counters. DirtyBytes is a gauge snapshot at stats
  // print time; the rest are monotonic. Flush counters track write-back
  // flush-to-source events (Task 4 wires the real flush).
  std::atomic<uint64_t> OmpFileStatsStageWritebackCaptures{0};
  std::atomic<uint64_t> OmpFileStatsStageWritebackCaptureBytes{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyBytes{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyFlushes{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyFlushBytes{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyFlushFailures{0};
  std::atomic<uint64_t> OmpFileStatsStagingEvictions{0};
  std::atomic<uint64_t> OmpFileStatsStageFreshnessGuardBypasses{0};
  std::atomic<uint64_t> OmpFileStatsDirtyOwnerForwardReads{0};
  std::atomic<uint64_t> OmpFileStatsDirtyOwnerForwardBytes{0};
  std::atomic<uint64_t> OmpFileStatsDirtyOwnerForwardFailures{0};
  std::atomic<uint64_t> OmpFileStatsCoherentReadRefreshes{0};
  std::atomic<uint64_t> OmpFileStatsCoherentReadRefreshFailures{0};
  std::atomic<uint64_t> OmpFileStatsCoherentReadRefreshUs{0};
  std::mutex MppEventMutex;
  std::unordered_map<uint64_t, EventTy> MppEvents;
  std::unordered_set<uint64_t> CompletedMppTokens;
};

static ProxyDevice *getActiveProxyDevice() {
  const std::lock_guard<std::mutex> Lock(ActiveProxyDeviceMutex);
  return ActiveProxyDevice;
}

extern "C" {
int ompfile_mpp_init() {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppInit();
}

int ompfile_mpp_submit(uint64_t Token) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppSubmit(Token);
}

int ompfile_mpp_open(const char *Path, int Flags, int Mode, int *Handle) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppOpen(Path, Flags, Mode, Handle);
}

int ompfile_mpp_open_on_rank(const char *Path, int Flags, int Mode, int Rank,
                             int *Handle) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppOpenOnRank(Path, Flags, Mode, Rank, Handle);
}

int ompfile_mpp_sched_request(const OmpFileIORequest *Request, const char *Path,
                              OmpFileIOPlan *Plan) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppSchedRequest(Request, Path, Plan);
}

int ompfile_mpp_sched_request_batch(
    const OmpFileIOBatchRequest *Request, const void *RequestPayload,
    uint64_t RequestPayloadBytes, OmpFileIOBatchPlan *Plan, void *PlanPayload,
    uint64_t PlanPayloadCapBytes, uint64_t *PlanPayloadOutBytes) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppSchedBatchRequest(Request, RequestPayload, RequestPayloadBytes,
                                  Plan, PlanPayload, PlanPayloadCapBytes,
                                  PlanPayloadOutBytes);
}

int ompfile_mpp_close(int Handle) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppClose(Handle);
}

int ompfile_mpp_handle_owner_rank(int Handle, int *RankOut) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppHandleOwnerRank(Handle, RankOut);
}

int ompfile_mpp_pread(int Handle, int64_t Offset, void *Buffer, uint64_t Size) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPread(Handle, Offset, Buffer, Size);
}

int ompfile_mpp_pread_ex(int Handle, int64_t Offset, void *Buffer,
                          uint64_t Size, uint64_t *BytesRead) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPreadEx(Handle, Offset, Buffer, Size, BytesRead);
}

int ompfile_mpp_pread_no_stage_ex(int Handle, int64_t Offset, void *Buffer,
                                  uint64_t Size, uint64_t *BytesRead) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPreadNoStageEx(Handle, Offset, Buffer, Size, BytesRead);
}

int ompfile_mpp_pwrite(int Handle, int64_t Offset, const void *Buffer,
                       uint64_t Size) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPwrite(Handle, Offset, Buffer, Size);
}


int ompfile_mpp_dirty_owner_pread_ex(int Handle, int SourceRank,
                                     uint64_t ExpectedVersion, int64_t Offset,
                                     void *Buffer, uint64_t Size,
                                     uint64_t *BytesRead) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppDirtyOwnerPreadEx(Handle, SourceRank, ExpectedVersion, Offset,
                                  Buffer, Size, BytesRead);
}

int ompfile_mpp_dirty_owner_pread_batch_ex(
    int Handle, int SourceRank,
    const OmpFileDirtyOwnerPreadBatchSegment *Segments, uint64_t SegmentCount,
    void *const *Buffers, uint64_t *BytesRead, int *Statuses, int *Errnos) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppDirtyOwnerPreadBatchEx(Handle, SourceRank, Segments,
                                       SegmentCount, Buffers, BytesRead,
                                       Statuses, Errnos);
}

int ompfile_mpp_dirty_owner_query_ex(int Handle, int SourceRank,
                                     uint64_t ExpectedVersion, int64_t Offset,
                                     uint64_t Size, int *StateOut) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppDirtyOwnerQueryEx(Handle, SourceRank, ExpectedVersion, Offset,
                                  Size, StateOut);
}

int ompfile_mpp_pwrite_ex(int Handle, int64_t Offset, const void *Buffer,
                          uint64_t Size, uint64_t *BytesWritten) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPwriteEx(Handle, Offset, Buffer, Size, BytesWritten);
}

int ompfile_mpp_stage_invalidate_path_key(uint64_t PathKey,
                                          uint64_t Generation,
                                          const char *Path) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppStageInvalidatePathKey(PathKey, Generation, Path);
}

int ompfile_mpp_freshness_query(const OmpFileFreshnessQueryRequest *Request,
                                OmpFileFreshnessQueryReply *Reply) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppFreshnessQuery(Request, Reply);
}

int ompfile_mpp_freshness_write_commit(uint64_t PathKey, int WriterRank,
                                       uint64_t TileId, int WriteThroughMode,
                                       uint64_t *CommittedVersion) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppFreshnessWriteCommit(PathKey, WriterRank, TileId,
                                     WriteThroughMode, CommittedVersion);
}

int ompfile_mpp_proxy_copy_tile(uint64_t PathKey, uint64_t TileId,
                                int SourceRank, int DestRank,
                                uint64_t Version) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppProxyCopyTile(PathKey, TileId, SourceRank, DestRank, Version);
}

int ompfile_mpp_freshness_mark_fresh(uint64_t PathKey, int Rank,
                                     uint64_t Version) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppFreshnessMarkFresh(PathKey, Rank, Version);
}

int ompfile_mpp_flush_dirty_tile(uint64_t PathKey, int *SourceRank,
                                 uint64_t *FlushedVersion) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppFlushDirtyTile(PathKey, SourceRank, FlushedVersion);
}

int ompfile_mpp_poll(uint64_t Token, int *Done) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPoll(Token, Done);
}

int ompfile_mpp_finalize() {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppFinalize();
}
} // extern "C"

#ifndef OMPFILE_PROXYDEVICE_NO_MAIN
int main(int argc, char **argv) {
  ProxyDevice PD;
  PD.runGateThread();
  return 0;
}
#endif
