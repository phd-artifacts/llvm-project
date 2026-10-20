//===-- ProxyOmpFile.h - Proxy-side libompfile server -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The libompfile half of the MPI proxy (llvm-offload-mpi-proxy-device): the
// OMPFILE_* event handlers, the proxy open cache and tracked fds, the
// per-proxy stage, the owner forwarding to peer proxies and the headnode
// coherence round trips. ProxyDevice owns one instance (ProxyDevice::OmpFile)
// and routes every isFileIoEvent() type to createFileIoEvent().
//
// Private to the proxy tool: only llvm-offload-mpi-proxy-device TUs include
// it. Its inline members (createRankEvent) report with that target's
// DEBUG_PREFIX, so a TU of another target must not include it.
//
// The member definitions live in ProxyOmpFileConfig.cpp (config, topology,
// init/teardown, trace), ProxyOmpFileStage.cpp (staging, local coherence,
// source I/O), ProxyOmpFileServer.cpp (event handlers, open cache, tracked
// fds, headnode plans), ProxyOmpFileStats.cpp (stats lines),
// ProxyOmpFileForward.cpp (owner forwarding, headnode round trips, the mpp*
// bridges, owner-bypass I/O) and ProxyOmpFileBridge.cpp (the extern C
// ompfile_mpp_* exports and the active-instance registry).
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_MPI_PROXY_OMPFILE_H
#define OFFLOAD_PLUGINS_NEXTGEN_MPI_PROXY_OMPFILE_H

// First: EventSystem.h defines MPICH_SKIP_MPICXX before it includes mpi.h.
#include "EventSystemOmpFile.h"

#include "OmpFileHeadnodeManager.h"
#include "Shared/APITypes.h"
#include "mpi.h"
#include "ompfile_config.h"
#include "ompfile_env.h"
#include "ompfile_extents.h"
#include "ompfile_keys.h"
#include "omptarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <fnmatch.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

struct ProxyOmpFile;

// The instance the extern C ompfile_mpp_* exports forward to
// (ProxyOmpFileBridge.cpp). initialize() publishes it and deinitialize()
// withdraws it; clear only drops the given instance.
ProxyOmpFile *getActiveProxyOmpFile();
void setActiveProxyOmpFile(ProxyOmpFile *Instance);
void clearActiveProxyOmpFile(ProxyOmpFile *Instance);

struct ProxyOmpFile {
  struct OmpFileHandleEntry {
    int Rank = -1;
    int RemoteHandle = -1;
    // Source path + open flags, retained so the issuing proxy can take the
    // local disjoint-write bypass (LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE): write
    // known-disjoint ranges straight to the shared PFS from this node instead
    // of forwarding every byte to the file's single aggregator/owner rank.
    std::string Path;
    int Flags = 0;
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

  // The extent algebra lives in ompfile_extents.h (unit-tested); the
  // members below forward to it so the call sites read as before.
  using OmpFileStageExtent = ompfile::extents::Extent;

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
    // One watermark flush per entry at a time (LIBOMPFILE_STAGE_FLUSH_COALESCE).
    bool FlushInProgress = false;
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

  // One coalescing window: a write-combine buffer or a readahead cache for a
  // single path (declared here, before the helpers that take it by
  // reference; the backing maps live with the other OmpFileLocal* members).
  struct LocalCoalesceBuf {
    std::unique_ptr<char[]> Data;
    int64_t Base = -1; // file offset of Data[0]; -1 = empty/invalid
    uint64_t Len = 0;
    uint64_t Cap = 0;
  };

  // Construction reads the proxy knobs (the Envar members and
  // OmpFileConfig) in the old ProxyDevice member order. initialize() is
  // the old constructor tail and runs after EventSystem.initialize() and
  // PluginManager.init(); deinitialize() is the old destructor head and
  // runs before EventSystem.deinitialize().
  explicit ProxyOmpFile(EventSystemTy &ES);
  void initialize();
  void deinitialize();

  // The handler for one isFileIoEvent() type, as the gate thread used to
  // create it; an empty EventTy for any other type.
  EventTy createFileIoEvent(EventTypeTy Type,
                            MPIRequestManagerTy &&RequestManager);

  static std::string envStringOrDefault(const char *Name,
                                        const char *DefaultValue) {
    return ompfile::env::stringOr(Name, DefaultValue);
  }

  static uint64_t elapsedMicros(std::chrono::steady_clock::time_point Start,
                                std::chrono::steady_clock::time_point End) {
    if (End <= Start)
      return 0;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(End - Start)
            .count());
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

  bool shouldSyncStagePopulate() const {
    return OmpFileStageSyncPolicy == "always";
  }

  bool useWindowedStagePopulate() const {
    return OmpFileStagePopulateMode != "full";
  }

  uint64_t saturatingAdd(uint64_t Base, uint64_t Delta) const {
    return ompfile::extents::saturatingAdd(Base, Delta);
  }

  uint64_t alignDown(uint64_t Value, uint64_t Alignment) const {
    return ompfile::extents::alignDown(Value, Alignment);
  }

  uint64_t alignUp(uint64_t Value, uint64_t Alignment) const {
    return ompfile::extents::alignUp(Value, Alignment);
  }

  bool isCoveredByExtents(const std::vector<OmpFileStageExtent> &Extents,
                          uint64_t Begin, uint64_t End) const {
    return ompfile::extents::isCoveredByExtents(Extents, Begin, End);
  }

  bool overlapsAnyExtent(const std::vector<OmpFileStageExtent> &Extents,
                         uint64_t Begin, uint64_t End) const {
    return ompfile::extents::overlapsAnyExtent(Extents, Begin, End);
  }

  void addCoveredExtent(std::vector<OmpFileStageExtent> &Extents,
                        uint64_t Begin, uint64_t End) {
    ompfile::extents::addCoveredExtent(Extents, Begin, End);
  }

  uint64_t removeCoveredRange(std::vector<OmpFileStageExtent> &Extents,
                              uint64_t Begin, uint64_t End) {
    return ompfile::extents::removeCoveredRange(Extents, Begin, End);
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

  bool isWorkerRank(int Rank) const {
    return Rank >= 0 && Rank < EventSystem.numWorkerRanks();
  }

  int getHeadnodeRank() const { return 0; }

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
    auto &EventComm = EventSystem.getNewEventComm(EventTag, EventSystem.LocalRank);
    const int MPITag = EventSystem.getEventMPITag(EventTag, EventSystem.LocalRank);
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

  // ProxyOmpFileConfig.cpp: config, topology and trace.
  [[noreturn]] void fatalStageConfig(const char *Reason);
  bool validateStageConfigFast();
  void traceOmpFile(const char *Fmt, ...) const;

  // ProxyOmpFileStage.cpp: staging, local coherence and source I/O.
  void rememberWritebackFreshnessCommit(uint64_t PathKey, uint64_t Version);
  bool consumeDuplicateWriteThroughFreshnessCommit(uint64_t PathKey,
                                                   uint64_t &VersionOut);
  uint64_t lastLocalFreshnessVersionForPathKey(uint64_t PathKey);
  uint64_t configuredStageTileBytes() const;
  bool dirtyOwnerExpectedVersionMatchesRange(
      const std::string &SourcePath, uint64_t Begin, uint64_t End,
      uint64_t ExpectedVersion, const OmpFileStageEntry &Entry);
  bool completeDerivedTileFlushesOnHeadnode(
      uint64_t PathKey, const std::vector<OmpFileStageExtent> &Extents,
      uint64_t TileBytes, const std::string &SourcePath);
  bool stageFreshnessGuardPermitsLocal(
      const std::string &SourcePath,
      const std::shared_ptr<OmpFileStageEntry> &StageEntry);
  uint64_t markDirtyRangeLocked(OmpFileStageEntry &Entry,
                                uint64_t Begin, uint64_t End);
  uint64_t clearDirtyRangeLocked(OmpFileStageEntry &Entry,
                                 uint64_t Begin, uint64_t End);
  bool flushLocalDirtyStageForPathKey(uint64_t BasePathKey);
  bool syncDeferredWritesForPathKey(uint64_t BasePathKey);
  bool flushLocalDirtyTileToSource(uint64_t TilePathKey, uint64_t Version);
  bool flushDirtyRangesToSource(OmpFileStageEntry &Entry, int SourceFd,
                                const std::string &SourcePath,
                                bool DeferCloseMetadataCompletion = false);
  bool shouldInvalidateStageOnOpen(int Flags) const;
  std::string makeStageFilePath(const std::string &SourcePath) const;
  bool ensureDirectoryTree(const std::string &Path, int &ErrnoOut) const;
  bool getStageEntryForPath(const std::string &SourcePath,
                            std::shared_ptr<OmpFileStageEntry> &EntryOut);
  bool hasStageFreeSpace(int &ErrnoOut) const;
  bool populateStageRange(const std::string &SourcePath, int StageFd,
                          uint64_t Offset, uint64_t Size,
                          const std::vector<OmpFileStageExtent> &SkipExtents,
                          uint64_t &PopulateBeginOut,
                          uint64_t &PopulateEndOut,
                          uint64_t &SourceSizeOut,
                          OmpFileStagePopulateStats &Stats,
                          int &ErrnoOut) const;
  void invalidateStageEntryLocked(const std::string &SourcePath,
                                  bool CountInvalidation,
                                  uint64_t InvalidatedBytes = 0,
                                  bool FullInvalidation = true);
  void invalidateStageForPath(const std::string &SourcePath);
  void invalidateStageRangeForPath(const std::string &SourcePath,
                                   uint64_t Offset, uint64_t Size);
  bool ensureStageEntryForPath(const std::string &SourcePath, uint64_t Offset,
                               uint64_t Size,
                               std::shared_ptr<OmpFileStageEntry> *EntryOut,
                               int &StageFd, int &ErrnoOut);
  bool preadWithOptionalStage(int Fd, int64_t Offset, void *Buffer, uint64_t Size,
                              uint64_t *BytesReadOut, int &ErrnoOut,
                              bool AllowStage = true);
  bool preadDirtyOwnerStageOnly(int Fd, int64_t Offset, void *Buffer,
                                  uint64_t Size, uint64_t ExpectedVersion,
                                  uint64_t *BytesReadOut, int &ErrnoOut);
  bool queryDirtyOwnerStageRangeOnly(int Fd, int64_t Offset, uint64_t Size,
                                     uint64_t ExpectedVersion, int &StateOut,
                                     int &ErrnoOut);
  bool preadFully(int Fd, int64_t Offset, void *Buffer, uint64_t Size,
                  uint64_t *BytesReadOut, uint64_t *ElapsedUsOut,
                  int &ErrnoOut);
  void recordStageWriteFsync(std::chrono::steady_clock::time_point Start);
  void recordSourcePwrite(uint64_t RequestedBytes, uint64_t BytesWritten,
                          uint64_t ElapsedUs);
  bool fdatasyncSourceTimed(int Fd);
  bool pwriteFully(int Fd, int64_t Offset, const void *Buffer, uint64_t Size,
                   uint64_t *BytesWrittenOut, uint64_t *ElapsedUsOut,
                   int &ErrnoOut);
  void updateStageCoverageForWrite(const std::shared_ptr<OmpFileStageEntry> &Entry,
                                   uint64_t Offset, uint64_t Size,
                                   uint64_t SourceSize,
                                   bool SourceSizeKnown);
  bool pwriteWithOptionalStage(int Fd, int64_t Offset, const void *Buffer,
                               uint64_t Size, uint64_t *BytesWrittenOut,
                               int &ErrnoOut);
  void drainOmpFileStageCache();

  // ProxyOmpFileServer.cpp: OMPFILE_* handlers, open cache, tracked fds, plans.
  EventTy ompfilePing(MPIRequestManagerTy RequestManager);
  void trackOmpFileFd(int Fd, const std::string &Path, int Flags, int Mode);
  bool getTrackedOmpFileFdPath(int Fd, std::string &Path) const;
  bool getTrackedOmpFileFdEntry(int Fd, OmpFileTrackedFdEntry &Entry) const;
  void eraseTrackedOmpFileFd(int Fd);
  bool shouldRefreshTrackedOmpFileFdForRead(
      const OmpFileTrackedFdEntry &Entry) const;
  bool refreshTrackedOmpFileFdForRead(int Fd, int &ErrnoOut);
  std::string getOmpFileOpenCacheKey(const char *Path, int Flags, int Mode) const;
  void reportOmpFileOpenFailureDiagnostics(const char *Path, int Flags,
                                           int Mode, int OpenErrno);
  int openWithOptionalCache(const char *Path, int Flags, int Mode, int &ErrnoOut);
  int closeWithOptionalCache(int Fd, int &ErrnoOut);
  void drainOmpFileOpenCache();
  EventTy ompfileOpen(MPIRequestManagerTy RequestManager);
  EventTy ompfileClose(MPIRequestManagerTy RequestManager);
  EventTy ompfilePread(MPIRequestManagerTy RequestManager,
                       bool AllowStage = true);
  EventTy ompfileDirtyOwnerPread(MPIRequestManagerTy RequestManager);
  EventTy ompfileDirtyOwnerPreadBatch(MPIRequestManagerTy RequestManager);
  EventTy ompfileDirtyOwnerQuery(MPIRequestManagerTy RequestManager);
  EventTy ompfilePwrite(MPIRequestManagerTy RequestManager);
  EventTy ompfileFreshnessQuery(MPIRequestManagerTy RequestManager);
  EventTy ompfileFreshnessWriteCommit(MPIRequestManagerTy RequestManager);
  EventTy ompfileFlushDirtyTile(MPIRequestManagerTy RequestManager);
  OmpFileIOPlan buildSchedulePlan(const OmpFileIORequest &Request,
                                  const char *Path);
  bool buildScheduleBatchPlan(const OmpFileIOBatchRequest &Request,
                              const void *Payload, uint64_t PayloadBytes,
                              OmpFileIOBatchPlan &Plan,
                              std::vector<uint8_t> &PlanPayload);
  EventTy ompfileSchedRequest(MPIRequestManagerTy RequestManager);
  EventTy ompfileSchedBatchRequest(MPIRequestManagerTy RequestManager);

  // ProxyOmpFileStats.cpp: the OMPFile and proxy_event_stats lines.
  void reportProxyEventStats(const char *Scope);
  void reportOmpFileStats(const char *Scope);

  // ProxyOmpFileForward.cpp: owner forwarding, round trips, mpp*, owner-bypass.
  bool waitForEvent(EventTy &Event, const char *OpName);
  bool dispatchSchedRequest(const OmpFileIORequest &Request, const char *Path,
                            OmpFileIOPlan &Plan);
  bool dispatchSchedBatchRequest(const OmpFileIOBatchRequest &Request,
                                 const void *RequestPayload,
                                 uint64_t RequestPayloadBytes,
                                 OmpFileIOBatchPlan &Plan, void *PlanPayload,
                                 uint64_t PlanPayloadCapBytes,
                                 uint64_t &PlanPayloadOutBytes);
  int registerRemoteHandle(int Rank, int RemoteHandle, const char *Path = nullptr,
                           int Flags = 0);
  bool findRemoteHandle(int LocalHandle, OmpFileHandleEntry &Entry);
  bool eraseRemoteHandle(int LocalHandle, OmpFileHandleEntry &Entry);
  bool openOnRank(int Rank, const char *Path, int Flags, int Mode,
                  int &RemoteHandle);
  bool closeOnRank(int Rank, int RemoteHandle);
  bool preadOnRank(int Rank, int RemoteHandle, int64_t Offset, void *Buffer,
                   uint64_t Size, uint64_t *BytesReadOut = nullptr,
                   bool AllowStage = true);
  bool pwriteOnRank(int Rank, int RemoteHandle, int64_t Offset,
                    const void *Buffer, uint64_t Size,
                    uint64_t *BytesWrittenOut);
  bool freshnessQueryOnHeadnode(const OmpFileFreshnessQueryRequest &Request,
                                OmpFileFreshnessQueryReply &Reply);
  bool freshnessWriteCommitOnHeadnode(uint64_t PathKey, int WriterRank,
                                      uint64_t TileId, bool WriteThroughMode,
                                      uint64_t &CommittedVersionOut);
  bool flushDirtyTileOnHeadnode(uint64_t PathKey, int &SourceRankOut,
                                uint64_t &FlushedVersionOut);
  bool commitPathKeyOnAllWorkers(uint64_t PathKey);
  bool completeDirtyFlushOnHeadnode(uint64_t PathKey, int SourceRank,
                                    uint64_t Version, bool Success);
  int selectAggregatorRankForOpen(const char *Path, int Mode);
  int mppInit();
  int mppSubmit(uint64_t Token);
  int mppPoll(uint64_t Token, int *Done);
  int mppOpen(const char *Path, int Flags, int Mode, int *Handle);
  int mppOpenOnRank(const char *Path, int Flags, int Mode, int Rank,
                    int *Handle);
  int mppClose(int Handle);
  int mppHandleOwnerRank(int Handle, int *RankOut);
  int mppPreadEx(int Handle, int64_t Offset, void *Buffer, uint64_t Size,
                 uint64_t *BytesRead);
  int mppPreadNoStageEx(int Handle, int64_t Offset, void *Buffer, uint64_t Size,
                        uint64_t *BytesRead);
  int mppDirtyOwnerPreadEx(int Handle, int SourceRank,
                           uint64_t ExpectedVersion, int64_t Offset,
                           void *Buffer, uint64_t Size, uint64_t *BytesRead);
  int mppDirtyOwnerPreadBatchEx(
      int Handle, int SourceRank,
      const OmpFileDirtyOwnerPreadBatchSegment *Segments, uint64_t SegmentCount,
      void *const *Buffers, uint64_t *BytesRead, int *Statuses,
      int *Errnos);
  int mppDirtyOwnerQueryEx(int Handle, int SourceRank,
                           uint64_t ExpectedVersion, int64_t Offset,
                           uint64_t Size, int *StateOut);
  int getLocalWriteFd(const std::string &Path);
  int getLocalReadFd(const std::string &Path);
  void closeLocalReadFds();
  void invalidateLocalReadCache(const std::string &Path);
  void invalidateLocalReadCaches();
  bool flushLocalWriteBufLocked(int Fd, LocalCoalesceBuf &Buf, int &ErrnoOut);
  bool flushLocalWriteBufs();
  bool preadStriped(int Fd, int64_t Offset, void *Buffer, uint64_t Size,
                    uint64_t *BytesReadOut);
  bool tryLocalDisjointPread(const OmpFileHandleEntry &Entry, int64_t Offset,
                             void *Buffer, uint64_t Size,
                             uint64_t *BytesRead);
  void syncLocalWriteFds();
  void closeLocalWriteFds();
  bool pwriteStriped(int Fd, int64_t Offset, const void *Buffer, uint64_t Size,
                     uint64_t *BytesWrittenOut, uint64_t *ElapsedUsOut,
                     int &ErrnoOut);
  bool tryLocalDisjointPwrite(const OmpFileHandleEntry &Entry, int64_t Offset,
                              const void *Buffer, uint64_t Size,
                              uint64_t *BytesWritten,
                              bool *HardFail = nullptr);
  int mppPwriteEx(int Handle, int64_t Offset, const void *Buffer, uint64_t Size,
                  uint64_t *BytesWritten);
  int mppSchedRequest(const OmpFileIORequest *Request, const char *Path,
                      OmpFileIOPlan *Plan);
  int mppFreshnessQuery(const OmpFileFreshnessQueryRequest *Request,
                        OmpFileFreshnessQueryReply *Reply);
  int mppFreshnessWriteCommit(uint64_t PathKey, int WriterRank, uint64_t TileId,
                              int WriteThroughMode,
                              uint64_t *CommittedVersion);
  int mppFlushDirtyTile(uint64_t PathKey, int *SourceRank,
                        uint64_t *FlushedVersion);
  int mppSchedBatchRequest(const OmpFileIOBatchRequest *Request,
                           const void *RequestPayload,
                           uint64_t RequestPayloadBytes,
                           OmpFileIOBatchPlan *Plan, void *PlanPayload,
                           uint64_t PlanPayloadCapBytes,
                           uint64_t *PlanPayloadOutBytes);
  int mppFinalize();

private:
  EventSystemTy &EventSystem;
  IntEnvar OmpFileOpenEioRetries;
  IntEnvar OmpFileOpenEioBackoffUs;
  BoolEnvar OmpFileOpenCacheEnable;
  BoolEnvar OmpFileOpenCacheKeepOpen;
  BoolEnvar OmpFileOptStats;
  // Create-on-write open semantics (default on): writable proxy opens add
  // O_CREAT so a fresh checkpoint write creates its file, removing the ENOENT
  // retry storm that deadlocked concurrent fresh opens. Set
  // LIBOMPFILE_OPEN_CREATE_ON_WRITE=0 to restore the pre-fix behavior (e.g. for
  // an A/B or if a workload needs a writable open to fail on a missing file).
  BoolEnvar OmpFileOpenCreateOnWrite;
  BoolEnvar OmpFileForceBlockingPwrite;
  Int64Envar OmpFileMPIFragmentSize;
  // LIBOMPFILE_* knobs, parsed once (ompfile_config.h); every OmpFile* member
  // below is initialized from it.
  const ompfile::config::ProxyConfig OmpFileConfig;
  // Owner-bypass: when set, an issuing proxy writes known-disjoint ranges of a
  // shared file directly to the local PFS mount instead of forwarding every
  // pwrite to the file's single aggregator/owner rank. The runtime exposes the
  // byte ranges; the app asserts disjointness by enabling this. Safe on a
  // parallel FS (POSIX permits concurrent disjoint pwrite); do not use on NFS.
  bool OmpFileLocalDisjointWrite = false;
  std::mutex OmpFileLocalWriteFdMutex;
  std::unordered_map<std::string, int> OmpFileLocalWriteFds; // path -> O_WRONLY fd
  std::atomic<uint64_t> OmpFileStatsLocalDisjointWrites{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointBytes{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointFallbacks{0};
  // Read counterpart (LIBOMPFILE_OPT_LOCAL_DISJOINT_READ): serve preads from a
  // process-local O_RDONLY fd instead of the owner funnel. Same app contract
  // (disjoint accesses, data visible in the PFS); fail-closed to the forward
  // path on open failure, error, or short read.
  bool OmpFileLocalDisjointRead = false;
  // LIBOMPFILE_OPT_COHERENT_READ_REFRESH (default on): re-open + dup2 a shared
  // O_RDWR descriptor before each read so it observes another NFS client's
  // writes (close-to-open). The dup2 closes the old description, and NFS
  // flushes the inode's dirty pages on every close, so with unsynced writes
  // (fsync policy "close", no commit yet) each restore read pays the previous
  // store's write-back: 1.8 s per proxy over 320 reads on the rtm lane
  // against 37 ms with nothing dirty. Off is safe only when every writer of
  // a file is its owner proxy (no owner-bypass, no out-of-band writers).
  bool OmpFileCoherentReadRefresh = true;
  std::mutex OmpFileLocalReadFdMutex;
  std::unordered_map<std::string, int> OmpFileLocalReadFds; // path -> O_RDONLY fd
  std::atomic<uint64_t> OmpFileStatsLocalDisjointReads{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointReadBytes{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointReadFallbacks{0};
  // Coalescing layer for the owner-bypass (opt-in, bytes; 0 = off). The PFS
  // charges a ~fixed per-op latency (wekafs: ~3.5 ms) regardless of size, so
  // small sequential disjoint ops must be merged into large PFS ops to
  // compete with MPI-IO collective buffering. WRITE_COMBINE appends
  // contiguous bypass pwrites into a per-path buffer flushed on
  // discontinuity/capacity/read/sync/close; READAHEAD serves bypass preads
  // from a per-path window filled cap-bytes at a time. Same disjointness
  // contract as the bypass itself; deferred write errors surface at the
  // flush point (close/sync), like page-cache write-back.
  uint64_t OmpFileLocalDisjointWriteCombine = 0;
  uint64_t OmpFileLocalDisjointReadahead = 0;
  // Striped bypass writes (LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE_STREAMS=K):
  // one large bypass pwrite is cut into up to K contiguous stripes of at
  // least STRIPE_MIN_BYTES and issued concurrently from K threads on the
  // same descriptor. pwrite is positional, so a shared fd is safe, and the
  // stripes are disjoint by construction. Why it helps is a property of the
  // storage client, not of this code: on wekafs a file with several writing
  // clients goes write-through, and one client's bandwidth then scales with
  // its stream count (~4.2 -> ~6.0 GB/s at four, application/benchmarks/
  // node-stream-probe, job 415432) where a single stream is the cap. 1 (the
  // default) is exactly the previous path. Do not raise it on NFS, where the
  // per-client stream is the cap and more streams only interleave worse.
  uint64_t OmpFileLocalDisjointWriteStreams = 1;
  uint64_t OmpFileLocalDisjointStripeMinBytes = 16u << 20;
  // Read counterpart (LIBOMPFILE_OPT_LOCAL_DISJOINT_READ_STREAMS=K): the same
  // striping for a large bypass pread, sharing STRIPE_MIN_BYTES. Reads scale
  // further than writes with streams on wekafs (1.8 -> 5.7 GB/s at four,
  // ~7 at eight, same probe).
  uint64_t OmpFileLocalDisjointReadStreams = 1;
  std::atomic<uint64_t> OmpFileStatsLocalDisjointStripedWrites{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointStripes{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointStripedReads{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointReadStripes{0};
  std::mutex OmpFileLocalWriteBufMutex;
  std::unordered_map<std::string, LocalCoalesceBuf> OmpFileLocalWriteBufs;
  std::mutex OmpFileLocalReadCacheMutex;
  std::unordered_map<std::string, LocalCoalesceBuf> OmpFileLocalReadCaches;
  std::atomic<uint64_t> OmpFileStatsLocalDisjointCombinedWrites{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointCombineFlushes{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointReadaheadHits{0};
  std::atomic<uint64_t> OmpFileStatsLocalDisjointReadaheadFills{0};
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
  bool OmpFileStageFlushCoalesce = false;
  bool OmpFileTopologyLoaded = false;
  uint64_t OmpFileTopologyEntries = 0;
  std::string OmpFileStageRoot;
  std::string OmpFileSelectedStageClass;
  std::string OmpFileSharedStoragePath;
  std::string OmpFileSharedStorageClass;
  std::string OmpFileStageLocalHost;
  std::string OmpFileStageDecisionReason;
  std::string OmpFileTopologyLoadError;
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
  std::atomic<uint64_t> OmpFileStatsSourcePreadBytes{0};
  std::atomic<uint64_t> OmpFileStatsSourcePreadUs{0};
  std::atomic<uint64_t> OmpFileStatsStagedPreadUs{0};
  // Source-side write service, the mirror of the pread counters above. Covers
  // the write-through path, the write-back flush to source, and the
  // owner-bypass (express) writes, so every byte the runtime puts on the
  // shared filesystem is timed exactly once.
  std::atomic<uint64_t> OmpFileStatsSourcePwriteBytes{0};
  std::atomic<uint64_t> OmpFileStatsSourcePwriteOps{0};
  std::atomic<uint64_t> OmpFileStatsSourcePwriteUs{0};
  std::atomic<uint64_t> OmpFileStatsSourceFsyncOps{0};
  std::atomic<uint64_t> OmpFileStatsSourceFsyncUs{0};
  std::atomic<uint64_t> OmpFileStatsStagingInvalidations{0};
  std::atomic<uint64_t> OmpFileStatsStagingRangeInvalidations{0};
  std::atomic<uint64_t> OmpFileStatsStagingFullInvalidations{0};
  std::atomic<uint64_t> OmpFileStatsStagingInvalidatedBytes{0};
  std::atomic<uint64_t> OmpFileStatsStagingWriteBypassCount{0};
  std::atomic<uint64_t> OmpFileStatsStageWriteFailures{0};
  // stats_schema=2: OmpFileStatsStageWriteUs is now the stage device pwrite
  // alone. The stage-side fdatasync and the write-back flush window used to be
  // folded into it and are separate terms below, so a schema-1 bundle's
  // stage_write_us_total is not comparable with a schema-2 one.
  std::atomic<uint64_t> OmpFileStatsStageWriteUs{0};
  std::atomic<uint64_t> OmpFileStatsStageWriteFsyncOps{0};
  std::atomic<uint64_t> OmpFileStatsStageWriteFsyncUs{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyFlushUs{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyFlushReadUs{0};
  // Write-back staging counters. DirtyBytes is a gauge snapshot at stats
  // print time; the rest are monotonic. Flush counters track write-back
  // flush-to-source events (Task 4 wires the real flush).
  std::atomic<uint64_t> OmpFileStatsStageWritebackCaptures{0};
  std::atomic<uint64_t> OmpFileStatsStageWritebackCaptureBytes{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyBytes{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyFlushes{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyFlushBytes{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyFlushFailures{0};
  std::atomic<uint64_t> OmpFileStatsStageDirtyFlushCoalesced{0};
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

#endif // OFFLOAD_PLUGINS_NEXTGEN_MPI_PROXY_OMPFILE_H
