//===-- ProxyOmpFileForward.cpp - Proxy owner forwarding --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Everything a proxy does on behalf of a target region running on it: the
// remote handle table, open/close/pread/pwrite on the owner rank, the
// headnode coherence round trips, the mpp* bridges behind the extern C
// exports, and the owner-bypass local disjoint I/O.
//
//===----------------------------------------------------------------------===//

#include "ProxyOmpFile.h"

bool ProxyOmpFile::waitForEvent(EventTy &Event, const char *OpName) {
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

bool ProxyOmpFile::dispatchSchedRequest(const OmpFileIORequest &Request,
                                        const char *Path, OmpFileIOPlan &Plan) {
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

bool ProxyOmpFile::dispatchSchedBatchRequest(
    const OmpFileIOBatchRequest &Request, const void *RequestPayload,
    uint64_t RequestPayloadBytes, OmpFileIOBatchPlan &Plan, void *PlanPayload,
    uint64_t PlanPayloadCapBytes, uint64_t &PlanPayloadOutBytes) {
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

int ProxyOmpFile::registerRemoteHandle(int Rank, int RemoteHandle,
                                       const char *Path, int Flags) {
  const int LocalHandle =
      NextOmpFileHandle.fetch_add(1, std::memory_order_relaxed);
  const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
  OmpFileHandleEntry Entry{};
  Entry.Rank = Rank;
  Entry.RemoteHandle = RemoteHandle;
  if (Path)
    Entry.Path = Path;
  Entry.Flags = Flags;
  OmpFileHandles[LocalHandle] = std::move(Entry);
  traceOmpFile("registerRemoteHandle local=%d rank=%d remote=%d map_size=%zu\n",
               LocalHandle, Rank, RemoteHandle, OmpFileHandles.size());
  return LocalHandle;
}

bool ProxyOmpFile::findRemoteHandle(int LocalHandle,
                                    OmpFileHandleEntry &Entry) {
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

bool ProxyOmpFile::eraseRemoteHandle(int LocalHandle,
                                     OmpFileHandleEntry &Entry) {
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

bool ProxyOmpFile::openOnRank(int Rank, const char *Path, int Flags, int Mode,
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

bool ProxyOmpFile::closeOnRank(int Rank, int RemoteHandle) {
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

bool ProxyOmpFile::preadOnRank(int Rank, int RemoteHandle, int64_t Offset,
                               void *Buffer, uint64_t Size,
                               uint64_t *BytesReadOut, bool AllowStage) {
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

bool ProxyOmpFile::pwriteOnRank(int Rank, int RemoteHandle, int64_t Offset,
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

bool ProxyOmpFile::freshnessQueryOnHeadnode(
    const OmpFileFreshnessQueryRequest &Request,
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

bool ProxyOmpFile::freshnessWriteCommitOnHeadnode(
    uint64_t PathKey, int WriterRank, uint64_t TileId, bool WriteThroughMode,
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

bool ProxyOmpFile::flushDirtyTileOnHeadnode(uint64_t PathKey,
                                            int &SourceRankOut,
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

// Commit issued on this proxy (from inside a target region): flush this
// process, then ask every peer proxy to do the same for the file, the way the
// origin's commit broadcasts Action 3. The file's writes live on whichever
// proxy owns it, which the HEADNODE may place on a peer, so a local-only
// flush would return with the owner's deferred fdatasyncs (fsync policy
// "close") and any dirty write-back stage still pending. Every peer is asked,
// not only the owner: a proxy holding nothing for the file answers at once.
bool ProxyOmpFile::commitPathKeyOnAllWorkers(uint64_t PathKey) {
  if (PathKey == 0) {
    errno = EINVAL;
    return false;
  }
  int FirstErrno = 0;
  if (!flushLocalDirtyStageForPathKey(PathKey))
    FirstErrno = errno != 0 ? errno : EIO;

  for (int Rank = 0; Rank < EventSystem.numWorkerRanks(); ++Rank) {
    if (Rank == EventSystem.LocalRank)
      continue;
    OmpFileFlushDirtyTileRequest Request{};
    Request.Action = 3;
    Request.PathKey = PathKey;
    OmpFileFlushDirtyTileReply Reply{};
    EventTy Event = createRankEvent(OriginEvents::ompfileFlushDirtyTile,
                                    EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE,
                                    Rank, /*TargetDeviceId=*/0, &Request,
                                    &Reply);
    if (!waitForEvent(Event, "commit_stage_peer")) {
      if (FirstErrno == 0)
        FirstErrno = EIO;
      continue;
    }
    if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      if (FirstErrno == 0)
        FirstErrno = EPROTO;
    } else if (Reply.Status != 0 && FirstErrno == 0) {
      FirstErrno = Reply.Errno != 0 ? Reply.Errno : EIO;
    }
  }
  errno = FirstErrno;
  return FirstErrno == 0;
}

bool ProxyOmpFile::completeDirtyFlushOnHeadnode(uint64_t PathKey,
                                                int SourceRank,
                                                uint64_t Version,
                                                bool Success) {
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

int ProxyOmpFile::selectAggregatorRankForOpen(const char *Path, int Mode) {
  int Rank = isWorkerRank(EventSystem.LocalRank) ? EventSystem.LocalRank : 0;
  if (!OmpFileHeadnodeScheduler)
    return Rank;

  OmpFileIORequest Request{};
  Request.RequestId =
      NextSchedRequestId.fetch_add(1, std::memory_order_relaxed);
  Request.Op = OmpFileIOOp::OPEN;
  // A placement query only. The headnode counts an OPEN plan as a live
  // handle until a CLOSE names its rank, and nothing sends that CLOSE for
  // this pick: a libompfile client that reached ompfile_mpp_open already
  // holds the counted plan for this logical open (or holds none) and its
  // CLOSE settles that one.
  Request.Flags = OMPFILE_IO_REQUEST_PLACEMENT_ONLY;
  Request.Mode = Mode;
  Request.ClientRank = EventSystem.LocalRank;
  Request.PathSize = Path ? static_cast<uint32_t>(std::strlen(Path) + 1) : 0;

  OmpFileIOPlan Plan{};
  if (!dispatchSchedRequest(Request, Path, Plan)) {
    // Not DP(): that compiles out of a release plugin, and a silently
    // re-routed open reads as a performance anomaly rather than a fault.
    fprintf(stderr,
            "MPIProxyDevice --> OMPFile scheduler fallback rank=%d "
            "reason=request_failed errno=%d path=%s using_rank=%d\n",
            EventSystem.LocalRank, errno, Path ? Path : "(null)", Rank);
    return Rank;
  }

  if (isWorkerRank(Plan.AggregatorRank))
    return Plan.AggregatorRank;
  fprintf(stderr,
          "MPIProxyDevice --> OMPFile scheduler fallback rank=%d "
          "reason=plan_rank_not_a_worker plan_rank=%d status=%d errno=%d "
          "path=%s using_rank=%d\n",
          EventSystem.LocalRank, Plan.AggregatorRank, Plan.Status, Plan.Errno,
          Path ? Path : "(null)", Rank);
  return Rank;
}

int ProxyOmpFile::mppInit() {
  DP("ompfile_mpp_init via proxy runtime (rank=%d, world=%d)\n",
     EventSystem.LocalRank, EventSystem.WorldSize);
  if (EventSystem.LocalRank == getHeadnodeRank())
    OmpFileHeadnodeManager::instance().initialize(
        EventSystem.numWorkerRanks(), getHeadnodeRank());
  const auto State = EventSystem.EventSystemState.load();
  if (State == EventSystemStateTy::RUNNING ||
      State == EventSystemStateTy::INITIALIZED)
    return OFFLOAD_SUCCESS;
  return OFFLOAD_FAIL;
}

int ProxyOmpFile::mppSubmit(uint64_t Token) {
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

int ProxyOmpFile::mppPoll(uint64_t Token, int *Done) {
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

int ProxyOmpFile::mppOpen(const char *Path, int Flags, int Mode, int *Handle) {
  if (!Path || !Handle)
    return OFFLOAD_FAIL;

  traceOmpFile("mppOpen enter path=%s flags=0x%x mode=%o\n", Path, Flags,
               Mode);
  const int Rank = selectAggregatorRankForOpen(Path, Mode);
  int RemoteHandle = -1;
  if (!openOnRank(Rank, Path, Flags, Mode, RemoteHandle))
    return OFFLOAD_FAIL;

  *Handle = registerRemoteHandle(Rank, RemoteHandle, Path, Flags);
  traceOmpFile("mppOpen exit local=%d rank=%d remote=%d\n", *Handle, Rank,
               RemoteHandle);
  return OFFLOAD_SUCCESS;
}

int ProxyOmpFile::mppOpenOnRank(const char *Path, int Flags, int Mode, int Rank,
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

  *Handle = registerRemoteHandle(Rank, RemoteHandle, Path, Flags);
  traceOmpFile("mppOpenOnRank exit local=%d rank=%d remote=%d\n", *Handle,
               Rank, RemoteHandle);
  return OFFLOAD_SUCCESS;
}

int ProxyOmpFile::mppClose(int Handle) {
  traceOmpFile("mppClose enter local=%d\n", Handle);
  OmpFileHandleEntry Entry{};
  if (!eraseRemoteHandle(Handle, Entry))
    return OFFLOAD_FAIL;
  // Owner-bypass durability: if any writes for this file were serviced
  // locally, flush the local write fds at the logical close (covers the
  // "close" fsync policy, which defers per-write fdatasync).
  if (OmpFileLocalDisjointWrite && (Entry.Flags & O_ACCMODE) != O_RDONLY) {
    // Deferred write-combine errors surface here, like page-cache
    // write-back at close.
    if (!flushLocalWriteBufs())
      return OFFLOAD_FAIL;
    syncLocalWriteFds();
  }
  // Close is a phase boundary: other ranks may rewrite this path next, so
  // drop any readahead windows.
  if (OmpFileLocalDisjointRead)
    invalidateLocalReadCaches();
  if (!closeOnRank(Entry.Rank, Entry.RemoteHandle))
    return OFFLOAD_FAIL;
  traceOmpFile("mppClose exit local=%d rank=%d remote=%d\n", Handle,
               Entry.Rank, Entry.RemoteHandle);
  return OFFLOAD_SUCCESS;
}

int ProxyOmpFile::mppHandleOwnerRank(int Handle, int *RankOut) {
  if (!RankOut)
    return EINVAL;
  *RankOut = -1;
  OmpFileHandleEntry Entry{};
  if (!findRemoteHandle(Handle, Entry))
    return errno != 0 ? errno : EIO;
  *RankOut = Entry.Rank;
  return OFFLOAD_SUCCESS;
}

int ProxyOmpFile::mppPreadEx(int Handle, int64_t Offset, void *Buffer,
                             uint64_t Size, uint64_t *BytesRead) {
  if (!BytesRead)
    return OFFLOAD_FAIL;
  *BytesRead = 0;
  OmpFileHandleEntry Entry{};
  if (!findRemoteHandle(Handle, Entry))
    return OFFLOAD_FAIL;
  if (tryLocalDisjointPread(Entry, Offset, Buffer, Size, BytesRead))
    return OFFLOAD_SUCCESS;
  if (!preadOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
                    BytesRead))
    return OFFLOAD_FAIL;
  return OFFLOAD_SUCCESS;
}

int ProxyOmpFile::mppPreadNoStageEx(int Handle, int64_t Offset, void *Buffer,
                                    uint64_t Size, uint64_t *BytesRead) {
  if (!BytesRead)
    return OFFLOAD_FAIL;
  *BytesRead = 0;
  OmpFileHandleEntry Entry{};
  if (!findRemoteHandle(Handle, Entry))
    return OFFLOAD_FAIL;
  if (tryLocalDisjointPread(Entry, Offset, Buffer, Size, BytesRead))
    return OFFLOAD_SUCCESS;
  if (!preadOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
                   BytesRead, /*AllowStage=*/false))
    return OFFLOAD_FAIL;
  return OFFLOAD_SUCCESS;
}

int ProxyOmpFile::mppDirtyOwnerPreadEx(int Handle, int SourceRank,
                                       uint64_t ExpectedVersion, int64_t Offset,
                                       void *Buffer, uint64_t Size,
                                       uint64_t *BytesRead) {
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

int ProxyOmpFile::mppDirtyOwnerPreadBatchEx(
    int Handle, int SourceRank,
    const OmpFileDirtyOwnerPreadBatchSegment *Segments, uint64_t SegmentCount,
    void *const *Buffers, uint64_t *BytesRead, int *Statuses, int *Errnos) {
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

int ProxyOmpFile::mppDirtyOwnerQueryEx(int Handle, int SourceRank,
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

// Owner-bypass: get (or lazily open) a process-local O_WRONLY fd to a shared
// file path, so the issuing proxy can pwrite disjoint ranges to the parallel
// FS directly instead of forwarding to the file's aggregator rank. Returns
// -1 on failure (caller then falls back to the forward path).
int ProxyOmpFile::getLocalWriteFd(const std::string &Path) {
  if (Path.empty())
    return -1;
  const std::lock_guard<std::mutex> Lock(OmpFileLocalWriteFdMutex);
  auto It = OmpFileLocalWriteFds.find(Path);
  if (It != OmpFileLocalWriteFds.end())
    return It->second;
  const int Fd = ::open(Path.c_str(), O_WRONLY);
  if (Fd >= 0)
    OmpFileLocalWriteFds[Path] = Fd;
  return Fd;
}

// Owner-bypass read counterpart: get (or lazily open) a process-local
// O_RDONLY fd so the issuing proxy can pread directly from the parallel FS
// instead of funneling through the file's aggregator rank. Same app
// contract as the write bypass (LIBOMPFILE_OPT_LOCAL_DISJOINT_WRITE):
// accesses are disjoint and the data being read is visible in the PFS
// (flushed / written-through). -1 on failure -> caller uses forward path.
int ProxyOmpFile::getLocalReadFd(const std::string &Path) {
  if (Path.empty())
    return -1;
  const std::lock_guard<std::mutex> Lock(OmpFileLocalReadFdMutex);
  auto It = OmpFileLocalReadFds.find(Path);
  if (It != OmpFileLocalReadFds.end())
    return It->second;
  const int Fd = ::open(Path.c_str(), O_RDONLY);
  if (Fd >= 0)
    OmpFileLocalReadFds[Path] = Fd;
  return Fd;
}

void ProxyOmpFile::closeLocalReadFds() {
  invalidateLocalReadCaches();
  const std::lock_guard<std::mutex> Lock(OmpFileLocalReadFdMutex);
  for (auto &KV : OmpFileLocalReadFds)
    if (KV.second >= 0)
      ::close(KV.second);
  OmpFileLocalReadFds.clear();
}

// Readahead cache invalidation: a bypass write to the path, a close (phase
// boundary), or shutdown makes any cached window untrustworthy.
void ProxyOmpFile::invalidateLocalReadCache(const std::string &Path) {
  const std::lock_guard<std::mutex> Lock(OmpFileLocalReadCacheMutex);
  auto It = OmpFileLocalReadCaches.find(Path);
  if (It != OmpFileLocalReadCaches.end()) {
    It->second.Base = -1;
    It->second.Len = 0;
  }
}

void ProxyOmpFile::invalidateLocalReadCaches() {
  const std::lock_guard<std::mutex> Lock(OmpFileLocalReadCacheMutex);
  for (auto &KV : OmpFileLocalReadCaches) {
    KV.second.Base = -1;
    KV.second.Len = 0;
  }
}

// Flush one write-combine buffer to its fd. Caller holds
// OmpFileLocalWriteBufMutex. False = flush failed (ErrnoOut set) and the
// buffered bytes are still pending — the caller must surface a hard error,
// never silently drop already-acked writes.
bool ProxyOmpFile::flushLocalWriteBufLocked(int Fd, LocalCoalesceBuf &Buf,
                                            int &ErrnoOut) {
  if (Buf.Base < 0 || Buf.Len == 0) {
    Buf.Base = -1;
    Buf.Len = 0;
    return true;
  }
  uint64_t Written = 0, ElapsedUs = 0;
  const bool WriteOk = pwriteFully(Fd, Buf.Base, Buf.Data.get(), Buf.Len,
                                   &Written, &ElapsedUs, ErrnoOut);
  recordSourcePwrite(Buf.Len, Written, ElapsedUs);
  if (!WriteOk)
    return false;
  if (writethroughFsyncEachEnabled())
    (void)fdatasyncSourceTimed(Fd);
  OmpFileStatsLocalDisjointCombineFlushes.fetch_add(
      1, std::memory_order_relaxed);
  traceOmpFile("localDisjointCombineFlush base=%lld len=%llu\n",
               static_cast<long long>(Buf.Base),
               static_cast<unsigned long long>(Buf.Len));
  Buf.Base = -1;
  Buf.Len = 0;
  return true;
}

// Flush every pending write-combine buffer (logical close, sync, shutdown).
// Returns false if any flush failed; deferred write errors surface to the
// caller like page-cache write-back errors surface at fsync/close.
bool ProxyOmpFile::flushLocalWriteBufs() {
  const std::lock_guard<std::mutex> Lock(OmpFileLocalWriteBufMutex);
  bool Ok = true;
  for (auto &KV : OmpFileLocalWriteBufs) {
    const int Fd = getLocalWriteFd(KV.first);
    int FlushErrno = 0;
    if (Fd < 0 || !flushLocalWriteBufLocked(Fd, KV.second, FlushErrno))
      Ok = false;
  }
  return Ok;
}

// Attempt the owner-bypass for one pread. Returns true only if the full
// requested range was read locally; any failure or short read falls back to
// the forward path (the owner's authoritative handle), fail-closed.
// A full-length bypass pread as up to OmpFileLocalDisjointReadStreams
// concurrent contiguous stripes (see pwriteStriped; same stripe rule).
// Returns false on any error or short read — the caller falls back to the
// owner, which is authoritative for in-flight writes and the true size.
bool ProxyOmpFile::preadStriped(int Fd, int64_t Offset, void *Buffer,
                                uint64_t Size, uint64_t *BytesReadOut) {
  uint64_t Streams = OmpFileLocalDisjointReadStreams;
  if (Streams < 1)
    Streams = 1;
  const uint64_t MinStripe =
      OmpFileLocalDisjointStripeMinBytes > 0 ? OmpFileLocalDisjointStripeMinBytes
                                             : 1;
  uint64_t K = Size / MinStripe;
  if (K > Streams)
    K = Streams;
  if (K < 1)
    K = 1;
  struct Stripe {
    int64_t Offset = 0;
    uint64_t Size = 0;
    uint64_t Read = 0;
    bool Ok = true;
  };
  std::vector<Stripe> Stripes(static_cast<size_t>(K));
  const uint64_t Base = Size / K;
  const uint64_t Rem = Size % K;
  uint64_t Cursor = 0;
  for (uint64_t I = 0; I < K; ++I) {
    Stripes[I].Offset = Offset + static_cast<int64_t>(Cursor);
    Stripes[I].Size = Base + (I < Rem ? 1 : 0);
    Cursor += Stripes[I].Size;
  }
  auto Run = [&](Stripe &S) {
    char *Dst = static_cast<char *>(Buffer) +
                static_cast<size_t>(S.Offset - Offset);
    while (S.Read < S.Size) {
      const ssize_t Bytes =
          ::pread(Fd, Dst + S.Read, static_cast<size_t>(S.Size - S.Read),
                  static_cast<off_t>(S.Offset + static_cast<int64_t>(S.Read)));
      if (Bytes <= 0) {
        S.Ok = false;
        return;
      }
      S.Read += static_cast<uint64_t>(Bytes);
    }
  };
  if (K == 1) {
    Run(Stripes[0]);
  } else {
    std::vector<std::thread> Threads;
    Threads.reserve(static_cast<size_t>(K - 1));
    for (uint64_t I = 1; I < K; ++I)
      Threads.emplace_back(Run, std::ref(Stripes[I]));
    Run(Stripes[0]);
    for (auto &T : Threads)
      T.join();
    OmpFileStatsLocalDisjointStripedReads.fetch_add(1,
                                                    std::memory_order_relaxed);
    OmpFileStatsLocalDisjointReadStripes.fetch_add(K,
                                                   std::memory_order_relaxed);
  }
  uint64_t Total = 0;
  for (auto &S : Stripes) {
    if (!S.Ok)
      return false;
    Total += S.Read;
  }
  if (BytesReadOut)
    *BytesReadOut = Total;
  return true;
}

bool ProxyOmpFile::tryLocalDisjointPread(const OmpFileHandleEntry &Entry,
                                         int64_t Offset, void *Buffer,
                                         uint64_t Size, uint64_t *BytesRead) {
  if (!OmpFileLocalDisjointRead || Entry.Path.empty())
    return false;
  const int Fd = getLocalReadFd(Entry.Path);
  if (Fd < 0) {
    OmpFileStatsLocalDisjointReadFallbacks.fetch_add(
        1, std::memory_order_relaxed);
    return false;
  }
  // Readahead: wekafs charges a ~fixed latency per op regardless of size,
  // so small sequential preads are served from a per-path window filled
  // cap-bytes at a time. Same disjointness contract as the bypass; the
  // window is invalidated on local bypass writes and at close.
  // Strict '<': an op exactly at cap gains nothing from the window (the
  // fill IS the op) and would pay an extra full-size memcpy — measured 12%
  // read loss at 16 MiB transfers with a 16 MiB cap (job 359377).
  if (OmpFileLocalDisjointReadahead != 0 &&
      Size < OmpFileLocalDisjointReadahead) {
    // Coherence with our own write-combine buffer: pending combined bytes
    // must reach the PFS before we read through the read fd.
    if (OmpFileLocalDisjointWriteCombine != 0) {
      const std::lock_guard<std::mutex> BufLock(OmpFileLocalWriteBufMutex);
      auto BufIt = OmpFileLocalWriteBufs.find(Entry.Path);
      if (BufIt != OmpFileLocalWriteBufs.end() && BufIt->second.Base >= 0) {
        const int WriteFd = getLocalWriteFd(Entry.Path);
        int FlushErrno = 0;
        if (WriteFd < 0 ||
            !flushLocalWriteBufLocked(WriteFd, BufIt->second, FlushErrno)) {
          OmpFileStatsLocalDisjointReadFallbacks.fetch_add(
              1, std::memory_order_relaxed);
          return false;
        }
      }
    }
    const std::lock_guard<std::mutex> Lock(OmpFileLocalReadCacheMutex);
    LocalCoalesceBuf &Cache = OmpFileLocalReadCaches[Entry.Path];
    if (!Cache.Data) {
      Cache.Cap = OmpFileLocalDisjointReadahead;
      Cache.Data.reset(new (std::nothrow) char[Cache.Cap]);
    }
    if (Cache.Data) {
      const bool Hit =
          Cache.Base >= 0 && Offset >= Cache.Base &&
          Offset + static_cast<int64_t>(Size) <=
              Cache.Base + static_cast<int64_t>(Cache.Len);
      if (!Hit) {
        uint64_t Filled = 0;
        while (Filled < Cache.Cap) {
          const ssize_t Bytes = ::pread(
              Fd, Cache.Data.get() + Filled,
              static_cast<size_t>(Cache.Cap - Filled),
              static_cast<off_t>(Offset + static_cast<int64_t>(Filled)));
          if (Bytes < 0) {
            Cache.Base = -1;
            Cache.Len = 0;
            OmpFileStatsLocalDisjointReadFallbacks.fetch_add(
                1, std::memory_order_relaxed);
            return false;
          }
          if (Bytes == 0)
            break; // EOF: partial window is fine if it covers the request
          Filled += static_cast<uint64_t>(Bytes);
        }
        if (Filled < Size) {
          // Short of the requested range: the owner's handle is
          // authoritative (in-flight forwarded writes, true size).
          Cache.Base = -1;
          Cache.Len = 0;
          OmpFileStatsLocalDisjointReadFallbacks.fetch_add(
              1, std::memory_order_relaxed);
          return false;
        }
        Cache.Base = Offset;
        Cache.Len = Filled;
        OmpFileStatsLocalDisjointReadaheadFills.fetch_add(
            1, std::memory_order_relaxed);
      } else {
        OmpFileStatsLocalDisjointReadaheadHits.fetch_add(
            1, std::memory_order_relaxed);
      }
      std::memcpy(Buffer,
                  Cache.Data.get() + (Offset - Cache.Base),
                  static_cast<size_t>(Size));
      if (BytesRead)
        *BytesRead = Size;
      OmpFileStatsLocalDisjointReads.fetch_add(1, std::memory_order_relaxed);
      OmpFileStatsLocalDisjointReadBytes.fetch_add(
          Size, std::memory_order_relaxed);
      traceOmpFile("localDisjointPread offset=%lld size=%llu (readahead)\n",
                   static_cast<long long>(Offset),
                   static_cast<unsigned long long>(Size));
      return true;
    }
    // Window allocation failed: fall through to the direct pread loop.
  }
  uint64_t TotalRead = 0;
  if (!preadStriped(Fd, Offset, Buffer, Size, &TotalRead)) {
    // Error or EOF-short: the owner's handle is authoritative (it sees
    // in-flight forwarded writes and the true size) — fall back.
    OmpFileStatsLocalDisjointReadFallbacks.fetch_add(
        1, std::memory_order_relaxed);
    return false;
  }
  if (BytesRead)
    *BytesRead = TotalRead;
  OmpFileStatsLocalDisjointReads.fetch_add(1, std::memory_order_relaxed);
  OmpFileStatsLocalDisjointReadBytes.fetch_add(Size,
                                               std::memory_order_relaxed);
  traceOmpFile("localDisjointPread offset=%lld size=%llu\n",
               static_cast<long long>(Offset),
               static_cast<unsigned long long>(Size));
  return true;
}

// fdatasync every local write fd (durability at logical close for the
// "close" fsync policy) without closing — fds are reused across the run and
// reaped in shutdown.
void ProxyOmpFile::syncLocalWriteFds() {
  const std::lock_guard<std::mutex> Lock(OmpFileLocalWriteFdMutex);
  for (auto &KV : OmpFileLocalWriteFds)
    if (KV.second >= 0)
      ::fdatasync(KV.second);
}

void ProxyOmpFile::closeLocalWriteFds() {
  flushLocalWriteBufs();
  const std::lock_guard<std::mutex> Lock(OmpFileLocalWriteFdMutex);
  for (auto &KV : OmpFileLocalWriteFds)
    if (KV.second >= 0) {
      ::fdatasync(KV.second);
      ::close(KV.second);
    }
  OmpFileLocalWriteFds.clear();
}

// One bypass pwrite, issued as up to OmpFileLocalDisjointWriteStreams
// concurrent contiguous stripes (see the member comment). Stripes below
// OmpFileLocalDisjointStripeMinBytes are not worth a thread, so the stripe
// count is the largest K <= streams with Size / K >= min; K == 1 is a plain
// pwriteFully on the calling thread and is what the default resolves to.
// Stats: every physical pwrite counts as a source op; the elapsed time
// recorded is the wall of the whole striped op, not the sum over stripes,
// so source_pwrite_us_total keeps meaning "time this proxy spent writing".
// A stripe that fails reports the first failing errno and the bytes that
// did land contiguously from the start, which is what the caller's partial
// write semantics expect.
bool ProxyOmpFile::pwriteStriped(int Fd, int64_t Offset, const void *Buffer,
                                 uint64_t Size, uint64_t *BytesWrittenOut,
                                 uint64_t *ElapsedUsOut, int &ErrnoOut) {
  uint64_t Streams = OmpFileLocalDisjointWriteStreams;
  if (Streams < 1)
    Streams = 1;
  const uint64_t MinStripe =
      OmpFileLocalDisjointStripeMinBytes > 0 ? OmpFileLocalDisjointStripeMinBytes
                                             : 1;
  uint64_t K = Size / MinStripe;
  if (K > Streams)
    K = Streams;
  if (K <= 1) {
    const bool Ok = pwriteFully(Fd, Offset, Buffer, Size, BytesWrittenOut,
                                ElapsedUsOut, ErrnoOut);
    recordSourcePwrite(Size, BytesWrittenOut ? *BytesWrittenOut : 0,
                       ElapsedUsOut ? *ElapsedUsOut : 0);
    return Ok;
  }

  struct Stripe {
    int64_t Offset = 0;
    uint64_t Size = 0;
    uint64_t Written = 0;
    uint64_t ElapsedUs = 0;
    int Errno = 0;
    bool Ok = true;
  };
  std::vector<Stripe> Stripes(static_cast<size_t>(K));
  const uint64_t Base = Size / K;
  const uint64_t Rem = Size % K;
  uint64_t Cursor = 0;
  for (uint64_t I = 0; I < K; ++I) {
    Stripes[I].Offset = Offset + static_cast<int64_t>(Cursor);
    Stripes[I].Size = Base + (I < Rem ? 1 : 0);
    Cursor += Stripes[I].Size;
  }
  const auto WallStart = std::chrono::steady_clock::now();
  std::vector<std::thread> Threads;
  Threads.reserve(static_cast<size_t>(K - 1));
  auto Run = [&](Stripe &S) {
    S.Ok = pwriteFully(Fd, S.Offset,
                       static_cast<const char *>(Buffer) +
                           static_cast<size_t>(S.Offset - Offset),
                       S.Size, &S.Written, &S.ElapsedUs, S.Errno);
  };
  for (uint64_t I = 1; I < K; ++I)
    Threads.emplace_back(Run, std::ref(Stripes[I]));
  Run(Stripes[0]); // the calling thread takes the first stripe
  for (auto &T : Threads)
    T.join();
  const uint64_t WallUs =
      elapsedMicros(WallStart, std::chrono::steady_clock::now());

  // Contiguous bytes landed from the start; first failure wins.
  uint64_t Contiguous = 0;
  bool AllOk = true;
  ErrnoOut = 0;
  for (auto &S : Stripes) {
    if (AllOk)
      Contiguous += S.Written;
    if (!S.Ok) {
      if (AllOk)
        ErrnoOut = S.Errno;
      AllOk = false;
    }
  }
  if (BytesWrittenOut)
    *BytesWrittenOut = Contiguous;
  if (ElapsedUsOut)
    *ElapsedUsOut = WallUs;
  OmpFileStatsSourcePwriteOps.fetch_add(K, std::memory_order_relaxed);
  OmpFileStatsSourcePwriteBytes.fetch_add(Contiguous,
                                          std::memory_order_relaxed);
  OmpFileStatsSourcePwriteUs.fetch_add(WallUs, std::memory_order_relaxed);
  OmpFileStatsLocalDisjointStripedWrites.fetch_add(1,
                                                   std::memory_order_relaxed);
  OmpFileStatsLocalDisjointStripes.fetch_add(K, std::memory_order_relaxed);
  return AllOk;
}

// Attempt the owner-bypass for one pwrite. Returns true if the write was
// fully serviced locally; false means the caller should use the forward
// path — unless *HardFail was set, in which case a deferred combine flush
// failed (already-acked bytes could not reach the PFS) and the caller must
// fail the operation instead of forwarding.
bool ProxyOmpFile::tryLocalDisjointPwrite(const OmpFileHandleEntry &Entry,
                                          int64_t Offset, const void *Buffer,
                                          uint64_t Size, uint64_t *BytesWritten,
                                          bool *HardFail) {
  if (HardFail)
    *HardFail = false;
  if (!OmpFileLocalDisjointWrite || Entry.Path.empty())
    return false;
  if ((Entry.Flags & O_ACCMODE) == O_RDONLY)
    return false;
  const int Fd = getLocalWriteFd(Entry.Path);
  if (Fd < 0) {
    OmpFileStatsLocalDisjointFallbacks.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (OmpFileLocalDisjointReadahead != 0)
    invalidateLocalReadCache(Entry.Path);
  // Write-combine: contiguous small bypass writes append into a per-path
  // buffer and hit the PFS as one large pwrite (wekafs pays ~fixed latency
  // per op). Flushed on discontinuity, capacity, reads, sync, and close.
  if (OmpFileLocalDisjointWriteCombine != 0) {
    const std::lock_guard<std::mutex> Lock(OmpFileLocalWriteBufMutex);
    if (Size < OmpFileLocalDisjointWriteCombine) {
      LocalCoalesceBuf &Buf = OmpFileLocalWriteBufs[Entry.Path];
      if (!Buf.Data) {
        Buf.Cap = OmpFileLocalDisjointWriteCombine;
        Buf.Data.reset(new (std::nothrow) char[Buf.Cap]);
      }
      if (Buf.Data) {
        const bool Contiguous =
            Buf.Base >= 0 &&
            Offset == Buf.Base + static_cast<int64_t>(Buf.Len);
        if (!Contiguous || Buf.Len + Size > Buf.Cap) {
          int FlushErrno = 0;
          if (!flushLocalWriteBufLocked(Fd, Buf, FlushErrno)) {
            errno = FlushErrno != 0 ? FlushErrno : EIO;
            if (HardFail)
              *HardFail = true;
            return false;
          }
        }
        if (Buf.Base < 0)
          Buf.Base = Offset;
        std::memcpy(Buf.Data.get() + Buf.Len, Buffer,
                    static_cast<size_t>(Size));
        Buf.Len += Size;
        if (BytesWritten)
          *BytesWritten = Size;
        OmpFileStatsLocalDisjointWrites.fetch_add(1,
                                                  std::memory_order_relaxed);
        OmpFileStatsLocalDisjointBytes.fetch_add(Size,
                                                 std::memory_order_relaxed);
        OmpFileStatsLocalDisjointCombinedWrites.fetch_add(
            1, std::memory_order_relaxed);
        traceOmpFile("localDisjointPwrite offset=%lld size=%llu (combined)\n",
                     static_cast<long long>(Offset),
                     static_cast<unsigned long long>(Size));
        return true;
      }
      // Buffer allocation failed: fall through to the direct pwrite.
    } else {
      // Large op: flush any pending buffer first so per-path ordering of
      // this proxy's own writes is preserved.
      auto It = OmpFileLocalWriteBufs.find(Entry.Path);
      if (It != OmpFileLocalWriteBufs.end()) {
        int FlushErrno = 0;
        if (!flushLocalWriteBufLocked(Fd, It->second, FlushErrno)) {
          errno = FlushErrno != 0 ? FlushErrno : EIO;
          if (HardFail)
            *HardFail = true;
          return false;
        }
      }
    }
  }
  uint64_t Written = 0, ElapsedUs = 0;
  int WriteErrno = 0;
  const bool WriteOk = pwriteStriped(Fd, Offset, Buffer, Size, &Written,
                                     &ElapsedUs, WriteErrno);
  if (!WriteOk) {
    errno = WriteErrno;
    OmpFileStatsLocalDisjointFallbacks.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (writethroughFsyncEachEnabled())
    (void)fdatasyncSourceTimed(Fd);
  if (BytesWritten)
    *BytesWritten = Written;
  OmpFileStatsLocalDisjointWrites.fetch_add(1, std::memory_order_relaxed);
  OmpFileStatsLocalDisjointBytes.fetch_add(Size, std::memory_order_relaxed);
  traceOmpFile("localDisjointPwrite offset=%lld size=%llu bytes=%llu\n",
               static_cast<long long>(Offset),
               static_cast<unsigned long long>(Size),
               static_cast<unsigned long long>(Written));
  return true;
}

int ProxyOmpFile::mppPwriteEx(int Handle, int64_t Offset, const void *Buffer,
                              uint64_t Size, uint64_t *BytesWritten) {
  if (!BytesWritten)
    return OFFLOAD_FAIL;
  *BytesWritten = 0;
  OmpFileHandleEntry Entry{};
  if (!findRemoteHandle(Handle, Entry))
    return OFFLOAD_FAIL;
  bool HardFail = false;
  if (tryLocalDisjointPwrite(Entry, Offset, Buffer, Size, BytesWritten,
                             &HardFail))
    return OFFLOAD_SUCCESS;
  if (HardFail)
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

int ProxyOmpFile::mppSchedRequest(const OmpFileIORequest *Request,
                                  const char *Path, OmpFileIOPlan *Plan) {
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

int ProxyOmpFile::mppFreshnessQuery(const OmpFileFreshnessQueryRequest *Request,
                                    OmpFileFreshnessQueryReply *Reply) {
  if (!Request || !Reply)
    return EINVAL;
  if (!freshnessQueryOnHeadnode(*Request, *Reply))
    return errno != 0 ? errno : EIO;
  if (Reply->Status != 0)
    return Reply->Errno != 0 ? Reply->Errno : EIO;
  return OFFLOAD_SUCCESS;
}

int ProxyOmpFile::mppFreshnessWriteCommit(uint64_t PathKey, int WriterRank,
                                          uint64_t TileId, int WriteThroughMode,
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

int ProxyOmpFile::mppFlushDirtyTile(uint64_t PathKey, int *SourceRank,
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

int ProxyOmpFile::mppSchedBatchRequest(const OmpFileIOBatchRequest *Request,
                                       const void *RequestPayload,
                                       uint64_t RequestPayloadBytes,
                                       OmpFileIOBatchPlan *Plan,
                                       void *PlanPayload,
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

int ProxyOmpFile::mppFinalize() {
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
