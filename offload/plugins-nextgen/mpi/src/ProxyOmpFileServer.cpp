//===-- ProxyOmpFileServer.cpp - Proxy OMPFILE_* handlers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The proxy-side handlers of the OMPFILE_* events (createFileIoEvent is the
// gate thread's entry), the open cache and tracked fds behind them, and
// the headnode plan builders.
//
//===----------------------------------------------------------------------===//

#include "ProxyOmpFile.h"

EventTy ProxyOmpFile::ompfilePing(MPIRequestManagerTy RequestManager) {
  uint64_t Token = 0;

  RequestManager.receive(&Token, 1, MPI_UINT64_T);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.send(&Token, 1, MPI_UINT64_T);
  RequestManager.send(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

void ProxyOmpFile::trackOmpFileFd(int Fd, const std::string &Path, int Flags,
                                  int Mode) {
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

bool ProxyOmpFile::getTrackedOmpFileFdPath(int Fd, std::string &Path) const {
  const std::lock_guard<std::mutex> Lock(OmpFileTrackedFdMutex);
  auto It = OmpFileTrackedFds.find(Fd);
  if (It == OmpFileTrackedFds.end())
    return false;
  Path = It->second.Path;
  return true;
}

bool ProxyOmpFile::getTrackedOmpFileFdEntry(
    int Fd, OmpFileTrackedFdEntry &Entry) const {
  const std::lock_guard<std::mutex> Lock(OmpFileTrackedFdMutex);
  auto It = OmpFileTrackedFds.find(Fd);
  if (It == OmpFileTrackedFds.end())
    return false;
  Entry = It->second;
  return true;
}

void ProxyOmpFile::eraseTrackedOmpFileFd(int Fd) {
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

bool ProxyOmpFile::shouldRefreshTrackedOmpFileFdForRead(
    const OmpFileTrackedFdEntry &Entry) const {
  if (!OmpFileCoherentReadRefresh)
    return false;
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

bool ProxyOmpFile::refreshTrackedOmpFileFdForRead(int Fd, int &ErrnoOut) {
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

std::string ProxyOmpFile::getOmpFileOpenCacheKey(const char *Path, int Flags,
                                                 int Mode) const {
  std::string Key = Path ? Path : "";
  Key.push_back('\x1f');
  Key += std::to_string(Flags);
  Key.push_back('\x1f');
  Key += std::to_string(Mode);
  return Key;
}

void ProxyOmpFile::reportOmpFileOpenFailureDiagnostics(const char *Path,
                                                       int Flags, int Mode,
                                                       int OpenErrno) {
  if (!OmpFileConfig.OpenFailureDiag)
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

int ProxyOmpFile::openWithOptionalCache(const char *Path, int Flags, int Mode,
                                        int &ErrnoOut) {
  ErrnoOut = 0;
  OmpFileStatsOpenRequests.fetch_add(1, std::memory_order_relaxed);

  if (!Path) {
    ErrnoOut = EINVAL;
    return -1;
  }

  // Fresh write-open create semantics. A writable open of a not-yet-existing
  // file must create it, exactly as native POSIX/MPI-IO checkpoint writers do
  // (open(O_CREAT|O_WRONLY, ...)). Without O_CREAT the proxy open returns
  // ENOENT, which ompfileOpen's transient-retry loop treats as retryable and
  // backs off on synchronously (std::this_thread::sleep_for) — monopolizing
  // the rank's single data-event handler for up to MaxEioRetries. A wave of
  // concurrent fresh opens (the file-per-rank checkpoint pattern) then starves
  // cross-rank OmpFile servicing until the case timeout: the "concurrent
  // fresh-open deadlock". Adding O_CREAT for writable opens removes the ENOENT
  // entirely, which is both the O_CREAT retry-storm fix and the deadlock fix,
  // and lets the driver drop its sequential pre-open workaround. Read-only
  // opens keep ENOENT (with the transient retry that legitimately rides out
  // NFS metadata propagation for files another rank is still creating).
  if (OmpFileOpenCreateOnWrite && (Flags & O_ACCMODE) != O_RDONLY &&
      !(Flags & O_CREAT)) {
    Flags |= O_CREAT;
    if (Mode == 0)
      Mode = 0666;
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

int ProxyOmpFile::closeWithOptionalCache(int Fd, int &ErrnoOut) {
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
    if (NeedDeferredSync && !fdatasyncSourceTimed(Fd)) {
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

void ProxyOmpFile::drainOmpFileOpenCache() {
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

EventTy ProxyOmpFile::ompfileOpen(MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::ompfileClose(MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::ompfilePread(MPIRequestManagerTy RequestManager,
                                   bool AllowStage) {
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

EventTy ProxyOmpFile::ompfileDirtyOwnerPread(
    MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::ompfileDirtyOwnerPreadBatch(
    MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::ompfileDirtyOwnerQuery(
    MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::ompfilePwrite(MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::ompfileFreshnessQuery(
    MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::ompfileFreshnessWriteCommit(
    MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::ompfileFlushDirtyTile(
    MPIRequestManagerTy RequestManager) {
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
  } else if (Request.Action == 3) {
    // Commit: flush this rank's dirty stage for the file, version-free.
    if (!flushLocalDirtyStageForPathKey(Request.PathKey)) {
      Reply.Status = -1;
      Reply.Errno = errno != 0 ? errno : EIO;
    } else {
      Reply.SourceRank = EventSystem.LocalRank;
    }
  } else {
    Reply.Status = -1;
    Reply.Errno = EINVAL;
  }

  RequestManager.send(&Reply, sizeof(Reply), MPI_BYTE);
  RequestManager.send(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

OmpFileIOPlan ProxyOmpFile::buildSchedulePlan(const OmpFileIORequest &Request,
                                              const char *Path) {
  OmpFileHeadnodeManager &Manager = OmpFileHeadnodeManager::instance();
  Manager.initialize(EventSystem.numWorkerRanks(), getHeadnodeRank());
  return Manager.planRequest(Request, Path, EventSystem.LocalRank);
}

bool ProxyOmpFile::buildScheduleBatchPlan(const OmpFileIOBatchRequest &Request,
                                          const void *Payload,
                                          uint64_t PayloadBytes,
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
  Manager.initialize(EventSystem.numWorkerRanks(), getHeadnodeRank());
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

EventTy ProxyOmpFile::ompfileSchedRequest(MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::ompfileSchedBatchRequest(
    MPIRequestManagerTy RequestManager) {
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

EventTy ProxyOmpFile::createFileIoEvent(EventTypeTy Type,
                                        MPIRequestManagerTy &&RequestManager) {
  using enum EventTypeTy;
  switch (Type) {
  case OMPFILE_PING:
    return ompfilePing(std::move(RequestManager));
  case OMPFILE_OPEN:
    return ompfileOpen(std::move(RequestManager));
  case OMPFILE_CLOSE:
    return ompfileClose(std::move(RequestManager));
  case OMPFILE_PREAD:
    return ompfilePread(std::move(RequestManager));
  case OMPFILE_PREAD_NO_STAGE:
    return ompfilePread(std::move(RequestManager), /*AllowStage=*/false);
  case OMPFILE_DIRTY_OWNER_PREAD:
    return ompfileDirtyOwnerPread(std::move(RequestManager));
  case OMPFILE_DIRTY_OWNER_PREAD_BATCH:
    return ompfileDirtyOwnerPreadBatch(std::move(RequestManager));
  case OMPFILE_DIRTY_OWNER_QUERY:
    return ompfileDirtyOwnerQuery(std::move(RequestManager));
  case OMPFILE_PWRITE:
    return ompfilePwrite(std::move(RequestManager));
  case OMPFILE_FRESHNESS_QUERY:
    return ompfileFreshnessQuery(std::move(RequestManager));
  case OMPFILE_FRESHNESS_WRITE_COMMIT:
    return ompfileFreshnessWriteCommit(std::move(RequestManager));
  case OMPFILE_FLUSH_DIRTY_TILE:
    return ompfileFlushDirtyTile(std::move(RequestManager));
  case OMPFILE_SCHED_REQUEST:
    return ompfileSchedRequest(std::move(RequestManager));
  case OMPFILE_SCHED_REQUEST_BATCH:
    return ompfileSchedBatchRequest(std::move(RequestManager));
  default:
    return EventTy{};
  }
}
