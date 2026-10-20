//===-- ProxyOmpFileStage.cpp - Proxy stage and source I/O ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The per-proxy stage (readthrough, write-through and write-back), its
// dirty-range bookkeeping and local freshness memory, the headnode-guarded
// local reads, and the timed source pread/pwrite/fdatasync helpers.
//
//===----------------------------------------------------------------------===//

#include "ProxyOmpFile.h"

void ProxyOmpFile::rememberWritebackFreshnessCommit(uint64_t PathKey,
                                                    uint64_t Version) {
  if (PathKey == 0 || Version == 0)
    return;
  std::lock_guard<std::mutex> Lock(OmpFileWritebackCommitMutex);
  auto &Pending = OmpFilePendingWritebackCommits[PathKey];
  ++Pending.Count;
  Pending.LastVersion = Version;
  OmpFileLocalFreshnessVersionByPath[PathKey] = Version;
}

bool ProxyOmpFile::consumeDuplicateWriteThroughFreshnessCommit(
    uint64_t PathKey, uint64_t &VersionOut) {
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
uint64_t ProxyOmpFile::lastLocalFreshnessVersionForPathKey(uint64_t PathKey) {
  if (PathKey == 0)
    return 0;
  std::lock_guard<std::mutex> Lock(OmpFileWritebackCommitMutex);
  auto It = OmpFileLocalFreshnessVersionByPath.find(PathKey);
  if (It == OmpFileLocalFreshnessVersionByPath.end())
    return 0;
  return It->second;
}

uint64_t ProxyOmpFile::configuredStageTileBytes() const {
  return OmpFileConfig.StageTileBytes;
}

bool ProxyOmpFile::dirtyOwnerExpectedVersionMatchesRange(
    const std::string &SourcePath, uint64_t Begin, uint64_t End,
    uint64_t ExpectedVersion, const OmpFileStageEntry &Entry) {
  if (ExpectedVersion == 0)
    return true;

  const uint64_t PathKey = ompfile::keys::pathKey(SourcePath);
  const uint64_t TileBytes = configuredStageTileBytes();
  if (PathKey != 0 && TileBytes != 0 && End > Begin &&
      Begin % TileBytes == 0 && End == Begin + TileBytes) {
    const uint64_t TileId = Begin / TileBytes;
    const uint64_t CompositeKey = ompfile::keys::tileKey(PathKey, TileId);
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

bool ProxyOmpFile::completeDerivedTileFlushesOnHeadnode(
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
      const uint64_t CompositeKey = ompfile::keys::tileKey(PathKey, TileId);
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
bool ProxyOmpFile::stageFreshnessGuardPermitsLocal(
    const std::string &SourcePath,
    const std::shared_ptr<OmpFileStageEntry> &StageEntry) {
  if (!isWritebackStageEnabled() || !OmpFileStageFreshnessGuard)
    return true;
  if (SourcePath.empty())
    return true;
  const uint64_t PathKey = ompfile::keys::pathKey(SourcePath);
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

// --- Write-back dirty-region helpers ---
// All mark/clear helpers require the caller to hold Entry->Mutex; the
// "Locked" suffix mirrors invalidateStageEntryLocked.
// flushDirtyRangesToSource is the real stage->source flush and is wired
// into closeWithOptionalCache; it is a no-op unless DirtyExtents is
// nonempty, so the write-through default is behavior-neutral.

// Record [Begin, End) as dirty in Entry, merging into DirtyExtents and
// advancing DirtyBytes. Returns the new total dirty byte count.
uint64_t ProxyOmpFile::markDirtyRangeLocked(OmpFileStageEntry &Entry,
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
uint64_t ProxyOmpFile::clearDirtyRangeLocked(OmpFileStageEntry &Entry,
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

// File-scoped counterpart to flushLocalDirtyTileToSource below: push every
// dirty stage range of this file to the source, with no freshness-version
// gate and no tile matching.
//
// The tile-scoped variant exists for the close-time freshness protocol,
// which can only act once a write has been freshness-committed and a version
// assigned. A commit issued mid-wave has no version yet, so that path
// rejects it with EINVAL before it ever looks at the stage - which is what
// made omp_file_commit unusable under write-back staging.
bool ProxyOmpFile::flushLocalDirtyStageForPathKey(uint64_t BasePathKey) {
  if (BasePathKey == 0) {
    errno = EINVAL;
    return false;
  }

  // A commit is a durability point under every fsync policy. With "close"
  // the write-through path left its fdatasync for the logical close, so
  // drain it here for this file, or a commit would return before the bytes
  // it promises are durable.
  if (!syncDeferredWritesForPathKey(BasePathKey))
    return false;

  std::vector<std::shared_ptr<OmpFileStageEntry>> Entries;
  {
    const std::lock_guard<std::mutex> Lock(OmpFileStageMutex);
    Entries.reserve(OmpFileStageEntries.size());
    for (auto &It : OmpFileStageEntries)
      Entries.push_back(It.second);
  }

  bool AllOk = true;
  for (const std::shared_ptr<OmpFileStageEntry> &Entry : Entries) {
    if (!Entry || Entry->SourcePath.empty())
      continue;
    if (ompfile::keys::pathKey(Entry->SourcePath) !=
        BasePathKey)
      continue;

    // Checked without holding the entry lock across the flush itself, which
    // takes that lock of its own.
    bool HasDirty = false;
    {
      const std::lock_guard<std::mutex> EntryLock(Entry->Mutex);
      for (const OmpFileStageExtent &Extent : Entry->DirtyExtents) {
        if (Extent.End > Extent.Begin) {
          HasDirty = true;
          break;
        }
      }
    }
    if (!HasDirty)
      continue; // nothing staged for this file; not an error

    const int SourceFd = ::open(Entry->SourcePath.c_str(), O_RDWR);
    if (SourceFd < 0) {
      AllOk = false;
      continue;
    }
    if (!flushDirtyRangesToSource(*Entry, SourceFd, Entry->SourcePath))
      AllOk = false;
    ::close(SourceFd);
  }

  return AllOk;
}

// fdatasync the write-through fds that policy "close" left unsynced for one
// file, and the owner-bypass write fd for it. Each fd leaves the unsynced set
// only once its sync succeeded, so a failed commit is retried by the close.
bool ProxyOmpFile::syncDeferredWritesForPathKey(uint64_t BasePathKey) {
  if (writethroughFsyncEachEnabled())
    return true;
  std::vector<int> Fds;
  {
    const std::lock_guard<std::mutex> UnsyncedLock(
        OmpFileUnsyncedWriteFdMutex);
    Fds.assign(OmpFileUnsyncedWriteFds.begin(),
               OmpFileUnsyncedWriteFds.end());
  }
  for (int Fd : Fds) {
    std::string Path;
    if (!getTrackedOmpFileFdPath(Fd, Path) ||
        ompfile::keys::pathKey(Path) != BasePathKey)
      continue;
    if (!fdatasyncSourceTimed(Fd))
      return false;
    const std::lock_guard<std::mutex> UnsyncedLock(
        OmpFileUnsyncedWriteFdMutex);
    OmpFileUnsyncedWriteFds.erase(Fd);
  }
  if (OmpFileLocalDisjointWrite) {
    if (!flushLocalWriteBufs())
      return false;
    const std::lock_guard<std::mutex> Lock(OmpFileLocalWriteFdMutex);
    for (auto &KV : OmpFileLocalWriteFds)
      if (KV.second >= 0 &&
          ompfile::keys::pathKey(KV.first) ==
              BasePathKey &&
          ::fdatasync(KV.second) != 0)
        return false;
  }
  return true;
}

bool ProxyOmpFile::flushLocalDirtyTileToSource(uint64_t TilePathKey,
                                               uint64_t Version) {
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
    const uint64_t BasePathKey = ompfile::keys::pathKey(Entry->SourcePath);
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
          if (ompfile::keys::tileKey(BasePathKey, TileId) != TilePathKey)
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
bool ProxyOmpFile::flushDirtyRangesToSource(OmpFileStageEntry &Entry,
                                            int SourceFd,
                                            const std::string &SourcePath,
                                            bool DeferCloseMetadataCompletion) {
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

  const uint64_t PathKey = ompfile::keys::pathKey(SourcePath);
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
    uint64_t FlushStageReadUs = 0;
    std::vector<char> Buffer;
    for (const OmpFileStageExtent &Extent : Extents) {
      assert(Extent.End > Extent.Begin && "empty dirty extent in vector");
      const uint64_t Len = Extent.End - Extent.Begin;
      if (Buffer.size() < Len)
        Buffer.resize(Len);
      uint64_t ReadTotal = 0;
      while (ReadTotal < Len) {
        const auto ReadStart = std::chrono::steady_clock::now();
        const ssize_t N = ::pread(Entry.StageFd, Buffer.data() + ReadTotal,
                                  static_cast<size_t>(Len - ReadTotal),
                                  static_cast<off_t>(Extent.Begin + ReadTotal));
        FlushStageReadUs +=
            elapsedMicros(ReadStart, std::chrono::steady_clock::now());
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
      uint64_t ExtentWriteUs = 0;
      while (WriteTotal < Len) {
        const auto WriteStart = std::chrono::steady_clock::now();
        const ssize_t N = ::pwrite(
            FlushSourceFd, Buffer.data() + WriteTotal,
            static_cast<size_t>(Len - WriteTotal),
            static_cast<off_t>(Extent.Begin + WriteTotal));
        ExtentWriteUs +=
            elapsedMicros(WriteStart, std::chrono::steady_clock::now());
        if (N <= 0) {
          const int WriteErrno = N < 0 ? errno : EIO;
          // The write-back flush writes the source too; without this the
          // staged arm reports no source write time at all and the two arms
          // are not comparable term by term.
          recordSourcePwrite(Len, WriteTotal, ExtentWriteUs);
          closeFlushFd();
          invalidateStageForPath(SourcePath);
          return failDirtyFlush("pwrite-source", WriteErrno, Extent.Begin,
                                Extent.End, Epoch);
        }
        WriteTotal += static_cast<uint64_t>(N);
      }
      recordSourcePwrite(Len, WriteTotal, ExtentWriteUs);
      FlushedBytes += Len;
    }

    if (!fdatasyncSourceTimed(FlushSourceFd)) {
      const int SyncErrno = errno;
      closeFlushFd();
      invalidateStageForPath(SourcePath);
      return failDirtyFlush("fdatasync-source", SyncErrno, 0, 0, Epoch);
    }

    bool CompletedViaDerivedTileKeys = false;
    if (Epoch > 0 && !SourcePath.empty()) {
      // The same distributed write-back predicate libompfile routes by:
      // DISTRIBUTE_WRITES=1 and DISTRIBUTE_RANKS > 1.
      const uint64_t TileBytes = configuredStageTileBytes();
      if (OmpFileConfig.Shared.WritebackDistributeActive && TileBytes > 0) {
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
    // The whole flush window used to be charged to stage_write_us_total,
    // which made that counter report stage device writes plus stage reads
    // plus source writes plus the headnode completion round trip under one
    // name. Keep it as its own term; its components are now separable as
    // window - flush_read - the source counters above.
    OmpFileStatsStageDirtyFlushReadUs.fetch_add(FlushStageReadUs,
                                                std::memory_order_relaxed);
    OmpFileStatsStageDirtyFlushUs.fetch_add(
        elapsedMicros(FlushStart, std::chrono::steady_clock::now()),
        std::memory_order_relaxed);
    closeFlushFd();
    return true;
  }

  closeFlushFd();
  return failDirtyFlush("stale-retry-exhausted", ESTALE);
}

bool ProxyOmpFile::shouldInvalidateStageOnOpen(int Flags) const {
  if (Flags & (O_CREAT | O_EXCL | O_TRUNC))
    return true;
  const int AccessMode = Flags & O_ACCMODE;
  return AccessMode == O_WRONLY || AccessMode == O_RDWR;
}

std::string ProxyOmpFile::makeStageFilePath(
    const std::string &SourcePath) const {
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

bool ProxyOmpFile::ensureDirectoryTree(const std::string &Path,
                                       int &ErrnoOut) const {
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

bool ProxyOmpFile::getStageEntryForPath(
    const std::string &SourcePath,
    std::shared_ptr<OmpFileStageEntry> &EntryOut) {
  const std::lock_guard<std::mutex> Lock(OmpFileStageMutex);
  auto It = OmpFileStageEntries.find(SourcePath);
  if (It == OmpFileStageEntries.end())
    return false;
  EntryOut = It->second;
  return true;
}

bool ProxyOmpFile::hasStageFreeSpace(int &ErrnoOut) const {
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

bool ProxyOmpFile::populateStageRange(
    const std::string &SourcePath, int StageFd, uint64_t Offset, uint64_t Size,
    const std::vector<OmpFileStageExtent> &SkipExtents,
    uint64_t &PopulateBeginOut, uint64_t &PopulateEndOut,
    uint64_t &SourceSizeOut, OmpFileStagePopulateStats &Stats,
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

void ProxyOmpFile::invalidateStageEntryLocked(const std::string &SourcePath,
                                              bool CountInvalidation,
                                              uint64_t InvalidatedBytes,
                                              bool FullInvalidation) {
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

void ProxyOmpFile::invalidateStageForPath(const std::string &SourcePath) {
  if (SourcePath.empty())
    return;
  const std::lock_guard<std::mutex> Lock(OmpFileStageMutex);
  invalidateStageEntryLocked(SourcePath, /*CountInvalidation=*/true);
}

void ProxyOmpFile::invalidateStageRangeForPath(const std::string &SourcePath,
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

bool ProxyOmpFile::ensureStageEntryForPath(
    const std::string &SourcePath, uint64_t Offset, uint64_t Size,
    std::shared_ptr<OmpFileStageEntry> *EntryOut, int &StageFd, int &ErrnoOut) {
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

bool ProxyOmpFile::preadWithOptionalStage(int Fd, int64_t Offset, void *Buffer,
                                          uint64_t Size, uint64_t *BytesReadOut,
                                          int &ErrnoOut, bool AllowStage) {
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

  uint64_t Bytes = 0;
  uint64_t PreadUs = 0;
  if (!preadFully(ReadFd, Offset, Buffer, Size, &Bytes, &PreadUs, ErrnoOut))
    return false;

  if (BytesReadOut)
    *BytesReadOut = Bytes;
  if (!AllowStage && Bytes > 0) {
    OmpFileStatsStageBypassReads.fetch_add(1, std::memory_order_relaxed);
    OmpFileStatsStageBypassBytes.fetch_add(Bytes, std::memory_order_relaxed);
  }
  if (ReadFd != Fd && Bytes > 0) {
    OmpFileStatsStagedReadHits.fetch_add(1, std::memory_order_relaxed);
    OmpFileStatsStagedReadBytes.fetch_add(Bytes, std::memory_order_relaxed);
    OmpFileStatsStagedPreadUs.fetch_add(PreadUs, std::memory_order_relaxed);
  } else if (Bytes > 0) {
    OmpFileStatsSourcePreadBytes.fetch_add(Bytes, std::memory_order_relaxed);
    OmpFileStatsSourcePreadUs.fetch_add(PreadUs, std::memory_order_relaxed);
  }
  return true;
}

bool ProxyOmpFile::preadDirtyOwnerStageOnly(int Fd, int64_t Offset,
                                            void *Buffer, uint64_t Size,
                                            uint64_t ExpectedVersion,
                                            uint64_t *BytesReadOut,
                                            int &ErrnoOut) {
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
  uint64_t Bytes = 0;
  if (!preadFully(StageFd, Offset, Buffer, Size, &Bytes,
                  /*ElapsedUsOut=*/nullptr, ErrnoOut))
    return false;
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
bool ProxyOmpFile::queryDirtyOwnerStageRangeOnly(int Fd, int64_t Offset,
                                                 uint64_t Size,
                                                 uint64_t ExpectedVersion,
                                                 int &StateOut, int &ErrnoOut) {
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

// A single ::pread transfers at most 2,147,479,552 bytes (2 GiB minus one
// page) on Linux, so any request above that returns short without being an
// error or EOF. Loop until the request is satisfied; a zero return is real
// EOF and legitimately ends the read early (the caller sees the partial
// count). Repro: 7.6 GB/rank HACC checkpoints, AMD job 370693.
bool ProxyOmpFile::preadFully(int Fd, int64_t Offset, void *Buffer,
                              uint64_t Size, uint64_t *BytesReadOut,
                              uint64_t *ElapsedUsOut, int &ErrnoOut) {
  ompfile::trace::Scope Trace(ompfile::trace::Domain::MPP, "fs-pread",
                              ompfile::trace::Color::Syscall, Size);
  ErrnoOut = 0;
  if (BytesReadOut)
    *BytesReadOut = 0;
  if (ElapsedUsOut)
    *ElapsedUsOut = 0;

  uint64_t TotalRead = 0;
  while (TotalRead < Size) {
    const auto ReadStart = std::chrono::steady_clock::now();
    const ssize_t BytesRead =
        ::pread(Fd, static_cast<char *>(Buffer) + TotalRead,
                static_cast<size_t>(Size - TotalRead),
                static_cast<off_t>(Offset + static_cast<int64_t>(TotalRead)));
    const auto ReadEnd = std::chrono::steady_clock::now();
    if (ElapsedUsOut)
      *ElapsedUsOut += elapsedMicros(ReadStart, ReadEnd);
    if (BytesRead < 0) {
      ErrnoOut = errno;
      if (BytesReadOut)
        *BytesReadOut = TotalRead;
      return false;
    }
    if (BytesRead == 0)
      break;
    TotalRead += static_cast<uint64_t>(BytesRead);
  }

  if (BytesReadOut)
    *BytesReadOut = TotalRead;
  return true;
}

// The stage-side durability barrier, kept out of OmpFileStatsStageWriteUs so
// that counter means the stage device pwrite and nothing else.
void ProxyOmpFile::recordStageWriteFsync(
    std::chrono::steady_clock::time_point Start) {
  OmpFileStatsStageWriteFsyncOps.fetch_add(1, std::memory_order_relaxed);
  OmpFileStatsStageWriteFsyncUs.fetch_add(
      elapsedMicros(Start, std::chrono::steady_clock::now()),
      std::memory_order_relaxed);
}

// Source-side write accounting. The read path has had
// `source_pread_us_total` since the staging work landed; the write path had
// no timer at all, so the ~31% of stage-off time it costs could only be
// inferred by solving the two arm equations. Device service and the
// durability barrier are recorded separately because the cost model charges
// them as distinct terms. Zero-length requests issue no syscall and are
// skipped, so the op count stays usable as a per-op divisor.
void ProxyOmpFile::recordSourcePwrite(uint64_t RequestedBytes,
                                      uint64_t BytesWritten,
                                      uint64_t ElapsedUs) {
  if (RequestedBytes == 0)
    return;
  OmpFileStatsSourcePwriteOps.fetch_add(1, std::memory_order_relaxed);
  OmpFileStatsSourcePwriteBytes.fetch_add(BytesWritten,
                                          std::memory_order_relaxed);
  OmpFileStatsSourcePwriteUs.fetch_add(ElapsedUs, std::memory_order_relaxed);
}

// fdatasync on an authoritative (source) fd, timed. Returns false with errno
// preserved, exactly like ::fdatasync, so call sites keep their error paths.
bool ProxyOmpFile::fdatasyncSourceTimed(int Fd) {
  ompfile::trace::Scope Trace(ompfile::trace::Domain::MPP, "fs-fdatasync",
                              ompfile::trace::Color::Syscall);
  const auto Start = std::chrono::steady_clock::now();
  const int Ret = ::fdatasync(Fd);
  const int SyncErrno = Ret != 0 ? errno : 0;
  OmpFileStatsSourceFsyncOps.fetch_add(1, std::memory_order_relaxed);
  OmpFileStatsSourceFsyncUs.fetch_add(
      elapsedMicros(Start, std::chrono::steady_clock::now()),
      std::memory_order_relaxed);
  if (Ret != 0) {
    errno = SyncErrno;
    return false;
  }
  return true;
}

bool ProxyOmpFile::pwriteFully(int Fd, int64_t Offset, const void *Buffer,
                               uint64_t Size, uint64_t *BytesWrittenOut,
                               uint64_t *ElapsedUsOut, int &ErrnoOut) {
  ompfile::trace::Scope Trace(ompfile::trace::Domain::MPP, "fs-pwrite",
                              ompfile::trace::Color::Syscall, Size);
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

void ProxyOmpFile::updateStageCoverageForWrite(
    const std::shared_ptr<OmpFileStageEntry> &Entry, uint64_t Offset,
    uint64_t Size, uint64_t SourceSize, bool SourceSizeKnown) {
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

bool ProxyOmpFile::pwriteWithOptionalStage(int Fd, int64_t Offset,
                                           const void *Buffer, uint64_t Size,
                                           uint64_t *BytesWrittenOut,
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
    const bool SourceWriteOk =
        pwriteFully(Fd, Offset, Buffer, Size, &SourceBytesWritten,
                    &SourceWriteUs, ErrnoOut);
    // Recorded before the failure branch: a partial write still consumed
    // device time, and dropping it would bias the fitted constant downward
    // exactly on the runs that went wrong.
    recordSourcePwrite(Size, SourceBytesWritten, SourceWriteUs);
    if (!SourceWriteOk) {
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
        if (!fdatasyncSourceTimed(Fd)) {
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
        // Charged to the stage fsync counter rather than folded into
        // StageWriteUs: the durability barrier is its own model term, and
        // hiding it here is what made stage_write_us_total read as ~35 ms
        // for a 1 MiB local write.
        recordStageWriteFsync(FsyncStart);
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

    const uint64_t PathKey = ompfile::keys::pathKey(SourcePath);
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
        const uint64_t CompositeKey = ompfile::keys::tileKey(PathKey, TileId);
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
      // LIBOMPFILE_STAGE_FLUSH_COALESCE=1: one watermark flush per entry
      // at a time. flushDirtyRangesToSource flushes every dirty range of
      // the entry, so when several writers cross the watermark together
      // (concurrent kernels on one proxy) the second and later flushes
      // re-read the same ranges and queue behind the source fdatasync for
      // nothing: sorgan 91 shows 30 flushes and 9.0 s of flush time at
      // P=8 with four handler threads where one thread needs 15 and
      // 2.0 s. A writer that finds a flush in progress leaves its bytes
      // dirty for the next watermark crossing or for close, which flushes
      // everything, so durability at close is unchanged.
      bool RunFlush = true;
      if (OmpFileStageFlushCoalesce) {
        std::lock_guard<std::mutex> EntryLock(HeldStageEntry->Mutex);
        if (HeldStageEntry->FlushInProgress) {
          RunFlush = false;
          OmpFileStatsStageDirtyFlushCoalesced.fetch_add(
              1, std::memory_order_relaxed);
        } else {
          HeldStageEntry->FlushInProgress = true;
        }
      }
      if (RunFlush) {
        const bool FlushOk =
            flushDirtyRangesToSource(*HeldStageEntry, Fd, SourcePath);
        if (OmpFileStageFlushCoalesce) {
          std::lock_guard<std::mutex> EntryLock(HeldStageEntry->Mutex);
          HeldStageEntry->FlushInProgress = false;
        }
        if (!FlushOk) {
          ErrnoOut = EIO;
          return false;
        }
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
      OmpFileStatsStageWriteUs.fetch_add(StageWriteUs,
                                         std::memory_order_relaxed);
      recordStageWriteFsync(FsyncStart);
      invalidateStageForPath(SourcePath);
      return true;
    }
    recordStageWriteFsync(FsyncStart);
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

void ProxyOmpFile::drainOmpFileStageCache() {
  const std::lock_guard<std::mutex> Lock(OmpFileStageMutex);
  for (auto &It : OmpFileStageEntries) {
    if (!It.second->StagePath.empty()) {
      (void)::unlink(It.second->StagePath.c_str());
      It.second->StagePath.clear();
    }
  }
  OmpFileStageEntries.clear();
}
