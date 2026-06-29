//===---- OmpFileHeadnodeManager.cpp - Global OMPFile scheduler -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OmpFileHeadnodeManager.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <limits>
#include <string>
#include <string_view>

OmpFileHeadnodeManager &OmpFileHeadnodeManager::instance() {
  static OmpFileHeadnodeManager Manager;
  return Manager;
}

uint64_t OmpFileHeadnodeManager::computePathKey(std::string_view Path) {
  constexpr uint64_t OffsetBasis = 1469598103934665603ULL;
  constexpr uint64_t Prime = 1099511628211ULL;

  uint64_t Hash = OffsetBasis;
  for (const unsigned char C : Path) {
    Hash ^= static_cast<uint64_t>(C);
    Hash *= Prime;
  }
  return Hash;
}

uint64_t OmpFileHeadnodeManager::parseUint64Env(const char *Name,
                                                 uint64_t DefaultValue) {
  const char *Env = std::getenv(Name);
  if (!Env)
    return DefaultValue;

  errno = 0;
  char *End = nullptr;
  const unsigned long long Value = std::strtoull(Env, &End, 10);
  if (errno != 0 || End == Env || (End && *End != '\0'))
    return DefaultValue;

  return static_cast<uint64_t>(Value);
}

bool OmpFileHeadnodeManager::parseBoolEnv(const char *Name,
                                          bool DefaultValue) {
  const char *Env = std::getenv(Name);
  if (!Env)
    return DefaultValue;

  if (Env[0] == '\0')
    return false;
  if (Env[0] == '0' && Env[1] == '\0')
    return false;
  return true;
}

bool OmpFileHeadnodeManager::isFreshnessTraceEnabled() const {
  return FreshnessTraceEnabled;
}

void OmpFileHeadnodeManager::emitFreshnessTrace(
    const char *Event, const std::string &PathKey, int Rank, int SrcRank,
    int DstRank, uint64_t Version, uint64_t LatestVersion,
    const char *Decision, const char *Result, const char *Reason,
    const char *TileId) const {
  if (!isFreshnessTraceEnabled())
    return;

  std::fprintf(stderr,
               "freshness_trace event=%s path_key=%s rank=%d src_rank=%d "
               "dst_rank=%d version=%llu latest_version=%llu decision=%s "
               "result=%s reason=%s tile_id=%s\n",
               Event ? Event : "unknown", PathKey.c_str(), Rank, SrcRank,
               DstRank, static_cast<unsigned long long>(Version),
               static_cast<unsigned long long>(LatestVersion),
               Decision ? Decision : "na", Result ? Result : "na",
               Reason ? Reason : "na", TileId ? TileId : "na");
}

void OmpFileHeadnodeManager::freshnessTraceLogUnlocked(const char *Format,
                                                       ...) const {
  if (!isFreshnessTraceEnabled())
    return;

  char Buffer[512] = {0};
  va_list Args;
  va_start(Args, Format);
  std::vsnprintf(Buffer, sizeof(Buffer), Format, Args);
  va_end(Args);

  emitFreshnessTrace("legacy", "na", -1, -1, -1, 0, 0, "na", "na",
                     Buffer, "na");
}

bool OmpFileHeadnodeManager::segmentRangesOverlap(int64_t ReadOffset,
                                                  uint64_t ReadSize,
                                                  int64_t WriteOffset,
                                                  uint64_t WriteSize) const {
  if (ReadSize == 0 || WriteSize == 0)
    return false;
  if (ReadOffset < 0 || WriteOffset < 0)
    return true;

  const uint64_t ReadStart = static_cast<uint64_t>(ReadOffset);
  const uint64_t WriteStart = static_cast<uint64_t>(WriteOffset);
  if (ReadStart > std::numeric_limits<uint64_t>::max() - ReadSize)
    return true;
  if (WriteStart > std::numeric_limits<uint64_t>::max() - WriteSize)
    return true;

  const uint64_t ReadEnd = ReadStart + ReadSize;
  const uint64_t WriteEnd = WriteStart + WriteSize;
  return ReadStart < WriteEnd && WriteStart < ReadEnd;
}

uint64_t OmpFileHeadnodeManager::nextWriteSequenceUnlocked() {
  return NextWriteSequence++;
}

void OmpFileHeadnodeManager::recordWriteVersionUnlocked(
    uint64_t PathKey, const OmpFileIORequest &Request) {
  if (PathKey == 0 || Request.Size == 0)
    return;

  WriteVersionEntry Entry{};
  Entry.Start = Request.Offset;
  Entry.Size = Request.Size;
  Entry.Sequence = nextWriteSequenceUnlocked();
  Entry.HasEpoch = (Request.HintFlags & OMPFILE_IO_HINT_HAS_EPOCH) != 0;
  Entry.EpochId = Entry.HasEpoch ? Request.EpochId : 0;
  Entry.HasTile = (Request.HintFlags & OMPFILE_IO_HINT_HAS_TILE) != 0;
  Entry.TileId = Entry.HasTile ? Request.TileId : 0;

  auto &Versions = PathWriteVersions[PathKey];
  Versions.push_back(Entry);
  constexpr size_t MaxHistoryPerPath = 8192;
  if (Versions.size() > MaxHistoryPerPath)
    Versions.erase(Versions.begin(), Versions.begin() + (Versions.size() - MaxHistoryPerPath));
}

bool OmpFileHeadnodeManager::hasRebalanceConflictUnlocked(
    const OmpFileIOBatchSegment &Segment, const char *&Reason,
    int &ReasonErrno) const {
  Reason = "none";
  ReasonErrno = 0;

  if (Segment.PathKey == 0) {
    Reason = "missing-path-key";
    ReasonErrno = ENOKEY;
    return true;
  }

  auto It = PathWriteVersions.find(Segment.PathKey);
  if (It == PathWriteVersions.end() || It->second.empty())
    return false;

  const bool ReadHasEpoch = (Segment.SegmentFlags & OMPFILE_IO_HINT_HAS_EPOCH) != 0;
  const bool ReadHasTile = (Segment.SegmentFlags & OMPFILE_IO_HINT_HAS_TILE) != 0;
  const uint64_t ReadEpoch = Segment.EpochId;

  bool SawOverlap = false;
  bool SawMissingWriteEpoch = false;
  bool SawNewerWriteEpoch = false;
  for (const WriteVersionEntry &Write : It->second) {
    if (!segmentRangesOverlap(Segment.Offset, Segment.Size, Write.Start,
                              Write.Size)) {
      continue;
    }

    if (ReadHasTile && Write.HasTile && Segment.TileId != Write.TileId)
      continue;

    SawOverlap = true;
    if (!Write.HasEpoch) {
      SawMissingWriteEpoch = true;
      continue;
    }
    if (ReadHasEpoch && Write.EpochId > ReadEpoch)
      SawNewerWriteEpoch = true;
  }

  if (SawOverlap) {
    if (!ReadHasEpoch) {
      Reason = "missing-read-epoch";
      ReasonErrno = ENOKEY;
      return true;
    }
    if (SawMissingWriteEpoch) {
      Reason = "missing-write-epoch";
      ReasonErrno = ENOKEY;
      return true;
    }
    if (SawNewerWriteEpoch) {
      Reason = "newer-write-epoch";
      ReasonErrno = ESTALE;
      return true;
    }
    Reason = "unsafe-write-overlap";
    ReasonErrno = EAGAIN;
    return true;
  }

  return false;
}

void OmpFileHeadnodeManager::initialize(int NewWorldSize, int NewHeadnodeRank) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  if (Initialized)
    return;

  WorldSize = std::max(NewWorldSize, 1);
  HeadnodeRank = NewHeadnodeRank;
  MaxAffinityLoadSkew = parseUint64Env(
      "LIBOMPFILE_SCHED_MAX_AFFINITY_LOAD_SKEW",
      std::numeric_limits<uint64_t>::max());
  BatchStatsReportEvery =
      std::max<uint64_t>(1, parseUint64Env("LIBOMPFILE_SCHED_STATS_REPORT_EVERY",
                                           128));
  UsePlannedLoadForSkew =
      parseUint64Env("LIBOMPFILE_SCHED_USE_PLANNED_LOAD_FOR_SKEW", 0) != 0;
  SpreadSamePathOpens =
      parseUint64Env("LIBOMPFILE_MPP_OPEN_SPREAD", 0) != 0;
  FreshnessTraceEnabled = parseBoolEnv("LIBOMPFILE_FRESHNESS_TRACE", false);
  ensureHandlersUnlocked();
  Initialized = true;
}

void OmpFileHeadnodeManager::resetForTesting() {
  const std::lock_guard<std::mutex> Lock(Mutex);
  Initialized = false;
  WorldSize = 1;
  HeadnodeRank = 0;
  MaxAffinityLoadSkew = 2;
  BatchStatsReportEvery = 128;
  UsePlannedLoadForSkew = false;
  SpreadSamePathOpens = false;
  FreshnessTraceEnabled = false;
  FlightplanTable.clear();
  GlobalFileTable.clear();
  PathAffinityTable.clear();
  PathWriteVersions.clear();
  TileFreshnessTable.clear();
  PendingTileCopies.clear();
  Handlers.clear();
  Stats = {};
  NextGlobalFileId = 1;
  NextHandlerTieBreaker = 0;
  NextWriteSequence = 1;
}

bool OmpFileHeadnodeManager::commitTileFreshnessWriteForTesting(
    uint64_t PathKey, int WriterRank, bool WriteThroughMode,
    uint64_t &CommittedVersionOut) {
  return commitTileFreshnessWrite(PathKey, WriterRank, WriteThroughMode,
                                  CommittedVersionOut);
}

bool OmpFileHeadnodeManager::commitTileFreshnessWrite(
    uint64_t PathKey, int WriterRank, bool WriteThroughMode,
    uint64_t &CommittedVersionOut) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  return commitTileFreshnessWriteUnlocked(PathKey, WriterRank, WriteThroughMode,
                                          CommittedVersionOut);
}

bool OmpFileHeadnodeManager::markTileFresh(uint64_t PathKey, int Rank,
                                           uint64_t Version) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  return markTileFreshUnlocked(PathKey, Rank, Version);
}

bool OmpFileHeadnodeManager::registerTileCopy(uint64_t PathKey, int SourceRank,
                                              int DestRank, uint64_t Version) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  return registerTileCopyUnlocked(PathKey, SourceRank, DestRank, Version);
}

bool OmpFileHeadnodeManager::markTileFreshFromCopy(uint64_t PathKey, int Rank,
                                                   uint64_t Version) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  return markTileFreshFromCopyUnlocked(PathKey, Rank, Version);
}

bool OmpFileHeadnodeManager::flushDirtyTile(uint64_t PathKey,
                                            int &SourceRankOut,
                                            uint64_t &FlushedVersionOut) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  return flushDirtyTileUnlocked(PathKey, SourceRankOut, FlushedVersionOut);
}

bool OmpFileHeadnodeManager::completeDirtyFlush(uint64_t PathKey,
                                                int SourceRank,
                                                uint64_t Version,
                                                bool Success) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  return completeDirtyFlushUnlocked(PathKey, SourceRank, Version, Success);
}

bool OmpFileHeadnodeManager::markTileFreshForTesting(uint64_t PathKey, int Rank,
                                                     uint64_t Version) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  return markTileFreshUnlocked(PathKey, Rank, Version);
}

bool OmpFileHeadnodeManager::evaluateTileFreshnessQueryForTesting(
    uint64_t PathKey, uint64_t LocalVersion, int RequesterRank,
    OmpFileFreshnessDecision &DecisionOut, int &SourceRankOut,
    uint64_t &SelectedVersionOut) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  return evaluateTileFreshnessQueryUnlocked(PathKey, LocalVersion, RequesterRank,
                                            DecisionOut, SourceRankOut,
                                             SelectedVersionOut);
}

bool OmpFileHeadnodeManager::handleFreshnessQueryRequest(
    const OmpFileFreshnessQueryRequest &Request,
    OmpFileFreshnessQueryReply &Reply) {
  Reply = {};
  Reply.AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
  if (Request.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
    Reply.Status = -1;
    Reply.Errno = EPROTO;
    return true;
  }

  OmpFileFreshnessDecision Decision = OmpFileFreshnessDecision::WAIT_OR_FAIL;
  int SourceRank = -1;
  uint64_t SelectedVersion = 0;
  if (!evaluateTileFreshnessQueryForTesting(Request.PathKey, Request.LocalVersion,
                                            Request.RequesterRank, Decision,
                                            SourceRank, SelectedVersion)) {
    Reply.Status = -1;
    Reply.Errno = errno != 0 ? errno : EIO;
    return true;
  }

  Reply.Decision = static_cast<uint32_t>(Decision);
  Reply.SourceRank = SourceRank;
  Reply.SelectedVersion = SelectedVersion;
  Reply.Status = 0;
  Reply.Errno = 0;
  return true;
}

bool OmpFileHeadnodeManager::clearTileFreshRanksForTesting(uint64_t PathKey) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  if (PathKey == 0) {
    errno = EINVAL;
    return false;
  }
  auto It = TileFreshnessTable.find(PathKey);
  if (It == TileFreshnessTable.end()) {
    errno = ENOKEY;
    return false;
  }
  It->second.FreshRanks.clear();
  return true;
}

bool OmpFileHeadnodeManager::commitTileFreshnessWriteUnlocked(
    uint64_t PathKey, int WriterRank, bool WriteThroughMode,
    uint64_t &CommittedVersionOut) {
  const std::string PathKeyStr = std::to_string(PathKey);
  CommittedVersionOut = 0;
  if (PathKey == 0 || !isWorkerRankUnlocked(WriterRank)) {
    emitFreshnessTrace("write_commit", PathKeyStr, WriterRank, WriterRank,
                       WriterRank, 0, 0, "", "fail", "invalid-args", "na");
    errno = EINVAL;
    return false;
  }

  auto &Entry = TileFreshnessTable[PathKey];
  if (Entry.LatestVersion == std::numeric_limits<uint64_t>::max()) {
    emitFreshnessTrace("write_commit", PathKeyStr, WriterRank, WriterRank,
                       WriterRank, Entry.LatestVersion, Entry.LatestVersion, "",
                       "fail", "version-overflow", "na");
    errno = EOVERFLOW;
    return false;
  }
  ++Entry.LatestVersion;
  auto PendingIt = PendingTileCopies.find(PathKey);
  if (PendingIt != PendingTileCopies.end()) {
    auto &Pending = PendingIt->second;
    Pending.erase(std::remove_if(Pending.begin(), Pending.end(),
                                 [&](const PendingTileCopy &Copy) {
                                   return Copy.Version < Entry.LatestVersion;
                                 }),
                  Pending.end());
    if (Pending.empty())
      PendingTileCopies.erase(PendingIt);
  }
  Entry.FreshRanks.clear();
  Entry.FreshRanks.insert(WriterRank);
  if (WriteThroughMode) {
    Entry.PfsVersion = Entry.LatestVersion;
    Entry.Dirty = false;
  } else {
    Entry.Dirty = true;
  }
  CommittedVersionOut = Entry.LatestVersion;
  emitFreshnessTrace("write_commit", PathKeyStr, WriterRank, WriterRank,
                     WriterRank, CommittedVersionOut, Entry.LatestVersion, "",
                     "ok", WriteThroughMode ? "write-through" : "write-back",
                     "na");
  return true;
}

bool OmpFileHeadnodeManager::markTileFreshUnlocked(uint64_t PathKey, int Rank,
                                                   uint64_t Version) {
  const std::string PathKeyStr = std::to_string(PathKey);
  if (PathKey == 0 || !isWorkerRankUnlocked(Rank) || Version == 0) {
    emitFreshnessTrace("mark_fresh", PathKeyStr, Rank, Rank, Rank, Version, 0,
                       "", "fail", "invalid-args", "na");
    errno = EINVAL;
    return false;
  }
  auto It = TileFreshnessTable.find(PathKey);
  if (It == TileFreshnessTable.end()) {
    emitFreshnessTrace("mark_fresh", PathKeyStr, Rank, Rank, Rank, Version, 0,
                       "", "fail", "missing-entry", "na");
    errno = ENOKEY;
    return false;
  }
  if (It->second.LatestVersion != Version) {
    emitFreshnessTrace("mark_fresh", PathKeyStr, Rank, Rank, Rank, Version,
                       It->second.LatestVersion, "", "fail",
                       "version-mismatch", "na");
    errno = ESTALE;
    return false;
  }
  It->second.FreshRanks.insert(Rank);
  emitFreshnessTrace("mark_fresh", PathKeyStr, Rank, Rank, Rank, Version,
                     It->second.LatestVersion, "", "ok", "fresh", "na");
  return true;
}

bool OmpFileHeadnodeManager::registerTileCopyUnlocked(uint64_t PathKey,
                                                      int SourceRank,
                                                      int DestRank,
                                                      uint64_t Version) {
  const std::string PathKeyStr = std::to_string(PathKey);
  if (PathKey == 0 || !isWorkerRankUnlocked(SourceRank) ||
      !isWorkerRankUnlocked(DestRank) || Version == 0) {
    emitFreshnessTrace("register_copy", PathKeyStr, DestRank, SourceRank,
                       DestRank, Version, 0, "copy", "fail",
                       "invalid-args", "na");
    errno = EINVAL;
    return false;
  }
  auto It = TileFreshnessTable.find(PathKey);
  if (It == TileFreshnessTable.end()) {
    emitFreshnessTrace("register_copy", PathKeyStr, DestRank, SourceRank,
                       DestRank, Version, 0, "copy", "fail",
                       "missing-entry", "na");
    errno = ENOKEY;
    return false;
  }
  const TileFreshnessEntry &Entry = It->second;
  if (Entry.LatestVersion != Version) {
    emitFreshnessTrace("register_copy", PathKeyStr, DestRank, SourceRank,
                       DestRank, Version, Entry.LatestVersion, "copy", "fail",
                       "version-mismatch", "na");
    errno = ESTALE;
    return false;
  }
  if (Entry.FreshRanks.find(SourceRank) == Entry.FreshRanks.end()) {
    emitFreshnessTrace("register_copy", PathKeyStr, DestRank, SourceRank,
                       DestRank, Version, Entry.LatestVersion, "copy", "fail",
                       "source-not-fresh", "na");
    errno = EHOSTUNREACH;
    return false;
  }

  auto &Pending = PendingTileCopies[PathKey];
  for (const PendingTileCopy &Copy : Pending) {
    if (Copy.DestRank == DestRank && Copy.Version == Version)
      return true;
  }
  Pending.push_back({SourceRank, DestRank, Version});
  emitFreshnessTrace("register_copy", PathKeyStr, DestRank, SourceRank,
                     DestRank, Version, Entry.LatestVersion, "copy", "ok",
                     "pending-added", "na");
  return true;
}

bool OmpFileHeadnodeManager::markTileFreshFromCopyUnlocked(uint64_t PathKey,
                                                           int Rank,
                                                           uint64_t Version) {
  if (PathKey == 0 || !isWorkerRankUnlocked(Rank) || Version == 0) {
    errno = EINVAL;
    return false;
  }
  auto It = TileFreshnessTable.find(PathKey);
  if (It == TileFreshnessTable.end()) {
    errno = ENOKEY;
    return false;
  }
  if (It->second.LatestVersion != Version) {
    errno = ESTALE;
    return false;
  }

  auto PendingIt = PendingTileCopies.find(PathKey);
  if (PendingIt == PendingTileCopies.end()) {
    errno = ENOKEY;
    return false;
  }

  auto &Pending = PendingIt->second;
  Pending.erase(std::remove_if(Pending.begin(), Pending.end(),
                               [&](const PendingTileCopy &Copy) {
                                 return Copy.Version < Version;
                               }),
                Pending.end());
  for (auto VecIt = Pending.begin(); VecIt != Pending.end(); ++VecIt) {
    if (VecIt->DestRank == Rank && VecIt->Version == Version) {
      It->second.FreshRanks.insert(Rank);
      Pending.erase(VecIt);
      if (Pending.empty())
        PendingTileCopies.erase(PendingIt);
      return true;
    }
  }

  errno = ENOKEY;
  return false;
}

bool OmpFileHeadnodeManager::flushDirtyTileUnlocked(uint64_t PathKey,
                                                    int &SourceRankOut,
                                                    uint64_t &FlushedVersionOut) {
  const std::string PathKeyStr = std::to_string(PathKey);
  SourceRankOut = -1;
  FlushedVersionOut = 0;
  if (PathKey == 0) {
    emitFreshnessTrace("flush_prepare", PathKeyStr, -1, -1, -1, 0, 0, "",
                       "fail", "invalid-args", "na");
    errno = EINVAL;
    return false;
  }

  auto It = TileFreshnessTable.find(PathKey);
  if (It == TileFreshnessTable.end()) {
    emitFreshnessTrace("flush_prepare", PathKeyStr, -1, -1, -1, 0, 0, "",
                       "fail", "missing-entry", "na");
    errno = ENOKEY;
    return false;
  }

  TileFreshnessEntry &Entry = It->second;
  if (!Entry.Dirty || Entry.PfsVersion == Entry.LatestVersion) {
    FlushedVersionOut = Entry.PfsVersion;
    emitFreshnessTrace("flush_done", PathKeyStr, SourceRankOut, SourceRankOut,
                       SourceRankOut, FlushedVersionOut, Entry.LatestVersion, "",
                       "ok", Entry.Dirty ? "already-pfs-current" : "not-dirty",
                       "na");
    return true;
  }

  for (int Rank : Entry.FreshRanks) {
    if (isWorkerRankUnlocked(Rank)) {
      emitFreshnessTrace("flush_prepare", PathKeyStr, Rank, Rank, Rank,
                         Entry.LatestVersion, Entry.LatestVersion, "", "ok",
                         "", "na");
      SourceRankOut = Rank;
      FlushedVersionOut = Entry.LatestVersion;
      return true;
    }
  }

  emitFreshnessTrace("flush_prepare", PathKeyStr, -1, -1, -1,
                     Entry.LatestVersion, Entry.LatestVersion, "", "fail",
                     "no-fresh-rank", "na");
  errno = ENOKEY;
  return false;
}

bool OmpFileHeadnodeManager::completeDirtyFlushUnlocked(uint64_t PathKey,
                                                        int SourceRank,
                                                        uint64_t Version,
                                                        bool Success) {
  const std::string PathKeyStr = std::to_string(PathKey);
  if (PathKey == 0 || !isWorkerRankUnlocked(SourceRank) || Version == 0) {
    emitFreshnessTrace("flush_fail", PathKeyStr, SourceRank, SourceRank,
                       SourceRank, Version, 0, "", "fail", "invalid-args",
                       "na");
    errno = EINVAL;
    return false;
  }
  auto It = TileFreshnessTable.find(PathKey);
  if (It == TileFreshnessTable.end()) {
    emitFreshnessTrace("flush_fail", PathKeyStr, SourceRank, SourceRank,
                       SourceRank, Version, 0, "", "fail", "missing-entry",
                       "na");
    errno = ENOKEY;
    return false;
  }

  TileFreshnessEntry &Entry = It->second;
  if (Version > Entry.LatestVersion) {
    emitFreshnessTrace("flush_fail", PathKeyStr, SourceRank, SourceRank,
                       SourceRank, Version, Entry.LatestVersion, "", "fail",
                       "future-version", "na");
    errno = ESTALE;
    return false;
  }
  if (Entry.FreshRanks.find(SourceRank) == Entry.FreshRanks.end()) {
    emitFreshnessTrace("flush_fail", PathKeyStr, SourceRank, SourceRank,
                       SourceRank, Version, Entry.LatestVersion, "", "fail",
                       "source-not-fresh", "na");
    errno = EHOSTUNREACH;
    return false;
  }
  if (!Success) {
    emitFreshnessTrace("flush_fail", PathKeyStr, SourceRank, SourceRank,
                       SourceRank, Version, Entry.LatestVersion, "", "fail",
                       "flush-reported-failure", "na");
    errno = EIO;
    return false;
  }

  Entry.PfsVersion = Entry.LatestVersion;
  Entry.Dirty = false;
  emitFreshnessTrace("flush_done", PathKeyStr, SourceRank, SourceRank,
                     SourceRank, Version, Entry.LatestVersion, "", "ok",
                     Version == Entry.LatestVersion ? "" : "accepted-stale-owner",
                     "na");
  return true;
}

bool OmpFileHeadnodeManager::evaluateTileFreshnessQueryUnlocked(
    uint64_t PathKey, uint64_t LocalVersion, int RequesterRank,
    OmpFileFreshnessDecision &DecisionOut, int &SourceRankOut,
    uint64_t &SelectedVersionOut) const {
  const std::string PathKeyStr = std::to_string(PathKey);
  auto DecisionToString = [](OmpFileFreshnessDecision D) {
    switch (D) {
    case OmpFileFreshnessDecision::USE_LOCAL:
      return "use_local";
    case OmpFileFreshnessDecision::READ_PFS:
      return "read_pfs";
    case OmpFileFreshnessDecision::COPY_FROM_RANK:
      return "copy_from_rank";
    case OmpFileFreshnessDecision::WAIT_OR_FAIL:
      return "wait_or_fail";
    }
    return "unknown";
  };
  DecisionOut = OmpFileFreshnessDecision::WAIT_OR_FAIL;
  SourceRankOut = -1;
  SelectedVersionOut = 0;
  if (PathKey == 0 || !isWorkerRankUnlocked(RequesterRank)) {
    emitFreshnessTrace("query", PathKeyStr, RequesterRank, -1, RequesterRank,
                       LocalVersion, 0, "invalid", "fail", "invalid-args",
                       "na");
    errno = EINVAL;
    return false;
  }

  auto It = TileFreshnessTable.find(PathKey);
  if (It == TileFreshnessTable.end()) {
    DecisionOut = OmpFileFreshnessDecision::READ_PFS;
    emitFreshnessTrace("query", PathKeyStr, RequesterRank, SourceRankOut,
                       RequesterRank, SelectedVersionOut, SelectedVersionOut,
                       DecisionToString(DecisionOut), "ok", "missing-entry",
                       "na");
    return true;
  }
  const TileFreshnessEntry &Entry = It->second;
  SelectedVersionOut = Entry.LatestVersion;

  if (LocalVersion == Entry.LatestVersion &&
      Entry.FreshRanks.find(RequesterRank) != Entry.FreshRanks.end()) {
    DecisionOut = OmpFileFreshnessDecision::USE_LOCAL;
    SourceRankOut = RequesterRank;
    emitFreshnessTrace("query", PathKeyStr, RequesterRank, SourceRankOut,
                       RequesterRank, SelectedVersionOut, Entry.LatestVersion,
                       DecisionToString(DecisionOut), "ok", "", "na");
    return true;
  }

  if (!Entry.Dirty && Entry.PfsVersion == Entry.LatestVersion) {
    DecisionOut = OmpFileFreshnessDecision::READ_PFS;
    emitFreshnessTrace("query", PathKeyStr, RequesterRank, SourceRankOut,
                       RequesterRank, SelectedVersionOut, Entry.LatestVersion,
                       DecisionToString(DecisionOut), "ok", "", "na");
    return true;
  }

  for (int Rank : Entry.FreshRanks) {
    if (isWorkerRankUnlocked(Rank)) {
      DecisionOut = OmpFileFreshnessDecision::COPY_FROM_RANK;
      SourceRankOut = Rank;
      emitFreshnessTrace("query", PathKeyStr, RequesterRank, SourceRankOut,
                         RequesterRank, SelectedVersionOut,
                         Entry.LatestVersion, DecisionToString(DecisionOut),
                         "ok", "", "na");
      return true;
    }
  }

  DecisionOut = OmpFileFreshnessDecision::WAIT_OR_FAIL;
  emitFreshnessTrace("query", PathKeyStr, RequesterRank, SourceRankOut,
                     RequesterRank, SelectedVersionOut, Entry.LatestVersion,
                     DecisionToString(DecisionOut), "ok", "no-fresh-rank",
                     "na");
  return true;
}

bool OmpFileHeadnodeManager::classifyRebalanceConflictForTesting(
    const OmpFileIOBatchSegment &Segment, const char *&Reason,
    int &ReasonErrno) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  return hasRebalanceConflictUnlocked(Segment, Reason, ReasonErrno);
}

bool OmpFileHeadnodeManager::isWorkerRankUnlocked(int Rank) const {
  const int WorkerCount = std::max(WorldSize - 1, 0);
  return Rank >= 0 && Rank < WorkerCount;
}

bool OmpFileHeadnodeManager::isHeadnodeRequestUnlocked(int LocalRank) const {
  return LocalRank == HeadnodeRank;
}

void OmpFileHeadnodeManager::ensureHandlersUnlocked() {
  if (!Handlers.empty())
    return;

  const int WorkerCount = std::max(WorldSize - 1, 0);
  if (WorkerCount == 0) {
    Handlers.push_back({0, 0});
    return;
  }

  Handlers.reserve(WorkerCount);
  for (int Rank = 0; Rank < WorkerCount; ++Rank)
    Handlers.push_back({Rank, 0});
}

uint64_t OmpFileHeadnodeManager::minInFlightUnlocked() const {
  if (Handlers.empty())
    return 0;

  uint64_t MinLoad = std::numeric_limits<uint64_t>::max();
  for (const HandlerInfo &H : Handlers)
    MinLoad = std::min(MinLoad, H.InFlight);
  return MinLoad;
}

uint64_t OmpFileHeadnodeManager::inFlightForRankUnlocked(int Rank) const {
  for (const HandlerInfo &H : Handlers) {
    if (H.Rank == Rank)
      return effectiveLoadUnlocked(H);
  }
  return std::numeric_limits<uint64_t>::max();
}

uint64_t OmpFileHeadnodeManager::effectiveLoadUnlocked(
    const HandlerInfo &H) const {
  return UsePlannedLoadForSkew ? H.PlannedTotal : H.InFlight;
}

void OmpFileHeadnodeManager::bumpHandlerLoadUnlocked(int Rank) {
  for (HandlerInfo &H : Handlers) {
    if (H.Rank == Rank) {
      H.InFlight += 1;
      H.PlannedTotal += 1;
      return;
    }
  }
}

void OmpFileHeadnodeManager::noteBatchPlanUnlocked(int Rank) {
  for (HandlerInfo &H : Handlers) {
    if (H.Rank == Rank) {
      H.PlannedTotal += 1;
      return;
    }
  }
}

void OmpFileHeadnodeManager::registerPathAffinityUnlocked(uint64_t PathKey,
                                                          int Rank) {
  if (PathKey == 0 || !isWorkerRankUnlocked(Rank))
    return;
  PathAffinityTable[PathKey] = Rank;
}

int OmpFileHeadnodeManager::pickLeastLoadedRankUnlocked(int ClientRank) {
  ensureHandlersUnlocked();

  if (isWorkerRankUnlocked(ClientRank))
    return ClientRank;

  if (Handlers.empty())
    return 0;

  uint64_t BestLoad = std::numeric_limits<uint64_t>::max();
  int BestRank = Handlers.front().Rank;
  const uint64_t StartIdx = NextHandlerTieBreaker++ % Handlers.size();

  for (size_t I = 0; I < Handlers.size(); ++I) {
    const size_t Idx = (StartIdx + I) % Handlers.size();
    const HandlerInfo &H = Handlers[Idx];
    const uint64_t Load = effectiveLoadUnlocked(H);
    if (Load < BestLoad) {
      BestLoad = Load;
      BestRank = H.Rank;
    }
  }

  return BestRank;
}

int OmpFileHeadnodeManager::pickRankForPathKeyUnlocked(uint64_t PathKey,
                                                        bool HasPathKey,
                                                        int ClientRank,
                                                        bool &AffinityHit,
                                                        bool &Rebalanced) {
  AffinityHit = false;
  Rebalanced = false;
  ensureHandlersUnlocked();

  if (!HasPathKey || PathKey == 0)
    return pickLeastLoadedRankUnlocked(ClientRank);

  const int LeastLoadedRank = pickLeastLoadedRankUnlocked(/*ClientRank=*/-1);
  auto It = PathAffinityTable.find(PathKey);
  if (It == PathAffinityTable.end()) {
    registerPathAffinityUnlocked(PathKey, LeastLoadedRank);
    return LeastLoadedRank;
  }

  const int PreferredRank = It->second;
  if (!isWorkerRankUnlocked(PreferredRank)) {
    It->second = LeastLoadedRank;
    Rebalanced = true;
    return LeastLoadedRank;
  }

  const uint64_t PreferredLoad = inFlightForRankUnlocked(PreferredRank);
  const uint64_t LeastLoad = inFlightForRankUnlocked(LeastLoadedRank);
  const bool CanMeasureLoads =
      PreferredLoad != std::numeric_limits<uint64_t>::max() &&
      LeastLoad != std::numeric_limits<uint64_t>::max();
  if (CanMeasureLoads &&
      MaxAffinityLoadSkew != std::numeric_limits<uint64_t>::max() &&
      PreferredRank != LeastLoadedRank &&
      PreferredLoad > (LeastLoad + MaxAffinityLoadSkew)) {
    It->second = LeastLoadedRank;
    Rebalanced = true;
    return LeastLoadedRank;
  }

  AffinityHit = true;
  // Keep an existing path pinned to the same worker so cached opens and
  // follow-on reads/writes observe a single aggregator for that file.
  return PreferredRank;
}

OmpFileIOPlan OmpFileHeadnodeManager::planRequest(const OmpFileIORequest &Request,
                                                  const char *Path,
                                                  int LocalRank) {
  const std::lock_guard<std::mutex> Lock(Mutex);

  OmpFileIOPlan Plan{};
  Plan.RequestId = Request.RequestId;
  Plan.Offset = Request.Offset;
  Plan.Size = Request.Size;
  Plan.RemoteHandle = -1;
  Plan.Status = 0;
  Plan.Errno = 0;

  if (!Initialized) {
    Plan.Status = -1;
    Plan.Errno = EHOSTDOWN;
    return Plan;
  }

  if (!isHeadnodeRequestUnlocked(LocalRank)) {
    Plan.Status = -1;
    Plan.Errno = EPERM;
    return Plan;
  }

  int HandlerRank = pickLeastLoadedRankUnlocked(Request.ClientRank);

  if (Request.Op == OmpFileIOOp::OPEN) {
    if (Request.PathSize == 0 || !Path) {
      Plan.Status = -1;
      Plan.Errno = EINVAL;
      return Plan;
    }

    std::string_view PathView(Path, Request.PathSize > 0 ? Request.PathSize - 1
                                                         : 0);
    auto [It, Inserted] = GlobalFileTable.try_emplace(std::string(PathView));
    GlobalFileEntry &Entry = It->second;
    const uint64_t PathKey = computePathKey(PathView);

    if (Inserted)
      Entry.GlobalFileId = NextGlobalFileId++;

    Entry.OpenRequests += 1;
    Entry.PathKey = PathKey;
    Entry.HasPathKey = true;

    bool AffinityHit = false;
    bool Rebalanced = false;
    if (Inserted) {
      HandlerRank = pickRankForPathKeyUnlocked(PathKey, /*HasPathKey=*/true,
                                               Request.ClientRank, AffinityHit,
                                               Rebalanced);
      Entry.PreferredAggregatorRank = HandlerRank;
      registerPathAffinityUnlocked(PathKey, HandlerRank);
    } else {
      if (SpreadSamePathOpens) {
        HandlerRank = pickLeastLoadedRankUnlocked(Request.ClientRank);
        if (isWorkerRankUnlocked(Entry.PreferredAggregatorRank) &&
            HandlerRank == Entry.PreferredAggregatorRank) {
          AffinityHit = true;
        } else {
          Rebalanced = isWorkerRankUnlocked(Entry.PreferredAggregatorRank);
        }
      } else {
        // Multiple concurrent OPEN requests for the same path must keep routing
        // to one stable aggregator rank unless the explicit spread policy is on.
        if (isWorkerRankUnlocked(Entry.PreferredAggregatorRank)) {
          HandlerRank = Entry.PreferredAggregatorRank;
          AffinityHit = true;
        } else {
          HandlerRank = pickLeastLoadedRankUnlocked(Request.ClientRank);
          Entry.PreferredAggregatorRank = HandlerRank;
          registerPathAffinityUnlocked(PathKey, HandlerRank);
          Rebalanced = true;
        }
      }
    }

    if (AffinityHit)
      Plan.PlanFlags |= OMPFILE_BATCH_PLAN_FILE_AFFINITY;
    if (Rebalanced)
      Plan.PlanFlags |= OMPFILE_BATCH_PLAN_REBALANCED;

    Plan.RemoteHandle = static_cast<int32_t>(Entry.GlobalFileId & 0x7fffffffU);

    const std::string PathKeyStr = std::to_string(PathKey);
    emitFreshnessTrace("compat_sched_open", PathKeyStr, Request.ClientRank,
                       Request.ClientRank, HandlerRank, 0, 0, "route", "ok",
                       "legacy_entrypoint", "na");
  } else if (Request.Op == OmpFileIOOp::PWRITE) {
    uint64_t PathKey = 0;
    bool HasPathKey = false;
    if (Request.PathSize > 0 && Path) {
      std::string_view PathView(Path,
                                Request.PathSize > 0 ? Request.PathSize - 1 : 0);
      if (!PathView.empty()) {
        PathKey = computePathKey(PathView);
        HasPathKey = true;
      }
    }

    if (HasPathKey) {
      recordWriteVersionUnlocked(PathKey, Request);
      const std::string PathKeyStr = std::to_string(PathKey);
      emitFreshnessTrace("compat_write_version", PathKeyStr, Request.ClientRank,
                         Request.ClientRank, HandlerRank, 0, 0, "record",
                         "ok", "legacy_entrypoint", "na");
      fprintf(stderr,
              "[ompfile-mpp][sched] write-version path_key=%llu offset=%lld "
              "size=%llu has_epoch=%d epoch=%llu has_tile=%d tile=%llu\n",
              static_cast<unsigned long long>(PathKey),
              static_cast<long long>(Request.Offset),
              static_cast<unsigned long long>(Request.Size),
              static_cast<int>((Request.HintFlags & OMPFILE_IO_HINT_HAS_EPOCH) != 0),
              static_cast<unsigned long long>(Request.EpochId),
              static_cast<int>((Request.HintFlags & OMPFILE_IO_HINT_HAS_TILE) != 0),
              static_cast<unsigned long long>(Request.TileId));
    } else {
      fprintf(stderr,
              "[ompfile-mpp][sched] write-version skipped-metadata file=%d "
              "offset=%lld size=%llu has_path=%d\n",
              Request.FileHandle, static_cast<long long>(Request.Offset),
              static_cast<unsigned long long>(Request.Size),
              static_cast<int>(Request.PathSize > 0 && Path));
    }
  }

  if (!isWorkerRankUnlocked(HandlerRank)) {
    Plan.Status = -1;
    Plan.Errno = EHOSTUNREACH;
    return Plan;
  }

  Plan.AggregatorRank = HandlerRank;

  FlightplanTable[Plan.RequestId] = Plan;
  bumpHandlerLoadUnlocked(HandlerRank);

  return Plan;
}

bool OmpFileHeadnodeManager::planBatchRequest(
    const OmpFileIOBatchRequest &Request, const OmpFileIOBatchSegment *Segments,
    OmpFileIOBatchPlan &Plan, std::vector<OmpFileIOBatchPlanEntry> &Entries,
    int LocalRank) {
  const std::lock_guard<std::mutex> Lock(Mutex);

  Plan = {};
  Plan.AbiVersion = OMPFILE_SCHED_BATCH_ABI_VERSION;
  Plan.BatchId = Request.BatchId;
  Plan.SegmentCount = Request.SegmentCount;
  Plan.PlanFlags = OMPFILE_BATCH_PLAN_BATCH_API;
  Entries.clear();

  if (!Initialized) {
    Plan.Status = -1;
    Plan.Errno = EHOSTDOWN;
    return true;
  }

  if (!isHeadnodeRequestUnlocked(LocalRank)) {
    Plan.Status = -1;
    Plan.Errno = EPERM;
    return true;
  }

  if (Request.AbiVersion != OMPFILE_SCHED_BATCH_ABI_VERSION) {
    Plan.Status = -1;
    Plan.Errno = EPROTO;
    return true;
  }

  if (Request.SegmentCount > 0 && !Segments) {
    Plan.Status = -1;
    Plan.Errno = EINVAL;
    return true;
  }

  const uint64_t EntryBytes =
      static_cast<uint64_t>(Request.SegmentCount) *
      static_cast<uint64_t>(sizeof(OmpFileIOBatchPlanEntry));
  if (EntryBytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    Plan.Status = -1;
    Plan.Errno = EOVERFLOW;
    return true;
  }
  Plan.PayloadBytes = static_cast<uint32_t>(EntryBytes);

  Entries.resize(Request.SegmentCount);
  assert(Entries.size() == Request.SegmentCount &&
         "Batch entry vector must match segment count.");
  const bool FailOnAnyError =
      (Request.RequestFlags & OMPFILE_BATCH_REQ_FAIL_ON_ANY_ERROR) != 0;

  bool SawError = false;
  int FirstErrno = 0;
  uint64_t AffinityHits = 0;
  uint64_t Rebalances = 0;
  bool StopScheduling = false;

  for (uint32_t I = 0; I < Request.SegmentCount; ++I) {
    const OmpFileIOBatchSegment &Segment = Segments[I];
    OmpFileIOBatchPlanEntry &Entry = Entries[I];
    Entry.SegmentId = Segment.SegmentId;
    Entry.Offset = Segment.Offset;
    Entry.Size = Segment.Size;
    Entry.PlanFlags = OMPFILE_BATCH_PLAN_BATCH_API;

    if (StopScheduling) {
      Entry.Status = -1;
      Entry.Errno = ECANCELED;
      continue;
    }

    bool AffinityHit = false;
    bool Rebalanced = false;
    const bool HasPathKey = Segment.PathKey != 0;
    bool HadPriorAffinity = false;
    int PriorAffinityRank = -1;
    if (HasPathKey) {
      auto AffinityIt = PathAffinityTable.find(Segment.PathKey);
      if (AffinityIt != PathAffinityTable.end()) {
        HadPriorAffinity = true;
        PriorAffinityRank = AffinityIt->second;
      }
    }
    const int HandlerRank = pickRankForPathKeyUnlocked(
        Segment.PathKey, HasPathKey, Segment.ClientRank, AffinityHit, Rebalanced);

    const char *RebalanceReason = "none";
    int RebalanceErrno = 0;
    bool RebalanceBlocked = false;
    if (Rebalanced) {
      RebalanceBlocked =
          hasRebalanceConflictUnlocked(Segment, RebalanceReason, RebalanceErrno);
      if (RebalanceBlocked && HasPathKey) {
        if (HadPriorAffinity)
          PathAffinityTable[Segment.PathKey] = PriorAffinityRank;
        else
          PathAffinityTable.erase(Segment.PathKey);
      }
    }

    if (!isWorkerRankUnlocked(HandlerRank)) {
      Entry.Status = -1;
      Entry.Errno = EHOSTUNREACH;
    } else {
      Entry.AggregatorRank = HandlerRank;
      Entry.RemoteHandle = -1;
      noteBatchPlanUnlocked(HandlerRank);
      if (AffinityHit) {
        Entry.PlanFlags |= OMPFILE_BATCH_PLAN_FILE_AFFINITY;
        ++AffinityHits;
      }
      if (Rebalanced) {
        Entry.PlanFlags |= OMPFILE_BATCH_PLAN_REBALANCED;
        ++Rebalances;
        if (RebalanceBlocked) {
          Entry.Status = -1;
          Entry.Errno = RebalanceErrno != 0 ? RebalanceErrno : ESTALE;
          fprintf(stderr,
                  "[ompfile-mpp][sched] rebalanced-plan blocked segment=%llu "
                  "path_key=%llu offset=%lld size=%llu reason=%s errno=%d "
                  "read_epoch=%llu read_has_epoch=%d\n",
                  static_cast<unsigned long long>(Segment.SegmentId),
                  static_cast<unsigned long long>(Segment.PathKey),
                  static_cast<long long>(Segment.Offset),
                  static_cast<unsigned long long>(Segment.Size),
                  RebalanceReason, Entry.Errno,
                  static_cast<unsigned long long>(Segment.EpochId),
                  static_cast<int>((Segment.SegmentFlags &
                                    OMPFILE_IO_HINT_HAS_EPOCH) != 0));
        }
      }
    }

    Plan.PlanFlags |= Entry.PlanFlags;
    if (Entry.Status != 0) {
      SawError = true;
      if (FirstErrno == 0)
        FirstErrno = Entry.Errno;
      if (FailOnAnyError)
        StopScheduling = true;
    }
  }

  if (SawError) {
    Plan.Status = -1;
    Plan.Errno = FirstErrno != 0 ? FirstErrno : EIO;
  }

  Stats.Requests += 1;
  Stats.Segments += Request.SegmentCount;
  Stats.AffinityHits += AffinityHits;
  Stats.Rebalances += Rebalances;
  maybeReportBatchStatsUnlocked();
  return true;
}

std::vector<OmpFileHeadnodeManager::HandlerInfo>
OmpFileHeadnodeManager::snapshotHandlersForTesting() {
  const std::lock_guard<std::mutex> Lock(Mutex);
  ensureHandlersUnlocked();
  return Handlers;
}

void OmpFileHeadnodeManager::maybeReportBatchStatsUnlocked() {
  if (BatchStatsReportEvery == 0 ||
      (Stats.Requests % BatchStatsReportEvery) != 0) {
    return;
  }

  // Keep reporting lightweight and bounded to avoid perturbing scheduling.
  fprintf(stderr,
          "[ompfile-mpp][sched] batch_requests=%llu batch_segments=%llu "
          "affinity_hits=%llu rebalances=%llu handlers=%zu\n",
          static_cast<unsigned long long>(Stats.Requests),
          static_cast<unsigned long long>(Stats.Segments),
          static_cast<unsigned long long>(Stats.AffinityHits),
          static_cast<unsigned long long>(Stats.Rebalances), Handlers.size());
}

void OmpFileHeadnodeManager::completeRequest(const OmpFileIOPlan &Plan) {
  const std::lock_guard<std::mutex> Lock(Mutex);

  auto It = FlightplanTable.find(Plan.RequestId);
  if (It == FlightplanTable.end())
    return;

  const int HandlerRank = It->second.AggregatorRank;
  for (HandlerInfo &H : Handlers) {
    if (H.Rank == HandlerRank && H.InFlight > 0) {
      H.InFlight -= 1;
      break;
    }
  }

  FlightplanTable.erase(It);
}
