//===---- OmpFileHeadnodeManager.h - Global OMPFile scheduler ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Headnode-only global OMPFile control plane. Tracks:
// - Global file table.
// - Flightplan table.
// - Aggregator selection policy.
//
//===----------------------------------------------------------------------===//

#ifndef _MPI_PROXY_OMPFILE_HEADNODE_MANAGER_H_
#define _MPI_PROXY_OMPFILE_HEADNODE_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "EventSystem.h"

class OmpFileHeadnodeManager {
public:
  struct HandlerInfo {
    int Rank = -1;
    uint64_t InFlight = 0;
    uint64_t PlannedTotal = 0;
  };

  struct GlobalFileEntry {
    uint64_t GlobalFileId = 0;
    uint64_t OpenRequests = 0;
    int PreferredAggregatorRank = -1;
    uint64_t PathKey = 0;
    bool HasPathKey = false;
  };

  struct WriteVersionEntry {
    int64_t Start = 0;
    uint64_t Size = 0;
    uint64_t EpochId = 0;
    uint64_t TileId = 0;
    uint64_t Sequence = 0;
    bool HasEpoch = false;
    bool HasTile = false;
  };

  struct BatchPlanStats {
    uint64_t Requests = 0;
    uint64_t Segments = 0;
    uint64_t AffinityHits = 0;
    uint64_t Rebalances = 0;
  };

  struct TileFreshnessEntry {
    uint64_t LatestVersion = 0;
    std::unordered_set<int> FreshRanks;
    bool Dirty = false;
    uint64_t PfsVersion = 0;
  };

  static OmpFileHeadnodeManager &instance();

  // Public path-key hash for callers outside the manager (e.g. the proxy
  // computing a path key for a dirty-flush completion). Mirrors the private
  // computePathKey so the canonical FNV-1a hashing stays in one place.
  static uint64_t computePathKeyForPath(std::string_view Path) {
    return computePathKey(Path);
  }

  void initialize(int WorldSize, int HeadnodeRank);

  OmpFileIOPlan planRequest(const OmpFileIORequest &Request, const char *Path,
                            int LocalRank);

  bool planBatchRequest(const OmpFileIOBatchRequest &Request,
                        const OmpFileIOBatchSegment *Segments,
                        OmpFileIOBatchPlan &Plan,
                        std::vector<OmpFileIOBatchPlanEntry> &Entries,
                        int LocalRank);

  bool commitTileFreshnessWrite(uint64_t PathKey, int WriterRank,
                                bool WriteThroughMode,
                                uint64_t &CommittedVersionOut);
  bool markTileFresh(uint64_t PathKey, int Rank, uint64_t Version);
  bool registerTileCopy(uint64_t PathKey, int SourceRank, int DestRank,
                        uint64_t Version);
  bool markTileFreshFromCopy(uint64_t PathKey, int Rank, uint64_t Version);
  bool flushDirtyTile(uint64_t PathKey, int &SourceRankOut,
                      uint64_t &FlushedVersionOut);
  bool completeDirtyFlush(uint64_t PathKey, int SourceRank, uint64_t Version,
                         bool Success);

  bool handleFreshnessQueryRequest(
      const OmpFileFreshnessQueryRequest &Request,
      OmpFileFreshnessQueryReply &Reply);

  void completeRequest(const OmpFileIOPlan &Plan);

#if defined(OMPFILE_ENABLE_TEST_ACCESS)
  // Direct-linked regression seam; omitted from production plugin builds.
  bool classifyRebalanceConflictForTesting(const OmpFileIOBatchSegment &Segment,
                                           const char *&Reason,
                                           int &ReasonErrno);
  bool evaluateTileFreshnessQueryForTesting(
      uint64_t PathKey, uint64_t LocalVersion, int RequesterRank,
      OmpFileFreshnessDecision &DecisionOut, int &SourceRankOut,
      uint64_t &SelectedVersionOut);
  bool clearTileFreshRanksForTesting(uint64_t PathKey);
  std::vector<HandlerInfo> snapshotHandlersForTesting();
  void resetForTesting();
#endif // OMPFILE_ENABLE_TEST_ACCESS

  bool isFreshnessTraceEnabled() const;
  void emitFreshnessTrace(const char *Event, const std::string &PathKey,
                          int Rank, int SrcRank, int DstRank,
                          uint64_t Version, uint64_t LatestVersion,
                          const char *Decision, const char *Result,
                          const char *Reason, const char *TileId) const;

private:
  OmpFileHeadnodeManager() = default;

  static uint64_t computePathKey(std::string_view Path);
  static uint64_t parseUint64Env(const char *Name, uint64_t DefaultValue);
  static bool parseBoolEnv(const char *Name, bool DefaultValue);
  void freshnessTraceLogUnlocked(const char *Format, ...) const;
  bool segmentRangesOverlap(int64_t ReadOffset, uint64_t ReadSize,
                            int64_t WriteOffset, uint64_t WriteSize) const;
  uint64_t nextWriteSequenceUnlocked();
  void recordWriteVersionUnlocked(uint64_t PathKey,
                                  const OmpFileIORequest &Request);
  bool hasRebalanceConflictUnlocked(const OmpFileIOBatchSegment &Segment,
                                    const char *&Reason,
                                    int &ReasonErrno) const;
  bool isWorkerRankUnlocked(int Rank) const;
  bool isHeadnodeRequestUnlocked(int LocalRank) const;
  uint64_t effectiveLoadUnlocked(const HandlerInfo &H) const;
  uint64_t minInFlightUnlocked() const;
  uint64_t inFlightForRankUnlocked(int Rank) const;
  void bumpHandlerLoadUnlocked(int Rank);
  void noteBatchPlanUnlocked(int Rank);
  void registerPathAffinityUnlocked(uint64_t PathKey, int Rank);
  bool commitTileFreshnessWriteUnlocked(uint64_t PathKey, int WriterRank,
                                        bool WriteThroughMode,
                                        uint64_t &CommittedVersionOut);
  bool markTileFreshUnlocked(uint64_t PathKey, int Rank, uint64_t Version);
  bool registerTileCopyUnlocked(uint64_t PathKey, int SourceRank, int DestRank,
                                uint64_t Version);
  bool markTileFreshFromCopyUnlocked(uint64_t PathKey, int Rank,
                                     uint64_t Version);
  bool flushDirtyTileUnlocked(uint64_t PathKey, int &SourceRankOut,
                              uint64_t &FlushedVersionOut);
  bool completeDirtyFlushUnlocked(uint64_t PathKey, int SourceRank,
                                  uint64_t Version, bool Success);
  bool evaluateTileFreshnessQueryUnlocked(
      uint64_t PathKey, uint64_t LocalVersion, int RequesterRank,
      OmpFileFreshnessDecision &DecisionOut, int &SourceRankOut,
      uint64_t &SelectedVersionOut) const;
  int pickRankForPathKeyUnlocked(uint64_t PathKey, bool HasPathKey,
                                 int ClientRank, bool &AffinityHit,
                                 bool &Rebalanced);
  int pickLeastLoadedRankUnlocked(int ClientRank);
  void ensureHandlersUnlocked();
  void maybeReportBatchStatsUnlocked();

  std::mutex Mutex;
  bool Initialized = false;
  int WorldSize = 1;
  int HeadnodeRank = 0;
  uint64_t MaxAffinityLoadSkew = 2;
  uint64_t BatchStatsReportEvery = 128;
  bool UsePlannedLoadForSkew = false;
  bool SpreadSamePathOpens = false;
  bool FreshnessTraceEnabled = false;

  std::unordered_map<uint64_t, OmpFileIOPlan> FlightplanTable;
  std::unordered_map<std::string, GlobalFileEntry> GlobalFileTable;
  std::unordered_map<uint64_t, int> PathAffinityTable;
  std::unordered_map<uint64_t, std::vector<WriteVersionEntry>> PathWriteVersions;
  std::unordered_map<uint64_t, TileFreshnessEntry> TileFreshnessTable;
  struct PendingTileCopy {
    int SourceRank = -1;
    int DestRank = -1;
    uint64_t Version = 0;
  };
  std::unordered_map<uint64_t, std::vector<PendingTileCopy>> PendingTileCopies;
  std::vector<HandlerInfo> Handlers;
  BatchPlanStats Stats;
  uint64_t NextGlobalFileId = 1;
  uint64_t NextHandlerTieBreaker = 0;
  uint64_t NextWriteSequence = 1;
};

#endif // _MPI_PROXY_OMPFILE_HEADNODE_MANAGER_H_
