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
// - Per-rank counts: the cumulative plan counts that every pick, OPEN
//   included, weighs as the rank's load, and live open handles, kept for
//   accounting only (an OPEN plan adds one on the rank it names; the CLOSE
//   carrying that rank in OmpFileIORequest::AggregatorRank removes it); no
//   pick weighs them.
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

#include "EventSystemOmpFile.h"

class OmpFileHeadnodeManager {
public:
  struct HandlerInfo {
    int Rank = -1;
    // Accounting only, never a pick's load: OPEN plans placed on this rank
    // whose CLOSE has not arrived. A PLACEMENT_ONLY OPEN adds none. Read by
    // the LIBOMPFILE_DEBUG_TRACE lines and the tests; a future scheduler
    // signal.
    uint64_t OpenHandles = 0;
    // The load of every pick by default (scalar OPEN/PREAD/PWRITE/CLOSE,
    // batch segments): scalar plans placed on this rank, every OPEN included,
    // batch segments not; monotonic.
    uint64_t PlansIssued = 0;
    // Every plan placed on this rank (any scalar op, plus one per batch
    // segment); monotonic. The load of every pick under
    // LIBOMPFILE_SCHED_USE_PLANNED_LOAD_FOR_SKEW=1.
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

  /// WorkerCount is the size of the proxy partition (ranks [0, WorkerCount)
  /// are the I/O workers this manager may hand out); origin ranks above it
  /// are clients, never aggregators.
  void initialize(int WorkerCount, int HeadnodeRank);

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
  bool flushDirtyTile(uint64_t PathKey, int &SourceRankOut,
                      uint64_t &FlushedVersionOut);
  bool completeDirtyFlush(uint64_t PathKey, int SourceRank, uint64_t Version,
                         bool Success);

  bool handleFreshnessQueryRequest(
      const OmpFileFreshnessQueryRequest &Request,
      OmpFileFreshnessQueryReply &Reply);

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
  // Every pick weighs PlansIssued (PlannedTotal under
  // LIBOMPFILE_SCHED_USE_PLANNED_LOAD_FOR_SKEW); OpenHandles is accounting
  // only and never steers placement.
  uint64_t effectiveLoadUnlocked(const HandlerInfo &H) const;
  uint64_t loadForRankUnlocked(int Rank) const;
  void noteScalarPlanUnlocked(int Rank);
  void notePlannedUnlocked(int Rank);
  void addOpenHandleUnlocked(int Rank);
  void releaseOpenHandleUnlocked(int Rank);
  void registerPathAffinityUnlocked(uint64_t PathKey, int Rank);
  bool commitTileFreshnessWriteUnlocked(uint64_t PathKey, int WriterRank,
                                        bool WriteThroughMode,
                                        uint64_t &CommittedVersionOut);
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
  int WorkerCount = 1;
  int HeadnodeRank = 0;
  uint64_t MaxAffinityLoadSkew = 2;
  uint64_t BatchStatsReportEvery = 128;
  bool UsePlannedLoadForSkew = false;
  bool SpreadSamePathOpens = false;
  bool FreshnessTraceEnabled = false;
  // LIBOMPFILE_DEBUG_TRACE=1: the [ompfile-mpp][sched] open-handles lines.
  bool SchedTraceEnabled = false;

  std::unordered_map<std::string, GlobalFileEntry> GlobalFileTable;
  std::unordered_map<uint64_t, int> PathAffinityTable;
  std::unordered_map<uint64_t, std::vector<WriteVersionEntry>> PathWriteVersions;
  std::unordered_map<uint64_t, TileFreshnessEntry> TileFreshnessTable;
  std::vector<HandlerInfo> Handlers;
  BatchPlanStats Stats;
  uint64_t NextGlobalFileId = 1;
  uint64_t NextHandlerTieBreaker = 0;
  uint64_t NextWriteSequence = 1;
};

#endif // _MPI_PROXY_OMPFILE_HEADNODE_MANAGER_H_
