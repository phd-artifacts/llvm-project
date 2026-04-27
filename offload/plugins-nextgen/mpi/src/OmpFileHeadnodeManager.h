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

  static OmpFileHeadnodeManager &instance();

  void initialize(int WorldSize, int HeadnodeRank);

  OmpFileIOPlan planRequest(const OmpFileIORequest &Request, const char *Path,
                            int LocalRank);

  bool planBatchRequest(const OmpFileIOBatchRequest &Request,
                        const OmpFileIOBatchSegment *Segments,
                        OmpFileIOBatchPlan &Plan,
                        std::vector<OmpFileIOBatchPlanEntry> &Entries,
                        int LocalRank);

  void completeRequest(const OmpFileIOPlan &Plan);
  std::vector<HandlerInfo> snapshotHandlersForTesting();

private:
  OmpFileHeadnodeManager() = default;

  static uint64_t computePathKey(std::string_view Path);
  static uint64_t parseUint64Env(const char *Name, uint64_t DefaultValue);
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
  uint64_t minInFlightUnlocked() const;
  uint64_t inFlightForRankUnlocked(int Rank) const;
  void bumpHandlerLoadUnlocked(int Rank);
  void noteBatchPlanUnlocked(int Rank);
  void registerPathAffinityUnlocked(uint64_t PathKey, int Rank);
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

  std::unordered_map<uint64_t, OmpFileIOPlan> FlightplanTable;
  std::unordered_map<std::string, GlobalFileEntry> GlobalFileTable;
  std::unordered_map<uint64_t, int> PathAffinityTable;
  std::unordered_map<uint64_t, std::vector<WriteVersionEntry>> PathWriteVersions;
  std::vector<HandlerInfo> Handlers;
  BatchPlanStats Stats;
  uint64_t NextGlobalFileId = 1;
  uint64_t NextHandlerTieBreaker = 0;
  uint64_t NextWriteSequence = 1;
};

#endif // _MPI_PROXY_OMPFILE_HEADNODE_MANAGER_H_
