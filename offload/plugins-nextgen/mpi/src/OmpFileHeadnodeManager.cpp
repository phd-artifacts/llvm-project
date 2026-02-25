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

void OmpFileHeadnodeManager::initialize(int NewWorldSize, int NewHeadnodeRank) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  if (Initialized)
    return;

  WorldSize = std::max(NewWorldSize, 1);
  HeadnodeRank = NewHeadnodeRank;
  MaxAffinityLoadSkew =
      parseUint64Env("LIBOMPFILE_SCHED_MAX_AFFINITY_LOAD_SKEW", 2);
  BatchStatsReportEvery =
      std::max<uint64_t>(1, parseUint64Env("LIBOMPFILE_SCHED_STATS_REPORT_EVERY",
                                           128));
  ensureHandlersUnlocked();
  Initialized = true;
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
      return H.InFlight;
  }
  return std::numeric_limits<uint64_t>::max();
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
    if (H.InFlight < BestLoad) {
      BestLoad = H.InFlight;
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

  AffinityHit = true;
  const uint64_t PreferredLoad = inFlightForRankUnlocked(PreferredRank);
  const uint64_t MinLoad = minInFlightUnlocked();
  if (PreferredLoad <= MinLoad + MaxAffinityLoadSkew)
    return PreferredRank;

  if (LeastLoadedRank != PreferredRank) {
    It->second = LeastLoadedRank;
    Rebalanced = true;
  }
  return LeastLoadedRank;
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
    HandlerRank = pickRankForPathKeyUnlocked(PathKey, /*HasPathKey=*/true,
                                             Request.ClientRank, AffinityHit,
                                             Rebalanced);
    Entry.PreferredAggregatorRank = HandlerRank;
    registerPathAffinityUnlocked(PathKey, HandlerRank);

    if (AffinityHit)
      Plan.PlanFlags |= OMPFILE_BATCH_PLAN_FILE_AFFINITY;
    if (Rebalanced)
      Plan.PlanFlags |= OMPFILE_BATCH_PLAN_REBALANCED;

    Plan.RemoteHandle = static_cast<int32_t>(Entry.GlobalFileId & 0x7fffffffU);
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
    const int HandlerRank = pickRankForPathKeyUnlocked(
        Segment.PathKey, Segment.PathKey != 0, Segment.ClientRank, AffinityHit,
        Rebalanced);

    if (!isWorkerRankUnlocked(HandlerRank)) {
      Entry.Status = -1;
      Entry.Errno = EHOSTUNREACH;
    } else {
      Entry.AggregatorRank = HandlerRank;
      Entry.RemoteHandle = -1;
      bumpHandlerLoadUnlocked(HandlerRank);
      if (AffinityHit) {
        Entry.PlanFlags |= OMPFILE_BATCH_PLAN_FILE_AFFINITY;
        ++AffinityHits;
      }
      if (Rebalanced) {
        Entry.PlanFlags |= OMPFILE_BATCH_PLAN_REBALANCED;
        ++Rebalances;
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
