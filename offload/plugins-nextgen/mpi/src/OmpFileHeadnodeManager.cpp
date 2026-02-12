//===---- OmpFileHeadnodeManager.cpp - Global OMPFile scheduler -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OmpFileHeadnodeManager.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <string_view>

OmpFileHeadnodeManager &OmpFileHeadnodeManager::instance() {
  static OmpFileHeadnodeManager Manager;
  return Manager;
}

void OmpFileHeadnodeManager::initialize(int NewWorldSize, int NewHeadnodeRank) {
  const std::lock_guard<std::mutex> Lock(Mutex);
  if (Initialized)
    return;

  WorldSize = std::max(NewWorldSize, 1);
  HeadnodeRank = NewHeadnodeRank;
  ensureHandlersUnlocked();
  Initialized = true;
}

bool OmpFileHeadnodeManager::isWorkerRankUnlocked(int Rank) const {
  const int WorkerCount = std::max(WorldSize - 1, 0);
  return Rank >= 0 && Rank < WorkerCount;
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

  if (LocalRank != HeadnodeRank) {
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

    std::string_view PathView(Path, Request.PathSize > 0 ? Request.PathSize - 1 : 0);
    auto [It, Inserted] = GlobalFileTable.try_emplace(std::string(PathView));
    GlobalFileEntry &Entry = It->second;

    if (Inserted)
      Entry.GlobalFileId = NextGlobalFileId++;

    Entry.OpenRequests += 1;

    if (isWorkerRankUnlocked(Entry.PreferredAggregatorRank))
      HandlerRank = Entry.PreferredAggregatorRank;
    else
      Entry.PreferredAggregatorRank = HandlerRank;

    Plan.RemoteHandle = static_cast<int32_t>(Entry.GlobalFileId & 0x7fffffffU);
  }

  if (!isWorkerRankUnlocked(HandlerRank)) {
    Plan.Status = -1;
    Plan.Errno = EHOSTUNREACH;
    return Plan;
  }

  Plan.AggregatorRank = HandlerRank;

  FlightplanTable[Plan.RequestId] = Plan;
  for (HandlerInfo &H : Handlers) {
    if (H.Rank == HandlerRank) {
      H.InFlight += 1;
      break;
    }
  }

  return Plan;
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
