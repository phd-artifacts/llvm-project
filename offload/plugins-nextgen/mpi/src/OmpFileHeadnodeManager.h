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

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "EventSystem.h"

class OmpFileHeadnodeManager {
public:
  struct HandlerInfo {
    int Rank = -1;
    uint64_t InFlight = 0;
  };

  struct GlobalFileEntry {
    uint64_t GlobalFileId = 0;
    uint64_t OpenRequests = 0;
    int PreferredAggregatorRank = -1;
  };

  static OmpFileHeadnodeManager &instance();

  void initialize(int WorldSize, int HeadnodeRank);

  OmpFileIOPlan planRequest(const OmpFileIORequest &Request, const char *Path,
                            int LocalRank);

  void completeRequest(const OmpFileIOPlan &Plan);

private:
  OmpFileHeadnodeManager() = default;

  bool isWorkerRankUnlocked(int Rank) const;
  int pickLeastLoadedRankUnlocked(int ClientRank);
  void ensureHandlersUnlocked();

  std::mutex Mutex;
  bool Initialized = false;
  int WorldSize = 1;
  int HeadnodeRank = 0;

  std::unordered_map<uint64_t, OmpFileIOPlan> FlightplanTable;
  std::unordered_map<std::string, GlobalFileEntry> GlobalFileTable;
  std::vector<HandlerInfo> Handlers;
  uint64_t NextGlobalFileId = 1;
  uint64_t NextHandlerTieBreaker = 0;
};

#endif // _MPI_PROXY_OMPFILE_HEADNODE_MANAGER_H_
