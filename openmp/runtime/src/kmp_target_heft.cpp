//===---------------------- kmp_target_heft.cpp - HEFT scheduler
//---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Heterogeneous Earliest Finish Time scheduling algorithm.
//
//===----------------------------------------------------------------------===//

#include "kmp_target_heft.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include <set>
#include <deque>
#include <array>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>

namespace {
unsigned ioTaskKinds();
constexpr kmp_int32 IoTaskPrefetchKind = 1; // IoTaskPrefetch below
} // namespace

namespace detail {

// Implementation of HEFTContext
// =============================================================================

HEFTContext::HEFTContext(kmp_tdg_info *tdg)
    : CommMultiplier(1.0), DefaultCompCost(100.0), DefaultCommCost(1.0),
      NumThreads(1), tdg(tdg) {

  if (tdg != nullptr)
    computeUpwardRanks();
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_COMM_COEF"))
    CommMultiplier = std::stod(EnvStr);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_COMP_ESTIMATOR"))
    CompEstimatorMin = std::string(EnvStr) == "min";
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_COMP_COST"))
    DefaultCompCost = std::stod(EnvStr);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_COMM_COST"))
    DefaultCommCost = std::stod(EnvStr);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_LOG"))
    HEFTLogPath = EnvStr;
  if (char *EnvStr = std::getenv("OMPCLUSTER_NUM_EXEC_EVENT_HANDLERS"))
    NumThreads = std::stoi(EnvStr);
  if (char *EnvStr = std::getenv("OMPCLUSTER_INTERPOLATION_TYPE"))
    heterogeneousInterpolation = std::stoi(EnvStr);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_IO"))
    IoEnabled = std::atoi(EnvStr) != 0;
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_IO_LOG"))
    IoLog = EnvStr[0] != '\0' && EnvStr[0] != '0';
  // Rates are given per MiB in the environment, kept per byte here.
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_IO_LOCAL_US_PER_MB"))
    IoLocalUsPerByteEnv = std::stod(EnvStr) / (1 << 20);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_IO_REMOTE_US_PER_MB"))
    IoRemoteUsPerByteEnv = std::stod(EnvStr) / (1 << 20);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_IO_REMOTE_MIN_RATIO"))
    IoRemoteMinRatio = std::stod(EnvStr);
  if (char *EnvStr = std::getenv("OMPCLUSTER_HEFT_IO_PREFETCH_PRICING"))
    IoPrefetchPricing = std::atoi(EnvStr) != 0;

  buildDeviceTypeMap();
}

void HEFTContext::computeUpwardRanks() {

  for (auto CC : ComputationCost) {
    KG_TRACE(10, ("[Inter HEFT-after completion]: Task %d:\n",
                  KMP_TASKPTR_TO_TASKID(CC.first)));
    for (kmp_int32 device = 0; device < __kmpc_get_num_devices(); device++) {
      auto dit = CC.second.find(device);
      double val = (dit != CC.second.end()) ? dit->second : -2.0;
      KG_TRACE(10, ("device %d : value: %f ", device, val));
    }
    KG_TRACE(10, ("\n"));
  }

  KG_TRACE(10, ("[HEFT] Computing upward rank for all tasks.\n"));

  assert(tdg != nullptr && "The task map should not be NULL");
  for (auto *Task : ReverseTopologicalOrder(tdg)) {
    double MaxSuccUrankComm = 0.0;
    for (kmp_int32 idx_succ = 0; idx_succ < Task->nsuccessors; idx_succ++) {
      auto Succ = &(tdg->record_map[Task->successors[idx_succ]]);
      MaxSuccUrankComm =
          std::max(MaxSuccUrankComm,
                   UpwardRank[Succ] + computeAvgCommCost(*Task, *Succ));
    }
    // The recorded compute time leaves file I/O out; the rank puts it back
    // at the local rate (the cost of the placement the term aims for), so a
    // producer outranks its consumer by its own I/O as well and HEFT places
    // producers before measurement noise lets a consumer jump ahead.
    UpwardRank[Task] =
        computeAvgCompCost(*Task) + localIoCost(Task) + MaxSuccUrankComm;
  }
}

double HEFTContext::localIoCost(TaskPtr Task) const {
  if (!IoEnabled)
    return 0.0;
  const kmp_int32 Node = KMP_TASKPTR_TO_TDG_TASKID(Task);
  if (Node < 0 || static_cast<size_t>(Node) >= IoRanges.size())
    return 0.0;
  double Bytes = 0.0;
  for (const kmp_task_io_entry_t &E : IoRanges[Node])
    Bytes += static_cast<double>(E.bytes);
  return Bytes * IoLocalUsPerByte;
}

double HEFTContext::computeEST(TaskPtr Task, DeviceId Device) {
  double MaxPredEST = 0.0;
  // Iterate over predecessors
  for (kmp_int32 idx_pred = 0; idx_pred < Task->npredecessors; idx_pred++) {

    // Ignore deferred tasks for now
    if (isDeferredTask(*Task))
      continue;
    //    if((*get_schedule_map(tdg->tdg_id)).count(KMP_TDG_TASKID_TO_TASKID(tdg,
    //    Task->predecessors[idx_pred])) != 1)
    //	printf("Task %d, predecessor
    //%d\n",KMP_TASKPTR_TO_TDG_TASKID(Task),Task->predecessors[idx_pred]);
    assert((*get_schedule_map(tdg->tdg_id))
                   .count(KMP_TDG_TASKID_TO_TASKID(
                       tdg, Task->predecessors[idx_pred])) == 1 &&
           "Predecessor should already be scheduled!");

    DeviceId DevicePred(
        (*get_schedule_map(tdg->tdg_id))[KMP_TDG_TASKID_TO_TASKID(
            tdg, Task->predecessors[idx_pred])]);
    auto Pred = &(tdg->record_map[Task->predecessors[idx_pred]]);
    auto Candidate =
        AFT[Pred] + computeCommCost(*Pred, DevicePred, *Task, Device);

    if (Candidate > MaxPredEST)
      MaxPredEST = Candidate;
  }

  // Find the earliest start time if task is scheduled on device.
  EST[Task][Device] = std::max(DeviceAvail[Device], MaxPredEST);
  return EST[Task][Device];
}

double HEFTContext::computeEFT(TaskPtr Task, DeviceId Device) {
  // Calculate EST for task id, rank pair if it does not exist
  if (EST.count(Task) == 0 || EST[Task].count(Device) == 0)
    computeEST(Task, Device);

  // The recorded compute time leaves file I/O out (__kmp_task_io_complete);
  // its cost depends on where the files are owned, so it is added here.
  EFT[Task][Device] = computeCompCost(*Task, Device) +
                      computeIoCost(Task, Device) + EST[Task][Device];
  return EFT[Task][Device];
}

void HEFTContext::loadIoModel() {
  IoRanges.clear();
  IoOwner.clear();
  IoPrefetched.clear();
  IoTasks = IoBytes = IoRemoteBytes = 0;
  IoCostUs = 0.0;
  if (!IoEnabled && !IoLog)
    return;
  __kmp_task_io_graph_ranges(tdg, &IoRanges);

  // Defaults before any measurement: 1 GB/s on the owner, a third of that
  // away from it. A measured rate replaces its default; a single measured
  // rate sets the other one through the same ratio.
  constexpr double DefaultLocal = 1e-3, DefaultRatio = 3.0;
  double Local = -1.0, Remote = -1.0;
  __kmp_task_io_rates(&Local, &Remote);
  IoLocalSource = Local >= 0.0 ? "measured" : "default";
  IoRemoteSource = Remote >= 0.0 ? "measured" : "default";
  if (Local < 0.0)
    Local = Remote >= 0.0 ? Remote / DefaultRatio : DefaultLocal;
  if (Remote < 0.0)
    Remote = Local * DefaultRatio;
  if (IoRemoteMinRatio > 0.0 && Remote < Local * IoRemoteMinRatio) {
    Remote = Local * IoRemoteMinRatio;
    IoRemoteSource = "floor";
  }
  if (IoLocalUsPerByteEnv >= 0.0) {
    Local = IoLocalUsPerByteEnv;
    IoLocalSource = "env";
  }
  if (IoRemoteUsPerByteEnv >= 0.0) {
    Remote = IoRemoteUsPerByteEnv;
    IoRemoteSource = "env";
  }
  IoLocalUsPerByte = Local;
  IoRemoteUsPerByte = Remote;

  // With prefetch tasks, a remote read of a file one predecessor writes is
  // read into the reader's proxy right after the writer (addIoTasks), so the
  // reader pays the local rate wherever it runs and the file gives no reason
  // to prefer a device. Mirrors addIoTasks' condition, by file rather than
  // by range.
  // The recording run inserts no I/O tasks (addIoTasks), so it keeps the
  // remote price.
  if (!IoEnabled || !IoPrefetchPricing ||
      (ioTaskKinds() & (1u << IoTaskPrefetchKind)) == 0 ||
      __kmp_tdg_is_recording(tdg->tdg_status))
    return;
  const kmp_int32 Normal = KMP_ATOMIC_LD_RLX(&tdg->num_normal_tasks);
  auto writesKey = [&](kmp_int32 N, kmp_uint64 Key) {
    if (N < 0 || static_cast<size_t>(N) >= IoRanges.size())
      return false;
    for (const kmp_task_io_entry_t &E : IoRanges[N])
      if (E.is_write && E.path_key == Key)
        return true;
    return false;
  };
  for (kmp_int32 N = 0; N < Normal && static_cast<size_t>(N) < IoRanges.size();
       ++N) {
    const kmp_node_info &Node = tdg->record_map[N];
    if (Node.task == nullptr)
      continue;
    for (const kmp_task_io_entry_t &E : IoRanges[N]) {
      if (E.is_write || E.bytes == 0 || writesKey(N, E.path_key))
        continue;
      kmp_int32 Writers = 0;
      for (kmp_int32 I = 0; I < Node.npredecessors; ++I) {
        const kmp_int32 P = Node.predecessors[I];
        const kmp_int32 W = P < Normal ? P : tdg->record_map[P].predecessors[0];
        Writers += writesKey(W, E.path_key) ? 1 : 0;
      }
      if (Writers == 1)
        IoPrefetched[N].insert(E.path_key);
    }
  }
}

namespace {
/// True when the task writes the file and never reads it, so its first
/// access is a write and libompfile binds the file's owner to its device.
bool writesBeforeReading(const std::vector<kmp_task_io_entry_t> &Ranges,
                         kmp_uint64 PathKey) {
  bool Writes = false;
  for (const kmp_task_io_entry_t &E : Ranges) {
    if (E.path_key != PathKey)
      continue;
    if (!E.is_write)
      return false;
    Writes = true;
  }
  return Writes;
}
} // namespace

double HEFTContext::computeIoCost(TaskPtr Task, DeviceId Device,
                                  kmp_uint64 *RemoteBytes) const {
  if (RemoteBytes != nullptr)
    *RemoteBytes = 0;
  // Tasks on the head node do not run on a proxy and own nothing.
  if ((!IoEnabled && RemoteBytes == nullptr) || Device.Rank < 1)
    return 0.0;
  const kmp_int32 Node = KMP_TASKPTR_TO_TDG_TASKID(Task);
  if (Node < 0 || static_cast<size_t>(Node) >= IoRanges.size())
    return 0.0;
  const auto &Ranges = IoRanges[Node];
  const kmp_int32 Dev = Device.Rank - 1;
  const auto Prefetched = IoPrefetched.find(Node);
  double Cost = 0.0;
  for (const kmp_task_io_entry_t &E : Ranges) {
    if (E.bytes == 0)
      continue;
    kmp_int32 Owner = E.owner_rank;
    const auto It = IoOwner.find(E.path_key);
    if (It != IoOwner.end())
      Owner = It->second;
    else if (writesBeforeReading(Ranges, E.path_key))
      Owner = Dev;
    // An unknown owner (a declared range of a file not yet open) is priced
    // as local: it gives no reason to prefer a device.
    const bool Remote = Owner >= 0 && Owner != Dev;
    // A prefetched read is still counted as remote bytes, but costs the
    // reader the local rate.
    const bool Prefetch = Remote && Prefetched != IoPrefetched.end() &&
                          Prefetched->second.count(E.path_key) != 0;
    Cost += static_cast<double>(E.bytes) *
            (Remote && !Prefetch ? IoRemoteUsPerByte : IoLocalUsPerByte);
    if (Remote && RemoteBytes != nullptr)
      *RemoteBytes += E.bytes;
  }
  return IoEnabled ? Cost : 0.0;
}

void HEFTContext::noteIoPlacement(TaskPtr Task, DeviceId Device) {
  if (Device.Rank < 1)
    return;
  const kmp_int32 Node = KMP_TASKPTR_TO_TDG_TASKID(Task);
  if (Node < 0 || static_cast<size_t>(Node) >= IoRanges.size() ||
      IoRanges[Node].empty())
    return;
  const auto &Ranges = IoRanges[Node];
  kmp_uint64 Remote = 0;
  const double Cost = computeIoCost(Task, Device, &Remote);
  ++IoTasks;
  IoRemoteBytes += Remote;
  IoCostUs += Cost;
  const kmp_int32 Dev = Device.Rank - 1;
  for (const kmp_task_io_entry_t &E : Ranges) {
    IoBytes += E.bytes;
    if (IoOwner.count(E.path_key) != 0)
      continue;
    if (writesBeforeReading(Ranges, E.path_key))
      IoOwner[E.path_key] = Dev;
    else if (E.owner_rank >= 0)
      IoOwner[E.path_key] = E.owner_rank;
  }
  if (IoLog)
    fprintf(stderr,
            "[heft-io] task=%d device=%d io_us=%.1f remote_bytes=%llu\n", Node,
            Dev, Cost, static_cast<unsigned long long>(Remote));
}

void HEFTContext::reportIoModel(kmp_int32 TdgId) const {
  if (!IoLog)
    return;
  fprintf(stderr,
          "[heft-io] summary tdg=%d enabled=%d io_tasks=%llu bytes=%llu "
          "remote_bytes=%llu io_cost_us=%.0f local_us_per_mb=%.1f local_src=%s "
          "remote_us_per_mb=%.1f remote_src=%s\n",
          TdgId, IoEnabled ? 1 : 0, static_cast<unsigned long long>(IoTasks),
          static_cast<unsigned long long>(IoBytes),
          static_cast<unsigned long long>(IoRemoteBytes), IoCostUs,
          IoLocalUsPerByte * (1 << 20), IoLocalSource,
          IoRemoteUsPerByte * (1 << 20), IoRemoteSource);
  fflush(stderr);
}

double HEFTContext::scheduleTask(TaskPtr Task, DeviceId Device) {

  (*get_schedule_map(tdg->tdg_id))[KMP_TASKPTR_TO_TASKID(Task)] = {
      .state = TaskState::SCHEDULED,
      .rank = Device.Rank,
      .thread = Device.Thread,
      .translated_rank = -1,
      .task = Task,
  };

  assert((*get_schedule_map(tdg->tdg_id))[KMP_TASKPTR_TO_TASKID(Task)].state ==
         TaskState::SCHEDULED);
  AST[Task] = EST[Task][Device];
  AFT[Task] = EFT[Task][Device];
  noteIoPlacement(Task, Device);
  return AFT[Task];
}

double HEFTContext::scheduleTaskOnBestRank(TaskPtr Task) {
  const auto &EFTs = EFT.at(Task);

  // A NaN EFT compares false both ways, so std::min_element keeps it when it
  // is visited first; rank non-finite values last instead.
  auto Key = [](double V) {
    return std::isfinite(V) ? V : std::numeric_limits<double>::infinity();
  };
  auto BestRank = std::min_element(EFTs.begin(), EFTs.end(),
                                   [&Key](const auto &L, const auto &R) {
                                     return Key(L.second) < Key(R.second);
                                   });

  DeviceAvail[BestRank->first] = scheduleTask(Task, BestRank->first);
  return DeviceAvail[BestRank->first];
}

void HEFTContext::forceScheduleTask(TaskPtr Task, DeviceId Device) {
  computeEFT(Task, Device);
  DeviceAvail[Device] = scheduleTask(Task, Device);
}

void HEFTContext::deferTask(TaskPtr Task) {
  // Do not change state of FAILED tasks during the reschedule
  if ((*get_schedule_map(tdg->tdg_id))[KMP_TASKPTR_TO_TASKID(Task)].state !=
      TaskState::FAILED)
    (*get_schedule_map(tdg->tdg_id))[KMP_TASKPTR_TO_TASKID(Task)].state =
        TaskState::DEFERRED;
}

PriorityQueue HEFTContext::buildQueue() const {
  PriorityQueue Pqueue{UpwardRank.begin(), UpwardRank.end()};

  std::sort(Pqueue.begin(), Pqueue.end(),
            [](const auto &L, const auto &R) { return L.second > R.second; });

  return Pqueue;
}

std::vector<DeviceId> HEFTContext::buildDeviceList(kmp_int32 NumRanks) const {
  std::vector<DeviceId> Devices;
  Devices.reserve(NumRanks);
  for (kmp_int32 R = 1; R <= NumRanks; R++)
    for (kmp_int32 H = 0; H < NumThreads; H++)
      Devices.emplace_back(R, H);
  return Devices;
}

void HEFTContext::clear() {
  EST.clear();
  EFT.clear();
  AST.clear();
  AFT.clear();
  UpwardRank.clear();
  DeviceAvail.clear();
  CommunicationCost.clear();
  AverageComputationCost.clear();
  ComputationCost.clear();
}

void HEFTContext::clearCompCost() {
  AverageComputationCost.clear();
  ComputationCost.clear();
}

void HEFTContext::setTaskMap(kmp_int32 tdg_id) {
  tdg = __kmpc_find_tdg(tdg_id);
  assert(tdg != nullptr && "Expected a non-null task map!");
}

double HEFTContext::computeCommCost(const kmp_node_info &A, DeviceId DeviceA,
                                    const kmp_node_info &B, DeviceId DeviceB) {
  // If the tasks are to execute on the same rank, we do not pay for the
  // communication because no data needs to be moved.
  if (DeviceA.Rank == DeviceB.Rank)
    return 0.0;

  // FIXME: take actual bandwidth between RankA and RankB into consideration
  // instead of just using the average cost.
  return computeAvgCommCost(A, B);
}

/// Search for all `out` dependencies of A which are also `in` dependencies of B
/// and accumulate the size of those dependencies (in bytes).
double HEFTContext::computeAvgCommCost(const kmp_node_info &A,
                                       const kmp_node_info &B) {

  double TotalBytes = 0.0;
  // If the communication cost was already computed, then use the cached value.
  auto ItA = CommunicationCost.find(&A);
  if (ItA != CommunicationCost.end()) {
    auto ItB = ItA->second.find(&B);
    if (ItB != ItA->second.end()) {
      // The cache holds bytes; scale them like a freshly computed value.
      return CommMultiplier * ItB->second;
    }
  }

  // Look for arguments which are common between tasks A and B and accumulate
  // their sizes in bytes.
  if (A.target_data != nullptr && B.target_data != nullptr) {
    for (kmp_int32 I = 0; I < A.target_data->num_args; I++) {
      for (kmp_int32 J = 0; J < B.target_data->num_args; J++) {
        if (A.target_data->args[I] == B.target_data->args[J]) {
          TotalBytes += A.target_data->arg_sizes[I];
        }
      }
    }
  }
  CommunicationCost[&A][&B] = TotalBytes;

  KG_TRACE(10, ("[HEFT] Dependency `%d -> %d` has average cost: %lf KB\n",
                KMP_TASKNODE_TO_TASKID(A), KMP_TASKNODE_TO_TASKID(B),
                TotalBytes / 1024.0));

  return CommMultiplier * TotalBytes;
}

double HEFTContext::computeAvgCompCost(const kmp_node_info &Task) {
  KG_TRACE(10, ("[HEFT] computeAvgCompCost (start) : tdg task id: %d\n",
                KMP_TASKNODE_TO_TASKID(Task)));
  // If computation cost was already computed, then use the cached value.
  auto It = AverageComputationCost.find(&Task);
  if (It != AverageComputationCost.end()) {
    KG_TRACE(10, ("[HEFT] computeAvgCompCost (end) : Using cached value\n"));
    return It->second;
  }

  // TODO Something went wrong, we should always have a value after
  // extrapolation.
  KMP_DEBUG_ASSERT(false);
  KG_TRACE(10, ("[HEFT] computeAvgCompCost (end) : Using default value\n"));
  return AverageComputationCost[&Task] = 100.0;
}

double HEFTContext::computeCompCost(const kmp_node_info &Task,
                                    DeviceId Device) {

  KG_TRACE(10,
           ("[HEFT] computeCompCost (start) : tdg task id: %d, device id %d\n",
            KMP_TASKNODE_TO_TASKID(Task), Device.Rank - 1));
  // If it exists, we use the recorded time from the tdg structure

  auto It = ComputationCost.find(&Task);
  if (It != ComputationCost.end()) {
    auto Value = It->second.find(Device.Rank - 1);
    if (Value != It->second.end()) {
      return Value->second;
    }
  }

  // If we don't have the task time for the precise device, use the average time
  // Something went wrong, we should always have a value after extrapolation.
  //  KMP_DEBUG_ASSERT(false);
  KG_TRACE(10, ("[HEFT] computeCompCost : Using Averaged Comp cost"));
  return ComputationCost[&Task][Device.Rank - 1] = computeAvgCompCost(Task);
}

void HEFTContext::computeAvgCompCosts() {
  // TODO should we construct it once then pass it as an argument?
  for (auto Taskptr : TopologicalOrder(tdg)) {
    double accumulated_time = 0.0;
    kmp_int32 profiled_device = 0;
    for (auto DeviceValue : ComputationCost[Taskptr]) {
      accumulated_time += DeviceValue.second;
      profiled_device++;
    }
    AverageComputationCost[Taskptr] = accumulated_time / profiled_device;
  }
}

void HEFTContext::homogeneousInterpolation() {
  computeAvgCompCosts();
  for (auto Taskptr : TopologicalOrder(tdg)) {
    double average_value = AverageComputationCost[Taskptr];
    for (kmp_int32 device = 0; device < __kmpc_get_num_devices(); device++) {
      if (!ComputationCost[Taskptr].count(device))
        ComputationCost[Taskptr][device] = average_value;
    }
  }
}

void HEFTContext::linearInterpolation() {
  // TODO:Limit the iteration number
  kmp_int32 device_number = __kmpc_get_num_devices();
  std::vector<std::vector<double>> beta(
      device_number,
      std::vector<double>(device_number,
                          -1.0)); // We want to compute the betas (proportional
                                  // coefficient between the devices' speeds)

  // Computation of the Betas

  for (kmp_int32 device_0 = 0; device_0 < device_number; device_0++) {
    for (kmp_int32 device_1 = device_0; device_1 < device_number; device_1++) {
      double sum = 0.0;
      kmp_int32 size = 0;
      for (auto Taskptr : TopologicalOrder(tdg)) {
        if (isTargetDataTask(Taskptr) ||
            !ComputationCost[Taskptr].count(device_0) ||
            !ComputationCost[Taskptr].count(device_1))
          continue;
        const double c0 = ComputationCost[Taskptr][device_0];
        const double c1 = ComputationCost[Taskptr][device_1];
        // Samples are floored at 1 us where they are recorded; this guard
        // keeps log(c1 / c0) finite if a zero ever reaches the matrix.
        if (!(c0 > 0.0) || !(c1 > 0.0))
          continue;
        sum += std::log(c1 / c0);
        size++;
        if (size > maxIteration)
          break;
      }
      if (size) {
        beta[device_0][device_1] = std::exp(sum / size);
        beta[device_1][device_0] = 1 / beta[device_0][device_1];
        KG_TRACE(10, ("[HEFT] Getting profile for beta %d, %d: %d\n", device_0,
                      device_1));
      } else {
        KG_TRACE(10,
                 ("[HEFT] No profile for beta %d, %d\n", device_0, device_1));
      }
    }
  }

  // Interpolation of the pivot Beta[0][X]

  double sum_b0 = 0.0;
  kmp_int32 size_b0 = 0;
  for (auto beta0 : beta[0]) {
    if (beta0 > 0.0) {
      sum_b0 += std::log(beta0);
      size_b0++;
    }
  }

  // No ratio against device 0 at all: assume equal speeds rather than 0/0.
  double geomean_b0 = size_b0 ? std::exp(sum_b0 / size_b0) : 1.0;

  std::deque<kmp_int32> devices_to_visite;
  for (kmp_int32 i = 0; i < device_number; ++i) {
    devices_to_visite.push_back(i);
  }
  while (!devices_to_visite.empty()) {
    kmp_int32 queue_size = devices_to_visite.size();
    for (kmp_int32 it = 0; it < queue_size; ++it) {
      kmp_int32 device_1 = devices_to_visite.front();
      devices_to_visite.pop_front();
      if (beta[0][device_1] != -1.0) {
        // We already have the value
        continue;
      }
      double sum = 0.0;
      kmp_int32 size = 0;
      for (kmp_int32 device_2 = 1;
           device_2 < device_number && size <= maxIteration; device_2++) {
        if (beta[0][device_2] != -1 && beta[device_2][device_1] != -1) {
          sum += std::log(beta[0][device_2] * beta[device_2][device_1]);
          size++;
        }
      }
      if (size) {
        beta[0][device_1] = std::exp(sum / size);
        beta[device_1][0] = 1 / beta[0][device_1];
      } else {
        devices_to_visite.push_back(
            device_1); // We were not able to determine a value for B0,d1, we
                       // will have to go back to it
      }
    }
    if (devices_to_visite.size() ==
        queue_size) { // We need to do a partial homogenous assumptions to
                      // continue computing the pivot
      beta[0][devices_to_visite.front()] = geomean_b0;
      beta[devices_to_visite.front()][0] =
          1 / beta[0][devices_to_visite.front()];
      devices_to_visite.pop_front();
    }
  }

  // Interpolation of the Betas[X][Y]
  for (kmp_int32 device_0 = 1; device_0 < device_number; device_0++) {
    std::vector<kmp_int32> device_list;
    // Searching for a suitable Beta for beta_0
    for (kmp_int32 device_1 = device_0 + 1; device_1 < device_number;
         device_1++) {
      if (beta[device_0][device_1] == -1.0) // We don't have the value yet, we
                                            // can't search another beta from it
        continue;
      device_list.push_back(device_1);
    }

    for (kmp_int32 device_1 = device_0 + 1; device_1 < device_number;
         device_1++) {
      if (beta[device_0][device_1] !=
          -1.0) // We already have a value, no need to complete
        continue;
      double sum = 0.0;
      kmp_int32 size = 0;
      for (kmp_int32 second_device : device_list) {
        sum += std::log(beta[device_0][second_device] * beta[second_device][0] *
                        beta[0][device_1]);
        size++;
        if (size > maxIteration)
          break;
      }
      // device_list only holds higher-numbered devices with a known beta, so
      // it is often empty for the last devices: sum / size would be 0/0 and
      // the NaN beta makes every interpolated cost on device_1 NaN, which
      // then wins scheduleTaskOnBestRank and draws every task to it. Take the
      // path through the pivot instead; it is set for every device by now.
      if (!size)
        ++InterpFallbackPairs;
      beta[device_0][device_1] =
          size ? std::exp(sum / size) : beta[device_0][0] * beta[0][device_1];
      beta[device_1][device_0] = 1 / beta[device_0][device_1];
    }
  }

  // We just need to compute the costs now

  for (auto Taskptr : TopologicalOrder(tdg)) {
    if (isTargetDataTask(Taskptr))
      continue;
    // Copy known profiled costs for the task
    std::unordered_map<kmp_int32, double> profiled_comp_cost;
    kmp_int32 size = ComputationCost[Taskptr].size();
    for (auto CC : ComputationCost[Taskptr]) {
      profiled_comp_cost[CC.first] = CC.second;
      KMP_DEBUG_ASSERT(CC.second > 0.0);
    }
    for (kmp_int32 device_0 = 0; device_0 < device_number; device_0++) {
      // We find a pair (task,device) that was not profiled
      if (ComputationCost[Taskptr].count(device_0))
        continue;
      // We take the average of all the path from the tasks that were profiled
      double sum = 0.0;
      for (auto [profiled_device, profiled_value] : profiled_comp_cost) {
        sum += std::log(profiled_value * beta[profiled_device][device_0]);
      }
      ComputationCost[Taskptr][device_0] = std::exp(sum / size);
    }
  }
}

void HEFTContext::heterogeneousLinearInterpolation() {

  for (int i = 0; i < device_restricted_CompCost.size(); i++) {
    if (device_restricted_CompCost[i].empty())
      continue;
    restrictedLinearInterpolation(
        device_restricted_CompCost[i],
        i); // TODO: We might have to pass an argument to have restricted
            // device-> actual device
  }

  completeheterogeneousInterpolation();
}

void HEFTContext::restrictedLinearInterpolation(
    ComputationMap restrictedComputationCost, int type) {

  const std::vector<kmp_int32> &device_list = restricted_to_device_map.at(type);
  kmp_int32 device_number = (kmp_int32)device_list.size();
  std::vector<std::vector<double>> beta(
      device_number,
      std::vector<double>(device_number,
                          -1.0)); // We want to compute the betas (proportional
                                  // coefficient between the devices' speeds)

  // Computation of the Betas

  for (kmp_int32 device_0 = 0; device_0 < device_number; device_0++) {
    for (kmp_int32 device_1 = device_0; device_1 < device_number; device_1++) {
      kmp_int32 actual_device_0 = device_list[device_0];
      kmp_int32 actual_device_1 = device_list[device_1];
      double sum = 0.0;
      kmp_int32 size = 0;
      for (auto Taskptr : TopologicalOrder(tdg)) {
        if (isTargetDataTask(Taskptr) ||
            !restrictedComputationCost[Taskptr].count(actual_device_0) ||
            !restrictedComputationCost[Taskptr].count(actual_device_1))
          continue;
        const double c0 = restrictedComputationCost[Taskptr][actual_device_0];
        const double c1 = restrictedComputationCost[Taskptr][actual_device_1];
        // Same zero-charge guard as linearInterpolation.
        if (!(c0 > 0.0) || !(c1 > 0.0))
          continue;
        sum += std::log(c1 / c0);
        size++;
        if (size > maxIteration)
          break;
      }
      if (size) {
        beta[device_0][device_1] = std::exp(sum / size);
        beta[device_1][device_0] = 1 / beta[device_0][device_1];
        KG_TRACE(10, ("[HEFT] Getting profile for beta %d, %d: %d\n",
                      actual_device_0, actual_device_1));
      } else {
        KG_TRACE(10, ("[HEFT] No profile for beta %d, %d\n", actual_device_0,
                      actual_device_1));
      }
    }
  }

  // Interpolation of the pivot Beta[0][X]

  double sum_b0 = 0.0;
  kmp_int32 size_b0 = 0;
  for (auto beta0 : beta[0]) {
    if (beta0 > 0.0) {
      sum_b0 += std::log(beta0);
      size_b0++;
    }
  }

  // No ratio against device 0 at all: assume equal speeds rather than 0/0.
  double geomean_b0 = size_b0 ? std::exp(sum_b0 / size_b0) : 1.0;

  std::deque<kmp_int32> devices_to_visite;
  for (kmp_int32 i = 0; i < device_number; ++i) {
    devices_to_visite.push_back(i);
  }
  while (!devices_to_visite.empty()) {
    kmp_int32 queue_size = devices_to_visite.size();
    for (kmp_int32 it = 0; it < queue_size; ++it) {
      kmp_int32 device_1 = devices_to_visite.front();
      devices_to_visite.pop_front();
      if (beta[0][device_1] != -1.0) {
        // We already have the value
        continue;
      }
      double sum = 0.0;
      kmp_int32 size = 0;
      for (kmp_int32 device_2 = 1;
           device_2 < device_number && size <= maxIteration; device_2++) {
        if (beta[0][device_2] != -1 && beta[device_2][device_1] != -1) {
          sum += std::log(beta[0][device_2] * beta[device_2][device_1]);
          size++;
        }
      }
      if (size) {
        beta[0][device_1] = std::exp(sum / size);
        beta[device_1][0] = 1 / beta[0][device_1];
      } else {
        devices_to_visite.push_back(
            device_1); // We were not able to determine a value for B0,d1, we
                       // will have to go back to it
      }
    }
    if (devices_to_visite.size() ==
        queue_size) { // We need to do a partial homogenous assumptions to
                      // continue computing the pivot
      beta[0][devices_to_visite.front()] = geomean_b0;
      beta[devices_to_visite.front()][0] =
          1 / beta[0][devices_to_visite.front()];
      devices_to_visite.pop_front();
    }
  }

  // Interpolation of the Betas[X][Y]
  for (kmp_int32 device_0 = 1; device_0 < device_number; device_0++) {
    std::vector<kmp_int32> device_list;
    // Searching for a suitable Beta for beta_0
    for (kmp_int32 device_1 = device_0 + 1; device_1 < device_number;
         device_1++) {
      if (beta[device_0][device_1] == -1.0) // We don't have the value yet, we
                                            // can't search another beta from it
        continue;
      device_list.push_back(device_1);
    }

    for (kmp_int32 device_1 = device_0 + 1; device_1 < device_number;
         device_1++) {
      if (beta[device_0][device_1] !=
          -1.0) // We already have a value, no need to complete
        continue;
      double sum = 0.0;
      kmp_int32 size = 0;
      for (kmp_int32 second_device : device_list) {
        sum += std::log(beta[device_0][second_device] * beta[second_device][0] *
                        beta[0][device_1]);
        size++;
        if (size > maxIteration)
          break;
      }
      // device_list only holds higher-numbered devices with a known beta, so
      // it is often empty for the last devices: sum / size would be 0/0 and
      // the NaN beta makes every interpolated cost on device_1 NaN, which
      // then wins scheduleTaskOnBestRank and draws every task to it. Take the
      // path through the pivot instead; it is set for every device by now.
      if (!size)
        ++InterpFallbackPairs;
      beta[device_0][device_1] =
          size ? std::exp(sum / size) : beta[device_0][0] * beta[0][device_1];
      beta[device_1][device_0] = 1 / beta[device_0][device_1];
    }
  }

  // We just need to compute the costs now
  for (auto Taskptr : TopologicalOrder(tdg)) {
    if (isTargetDataTask(Taskptr))
      continue;
    // Copy known profiled costs for the task
    std::unordered_map<kmp_int32, double> profiled_comp_cost;
    kmp_int32 size = 0;
    for (kmp_int32 device_0 = 0; device_0 < device_number; device_0++) {
      kmp_int32 actual_device_0 = device_list[device_0];
      auto it = restrictedComputationCost[Taskptr].find(actual_device_0);
      if (it == restrictedComputationCost[Taskptr].end() || it->second < 0.0)
        continue;
      profiled_comp_cost[device_0] = it->second;
      size++;
    }
    for (kmp_int32 device_0 = 0; device_0 < device_number; device_0++) {
      kmp_int32 actual_device_0 = device_list[device_0];
      // We find a pair (task,device) that was not profiled
      if (restrictedComputationCost[Taskptr].count(actual_device_0))
        continue;
      // We take the average of all the path from the tasks that were profiled
      double sum = 0.0;
      for (auto [profiled_device, profiled_value] : profiled_comp_cost) {
        sum += std::log(profiled_value * beta[profiled_device][device_0]);
      }
      // Without a profile on any device of this type there is nothing to
      // interpolate from; leave the pair absent (operator[] would insert a
      // 0.0 cost) so completeheterogeneousInterpolation fills it.
      if (size > 0)
        ComputationCost[Taskptr][actual_device_0] = std::exp(sum / size);
    }
  }
}

void HEFTContext::completeheterogeneousInterpolation() {

  int num_types = deviceTypeNumber;

  double mean = 1.0;
  std::vector<double> coef(deviceTypeNumber, 1.0);

  double sum_mean = 0;
  int size = 0;

  std::unordered_map<TaskPtr, std::vector<double>> task_type_log_sum;
  std::unordered_map<TaskPtr, std::vector<int>> task_type_count;

  std::vector<double> type_dev_log_sum(deviceTypeNumber, 0);
  std::vector<kmp_int64> type_dev_count(deviceTypeNumber, 0);

  for (auto Taskptr : TopologicalOrder(tdg)) {
    if (isTargetDataTask(Taskptr))
      continue;
    double task_log_sum = 0;
    int task_count = 0;
    task_type_log_sum[Taskptr].assign(deviceTypeNumber, 0);
    task_type_count[Taskptr].assign(deviceTypeNumber, 0);
    for (auto CC : ComputationCost[Taskptr]) {
      if (CC.second < 0.0)
        continue;
      int type = deviceTypeMap.at(CC.first);
      task_type_log_sum[Taskptr][type] += std::log(CC.second);
      task_type_count[Taskptr][type]++;
      task_log_sum += std::log(CC.second);
      task_count++;
    }
    if (task_count == 0)
      continue;

    double task_avg_log = task_log_sum / task_count;
    sum_mean += task_avg_log;

    for (int type = 0; type < deviceTypeNumber; ++type) {
      if (task_type_count[Taskptr][type] == 0)
        continue;
      double type_avg_log =
          task_type_log_sum[Taskptr][type] / task_type_count[Taskptr][type];
      type_dev_log_sum[type] += (type_avg_log - task_avg_log);
      type_dev_count[type]++;
    }
    ++size;
  }

  if (size)
    mean = std::exp(sum_mean / size);

  for (int type = 0; type < num_types; ++type)
    if (type_dev_count[type] > 0)
      coef[type] = std::exp(type_dev_log_sum[type] / type_dev_count[type]);

  for (auto Taskptr : TopologicalOrder(tdg)) {
    if (isTargetDataTask(Taskptr))
      continue;
    // 1 = some device of the type has a cost for this task; 0 = none does
    // (absent or negative), so the type is filled from the reference type.
    std::vector<int> presence(num_types, 0);
    for (auto CC : ComputationCost[Taskptr])
      if (CC.second >= 0.0)
        presence[deviceTypeMap.at(CC.first)] = 1;

    int referenceType = -1;
    double referenceEstimate = 0;
    for (int type = 0; type < num_types; ++type) {
      if (presence[type] == 1) {
        referenceType = type;
        referenceEstimate = task_type_count[Taskptr][type]
                                ? std::exp(task_type_log_sum[Taskptr][type] /
                                           task_type_count[Taskptr][type])
                                : mean * coef[type];
        break;
      }
    }

    for (int type = 0; type < num_types; ++type) {
      if (presence[type] != 0)
        continue;
      double estimate = (referenceType != -1) ? referenceEstimate * coef[type] /
                                                    coef[referenceType]
                                              : mean * coef[type];
      auto Devices = restricted_to_device_map.find(type);
      if (Devices == restricted_to_device_map.end())
        continue;
      for (kmp_int32 Device : Devices->second) {
        auto It = ComputationCost[Taskptr].find(Device);
        if (It == ComputationCost[Taskptr].end() || It->second < 0.0)
          ComputationCost[Taskptr][Device] = estimate;
      }
    }
  }
}

void HEFTContext::printCompCostMatrices() {
  kmp_int32 device_number = __kmpc_get_num_devices();

  KG_TRACE(10,
           ("[Inter HEFT] ---- ComputationCost matrix (task x device) ----\n"));
  for (auto Taskptr : TopologicalOrder(tdg)) {
    std::string row = "[Inter HEFT] Task " +
                      std::to_string(KMP_TASKPTR_TO_TASKID(Taskptr)) + ": ";
    for (kmp_int32 device = 0; device < device_number; device++) {
      auto it = ComputationCost[Taskptr].find(device);
      row += "d" + std::to_string(device) + "=";
      row += (it != ComputationCost[Taskptr].end()) ? std::to_string(it->second)
                                                    : std::string("-");
      row += "  ";
    }
    KG_TRACE(10, ("%s\n", row.c_str()));
  }

  if (!heterogeneousInterpolation || deviceTypeNumber <= 1)
    return;

  for (int type = 0; type < deviceTypeNumber; type++) {
    KG_TRACE(10, ("[Inter HEFT] ---- device_restricted_CompCost[type=%d] "
                  "(task x device) ----\n",
                  type));
    if (device_restricted_CompCost[type].empty()) {
      KG_TRACE(10, ("[Inter HEFT]   (empty)\n"));
      continue;
    }
    for (auto &TaskEntry : device_restricted_CompCost[type]) {
      std::string row = "[Inter HEFT] Task " +
                        std::to_string(KMP_TASKPTR_TO_TASKID(TaskEntry.first)) +
                        ": ";
      for (kmp_int32 device = 0; device < device_number; device++) {
        auto it = TaskEntry.second.find(device);
        row += "d" + std::to_string(device) + "=";
        row += (it != TaskEntry.second.end()) ? std::to_string(it->second)
                                              : std::string("-");
        row += "  ";
      }
      KG_TRACE(10, ("%s\n", row.c_str()));
    }
  }
}

void HEFTContext::interpolateCompCost(kmp_int32 scheduling_iteration_counter,
                                      kmp_int32 scheduling_policy,
                                      kmp_int32 initial_iteration) {
  // TODO: Can we decorolate it from HEFTContext?
  KG_TRACE(10, ("[HEFT] interpolateCompCost (start): heterogeneous: %d, device "
                "type number: %d \n",
                heterogeneousInterpolation, deviceTypeNumber));
  __kmpc_print_profiled_comp_cost(tdg);
  if (heterogeneousInterpolation && deviceTypeNumber > 1) {
    for (auto &typeMap : device_restricted_CompCost)
      typeMap.clear();
  }

  InterpKind = "none";
  InterpFallbackPairs = 0;
  bool is_incomplete_profile = false;
  for (auto Taskptr : TopologicalOrder(tdg)) {
    for (kmp_int32 device = 0; device < __kmpc_get_num_devices(); device++) {
      int type = 0;
      if (heterogeneousInterpolation && deviceTypeNumber > 1) {
        auto it = deviceTypeMap.find(device);
        type = (it != deviceTypeMap.end()) ? it->second : 0;
        if (it == deviceTypeMap.end()) {
          KG_TRACE(10, ("[Inter HEFT]: device (id %d) : type not found in "
                        "device type map\n",
                        device));
        }
        KMP_DEBUG_ASSERT((size_t)type <
                         device_restricted_CompCost.size()); // canary
      }

      if (isTargetDataTask(*Taskptr)) {
        ComputationCost[Taskptr][device] = 1.0;
      } else {
        if (tdg->record_map[KMP_TASKPTR_TO_TDG_TASKID(Taskptr)]
                .task_time[device]
                .iteration_number) {
          KG_TRACE(10,
                   ("[Inter HEFT]: Adding task %d , device %d to CompCost\n",
                    KMP_TASKPTR_TO_TDG_TASKID(Taskptr), device));
          const kmp_recorded_task_time_t &Time =
              tdg->record_map[KMP_TASKPTR_TO_TDG_TASKID(Taskptr)]
                  .task_time[device];
          // min_time is -1 until a sample exists. Times are whole us and a
          // task's I/O is subtracted from its own (__kmp_task_io_complete), so
          // a short or I/O-bound task records 0; every interpolation takes a
          // log of these costs, so floor them at 1 us.
          const double Sample = std::max(
              1.0, static_cast<double>(CompEstimatorMin && Time.min_time >= 0
                                           ? Time.min_time
                                           : Time.mean_time));
          ComputationCost[Taskptr][device] = Sample;
          if (heterogeneousInterpolation && deviceTypeNumber > 1) {
            device_restricted_CompCost[deviceTypeMap[device]][Taskptr][device] =
                Sample;
          }
        } else {
          KG_TRACE(10, ("[Inter HEFT]: Task %d , device %d : no profile\n",
                        KMP_TASKPTR_TO_TDG_TASKID(Taskptr), device));
          is_incomplete_profile = true;
          if (!scheduling_iteration_counter) // No profile information : set the
                                             // task value to 100.0
            ComputationCost[Taskptr][device] = 100.0;
        }
      }
    }
  }

  printCompCostMatrices();
  // We could simplify these ifs with a switch into computeAvgCompCosts()

  if (!is_incomplete_profile || !scheduling_iteration_counter) {
    computeAvgCompCosts();
    KG_TRACE(10, ("[Inter HEFT] interpolateCompCost (end): iteration %d no "
                  "need for interpolation\n ",
                  scheduling_iteration_counter));
    return;
  }

  if (scheduling_iteration_counter == 1) {
    InterpKind = "homogeneous";
    homogeneousInterpolation();
    computeAvgCompCosts();
    KG_TRACE(10, ("[Inter HEFT] interpolateCompCost (end): iteration %d "
                  "homogeneous hypothesis\n ",
                  scheduling_iteration_counter));
    return;
  }

  if (heterogeneousInterpolation && deviceTypeNumber > 1 &&
      scheduling_iteration_counter > 3) {
    InterpKind = "heterogeneous";
    heterogeneousLinearInterpolation();
    computeAvgCompCosts();
    KG_TRACE(10, ("[Inter HEFT] interpolateCompCost (end): iteration %d "
                  "heterogeneous devices interpolation\n ",
                  scheduling_iteration_counter));
    return;
  }

  // general case
  InterpKind = "linear";
  linearInterpolation();
  computeAvgCompCosts();

  KG_TRACE(10, ("[Inter HEFT] interpolateCompCost (end): iteration %d "
                "homogeneous devices interpolation\n",
                scheduling_iteration_counter));
}

void HEFTContext::printInterpolation(kmp_int32 iteration) const {
  // A non-finite cost wins scheduleTaskOnBestRank's comparison or poisons
  // DeviceAvail; the gate fails on any (test-heft-comm-cost).
  kmp_int32 Costs = 0, NonFinite = 0;
  for (const auto &Row : ComputationCost)
    for (const auto &Cell : Row.second) {
      ++Costs;
      if (!std::isfinite(Cell.second))
        ++NonFinite;
    }
  fprintf(stderr,
          "[heft-io] interp tdg=%d iter=%d kind=%s fallback_pairs=%d "
          "costs=%d nonfinite_costs=%d\n",
          tdg ? tdg->tdg_id : -1, iteration, InterpKind, InterpFallbackPairs,
          Costs, NonFinite);
  fflush(stderr);
}

void HEFTContext::buildDeviceTypeMap() {
  // TODO: Do we want to merge it with DeviceList?
  deviceTypeMap.clear();
  device_restricted_map.clear();
  restricted_to_device_map.clear();
  device_restricted_CompCost.clear();

  kmp_int32 num_devices = __kmpc_get_num_devices();
  device_restricted_map.resize(num_devices);

  std::unordered_map<kmp_device_type, int> rawTypeToIndex;

  for (int device = 0; device < num_devices; device++) {
    kmp_device_type rawType = __kmpc_get_device_type(device);
    auto it_inserted =
        rawTypeToIndex.try_emplace(rawType, (int)rawTypeToIndex.size());
    int type = it_inserted.first->second;

    deviceTypeMap[device] = type;
    device_restricted_map[device] = type;
    restricted_to_device_map[type].push_back(device);
  }

  deviceTypeNumber = (kmp_int32)rawTypeToIndex.size();
  device_restricted_CompCost.resize(deviceTypeNumber);
  KMP_DEBUG_ASSERT(device_restricted_CompCost.size() ==
                   (size_t)deviceTypeNumber);
}

/// Dump internal HEFT information to a file.
void HEFTContext::dump(kmp_int32 NumDevices) {
  // FIXME: print information with rescheduling and memory tasks
  if (HEFTLogPath.empty())
    return;

  FILE *fp = fopen(HEFTLogPath.c_str(), "w");

  if (fp == NULL) {
    KG_TRACE(10, ("[HEFT] Failed to open log file.\n"));
    return;
  }

  auto Devices = buildDeviceList(NumDevices);

  KG_TRACE(10,
           ("[HEFT] Dumping HEFT stats to file '%s'\n", HEFTLogPath.c_str()));

  // FIXME: Print the header with correct columns for EST/EFT
  fprintf(fp,
          "%-4s  %-10s  %-10s  %-10s  %-10s  %-10s  %-7s  %-7s  %-7s  %-7s"
          "%-34s  %-34s\n",
          "ID", "FNID", "Rank", "AST", "AFT", "Dev", "Root", "Exit", "Data",
          "Classic", "EST", "EFT");

  std::vector<kmp_node_info *> TasksV;
  TasksV.reserve(tdg->num_tasks);
  for (kmp_int32 i = 0; i < tdg->num_tasks; i++) {
    TasksV.push_back(&(tdg->record_map[i]));
  }

  std::sort(TasksV.begin(), TasksV.end(),
            [&](kmp_node_info *A, kmp_node_info *B) {
              return UpwardRank[A] > UpwardRank[B];
            });

  // Iterate over tasks
  for (auto const *task_ptr : TasksV) {
    auto &task = *task_ptr;
    auto *sched_map = get_schedule_map(tdg->tdg_id);
    uint64_t ofid = (uint64_t)task.target_data->outlined_fn_id;
    auto task_id = KMP_TASKPTR_TO_TASKID(task_ptr);
    auto it = sched_map->find(task_id);
    int rank =
        (it != sched_map->end()) ? it->second.rank : -1; // -1 = "unknown"

    if (it == sched_map->end()) {
      KG_TRACE(10,
               ("Warning: task_id=%d not found in schedule map\n", task_id));
    }

    fprintf(fp,
            "%04d  0x%08lx  %10.2lf  %10.2lf  %10.2lf  %7d  %7d  %7d  %7d  %4d",
            task_id, ofid, UpwardRank[&task], AST[&task], AFT[&task], rank,
            isRootTask(task), isExitTask(task), isTargetDataTask(task),
            isClassicalTask(task));

    /* fprintf(fp,
             "%04d  0x%08lx  %10.2lf  %10.2lf  %10.2lf  %7d  %7d  %7d  %7d %4d",
             KMP_TASKPTR_TO_TASKID(task_ptr), ofid, UpwardRank[&task],
       AST[&task], AFT[&task],
             (*get_schedule_map(tdg->tdg_id))[KMP_TASKPTR_TO_TASKID(task_ptr)].rank,
       isRootTask(task), isExitTask(task), isTargetDataTask(task),
       isClassicalTask(task));*/
    for (const auto &Device : Devices)
      fprintf(fp, "  %10.2lf", EST[&task][Device]);
    for (const auto &Device : Devices)
      fprintf(fp, "  %10.2lf", EFT[&task][Device]);
    fprintf(fp, "\n");
  }

  fprintf(fp, "\n");
  fprintf(fp, "Dependencies\n");
  fprintf(fp, "-------------------------------------------------------\n");

  for (const auto &A : CommunicationCost) {
    auto id_A = KMP_TASKPTR_TO_TASKID(A.first);
    for (const auto &B : A.second) {
      auto id_B = KMP_TASKPTR_TO_TASKID(B.first);
      fprintf(fp, "%04d -> %04d : %10.4lf\n", id_A, id_B, B.second);
    }
  }

  fprintf(fp, "\n");
  fprintf(fp, "Schedule Map\n");
  fprintf(fp, "-------------------------------------------------------\n");

  for (const auto &s : (*get_schedule_map(tdg->tdg_id))) {
    fprintf(fp, "%04d: %04d %d %d\n", s.first,
            KMP_TASKPTR_TO_TASKID(s.second.task), s.second.rank,
            s.second.state);
  }

  fclose(fp);
}

} // namespace detail

// Implementation of HEFTScheduler
// =============================================================================

HEFTScheduler::DumpOptions::DumpOptions() {
  if (char *Value = std::getenv("OMPCLUSTER_HEFT_DUMP_EDGE_LABEL"))
    EdgeLabel = atoi(Value) > 0;
  if (char *Value = std::getenv("OMPCLUSTER_HEFT_DUMP_PRED"))
    Predecessors = atoi(Value) > 0;
  if (char *Value = std::getenv("OMPCLUSTER_HEFT_DUMP_FAKE_ENTRY"))
    FakeEntryNode = atoi(Value) > 0;
  if (char *Value = std::getenv("OMPCLUSTER_HEFT_DUMP_FAKE_EXIT"))
    FakeExitNode = atoi(Value) > 0;
}

/// This is the high-level implementation of the HEFT algorithm as is described
/// in the original paper.
///
/// \see https://ieeexplore.ieee.org/document/993206
void HEFTScheduler::scheduleGraph() {
  KG_TRACE(10, ("[HEFT] Scheduling graph with HEFT algorithm:\n"));
  using detail::DeviceId;
  acquireNewGraphPost();
  auto PQueue = Ctx.buildQueue();
  auto NumDevices = __kmpc_get_num_devices();
  auto Devices = Ctx.buildDeviceList(NumDevices);

  KG_TRACE(10, ("[HEFT] Number of devices: %d\n", NumDevices));
  while (not PQueue.empty()) {
    const auto *Task = PQueue.front().first;
    PQueue.pop_front();

    // Do not reschedule already executed or saved tasks
    const auto Data =
        (*get_schedule_map(tdg->tdg_id))[KMP_TASKPTR_TO_TASKID(Task)];
    if (Data.state > TaskState::SCHEDULED && Data.state != TaskState::FAILED)
      continue;

    // Defer the scheduling for target data tasks
    if (isTargetDataTask(Task)) {
      Ctx.deferTask(Task);
      continue;
    }

    // Classical tasks are pinned in the head node (rank=0)
    if (isClassicalTask(Task)) {
      Ctx.forceScheduleTask(Task, DeviceId::head());
      continue;
    }

    // If the user specifies a device for a task, it must be executed there
    if (hasSpecifiedDevice(Task) != 0) {
      Ctx.forceScheduleTask(Task, DeviceId(Task->target_data->device_id + 1));
      continue;
    }

    for (const auto &Device : Devices)
      Ctx.computeEFT(Task, Device);

    Ctx.scheduleTaskOnBestRank(Task);
  }
  // Schedule deferred tasks
  for (auto &Entry : (*get_schedule_map(tdg->tdg_id))) {
    auto &Data = Entry.second;
    const auto *Task = Data.task;

    // Do not reschedule already executed or saved tasks
    if (Data.state > TaskState::SCHEDULED && Data.state != TaskState::FAILED)
      continue;

    // These are already scheduled.
    if (not Ctx.isDeferredTask(*Task))
      continue;

    KG_TRACE(
        10,
        ("[HEFT] Scheduling deferred task %d: targetdata=%d root=%d exit=%d "
         "nsuccs=%d npreds=%d\n",
         KMP_TASKPTR_TO_TASKID(Task), isTargetDataTask(Task), isRootTask(Task),
         isExitTask(Task), Task->nsuccessors, Task->npredecessors));

    double MaxCost = 0.0;
    kmp_int32 Rank = 1;

    // We want the dataTask on the same device as its predecessor or successor
    // (depending if it's an exit or enter data map)
    if (Task->npredecessors == 0) {
      // Find the successor with highest communication cost
      for (kmp_int32 idx_succ = 0; idx_succ < Task->nsuccessors; idx_succ++) {
        auto Succ = &(tdg->record_map[Task->successors[idx_succ]]);
        auto SuccRank =
            (*get_schedule_map(tdg->tdg_id))[KMP_TASKPTR_TO_TASKID(Succ)].rank;

        if (Ctx.CommunicationCost[Task][Succ] > MaxCost &&
            !isClassicalTask(Succ) && SuccRank != -1) {
          MaxCost = Ctx.CommunicationCost[Task][Succ];
          Rank = SuccRank;
        }
      }
    } else {
      // Find the predecessor with highest communication cost
      for (kmp_int32 idx_pred = 0; idx_pred < Task->npredecessors; idx_pred++) {
        auto Pred = &(tdg->record_map[Task->predecessors[idx_pred]]);
        auto PredRank =
            (*get_schedule_map(tdg->tdg_id))[KMP_TASKPTR_TO_TASKID(Pred)].rank;

        if (Ctx.CommunicationCost[Pred][Task] > MaxCost &&
            !isClassicalTask(Pred) && PredRank != -1) {
          MaxCost = Ctx.CommunicationCost[Pred][Task];
          Rank = PredRank;
        }
      }
    }

    // Schedule this task to the same rank as the successor chosen before
    Ctx.forceScheduleTask(Task, DeviceId(Rank));
  }

  auto Predicate = [](const auto &Entry) {
    if (Entry.second.state >= TaskState::DISPATCHED)
      return true; // skip checking for executed/saved tasks
    return Entry.second.state == TaskState::SCHEDULED &&
                   isClassicalTask(Entry.second.task)
               ? Entry.second.rank == 0
               : Entry.second.rank > 0;
  };

  // Tasks can be DISPATCHED, SCHEDULED, EXECUTED or FAILED (because of
  // reschedule)

  assert(std::all_of((*get_schedule_map(tdg->tdg_id)).begin(),
                     (*get_schedule_map(tdg->tdg_id)).end(), Predicate) &&
         "Some tasks were not properly scheduled.\n");

  KG_TRACE(10, ("[HEFT]: HEFT (adding memory tasks)\n"));
  Ctx.reportIoModel(tdg->tdg_id);
  Ctx.dump(NumDevices);
  addMemoryTasks();
  addIoTasks();
  KG_TRACE(10, ("[HEFT] End HEFT schedule.\n"));
}

namespace {
// Mirrors OMPFILE_IO_TASK_* in libompfile's ompfile_graph_plan.h.
enum : kmp_int32 {
  IoTaskPrefetch = 1,
  IoTaskStage = 2,
  IoTaskFlush = 3,
  IoTaskReset = 4,
};

/// OMPCLUSTER_HEFT_IO_TASKS: a comma list of prefetch, stage, flush, or all.
unsigned ioTaskKinds() {
  static const unsigned Kinds = [] {
    const char *Env = std::getenv("OMPCLUSTER_HEFT_IO_TASKS");
    if (Env == nullptr)
      return 0u;
    std::string List(Env);
    unsigned K = 0;
    size_t Start = 0;
    while (Start <= List.size()) {
      size_t End = List.find(',', Start);
      if (End == std::string::npos)
        End = List.size();
      const std::string Item = List.substr(Start, End - Start);
      if (Item == "prefetch" || Item == "all")
        K |= 1u << IoTaskPrefetch;
      if (Item == "stage" || Item == "all")
        K |= 1u << IoTaskStage;
      if (Item == "flush" || Item == "all")
        K |= 1u << IoTaskFlush;
      Start = End + 1;
    }
    return K;
  }();
  return Kinds;
}

using IoTaskFn = int (*)(int, int, kmp_uint64, kmp_int64, kmp_uint64);
IoTaskFn ioTaskFn() {
  static const IoTaskFn Fn =
      reinterpret_cast<IoTaskFn>(KMP_DLSYM("ompfile_sched_io_task"));
  return Fn;
}

bool rangesOverlap(kmp_int64 A, kmp_uint64 AExt, kmp_int64 B,
                   kmp_uint64 BExt) {
  if (A < 0 || B < 0)
    return true; // a cursor call can touch anything
  return A < B + static_cast<kmp_int64>(BExt) &&
         B < A + static_cast<kmp_int64>(AExt);
}
} // namespace

/// Inserts file I/O tasks on the scheduled graph's edges. Each one runs on an
/// origin helper thread and asks a proxy to do the work (libompfile's
/// ompfile_sched_io_task):
///  - prefetch: a task reads a range its single writing predecessor wrote, on
///    a device that does not own the file. The range is read into the reading
///    proxy's prefetch cache right after the writer, so the reader's own read
///    is local; the reader's device can run something else meanwhile.
///  - stage: a task reads a file no task of the graph writes. The range is
///    read on the file's owner into its prefetch cache (and page cache) ahead
///    of the reader, off whatever predecessor edge the reader has.
///  - flush: after the last writer of a file, commit it on its owner. With
///    LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=deferred the writer's close skips
///    the sync, so the durability point leaves the writer and only that edge's
///    successor waits for it.
/// The proxies' prefetch caches are reset before the tasks are planned, so a
/// range left over from an earlier run is never served.
void HEFTScheduler::addIoTasks() {
  const unsigned Kinds = ioTaskKinds();
  const IoTaskFn Run = ioTaskFn();
  if (Kinds == 0 || Run == nullptr)
    return;
  // The recording run schedules the graph while its tasks execute; finished
  // tasks have already released their successor lists, so an edge task
  // cannot be spliced in. I/O tasks start with the first replay.
  if (__kmp_tdg_is_recording(tdg->tdg_status))
    return;
  (void)Run(IoTaskReset, -1, 0, 0, 0);

  const kmp_int32 Normal = KMP_ATOMIC_LD_RLX(&tdg->num_normal_tasks);
  const auto &Ranges = Ctx.IoRanges;
  auto rangesOf = [&](kmp_int32 Node) -> const std::vector<kmp_task_io_entry_t> * {
    if (Node < 0 || static_cast<size_t>(Node) >= Ranges.size())
      return nullptr;
    return &Ranges[Node];
  };
  auto writes = [&](kmp_int32 Node, kmp_uint64 Key, kmp_int64 Lo,
                    kmp_uint64 Ext) {
    const auto *R = rangesOf(Node);
    if (R == nullptr)
      return false;
    for (const kmp_task_io_entry_t &E : *R)
      if (E.is_write && E.path_key == Key &&
          rangesOverlap(E.offset, E.extent, Lo, Ext))
        return true;
    return false;
  };
  auto rankOf = [&](kmp_int32 Node) {
    const kmp_node_info &Info = tdg->record_map[Node];
    if (Info.task == nullptr)
      return -1;
    auto &Map = *get_schedule_map(tdg->tdg_id);
    auto It = Map.find(KMP_TASKNODE_TO_TASKID(Info));
    return It == Map.end() ? -1 : It->second.rank;
  };
  auto ownerOf = [&](kmp_uint64 Key, const std::vector<kmp_task_io_entry_t> &R) {
    auto It = Ctx.IoOwner.find(Key);
    if (It != Ctx.IoOwner.end())
      return It->second;
    for (const kmp_task_io_entry_t &E : R)
      if (E.path_key == Key && E.owner_rank >= 0)
        return E.owner_rank;
    return -1;
  };

  // Files some task of the graph writes, and the last writer of each. Tasks
  // are recorded after the tasks they depend on, so node order is a
  // topological order (and survives the record map moving when memory tasks
  // grow it, which pointer-keyed HEFT state does not).
  std::unordered_map<kmp_uint64, kmp_int32> LastWriter;
  for (kmp_int32 N = 0; N < Normal; ++N) {
    const auto *R = rangesOf(N);
    if (R == nullptr || tdg->record_map[N].task == nullptr)
      continue;
    for (const kmp_task_io_entry_t &E : *R)
      if (E.is_write)
        LastWriter[E.path_key] = N;
  }

  struct Plan {
    kmp_int32 Src, Dst, Kind, Device;
    kmp_uint64 Key;
    kmp_int64 Offset;
    kmp_uint64 Extent;
  };
  std::vector<Plan> Plans;
  kmp_uint64 Counts[4] = {0, 0, 0, 0};
  kmp_uint64 Bytes[4] = {0, 0, 0, 0};

  for (kmp_int32 T = 0; T < Normal; ++T) {
    const kmp_node_info &Node = tdg->record_map[T];
    if (Node.task == nullptr || !Node.is_target_task ||
        isTargetDataTask(&Node))
      continue;
    const kmp_int32 Rank = rankOf(T);
    const auto *R = rangesOf(T);
    if (Rank < 1 || R == nullptr || R->empty())
      continue;
    const kmp_int32 Dev = Rank - 1;
    // The task's read span per file.
    std::unordered_map<kmp_uint64, std::pair<kmp_int64, kmp_int64>> Reads;
    for (const kmp_task_io_entry_t &E : *R) {
      if (E.is_write || E.offset < 0 || E.extent == 0)
        continue;
      const kmp_int64 End = E.offset + static_cast<kmp_int64>(E.extent);
      auto It = Reads.find(E.path_key);
      if (It == Reads.end())
        Reads[E.path_key] = {E.offset, End};
      else {
        It->second.first = std::min(It->second.first, E.offset);
        It->second.second = std::max(It->second.second, End);
      }
    }
    for (const auto &KV : Reads) {
      const kmp_uint64 Key = KV.first;
      const kmp_int64 Lo = KV.second.first;
      const kmp_uint64 Ext = static_cast<kmp_uint64>(KV.second.second - Lo);
      // A task that also writes the range it reads keeps its own order.
      if (writes(T, Key, Lo, Ext))
        continue;
      kmp_int32 WriterEdge = -1, WriterEdges = 0;
      for (kmp_int32 I = 0; I < Node.npredecessors; ++I) {
        const kmp_int32 P = Node.predecessors[I];
        const kmp_int32 W =
            P < Normal ? P : tdg->record_map[P].predecessors[0];
        if (writes(W, Key, Lo, Ext)) {
          ++WriterEdges;
          WriterEdge = P;
        }
      }
      const kmp_int32 Owner = ownerOf(Key, *R);
      if (WriterEdges == 1 && (Kinds & (1u << IoTaskPrefetch)) && Owner >= 0 &&
          Owner != Dev) {
        Plans.push_back({WriterEdge, T, IoTaskPrefetch, Dev, Key, Lo, Ext});
      } else if (WriterEdges == 0 && LastWriter.count(Key) == 0 &&
                 (Kinds & (1u << IoTaskStage)) && Owner >= 0 &&
                 Node.npredecessors > 0) {
        Plans.push_back(
            {Node.predecessors[0], T, IoTaskStage, Owner, Key, Lo, Ext});
      }
    }
  }
  if (Kinds & (1u << IoTaskFlush)) {
    for (const auto &KV : LastWriter) {
      const kmp_int32 W = KV.second;
      const kmp_node_info &Node = tdg->record_map[W];
      if (Node.nsuccessors < 1)
        continue; // nothing to hang it on; the proxy syncs at shutdown
      const auto *R = rangesOf(W);
      kmp_int32 Owner = R != nullptr ? ownerOf(KV.first, *R) : -1;
      if (Owner < 0)
        Owner = rankOf(W) - 1;
      if (Owner < 0)
        continue;
      Plans.push_back({W, Node.successors[0], IoTaskFlush, Owner, KV.first, 0,
                       0});
    }
  }

  // Several tasks on one edge are chained: src -> io1 -> io2 -> dst.
  const kmp_int32 Gtid = __kmp_get_gtid();
  std::map<std::pair<kmp_int32, kmp_int32>, kmp_int32> Chain;
  for (const Plan &P : Plans) {
    auto &Src = Chain.emplace(std::make_pair(P.Src, P.Dst), P.Src).first->second;
    kmp_task_t *Inserted = __kmp_create_io_task(
        Gtid, tdg->record_map[Src].task, tdg->record_map[P.Dst].task, P.Kind,
        P.Device, P.Key, P.Offset, P.Extent);
    if (Inserted == nullptr)
      continue;
    Src = KMP_TASK_TO_TASKDATA(Inserted)->td_tdg_task_id;
    ++Counts[P.Kind];
    Bytes[P.Kind] += P.Extent;
  }
  if (Ctx.IoLog)
    fprintf(stderr,
            "[heft-io] io-tasks tdg=%d prefetch=%llu prefetch_bytes=%llu "
            "stage=%llu stage_bytes=%llu flush=%llu\n",
            tdg->tdg_id, static_cast<unsigned long long>(Counts[IoTaskPrefetch]),
            static_cast<unsigned long long>(Bytes[IoTaskPrefetch]),
            static_cast<unsigned long long>(Counts[IoTaskStage]),
            static_cast<unsigned long long>(Bytes[IoTaskStage]),
            static_cast<unsigned long long>(Counts[IoTaskFlush]));
}

/// Reset context and recompute upward ranks for the new graph
void HEFTScheduler::acquireNewGraphPost() {
  KG_TRACE(10, ("[HEFT]: AcquireNewGraphPost (start)\n"));
  Ctx.clear();
  Ctx.setTaskMap(__kmpc_curr_tdg_idx());
  Ctx.loadIoModel();
  dumpTaskGraph();
  Ctx.interpolateCompCost(scheduling_iteration_counter, initial_schedule_policy,
                          initial_iteration);
  if (Ctx.IoLog)
    Ctx.printInterpolation(scheduling_iteration_counter);
  Ctx.computeUpwardRanks();
  KG_TRACE(10, ("[HEFT]: AcquireNewGraphPost (end)\n"));
}

void HEFTScheduler::dumpTaskGraph() {
  // FIXME: Update for Memory Tasks
  if (graph_dump_dir_path.empty())
    return;

  if (tdg == nullptr) {
    KG_TRACE(10, ("[HEFT] Task map is null, skipping dump.\n"));
    return;
  }

  std::ofstream dot(graph_dump_dir_path + "_graph_" +
                    std::to_string(graph_counter) + ".dot");
  // Print header
  dot << "// Generated in HEFTScheduler\n"
      << "digraph TargetTaskGraph {\n"
      << "  labelloc=\"t\";\n"
      << "  label=\"HEFT Scheduler\";\n";

  // For each task
  for (kmp_int32 tid = 0; tid < tdg->num_tasks; tid++) {
    auto task = &(tdg->record_map[tid]);
    kmp_int32 trank =
        (*get_schedule_map(tdg->tdg_id))[KMP_TDG_TASKID_TO_TASKID(tdg, tid)]
            .rank;
    std::string tinfo =
        (*get_schedule_map(tdg->tdg_id))[KMP_TDG_TASKID_TO_TASKID(tdg, tid)]
            .source_location;
    kmp_int32 state = static_cast<kmp_int32>(
        (*get_schedule_map(tdg->tdg_id))[KMP_TDG_TASKID_TO_TASKID(tdg, tid)]
            .state);

    long task_id = KMP_TASKPTR_TO_TASKID(task);

    // Print graph node
    dot << "  V" << task_id << " [label=<<B>ID " << task_id
        << "</B><BR /><FONT POINT-SIZE=\"10\">Rank " << trank
        << "<BR /> State: " << state;

    if (!tinfo.empty()) {
      dot << "<BR />" << tinfo;
    }

    kmp_uint64 io_bytes = 0;
    kmp_int32 io_ranges = 0;
    double io_seconds = 0.0;
    if (__kmp_task_io_summary(task, &io_bytes, &io_ranges, &io_seconds)) {
      dot << "<BR />I/O: " << io_bytes << " B in " << io_ranges << " ranges, "
          << io_seconds << " s";
    }

    dot << "</FONT>>";

    if (isTargetDataTask(task))
      dot << " color=gray70 fontcolor=gray50";
    if (isRootTask(task) || isExitTask(task))
      dot << " shape=octagon";

    dot << "];\n";

    // Print forward arrows using the list of successors
    for (kmp_int32 idx = 0; idx < task->nsuccessors; idx++) {
      kmp_int32 sid = task->successors[idx];
      auto succ = &(tdg->record_map[task->successors[idx]]);
      kmp_int32 srank =
          (*get_schedule_map(tdg->tdg_id))[KMP_TDG_TASKID_TO_TASKID(tdg, sid)]
              .rank;
      const auto *color = detail::getEdgeColor(trank, srank);
      double CommCost = Ctx.CommunicationCost[task][succ];

      long succ_id = KMP_TASKPTR_TO_TASKID(succ);

      // Print graph edge
      dot << "  V" << task_id << " -> V" << succ_id << " [color=\"" << color
          << "\"";
      if (Dump.EdgeLabel)
        dot << " label=<" << CommCost << ">";
      dot << "];\n";
    }

    // Print backward arrows using the list of predecessors
    if (Dump.Predecessors) {
      for (kmp_int32 pid = 0; pid < task->npredecessors; pid++) {
        dot << "  V" << pid << " -> V" << task_id
            << " [style=dashed dir=back color=gray30];\n";
      }
    }
  }

  // Print a single fake entry node
  if (Dump.FakeEntryNode) {
    dot << "  Entry;\n";
    for (kmp_int32 rid = 0; rid < tdg->num_roots; rid++) {
      dot << "  Entry -> V" << rid << ";\n";
    }
  }

  // Print a single fake exit node
  if (Dump.FakeExitNode) {
    dot << "  Exit;\n";
    for (kmp_int32 tid = 0; tid < tdg->num_tasks; tid++) {
      auto task = &(tdg->record_map[tid]);
      long task_id = KMP_TASKPTR_TO_TASKID(task);
      if (isExitTask(task))
        dot << "  V" << task_id << " -> Exit;\n";
    }
  }
  dot << "}\n";
}

void HEFTScheduler::resetWeight() { Ctx.clearCompCost(); }
