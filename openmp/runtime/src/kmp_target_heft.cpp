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
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <deque>
#include <array>
#include <utility>
#include <cmath>

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
    UpwardRank[Task] = computeAvgCompCost(*Task) + MaxSuccUrankComm;
  }
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

  EFT[Task][Device] = computeCompCost(*Task, Device) + EST[Task][Device];
  return EFT[Task][Device];
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
  return AFT[Task];
}

double HEFTContext::scheduleTaskOnBestRank(TaskPtr Task) {
  const auto &EFTs = EFT.at(Task);

  auto BestRank = std::min_element(
      EFTs.begin(), EFTs.end(),
      [](const auto &L, const auto &R) { return L.second < R.second; });

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
        sum += std::log(ComputationCost[Taskptr][device_1] /
                        ComputationCost[Taskptr][device_0]);
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

  double geomean_b0 = std::exp(sum_b0 / size_b0);

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
      beta[device_0][device_1] = std::exp(sum / size);
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
        sum += std::log(restrictedComputationCost[Taskptr][actual_device_1] /
                        restrictedComputationCost[Taskptr][actual_device_0]);
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

  double geomean_b0 = std::exp(sum_b0 / size_b0);

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
      beta[device_0][device_1] = std::exp(sum / size);
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
          ComputationCost[Taskptr][device] = static_cast<double>(
              tdg->record_map[KMP_TASKPTR_TO_TDG_TASKID(Taskptr)]
                  .task_time[device]
                  .mean_time);
          if (heterogeneousInterpolation && deviceTypeNumber > 1) {
            device_restricted_CompCost[deviceTypeMap[device]][Taskptr][device] =
                static_cast<double>(
                    tdg->record_map[KMP_TASKPTR_TO_TDG_TASKID(Taskptr)]
                        .task_time[device]
                        .mean_time);
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
    homogeneousInterpolation();
    computeAvgCompCosts();
    KG_TRACE(10, ("[Inter HEFT] interpolateCompCost (end): iteration %d "
                  "homogeneous hypothesis\n ",
                  scheduling_iteration_counter));
    return;
  }

  if (heterogeneousInterpolation && deviceTypeNumber > 1 &&
      scheduling_iteration_counter > 3) {
    heterogeneousLinearInterpolation();
    computeAvgCompCosts();
    KG_TRACE(10, ("[Inter HEFT] interpolateCompCost (end): iteration %d "
                  "heterogeneous devices interpolation\n ",
                  scheduling_iteration_counter));
    return;
  }

  // general case
  linearInterpolation();
  computeAvgCompCosts();

  KG_TRACE(10, ("[Inter HEFT] interpolateCompCost (end): iteration %d "
                "homogeneous devices interpolation\n",
                scheduling_iteration_counter));
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
  Ctx.dump(NumDevices);
  addMemoryTasks();
  KG_TRACE(10, ("[HEFT] End HEFT schedule.\n"));
}

/// Reset context and recompute upward ranks for the new graph
void HEFTScheduler::acquireNewGraphPost() {
  KG_TRACE(10, ("[HEFT]: AcquireNewGraphPost (start)\n"));
  Ctx.clear();
  Ctx.setTaskMap(__kmpc_curr_tdg_idx());
  dumpTaskGraph();
  Ctx.interpolateCompCost(scheduling_iteration_counter, initial_schedule_policy,
                          initial_iteration);
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
