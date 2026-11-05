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

#include "kmp_target_legacy_heft.h"
// #include "profiler.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace detail {

// Implementation of LegacyHEFTContext
// =============================================================================

LegacyHEFTContext::LegacyHEFTContext(LegacyScheduleMap &Schedule,
                                     kmp_target_task_map_t *TaskMap)
    : CommMultiplier(1.0), DefaultCompCost(100.0), DefaultCommCost(1.0),
      NumThreads(1), TaskMap(TaskMap),
      Schedule(Schedule) /*, ProfileInfo({})*/ {

  if (TaskMap != nullptr)
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
}

void LegacyHEFTContext::computeUpwardRanks() {
  DP("[HEFT] Computing upward rank for all tasks.\n");
  assert(TaskMap != nullptr && "The task map should not be NULL");

  for (auto *Task : ReverseTopologicalOrder(TaskMap)) {
    double MaxSuccUrankComm = 0.0;
    for (auto &Succ : Successors(Task)) {
      MaxSuccUrankComm =
          std::max(MaxSuccUrankComm,
                   UpwardRank[&Succ] + computeAvgCommCost(*Task, Succ));
    }
    UpwardRank[Task] = computeAvgCompCost(*Task) + MaxSuccUrankComm;
  }
}

double LegacyHEFTContext::computeEST(LegacyTaskPtr Task,
                                     LegacyDeviceId Device) {
  double MaxPredEST = 0.0;

  // Iterate over predecessors
  for (const auto &Pred : Predecessors(Task)) {

    // Ignore deferred tasks for now
    if (isDeferredTask(*Task))
      continue;

    assert(Schedule.count(Pred.task_id) == 1 &&
           "Predecessor should already be scheduled!");

    LegacyDeviceId DevicePred(Schedule[Pred.task_id]);
    auto Candidate =
        AFT[&Pred] + computeCommCost(Pred, DevicePred, *Task, Device);

    if (Candidate > MaxPredEST)
      MaxPredEST = Candidate;
  }

  // Find the earliest start time if task is scheduled on device.
  EST[Task][Device] = std::max(DeviceAvail[Device], MaxPredEST);
  return EST[Task][Device];
}

double LegacyHEFTContext::computeEFT(LegacyTaskPtr Task,
                                     LegacyDeviceId Device) {
  // Calculate EST for task id, rank pair if it does not exist
  if (EST.count(Task) == 0 || EST[Task].count(Device) == 0)
    computeEST(Task, Device);

  EFT[Task][Device] = computeCompCost(*Task, Device) + EST[Task][Device];
  return EFT[Task][Device];
}

double LegacyHEFTContext::scheduleTask(LegacyTaskPtr Task,
                                       LegacyDeviceId Device) {
  // Do not change state of FAILED tasks during the reschedule
  bool IsRestartingTask =
      Schedule[Task->task_id].state == LegacyTaskState::FAILED;

  Schedule[Task->task_id] = {
      .state = IsRestartingTask ? LegacyTaskState::FAILED
                                : LegacyTaskState::SCHEDULED,
      .rank = Device.Rank,
      .thread = Device.Thread,
      .translated_rank = -1,
      .task = Task,
  };

  AST[Task] = EST[Task][Device];
  AFT[Task] = EFT[Task][Device];
  return AFT[Task];
}

double LegacyHEFTContext::scheduleTaskOnBestRank(LegacyTaskPtr Task) {
  const auto &EFTs = EFT.at(Task);

  auto BestRank = std::min_element(
      EFTs.begin(), EFTs.end(),
      [](const auto &L, const auto &R) { return L.second < R.second; });

  DeviceAvail[BestRank->first] = scheduleTask(Task, BestRank->first);
  return DeviceAvail[BestRank->first];
}

void LegacyHEFTContext::forceScheduleTask(LegacyTaskPtr Task,
                                          LegacyDeviceId Device) {
  computeEFT(Task, Device);
  scheduleTaskOnBestRank(Task);
}

void LegacyHEFTContext::deferTask(LegacyTaskPtr Task) {
  // Do not change state of FAILED tasks during the reschedule
  if (Schedule[Task->task_id].state != LegacyTaskState::FAILED)
    Schedule[Task->task_id].state = LegacyTaskState::DEFERRED;
}

LegacyPriorityQueue LegacyHEFTContext::buildQueue() const {
  LegacyPriorityQueue Pqueue{UpwardRank.begin(), UpwardRank.end()};
  std::sort(Pqueue.begin(), Pqueue.end(),
            [](const auto &L, const auto &R) { return L.second > R.second; });

  return Pqueue;
}

std::vector<LegacyDeviceId>
LegacyHEFTContext::buildDeviceList(int32_t NumRanks) const {
  std::vector<LegacyDeviceId> Devices;
  Devices.reserve(NumRanks);
  for (int32_t R = 1; R <= NumRanks; R++)
    for (int32_t H = 0; H < NumThreads; H++)
      Devices.emplace_back(R, H);
  return Devices;
}

void LegacyHEFTContext::clear() {
  EST.clear();
  EFT.clear();
  AST.clear();
  AFT.clear();
  UpwardRank.clear();
  DeviceAvail.clear();
  CommunicationCost.clear();
  ComputationCost.clear();
}

void LegacyHEFTContext::setTaskMap(kmp_target_task_map_t *NewTaskMap) {
  assert(NewTaskMap != nullptr && "Expected a non-null task map!");
  TaskMap = NewTaskMap;
}

double LegacyHEFTContext::computeCommCost(const kmp_target_task_t &A,
                                          LegacyDeviceId DeviceA,
                                          const kmp_target_task_t &B,
                                          LegacyDeviceId DeviceB) {
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
double LegacyHEFTContext::computeAvgCommCost(const kmp_target_task_t &A,
                                             const kmp_target_task_t &B) {
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
  for (int32_t I = 0; I < A.target_data.num_args; I++) {
    for (int32_t J = 0; J < B.target_data.num_args; J++) {
      if (A.target_data.args[I] == B.target_data.args[J]) {
        TotalBytes += A.target_data.arg_sizes[I];
      }
    }
  }

  CommunicationCost[&A][&B] = TotalBytes;

  DP("[HEFT] Dependency `%d -> %d` has average cost: %lf KB\n", A.task_id,
     B.task_id, TotalBytes / 1024.0);

  return CommMultiplier * TotalBytes;
}

double LegacyHEFTContext::computeAvgCompCost(const kmp_target_task_t &Task) {
  // If computation cost was already computed, then use the cached value.
  auto It = ComputationCost.find(&Task);
  if (It != ComputationCost.end()) {
    return It->second;
  }

  return ComputationCost[&Task] =
             (not isLegacyTargetDataTask(Task)) ? DefaultCompCost : 1.0;
}

double LegacyHEFTContext::computeCompCost(const kmp_target_task_t &Task,
                                          LegacyDeviceId Device) {
  // FIXME: the actual rank of the task is not being taken into account.
  // Therefore we are assuming the the task takes the same amount of time to be
  // computed independent of the device it is scheduled.
  return computeAvgCompCost(Task);
}

/// Dump internal HEFT information to a file.
void LegacyHEFTContext::dump(int32_t NumDevices) {
  if (HEFTLogPath.empty())
    return;

  FILE *fp = fopen(HEFTLogPath.c_str(), "w");

  if (fp == NULL) {
    DP("[HEFT] Failed to open log file.\n");
    return;
  }

  auto Devices = buildDeviceList(NumDevices);

  DP("[HEFT] Dumping HEFT stats to file '%s'\n", HEFTLogPath.c_str());

  // FIXME: Print the header with correct columns for EST/EFT
  fprintf(fp,
          "%-4s  %-10s  %-10s  %-10s  %-10s  %-10s  %-7s  %-7s  %-7s  %-7s"
          "%-34s  %-34s\n",
          "ID", "FNID", "Rank", "AST", "AFT", "Dev", "Root", "Exit", "Data",
          "Classic", "EST", "EFT");

  std::vector<kmp_target_task_t *> TasksV;
  TasksV.reserve(TaskMap->num_tasks);
  for (int i = 0; i < TaskMap->num_tasks; i++) {
    TasksV.push_back(&(TaskMap->tasks[i]));
  }

  std::sort(TasksV.begin(), TasksV.end(),
            [&](kmp_target_task_t *A, kmp_target_task_t *B) {
              return UpwardRank[A] > UpwardRank[B];
            });

  // Iterate over tasks
  for (auto const *task_ptr : TasksV) {
    auto &task = *task_ptr;
    uint64_t ofid = (uint64_t)task.target_data.outlined_fn_id;
    fprintf(fp,
            "%04d  0x%08lx  %10.2lf  %10.2lf  %10.2lf  %7d  %7d  %7d  %7d  %4d",
            task.task_id, ofid, UpwardRank[&task], AST[&task], AFT[&task],
            Schedule[task.task_id].rank, isLegacyRootTask(task),
            isLegacyExitTask(task), isLegacyTargetDataTask(task),
            isLegacyClassicalTask(task));
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
    auto id_A = A.first->task_id;
    for (const auto &B : A.second) {
      auto id_B = B.first->task_id;
      fprintf(fp, "%04d -> %04d : %10.4lf\n", id_A, id_B, B.second);
    }
  }

  fprintf(fp, "\n");
  fprintf(fp, "Schedule Map\n");
  fprintf(fp, "-------------------------------------------------------\n");

  for (const auto &s : Schedule) {
    fprintf(fp, "%04d: %04d %d %d\n", s.first, s.second.task->task_id,
            s.second.rank, s.second.state);
  }

  fclose(fp);
}

} // namespace detail

// Implementation of LegacyHEFTScheduler
// =============================================================================

LegacyHEFTScheduler::DumpOptions::DumpOptions() {
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
void LegacyHEFTScheduler::scheduleGraph() {

  using detail::LegacyDeviceId;
  auto PQueue = Ctx.buildQueue();
  auto NumDevices = omp_get_num_devices();
  auto Devices = Ctx.buildDeviceList(NumDevices);

  DP("[SCHEDULER] Scheduling graph with HEFT algorithm:\n"
     "[SCHEDULER] --> Number of tasks: %d\n"
     "[SCHEDULER] --> Number of ranks: %d\n",
     task_map->num_tasks, NumDevices);

  while (not PQueue.empty()) {
    const auto *Task = PQueue.front().first;
    PQueue.pop_front();
    // Do not reschedule already executed or saved tasks
    const auto Data = schedule_map[Task->task_id];
    if (Data.state > LegacyTaskState::SCHEDULED &&
        Data.state != LegacyTaskState::FAILED)
      continue;

    // Defer the scheduling for target data tasks
    if (isLegacyTargetDataTask(Task)) {
      Ctx.deferTask(Task);
      continue;
    }

    // Classical tasks are pinned in the head node (rank=0)
    if (isLegacyClassicalTask(Task)) {
      Ctx.forceScheduleTask(Task, LegacyDeviceId::head());
      continue;
    }

    for (const auto &Device : Devices)
      Ctx.computeEFT(Task, Device);

    Ctx.scheduleTaskOnBestRank(Task);
  }

  // Schedule deferred tasks
  for (auto &Entry : schedule_map) {
    auto &Data = Entry.second;
    const auto *Task = Data.task;

    // Do not reschedule already executed or saved tasks
    if (Data.state > LegacyTaskState::SCHEDULED &&
        Data.state != LegacyTaskState::FAILED)
      continue;

    // These are already scheduled.
    if (not Ctx.isDeferredTask(*Task))
      continue;

    DP("[HEFT] Scheduling deferred task %d: targetdata=%d root=%d exit=%d "
       "nsuccs=%d npreds=%d\n",
       Task->task_id, isLegacyTargetDataTask(Task), isLegacyRootTask(Task),
       isLegacyExitTask(Task), NumSuccessors(Task), NumPredecessors(Task));

    double MaxCost = 0.0;
    int32_t Rank = 1;

    // Find the successor with highest communication cost
    // NOTE: Exit target data tasks are temporary scheduled but the rank will
    // be overwritten later with PinTask function.
    for (const auto &Succ : Successors(Task)) {
      auto SuccRank = schedule_map[Succ.task_id].rank;

      // Save successor ranks of each scheduled data task to detect broadcast
      // opportunites
      dataTaskSuccessorRanks[Task->task_id].insert(SuccRank);

      if (Ctx.CommunicationCost[Task][&Succ] > MaxCost &&
          !isLegacyClassicalTask(Succ) && SuccRank != -1) {
        MaxCost = Ctx.CommunicationCost[Task][&Succ];
        Rank = SuccRank;
      }
    }

    // Schedule this task to the same rank as the successor chosen before
    Ctx.forceScheduleTask(Task, LegacyDeviceId(Rank));
  }

  auto Predicate = [](const auto &Entry) {
    if (Entry.second.state >= LegacyTaskState::DISPATCHED)
      return true; // skip checking for executed/saved tasks
    return Entry.second.state == LegacyTaskState::SCHEDULED &&
                   isLegacyClassicalTask(Entry.second.task)
               ? Entry.second.rank == 0
               : Entry.second.rank > 0;
  };

  // Tasks can be DISPATCHED, SCHEDULED, EXECUTED or FAILED (because of
  // reschedule)

  assert(std::all_of(schedule_map.begin(), schedule_map.end(), Predicate) &&
         "Some tasks were not properly scheduled.");

  Ctx.dump(NumDevices);
  DP("[SCHEDULER] HEFT scheduled %d tasks to %d resources.\n",
     task_map->num_tasks, NumDevices);
}

/// Reset context and recompute upward ranks for the new graph
void LegacyHEFTScheduler::acquireNewGraphPost() {
  Ctx.clear();
  Ctx.setTaskMap(task_map);
  Ctx.computeUpwardRanks();
}

void LegacyHEFTScheduler::dumpTaskGraph() {
  if (graph_dump_dir_path.empty())
    return;

  if (task_map == nullptr) {
    DP("[HEFT] Task map is null, skipping dump.\n");
    return;
  }

  std::ofstream dot(graph_dump_dir_path + "_graph_" +
                    std::to_string(graph_counter) + ".dot");
  // Print header
  dot << "// Generated in LegacyHEFTScheduler\n"
      << "digraph TargetTaskGraph {\n"
      << "  labelloc=\"t\";\n"
      << "  label=\"HEFT Scheduler\";\n";

  // For each task
  for (const auto &task : Tasks(task_map)) {
    int32_t tid = task.task_id;
    int32_t trank = schedule_map[tid].rank;
    std::string tinfo = schedule_map[tid].source_location;
    int32_t state = static_cast<int32_t>(schedule_map[tid].state);

    // Print graph node
    dot << "  V" << tid << " [label=<<B>ID " << tid
        << "</B><BR /><FONT POINT-SIZE=\"10\">Rank " << trank
        << "<BR /> State: " << state;

    if (!tinfo.empty()) {
      dot << "<BR />" << tinfo;
    }

    dot << "</FONT>>";

    if (isLegacyTargetDataTask(task))
      dot << " color=gray70 fontcolor=gray50";
    if (isLegacyRootTask(task) || isLegacyExitTask(task))
      dot << " shape=octagon";

    dot << "];\n";

    // Print forward arrows using the list of successors
    for (const auto &succ : Successors(task)) {
      int32_t sid = succ.task_id;
      int32_t srank = schedule_map[sid].rank;
      const auto *color = detail::getLegacyEdgeColor(trank, srank);
      double CommCost = Ctx.CommunicationCost[&task][&succ];

      // Print graph edge
      dot << "  V" << tid << " -> V" << sid << " [color=\"" << color << "\"";
      if (Dump.EdgeLabel)
        dot << " label=<" << CommCost << ">";
      dot << "];\n";
    }

    // Print backward arrows using the list of predecessors
    if (Dump.Predecessors) {
      for (const auto &pred : Predecessors(task)) {
        const auto &pid = pred.task_id;
        dot << "  V" << pid << " -> V" << tid
            << " [style=dashed dir=back color=gray30];\n";
      }
    }
  }

  // Print a single fake entry node
  if (Dump.FakeEntryNode) {
    dot << "  Entry;\n";
    for (const auto *root : Roots(task_map))
      dot << "  Entry -> V" << root->task_id << ";\n";
  }

  // Print a single fake exit node
  if (Dump.FakeExitNode) {
    dot << "  Exit;\n";
    for (const auto &task : Tasks(task_map))
      if (isLegacyExitTask(task))
        dot << "  V" << task.task_id << " -> Exit;\n";
  }

  dot << "}\n";
}

// fixme Return the list of successor ranks of a given data task
std::set<int> LegacyHEFTScheduler::getDataTaskSuccessors(int32_t task_id) {
  return dataTaskSuccessorRanks[task_id];
}
