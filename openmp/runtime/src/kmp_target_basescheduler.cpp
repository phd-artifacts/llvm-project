//===-------------- basescheduler.cpp - Target-Task scheduler -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the base abstract scheduler.
//
//===----------------------------------------------------------------------===//

#include "kmp_target_basescheduler.h"
#include "kmp.h"
#include "kmp_debug.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <thread>
#include <unordered_set>
#include <stdio.h>

BaseScheduler::BaseScheduler()
    : remaining_target_tasks(0), non_nowait_counter(0),
      non_task_scheduler(NonTaskScheduler::STATIC), graph_dump_dir_path(),
      graph_counter(0), tdg(nullptr) {

  // Read environment variables
  if (char *env_str = std::getenv("OMPCLUSTER_TASK_GRAPH_DUMP_PATH")) {
    graph_dump_dir_path = env_str;
  }

  const char *str = std::getenv("OMPCLUSTER_NONTASK_SCHEDULER");
  std::string sched_str = (str != nullptr) ? str : "";

  if (sched_str == "roundrobin") {
    KG_TRACE(
        10,
        ("[NON-TASK SCHEDULER] Map synchronous regions using round-robin\n"));
    non_task_scheduler = NonTaskScheduler::ROUNDROBIN;
  } else {
    KG_TRACE(
        10,
        ("[NON-TASK SCHEDULER] Map synchronous regions to the first worker\n"));
  }

  // if policy = 0 (default), use selected scheduler for initial iterations;
  // if policy = 1, use RR with offset (equal to iteration's number)
  if (char *env_str = std::getenv("OMPCLUSTER_INITIAL_SCHEDULE_POLICY")) {
    initial_schedule_policy = std::stoi(env_str);
  } else {
    initial_schedule_policy = 0;
  }

  // Set up the number of initial iterations (after these initial iterations, we
  // schedule at least one)
  if (char *env_str = std::getenv("OMPCLUSTER_INITIAL_ITERATION_NUMBER")) {
    initial_iteration = std::stoi(env_str);
  } else {
    initial_iteration = 0;
  }

  // Reschedule every X iterations
  //  if 0 (default) do not reschedule
  if (char *env_str = std::getenv("OMPCLUSTER_RESCHEDULING_INTERVAL")) {
    rescheduling_interval = std::stoi(env_str);
  } else {
    rescheduling_interval = 0;
  }
  scheduling_iteration_counter = 0;
  isScheduleDone = 0;
}

BaseScheduler::~BaseScheduler() {
  // The task map can still be allocated due to a cancellation point.
  if (tdg != nullptr) {
    if (remaining_target_tasks != 0) {
      KG_TRACE(10, ("Warning: some target-tasks were discarted at the end of "
                    "the program, "
                    "probably due to a cancellation point.\n"));
    }
  }
}

void BaseScheduler::acquireNewGraph(kmp_tdg_info_t *new_tdg) {

  KG_TRACE(10, ("acquireNewGraph Start\n"));
  // Remove any internal state of the current task graph.

  if (tdg != nullptr) {
    // Returning if we're still in the same tdg
    if (tdg == new_tdg) {
      KG_TRACE(10, ("acquireNewGraph Exit : Same tdg\n"));
      return;
    }
    isScheduleDone = 0;
    (*get_schedule_map(tdg->tdg_id))
        .clear(); // TODO Do we really want to clear this structure?  Can it be
                  // shared between tdgs?

    if (remaining_target_tasks != 0) {
      KG_TRACE(10,
               ("Warning: some target-tasks were discarded in favor of a new "
                "task-graph, probably due to a cancellation point.\n"));
    }

    remaining_target_tasks = 0;
  }

  // Acquire a new task graph from the target task integration layer.
  tdg = new_tdg;
  assert(tdg != nullptr);
  // At graph acquisition, all recorded tasks are normal tasks.
  // Memory tasks, if any, are added later by the scheduler.
  tdg->num_normal_tasks.store(tdg->num_tasks.load());
  std::unordered_map<void *, kmp_int64> size_map;
  for (kmp_int32 idx = 0; idx < tdg->num_tasks; ++idx) {
    const kmp_node_info &task = tdg->record_map[idx];
    if (task.is_target_task) {
      // Add new target-tasks to the schedule map
      assertm((*get_schedule_map(tdg->tdg_id))
                      .count(KMP_TASKNODE_TO_TASKID(task)) == 0,
              "New tasks should not be already present in the schedule map.");
      (*get_schedule_map(tdg->tdg_id))[KMP_TASKNODE_TO_TASKID(task)] = {
          TaskState::CREATED, -1, -1, -1, &task};
      remaining_target_tasks++;
      // Correct size of globally mapped variables
      if (isTargetDataTask(task)) {
        for (kmp_int32 i = 0; i < task.target_data->num_args; i++) {
          // Backup initial mapping size
          size_map[task.target_data->args[i]] = task.target_data->arg_sizes[i];
        }
      } else {
        for (kmp_int32 i = 0; i < task.target_data->num_args; i++) {
          // Update mapping size when needed
          auto it = size_map.find(task.target_data->args[i]);
          if (it != size_map.end())
            task.target_data->arg_sizes[i] = it->second;
        }
      }
    }
  }
  if (graph_counter != 0) {
    graph_counter++;
  }
  // dumpTaskGraph();
  graph_counter++;
  hasAcquiredGraph.store(true);
  KG_TRACE(10, ("acquireNewGraph Exit\n"));
}

void BaseScheduler::acquireAndScheduleGraph(kmp_tdg_info_t *tdg) {
  // Acquire a new graph and reset the scheduler when we have no task scheduled.

  KG_TRACE(10, ("acquireAndScheduleGraph Start\n"));
  isScheduleDone = false;
  acquireNewGraph(tdg);
  assertm(tdg != nullptr, "Task Map should not be null at this point");

  // Classical OpenMP tasks do not require target-device scheduling.
  bool has_target_tasks = false;
  for (kmp_int32 i = 0; i < tdg->num_tasks; ++i) {
    if (tdg->record_map[i].is_target_task) {
      has_target_tasks = true;
      break;
    }
  }

  if (!has_target_tasks) {
    isScheduleDone = 1;
    return;
  }

  // Schedule the graph.
  if (initial_schedule_policy == 1 &&
      scheduling_iteration_counter < initial_iteration) {
    KG_TRACE(10, ("acquireAndScheduleGraph: RR selected\n"));
    scheduleGraphRR(scheduling_iteration_counter);
  } else {
    KG_TRACE(10, ("acquireAndScheduleGraph: Main Scheduler selected\n"));
    scheduleGraph();
  }
  isScheduleDone = 1;
  KG_TRACE(10, ("acquireAndScheduleGraph Exit\n"));
}

void BaseScheduler::acquireGraphOnly(kmp_tdg_info_t *new_tdg) {
  acquireNewGraph(new_tdg);
}

kmp_int32 BaseScheduler::scheduleTask(kmp_int32 task_id) {
  KG_TRACE(10, ("scheduleTask start, task_id: %d\n", task_id));
  kmp_int32 active_devices_number = __kmpc_get_num_devices();
  kmp_int32 scheduled_id = -1;
  // Tasks may execute during TDG recording, before graph acquisition. Such a
  // task has no scheduling decision: return -1 so the caller (libomptarget
  // Schedule()) keeps the task's device() clause, or device 0 without one.
  if (task_id != -1 && tdg == nullptr)
    return -1;

  if (task_id == -1) {
    // If the target region is not inside a task (non-nowait), use the
    // configured scheduling strategy
    if (non_task_scheduler == NonTaskScheduler::ROUNDROBIN) {
      scheduled_id = (non_nowait_counter++ % active_devices_number);
      KG_TRACE(50, ("[SCHEDULER] scheduleTask : task_id: %d, default round "
                    "robin: rank = %d\n",
                    task_id, scheduled_id));
    } else {
      // Static strategy as default: map to the first device
      scheduled_id = 0;
      KG_TRACE(50,
               ("[SCHEDULER] scheduleTask : task_id: %d, default : rank = %d\n",
                task_id, scheduled_id));
    }
  } else {

    if ((*get_schedule_map(tdg->tdg_id)).count(task_id) == 0) {
      // The task is not in the recorded schedule (serialization, or a task
      // created outside the recorded region): no decision, the caller keeps
      // the user-specified device or falls back to the first device.
      KG_TRACE(50, ("[SCHEDULER] scheduleTask : task_id: %d, task not "
                    "scheduled\n",
                    task_id));
      return -1;
    } else {
      // If the target region is inside a task (nowait), use the result of the
      // graph based scheduling algorithm.

      assert((*get_schedule_map(tdg->tdg_id)).count(task_id) == 1);

      preScheduleTask(task_id, active_devices_number);
      const TaskScheduleData &task_schedule =
          (*get_schedule_map(tdg->tdg_id))[task_id];

      KG_TRACE(50, ("[SCHEDULER] scheduleTask : task_id: %d, using scheduler : "
                    "rank = %d, schedule state: %d \n",
                    task_id, task_schedule.rank, task_schedule.state));
      assert(task_schedule.rank != -1);
      assert(task_schedule.state == TaskState::SCHEDULED);

      // A classical task is pinned to the head node (rank 0), which is not a
      // device, so a target region inside it has no decision either.
      if (task_schedule.rank == 0)
        return -1;

      scheduled_id = task_schedule.rank - 1;
      KG_TRACE(50, ("[SCHEDULER] scheduleTask : task_id: %d, using scheduler : "
                    "rank = %d\n",
                    task_id, scheduled_id));
    }
  }
  assert(scheduled_id >= 0);
  assert(scheduled_id < active_devices_number);
  return scheduled_id;
}

kmp_int32 BaseScheduler::get_scheduleTask(kmp_int32 task_id) {
  KG_TRACE(50, ("[SCHEDULER] get_scheduleTask: task_id: %d\n", task_id));
  // A task may finish during TDG recording, before the scheduler acquires it.
  // Use the default device when no graph has been acquired yet.
  if (tdg == nullptr)
    return 0;

  if (task_id == -1 || !(*get_schedule_map(tdg->tdg_id)).count(task_id))
    return 0;
  return (*get_schedule_map(tdg->tdg_id))[task_id].rank - 1;
}

void BaseScheduler::pinTask(kmp_int32 task_id, kmp_int32 DeviceId) {
  KG_TRACE(
      50, ("[SCHEDULER] pinTask: task_id %d; device: %d\n", task_id, DeviceId));
  if (task_id != -1) {
    TaskScheduleData &task_schedule = (*get_schedule_map(tdg->tdg_id))[task_id];
    task_schedule.rank = DeviceId + 1;
  }
}

// HEFT also puts classical tasks in the schedule map, pinned to the head node;
// remaining_target_tasks only counts the target tasks acquireNewGraph counted.
static bool countsAsTargetTask(const TaskScheduleData &task_schedule) {
  return task_schedule.task != nullptr && task_schedule.task->is_target_task;
}

void BaseScheduler::setTaskState(kmp_int32 task_id, TaskState new_state) {
  // A task may execute before the TDG has been acquired.
  if (tdg == nullptr)
    return;

  KG_TRACE(50, ("[SCHEDULER] setTaskState: task_id %d;\n", task_id));
  // assert(tdg != nullptr);
  if ((*get_schedule_map(tdg->tdg_id)).count(task_id) != 1) {
    return;
  }
  // KG_TRACE(10,("Task state fail: task %d. Initial task:
  // %d\n",task_id,tdg->initial_task_id));
  // assert((*get_schedule_map(tdg->tdg_id)).count(task_id) == 1);
  // assert(new_state >= TaskState::DISPATCHED);

  TaskScheduleData &task_schedule = (*get_schedule_map(tdg->tdg_id))[task_id];
  // If the current task state is FAILED, it means that the counter was already
  // decremented for this task, so we increment again to make the values
  // consistent
  if (task_schedule.state == TaskState::FAILED) {
    // A FAILED task cannot have its state changed directly to EXECUTED, it
    // needs to be DISPATCHED first
    // if (new_state != TaskState::DISPATCHED)
    remaining_target_tasks++;
  }

  task_schedule.state = new_state;
  if ((*get_schedule_map(tdg->tdg_id)).count(task_id) == 1 &&
      new_state == TaskState::EXECUTED && countsAsTargetTask(task_schedule)) {
    remaining_target_tasks--;
    KMP_DEBUG_ASSERT(remaining_target_tasks >= 0)
  }
}

void BaseScheduler::setTaskState(kmp_int32 task_id, kmp_int32 device_id,
                                 TaskState new_state) {
  if (tdg == nullptr)
    return;

  TaskScheduleData &task_schedule = (*get_schedule_map(tdg->tdg_id))[task_id];

  /*
   * schedule_map utilise des ranks commençant à 1,
   * tandis que les devices commencent à 0.
   */
  task_schedule.rank = device_id + 1;

  setTaskState(task_id, new_state);
}

// void BaseScheduler::setTaskSourceLocation(kmp_int32 task_id, ident_t *loc) {
//   TaskScheduleData &task_schedule = schedule_map[task_id];
//
//   SourceInfo tinfo(loc);
//   if (tinfo.isAvailible()) {
//     task_schedule.source_location = std::string(tinfo.getFilename()) + ":" +
//                                     std::to_string(tinfo.getLine());
//   }
// }

TaskScheduleData BaseScheduler::getTask(kmp_int32 task_id) {
  KG_TRACE(50, ("[SCHEDULER] getTask: task_id %d\n", task_id));
  if (task_id != -1) {
    if ((*get_schedule_map(tdg->tdg_id)).count(task_id) == 0) {
      return TaskScheduleData{.state = TaskState::UNDEFINED,
                              .rank = -1,
                              .translated_rank = -1,
                              .task = nullptr};
    }
    assert((*get_schedule_map(tdg->tdg_id)).count(task_id) == 1);

    return (*get_schedule_map(tdg->tdg_id))[task_id];
  }

  return TaskScheduleData{.state = TaskState::UNDEFINED,
                          .rank = -1,
                          .translated_rank = -1,
                          .task = nullptr};
}

kmp_target_dep BaseScheduler::getDep(kmp_int32 task_id, kmp_int64 addr) {
  auto dep = kmp_target_dep{.addr = -1, .dep_type = {.in = 0, .out = 0}};
  if (task_id != tdg->num_tasks) {
    kmp_node_info *task = &tdg->record_map[task_id];
    for (kmp_int32 i = 0; i < task->num_deps; ++i) {
      if (task->deps[i].addr == addr) {
        dep = task->deps[i];
        break;
      }
    }
  }

  return dep;
}

void BaseScheduler::addMemoryTasks() {

  kmp_int32 gtid = __kmp_get_gtid();
  KG_TRACE(10, ("[SCHEDULER-MT] Add Memory Tasks (start) gtid: %d\n", gtid));

  tdg->num_normal_tasks.store(tdg->num_tasks.load());
  KMP_ATOMIC_ST_RLX(&__kmp_tdg_task_id, tdg->num_tasks.load());
  KG_TRACE(10, ("addMemoryTasks: __kmp_tdg_task_id = %d\n",
                __kmp_tdg_task_id.load()));
  KG_TRACE(10,
           ("addMemoryTasks: tdg->num_normal_tasks = %d, tdg->num_tasks = %d\n",
            tdg->num_normal_tasks.load(), tdg->num_tasks.load()));

  for (auto &map_entry : (*get_schedule_map(tdg->tdg_id))) {
    TaskScheduleData &task_schedule = map_entry.second;
    const kmp_node_info *task = task_schedule.task;
    // if (isTargetDataTask(task))
    //  continue;
    kmp_int32 task_rank = task_schedule.rank - 1;
    kmp_int32 task_id = KMP_TASKPTR_TO_TDG_TASKID(task);
    KG_TRACE(50, ("[SCHEDULER-MT] searching successors for task %d; rank %d \n",
                  KMP_TASKPTR_TO_TASKID(task), task_rank));
    for (kmp_int32 idx_succ = 0; idx_succ < task->nsuccessors; idx_succ++) {
      kmp_node_info *successor = &(tdg->record_map[task->successors[idx_succ]]);

      if (isTargetDataTask(successor))
        continue;

      kmp_int32 successor_rank =
          get_scheduleTask(KMP_TASKPTR_TO_TASKID(successor));

      KG_TRACE(50, ("[SCHEDULER-MT] successor %d; rank %d \n",
                    KMP_TASKPTR_TO_TASKID(successor), successor_rank));
      if (successor_rank != task_rank || isTargetDataTask(task) ||
          isTargetDataTask(successor)) {

        if (task->target_data != nullptr && successor->target_data != nullptr) {

          for (kmp_int32 SrcDep = 0; SrcDep < task->num_deps; ++SrcDep) {

            if (!task->deps[SrcDep].dep_type.out)
              continue;

            const kmp_intptr_t DepAddr = task->deps[SrcDep].addr;

            bool SuccessorConsumes = false;

            for (kmp_int32 DstDep = 0; DstDep < successor->num_deps; ++DstDep) {

              if (successor->deps[DstDep].addr == DepAddr &&
                  successor->deps[DstDep].dep_type.in) {
                SuccessorConsumes = true;
                break;
              }
            }

            if (!SuccessorConsumes)
              continue;

            void *MemoryAddr = reinterpret_cast<void *>(DepAddr);

            kmp_int64 MemorySize = 0;

            for (kmp_int32 I = 0; I < task->target_data->num_args; ++I) {

              constexpr kmp_int64 MAPTYPE_LITERAL = 0x100;

              if (task->target_data->arg_types[I] & MAPTYPE_LITERAL)
                continue;

              if (task->target_data->args_base[I] != MemoryAddr)
                continue;

              if (task->target_data->args[I] == nullptr ||
                  task->target_data->arg_sizes[I] <= 0)
                continue;

              MemorySize = task->target_data->arg_sizes[I];
              break;
            }

            if (MemorySize <= 0)
              continue;

            // Use the device selected during recording as the transfer source.
            kmp_int32 InRank = task_rank;
            if (isTargetDataTask(task) && task->target_data &&
                task->target_data->recording_device_id >= 0) {
              InRank = static_cast<kmp_int32>(
                  task->target_data->recording_device_id);
            }
            const kmp_int32 OutRank = successor_rank;

            if (InRank == OutRank)
              continue;

            kmp_task_t *MemoryTask = __kmp_create_memory_task(
                gtid, task->task, successor->task, InRank, OutRank, MemoryAddr,
                MemorySize, true);

            if (MemoryTask) {
              // record_map may have moved if insertion enlarged it.
              if (task != &(tdg->record_map[task_id])) {
                task = &(tdg->record_map[task_id]);
                successor = &(tdg->record_map[task->successors[idx_succ]]);
              }
            }
          }
        }
      }
    }
  }

  KMP_ATOMIC_ST_RLX(&__kmp_tdg_task_id, 0);

  KG_TRACE(10, ("[SCHEDULER-MT] Add Memory Tasks (exit)\n"));
}

void BaseScheduler::waitScheduleGraph(kmp_int32 gtid) {

  // Do not block ordinary tasks before the first TDG is acquired,
  // or when no target tasks require scheduling.
  if (!hasAcquiredGraph.load() || !areTargetTaskRemaining())
    return;

  while (!(KMP_HIDDEN_HELPER_THREAD(gtid) &&
           TCR_4(__kmp_hidden_helper_team_done)) &&
         !isScheduleDone && areTargetTaskRemaining()) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

void BaseScheduler::setIsScheduleDone(bool state) { isScheduleDone = state; }

// Claims the scheduling for a thread
void BaseScheduler::setSchedulingThread(kmp_int32 gtid) {
  if (schedulingThread == -1)
    schedulingThread = gtid;
}

void BaseScheduler::resetSchedulingThread() { schedulingThread = -1; }

bool BaseScheduler::isSchedulingThread(kmp_int32 gtid) {
  return schedulingThread == gtid;
}

void BaseScheduler::resetTasksState() {
  KG_TRACE(10, ("resetTasksState start\n"));
  for (auto task : (*get_schedule_map(tdg->tdg_id))) {
    KG_TRACE(10, ("resetTasksState task %d\n", task.first));
    setTaskState(task.first, TaskState::SCHEDULED);
    if (countsAsTargetTask(task.second))
      remaining_target_tasks++;
  }
  KG_TRACE(10, ("resetTasksState end\n"));
}

void BaseScheduler::resetTasksRank() {
  KG_TRACE(10, ("resetTasksRank start\n"));
  for (auto &task : (*get_schedule_map(tdg->tdg_id))) {
    task.second.rank = -1;
    task.second.translated_rank = -1;
  }
  KG_TRACE(10, ("resetTasksRank end\n"));
}

void BaseScheduler::resetMemoryTasks() {
  KG_TRACE(10, ("resetMemoryTasks start\n"));
  for (kmp_int32 tdg_task_id = tdg->num_normal_tasks;
       tdg_task_id < tdg->num_tasks; tdg_task_id++) {
    (*get_schedule_map(tdg->tdg_id))
        .erase(KMP_TDG_TASKID_TO_TASKID(tdg, tdg_task_id));
    __kmpc_remove_task(&tdg->record_map[tdg_task_id]);
  }
  tdg->num_tasks.store(tdg->num_normal_tasks.load());
  KG_TRACE(10, ("resetMemoryTasks: __kmp_tdg_task_id = %d\n",
                __kmp_tdg_task_id.load()));
  KG_TRACE(
      10,
      ("resetMemoryTasks: tdg->num_normal_tasks = %d, tdg->num_tasks = %d\n",
       tdg->num_normal_tasks.load(), tdg->num_tasks.load()));
  KG_TRACE(10, ("resetMemoryTasks end\n"));
}

void BaseScheduler::scheduleGraphRR(kmp_int32 offset) {
  // Graph based round-robin algorithm.
  // With offset on the device
  KG_TRACE(10, ("ScheduleGraphRR(Start) \n"));
  kmp_int32 task_counter = 0;

  kmp_int32 NumDevices = __kmpc_get_num_devices();

  for (auto &map_entry : (*get_schedule_map(tdg->tdg_id))) {
    KG_TRACE(30, ("scheduleGraphRR: map_entry.first: %d\n", map_entry.first));
    TaskScheduleData &task_schedule = map_entry.second;
    auto &task = task_schedule.task;

    assert(task_schedule.task->is_target_task);
    assert(task_schedule.state <= TaskState::SCHEDULED);

    if (isTargetDataTask(task)) {
      // Defer the scheduling for target data tasks
      task_schedule.state = TaskState::DEFERRED;
    } else {
      task_schedule.state = TaskState::SCHEDULED;
      task_schedule.rank = ((task_counter++ + offset) % NumDevices) + 1;
    }
  }

  // Schedule deferred tasks
  for (auto &map_entry : (*get_schedule_map(tdg->tdg_id))) {
    TaskScheduleData &task_schedule = map_entry.second;
    auto &task = task_schedule.task;

    // These are already scheduled.
    if (task_schedule.state != TaskState::DEFERRED)
      continue;

    kmp_int32 Rank = 1;

    // Find an already scheduled successor
    // NOTE: Exit target data tasks are temporary scheduled but the rank will
    // be overwritten later with PinTask function.

    for (kmp_int32 idx_succ = 0; idx_succ < task->nsuccessors; idx_succ++) {
      auto Succ = &(tdg->record_map[task->successors[idx_succ]]);
      auto SuccRank =
          (*get_schedule_map(tdg->tdg_id))[KMP_TASKPTR_TO_TASKID(Succ)].rank;

      if (!isClassicalTask(Succ) && SuccRank != -1) {
        Rank = SuccRank;
        break;
      }
    }

    // Schedule this task to the same rank as the successor chosen before
    task_schedule.rank = Rank;
    task_schedule.state = TaskState::SCHEDULED;
  }

  KG_TRACE(10, ("[SCHEDULER]: ScheduleGraphRR (adding memory tasks)\n"));
  addMemoryTasks();

  // assert(std::none_of((*get_schedule_map(tdg->tdg_id)).begin(),
  // (*get_schedule_map(tdg->tdg_id)).end(),
  //                     [](const auto &entry) {
  //                       return entry.second.state < TaskState::SCHEDULED;
  //                     }));
  KG_TRACE(10, ("[SCHEDULER]: ScheduleGraphRR (end) \n"));
}

bool BaseScheduler::is_rescheduling() {
  KG_TRACE(10, ("[SCHEDULER]: __kmpc_is_rescheduling iteration %d  \n",
                scheduling_iteration_counter));
  if (scheduling_iteration_counter <= initial_iteration ||
      (rescheduling_interval &&
       (scheduling_iteration_counter - initial_iteration) %
               rescheduling_interval ==
           0)) {
    KG_TRACE(10, ("[SCHEDULER]: __kmpc_is_rescheduling true  \n"));
    return true;
  }
  KG_TRACE(10, ("[SCHEDULER]: __kmpc_is_rescheduling false  \n"));
  return false;
}

kmp_int32 BaseScheduler::inc_scheduling_iteration_counter() {
  return ++scheduling_iteration_counter;
}

kmp_int32 BaseScheduler::get_scheduling_iteration_counter() {
  return scheduling_iteration_counter;
}

void BaseScheduler::dumpTaskGraph() {
  if (graph_dump_dir_path.empty())
    return;

  std::ofstream dot(graph_dump_dir_path + "_graph_" +
                    std::to_string(graph_counter) + ".dot");
  graph_counter++;
  // Print header
  dot << "// Generated in BaseScheduler\n"
      << "digraph TargetTaskGraph {\n"
      << "  labelloc=\"t\";\n"
      << "  label=\"Generated in BaseScheduler\";\n";

  if (tdg == nullptr) {
    dot << "}\n";
    return;
  }

  for (kmp_int32 tid = 0; tid < tdg->num_tasks; tid++) {
    auto task = &(tdg->record_map[tid]);
    kmp_int32 trank = (*get_schedule_map(tdg->tdg_id))[tid].rank;
    std::string tinfo = (*get_schedule_map(tdg->tdg_id))[tid].source_location;

    int task_id = KMP_TASKPTR_TO_TASKID(task);

    // Print graph node
    dot << "  V" << task_id << " [label=<<B>ID " << task_id
        << "</B><BR /><FONT POINT-SIZE=\"10\">Rank " << trank
        << (((*get_schedule_map(tdg->tdg_id))[tid].state == TaskState::SAVED)
                ? "<BR />Saved\""
                : "<BR />Unsaved\"");

    if (!tinfo.empty()) {
      dot << "<BR />" << tinfo;
    }

    dot << "</FONT>>";

    if (isTargetDataTask(task))
      dot << " color=gray70 fontcolor=gray50";
    if (isRootTask(task) || isExitTask(task))
      dot << " shape=octagon";

    dot << "];\n";

    for (kmp_int32 idx = 0; idx < task->nsuccessors; idx++) {
      kmp_int32 sid = task->successors[idx];
      kmp_int32 srank = (*get_schedule_map(tdg->tdg_id))[sid].rank;
      const auto *color = detail::getEdgeColor(trank, srank);
      // Print graph edge
      dot << "  V" << tid << " -> V" << sid << " [color=\"" << color
          << "\"];\n";
    }
  }

  dot << "}\n";
}

// Task Iterators
// ======================================================================
std::list<TaskPtr> TopologicalOrder(kmp_tdg_info &tdg) {
  // Set of all nodes with no incoming edge
  std::unordered_set<TaskPtr> Set;
  // Hash map to keep track of incoming edges
  std::unordered_map<TaskPtr, std::list<TaskPtr>> Incoming;
  // Empty list that will contain the sorted elements
  std::list<TaskPtr> List;

  Set.reserve(tdg.num_tasks);
  Incoming.reserve(tdg.num_tasks);

  // Initialization
  for (kmp_int32 idx = 0; idx < tdg.num_tasks; idx++) {
    TaskPtr Task = &(tdg.record_map[idx]);
    if (Task->npredecessors == 0)
      Set.insert(Task);
    for (kmp_int32 idx_succ = 0; idx_succ < Task->nsuccessors; idx_succ++) {
      auto Succ = &(tdg.record_map[Task->successors[idx_succ]]);
      Incoming[Succ].push_back(Task);
    }
  }

  // Main algorithm
  while (not Set.empty()) {
    TaskPtr Task = *(Set.begin());
    Set.erase(Task);
    List.push_back(Task);
    for (kmp_int32 idx_succ = 0; idx_succ < Task->nsuccessors; idx_succ++) {
      auto Succ = &(tdg.record_map[Task->successors[idx_succ]]);
      Incoming[Succ].remove(Task);
      if (Incoming[Succ].empty())
        Set.insert(Succ);
    }
  }

  assert(std::all_of(Incoming.begin(), Incoming.end(),
                     [](const auto &pair) { return pair.second.empty(); }) &&
         "There should be no cyclic dependencies in the task graph");

  assert(std::all_of(List.begin(), List.end(),
                     [](TaskPtr item) { return item != nullptr; }) &&
         "There should be no null pointers in the topologically ordered list");

  return List;
}

std::list<TaskPtr> TopologicalOrder(kmp_tdg_info *tdg) {
  assert(tdg != nullptr);
  return TopologicalOrder(*tdg);
}

std::list<TaskPtr> ReverseTopologicalOrder(kmp_tdg_info &tdg) {
  auto List = TopologicalOrder(tdg);
  List.reverse();
  return List;
}

std::list<TaskPtr> ReverseTopologicalOrder(kmp_tdg_info *tdg) {
  assert(tdg != nullptr);
  return ReverseTopologicalOrder(*tdg);
}
