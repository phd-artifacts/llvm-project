//===----------------------- kmp_target_heft.h - HEFT scheduler
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Heterogeneous Earliest Finish Time algorithm.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_HEFT_H
#define _OMPTARGET_HEFT_H

#include "kmp_target_basescheduler.h"
#include "kmp.h"

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declaration
class HEFTScheduler;

namespace detail {

// Types
// =============================================================================

/// Priority queue, ordered from the higher upward rank to the lowest.
using PriorityQueue = std::deque<std::pair<TaskPtr, double>>;

// Classes
// =============================================================================

/// Identifies which rank, thread pair the task should be executed at.
struct DeviceId {
  /// The MPI rank of the worker node.
  kmp_int32 Rank;
  /// The ID of the event handler thread.
  kmp_int32 Thread;

  explicit DeviceId(kmp_int32 Rank, kmp_int32 Thread = -1)
      : Rank(Rank), Thread(Thread) {}

  explicit DeviceId(const TaskScheduleData &Data)
      : Rank(Data.rank), Thread(Data.thread) {}

  ~DeviceId() = default;

  static DeviceId head() { return DeviceId{0, -1}; }

  bool operator==(const DeviceId &Other) const {
    return Rank == Other.Rank && Thread == Other.Thread;
  }

  /// Hashing function for this type so we can use it as unordered map keys
  struct Hash {
    size_t operator()(const DeviceId &Device) const {
      return std::hash<kmp_int32>()(Device.Rank) ^
             std::hash<kmp_int32>()(Device.Thread);
    }
  };
};

/// Context of the HEFT scheduler.
///
/// This is the class that does all of the heavy lifting in terms of computing
/// the intermediate values for the HEFT algorithm. For clarification about what
/// each of these data structures mean, refer to the original HEFT paper.
///
/// \note As an implementation detail, this class makes heavy use of
/// `std::unordered_map` (hash) maps because of its almost-constant lookup time,
/// in constrast to `std::map`'s logarithm lookup.
///
/// \see HEFTScheduler
/// \see https://ieeexplore.ieee.org/document/993206
class HEFTContext {
public:
  /// Map a (task, device) pair to a timepoint.
  ///
  /// This table is filled by the scheduler in order to find the device which
  /// minimizes the earliest finish time.
  using EarliestTimeMap =
      std::unordered_map<TaskPtr,
                         std::unordered_map<DeviceId, double, DeviceId::Hash>>;

  /// Maps a task to a timepoint.
  ///
  /// This table indicates the actual start or finish time of a task after being
  /// scheduled.
  using ActualTimeMap = std::unordered_map<TaskPtr, double>;

  /// Device availability map.
  ///
  /// Maps a device ID to the next timepoint it will be available for executing
  /// a new task.
  using DeviceAvailMap = std::unordered_map<DeviceId, double, DeviceId::Hash>;

  /// Upward rank map.
  ///
  /// Maps a task to its upward rank priority.
  using UpwardRankMap = std::unordered_map<TaskPtr, double>;

  /// Communication map.
  ///
  /// Indicates how many bytes should be transferred from one task to another.
  using CommunicationMap =
      std::unordered_map<TaskPtr, std::unordered_map<TaskPtr, double>>;

  /// Computation map.
  ///
  /// Indicates how long each task should take to execute after being scheduled.
  using AverageComputationMap = std::unordered_map<TaskPtr, double>;

  /// Computation map.
  ///
  /// Indicates how long each task should take to execute after being scheduled
  /// on each device.
  using ComputationMap =
      std::unordered_map<TaskPtr, std::unordered_map<kmp_int32, double>>;

  using kmp_device_type = int;

  friend class ::HEFTScheduler;

public:
  /// Default constructor.
  HEFTContext(kmp_tdg_info *tdg);

  /// Destructor.
  ~HEFTContext() = default;

  /// Compute upward rank for every task in the graph.
  ///
  /// This should be called **before** starting the scheduling process.
  void computeUpwardRanks();

  /// Compute the earliest finish time for a specific (task, rank) pair.
  ///
  /// \returns the computed EFT.
  double computeEFT(TaskPtr Task, DeviceId Device);

  /// Schedule the task on the best rank available based on the EFT
  /// calculations.
  ///
  /// \returns the actual finish time of the task.
  double scheduleTaskOnBestRank(TaskPtr Task);

  /// Force a task to be scheduled on a specific rank irrespective its
  /// predecessors.
  ///
  /// \param task the task
  /// \param rank the rank
  void forceScheduleTask(TaskPtr Task, DeviceId Device);

  /// Defer scheduling of task for later.
  ///
  /// \param task the task
  void deferTask(TaskPtr Task);

  /// Returns a boolean indicating if this task is deferred.
  ///
  /// \param task the task
  inline bool isDeferredTask(const kmp_node_info &Task) const {
    bool failed =
        isTargetDataTask(Task) &&
        (*get_schedule_map(tdg->tdg_id))[KMP_TASKNODE_TO_TASKID(Task)].state ==
            TaskState::FAILED;
    return (*get_schedule_map(tdg->tdg_id))[KMP_TASKNODE_TO_TASKID(Task)]
                   .state == TaskState::DEFERRED ||
           failed;
  }

  /// Build a priority queue from the upward rank map.
  ///
  /// \returns the priority queue built.
  PriorityQueue buildQueue() const;

  /// Build a list of device ids from the total number of ranks and handler
  /// threads.
  std::vector<DeviceId> buildDeviceList(kmp_int32 NumRanks) const;

  /// Clear the structures used by the context.
  void clear();

  void clearCompCost();

  /// Construct CompCost Matrix from profiled informations
  void interpolateCompCost(kmp_int32 scheduling_iteration_counter,
                           kmp_int32 scheduling_policy,
                           kmp_int32 initial_iteration);

  /// Set a new task map.
  void setTaskMap(kmp_int32 tdg_idx);

  /// Load every task's file ranges and the I/O rates for this pass, and forget
  /// the owners predicted by the previous pass.
  void loadIoModel();

  /// Predicted time a task spends in file I/O on a device: each range costs
  /// the local or the remote rate depending on whether the device will own
  /// the file. Zero when the I/O term is off.
  ///
  /// \param RemoteBytes if not null, receives the bytes predicted remote.
  /// File I/O of the task at the local rate (0 with the term off).
  double localIoCost(TaskPtr Task) const;
  double computeIoCost(TaskPtr Task, DeviceId Device,
                       kmp_uint64 *RemoteBytes = nullptr) const;

  /// Record where a scheduled task's files will be owned: a file the task
  /// writes before reading it moves its owner to the writer (libompfile binds
  /// a new file's owner to its first writing proxy).
  void noteIoPlacement(TaskPtr Task, DeviceId Device);

  /// Print the per-graph I/O summary when OMPCLUSTER_HEFT_IO_LOG is set.
  void reportIoModel(kmp_int32 TdgId) const;

  // Dump internal HEFT information to a file.
  void dump(kmp_int32 NumDevices);

private:
  void buildDeviceTypeMap();
  void computeAvgCompCosts();
  void homogeneousInterpolation();
  void linearInterpolation();
  void heterogeneousLinearInterpolation();
  void restrictedLinearInterpolation(ComputationMap, int);
  void completeheterogeneousInterpolation();
  void printCompCostMatrices();
  /// Compute the earliest start time for a specific (task, rank) pair.
  ///
  /// \returns the computed EST.
  double computeEST(TaskPtr Task, DeviceId Device);

  /// Commit a task to run on a specific rank.
  ///
  /// The task must not be rescheduled after commit.
  ///
  /// \returns the actual finish time of the task.
  double scheduleTask(TaskPtr Task, DeviceId Device);

  /// Estimate the communication cost between two adjacent tasks.
  ///
  /// \param A the current task
  /// \param RankA the rank where A's data should read from
  /// \param B the successor task
  /// \param RankB the rank where B's data should write to
  ///
  /// \returns the communication cost from A -> B.
  double computeCommCost(const kmp_node_info &A, DeviceId DeviceA,
                         const kmp_node_info &B, DeviceId DeviceB);

  /// Estimate the average communication cost between two adjacent tasks.
  ///
  /// \param A the current task
  /// \param B the successor task
  ///
  /// \returns the average communication cost from A -> B
  double computeAvgCommCost(const kmp_node_info &A, const kmp_node_info &B);

  /// Estimate the average computation cost of a task.
  ///
  /// \param task the task
  ///
  /// \retunrs the average computation cost
  double computeAvgCompCost(const kmp_node_info &Task);

  /// Estimate the average computation cost of a task running on a specific
  /// rank.
  ///
  /// \param task the task
  /// \param trank the rank of the task
  ///
  /// \retunrs the average computation cost
  double computeCompCost(const kmp_node_info &Task, DeviceId Device);

private:
  /// Coefficient of the communication cost.
  /// Having a higher coefficient means that communication will have more weight
  /// in relation to computation.
  double CommMultiplier;
  /// Default computation cost of tasks
  double DefaultCompCost;
  /// Default communication cost of tasks
  double DefaultCommCost;
  /// Number of threads in each node to choose from.
  kmp_int32 NumThreads;
  /// List of tasks to be scheduled
  kmp_tdg_info *tdg;
  /// Earliest start time
  EarliestTimeMap EST;
  /// Earliest finish time
  EarliestTimeMap EFT;
  /// Actual start time
  ActualTimeMap AST;
  /// Actual finish time
  ActualTimeMap AFT;
  /// Upward ranks
  UpwardRankMap UpwardRank;
  /// Device availability
  DeviceAvailMap DeviceAvail;
  /// Communication map
  CommunicationMap CommunicationCost;
  /// Average Computation map
  AverageComputationMap AverageComputationCost;
  /// Computation map
  ComputationMap ComputationCost;
  /// File where to dump HEFT information
  std::string HEFTLogPath;
  /// Using heterogeneous Interpolation
  bool heterogeneousInterpolation = 0;
  kmp_int32 deviceTypeNumber = 1;
  std::unordered_map<int, int> deviceTypeMap;
  std::vector<int> device_restricted_map;
  std::unordered_map<int, std::vector<int>> restricted_to_device_map;
  std::vector<ComputationMap> device_restricted_CompCost;
  kmp_int32 maxIteration = 100;

  /// File-locality term (OMPCLUSTER_HEFT_IO, on by default).
  bool IoEnabled = true;
  bool IoLog = false;
  /// Microseconds per byte for a range on the owner and away from it, and
  /// where each value came from ("measured", "default" or "env").
  double IoLocalUsPerByte = 0.0;
  double IoRemoteUsPerByte = 0.0;
  const char *IoLocalSource = "default";
  const char *IoRemoteSource = "default";
  /// Values set through the environment, or -1.
  double IoLocalUsPerByteEnv = -1.0;
  double IoRemoteUsPerByteEnv = -1.0;
  /// Lowest remote/local ratio a measured remote rate may take
  /// (OMPCLUSTER_HEFT_IO_REMOTE_MIN_RATIO, 0 disables the floor). The rate
  /// measured on the requester misses the owner proxy's service time and the
  /// interference with the owner's own tasks; at the measured 1.3x the term
  /// traded locality for balance and lost on RTM and Cholesky, and 24 was the
  /// lowest floor that gained on both on AMD.
  double IoRemoteMinRatio = 24.0;
  /// File ranges of every node, indexed by the node's TDG task id.
  std::vector<std::vector<kmp_task_io_entry_t>> IoRanges;
  /// Owner predicted for each file (path key) by the tasks scheduled so far.
  std::unordered_map<kmp_uint64, kmp_int32> IoOwner;
  /// Price reads a prefetch task will serve at the local rate
  /// (OMPCLUSTER_HEFT_IO_PREFETCH_PRICING, on by default; it applies only
  /// when OMPCLUSTER_HEFT_IO_TASKS includes prefetch).
  bool IoPrefetchPricing = true;
  /// Per node, the files it reads that exactly one predecessor writes and it
  /// does not write itself: the reads addIoTasks prefetches when remote.
  std::unordered_map<kmp_int32, std::unordered_set<kmp_uint64>> IoPrefetched;
  /// Totals of the chosen placement, for the summary line.
  kmp_uint64 IoTasks = 0;
  kmp_uint64 IoBytes = 0;
  kmp_uint64 IoRemoteBytes = 0;
  double IoCostUs = 0.0;
};

} // namespace detail

/// Implementation of the scheduler interface for HEFT.
///
/// This class provides a simple interface that external users can call to run
/// the HEFT scheduler.
///
/// \see HEFTContext
class HEFTScheduler final : public BaseScheduler {
public:
  /// Default constructor.
  HEFTScheduler() : Ctx(this->tdg) {}

  /// Destructor.
  ~HEFTScheduler() {}

private:
  /// Run HEFT scheduling algorithm.
  void scheduleGraph() override;

  /// Clear context after acquiring new graph.
  void acquireNewGraphPost() override;

  /// Do nothing.
  void preScheduleTask(kmp_int32 task_id, kmp_int32) override {}

  void resetWeight() override;

  /// Insert the file I/O tasks OMPCLUSTER_HEFT_IO_TASKS asks for (prefetch,
  /// stage, flush) on the edges of the scheduled graph.
  void addIoTasks();

public:
  /// Dump the task graph with HEFT-specific information.
  void dumpTaskGraph() override;

private:
  detail::HEFTContext Ctx;

private:
  struct DumpOptions {
    DumpOptions();
    /// Whether to dump the communication cost in the edges
    bool EdgeLabel = true;
    /// Whether to show predecessors or not
    bool Predecessors = false;
    /// Wether to dump a fake, unique entry node
    bool FakeEntryNode = false;
    /// Wether to dump a fake, unique exit node
    bool FakeExitNode = false;
  };

  DumpOptions Dump;
};

#endif /* _OMPTARGET_HEFT_H */
