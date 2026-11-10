/*
 * kmp_target.cpp -- OpenMP integration with target library
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "kmp_dmdas.h"

#include <cassert>
#include <cerrno>
#include <cstdint>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#if KMP_OS_UNIX
#include <dlfcn.h>
#endif
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <list>
#include <map>
#include <functional>

#define assertm(exp, msg) assert(((void)msg, exp));

// Legacy target-task graph support
// =============================================================================

// Address dep collision structure
typedef struct addr_ref {
  kmp_target_task_t *last_out = nullptr;
  std::list<kmp_target_task_t *> last_ins;
} addr_ref_t;

using TaskPtrMap = std::unordered_map<kmp_int32, kmp_target_task_t *>;

struct TargetTaskContext {
  std::unordered_map<kmp_int32, kmp_target_task_t> active_tasks;
  TaskPtrMap root_tasks;
  std::unordered_map<kmp_intptr_t, addr_ref_t> dep_history;
  std::unordered_map<kmp_target_task_map_t *, TaskPtrMap> user_hash_map;

  int graph_builder_gtid;
  std::mutex ctx_mutex;
};

static TargetTaskContext *gcontext = nullptr;
static std::atomic<int> gcontext_reference_count(0);

__attribute__((constructor)) static void init_target_task_context() {
  int old_count = gcontext_reference_count++;

  if (old_count == 0) {
    gcontext = new TargetTaskContext;
    gcontext->graph_builder_gtid = -1;
  } else {
    while (gcontext == nullptr)
      ;
  }

  KA_TRACE(20, ("init_target_task_context: new user registered, currently %d "
                "users depend on target-task integration",
                old_count + 1));
}

__attribute__((destructor)) static void deinit_target_task_context() {
  int old_count = gcontext_reference_count--;
  assert(old_count > 0);

  if (old_count == 1 && gcontext != nullptr) {
    delete gcontext;
    gcontext = nullptr;
  }

  KA_TRACE(20, ("deinit_target_task_context: user unregistered, currently %d "
                "users depend on target-task integration",
                old_count - 1));
}

// Helper functions
// =============================================================================
template <typename T> static T *thread_malloc(kmp_info_t *thread, size_t size) {
  // Use fast user space allocation if an OpenMP thread context is available.
  if (thread != nullptr) {
#if USE_FAST_MEMORY
    return (T *)__kmp_fast_allocate(thread, size * sizeof(T));
#else
    return (T *)__kmp_thread_malloc(thread, size * sizeof(T));
#endif
  }

  // Fallback to normal allocation otherwise.
  return (T *)__kmp_allocate(size * sizeof(T));
}

static void thread_free(kmp_info_t *thread, void *ptr) {
  if (thread != nullptr) {
#if USE_FAST_MEMORY
    __kmp_fast_free(thread, ptr);
#else
    __kmp_thread_free(thread, ptr);
#endif
  } else {
    __kmp_free(ptr);
  }
}

static void add_to_succ_list(kmp_info_t *thread, kmp_target_task_t *target_task,
                             kmp_target_task_t *succ_task,
                             bool update_pred = true) {
  auto *succ = thread_malloc<kmp_target_task_list_node_t>(thread, 1);
  succ->task = succ_task;
  succ->next = target_task->successors;
  target_task->successors = succ;

  auto *pred = thread_malloc<kmp_target_task_list_node_t>(thread, 1);
  pred->task = target_task;
  pred->next = succ_task->predecessors;
  succ_task->predecessors = pred;

  if (update_pred)
    succ_task->num_predecessors++;
}

static void remove_from_pred_list(kmp_info_t *thread, kmp_target_task_t *task,
                                  kmp_target_task_t *pred_task) {
  assertm(task != nullptr && pred_task != nullptr,
          "task and pred_task cannot be null");

  kmp_target_task_list_node_t *prev = nullptr;
  kmp_target_task_list_node_t *curr = task->predecessors;

  while (curr != nullptr && curr->task != pred_task) {
    prev = curr;
    curr = curr->next;
  }

  assertm(curr != nullptr && curr->task == pred_task,
          "Expected to find predecessor task in the current list");

  if (prev == nullptr)
    task->predecessors = curr->next;
  else
    prev->next = curr->next;

  thread_free(thread, curr);
}

// Returns the thread info of the current OpenMP thread. If the caller is not an
// OpenMP thread, returns nullptr.
static kmp_info_t *get_omp_thread() {
  // Get the current omp global thread id (valid if gtid >= 0).
  const int omp_gtid = __kmp_get_gtid();

  if (omp_gtid >= 0) {
    return __kmp_thread_from_gtid(omp_gtid);
  }

  return nullptr;
}

// Target task data
// =============================================================================
kmp_target_task_data::kmp_target_task_data(kmp_int64 device_id_p,
                                           void *outlined_fn_id_p,
                                           kmp_int32 num_args_p,
                                           void **args_base_p, void **args_p,
                                           kmp_int64 *arg_sizes_p,
                                           kmp_int64 *arg_types_p)
    : device_id(device_id_p), outlined_fn_id(outlined_fn_id_p),
      num_args(num_args_p) {
  allocBuffers();
  copyBuffers(args_base_p, args_p, arg_sizes_p, arg_types_p);
}

kmp_target_task_data::~kmp_target_task_data() { deleteBuffers(); }

void *kmp_target_task_data::operator new(size_t size) {
  // TODO: Check why we cannot use thread fast allocation without getting
  // address errors when freeing them.
  return __kmp_allocate(size);
}

void kmp_target_task_data::operator delete(void *ptr) { __kmp_free(ptr); }

kmp_target_task_data &
kmp_target_task_data::operator=(const kmp_target_task_data &other) {
  // Skip self assignment.
  if (this == &other) {
    return *this;
  }

  // Realloc internal arrays on size change.
  if (num_args != other.num_args) {
    deleteBuffers();
    num_args = other.num_args;
    allocBuffers();
  }

  // Assign and copy new data from other.
  device_id = other.device_id;
  recording_device_id = other.recording_device_id;
  outlined_fn_id = other.outlined_fn_id;
  copyBuffers(other.args_base, other.args, other.arg_sizes, other.arg_types);

  return *this;
}

void kmp_target_task_data::allocBuffers() {
  KG_TRACE(10, ("kmp_target_task_data::allocBuffers (start)\n"));
  if (num_args > 0) {
    // TODO: Check why we cannot use thread fast allocation without getting
    // address errors when freeing them.
    args_base = (void **)__kmp_allocate(sizeof(void *) * num_args);
    args = (void **)__kmp_allocate(sizeof(void *) * num_args);
    arg_sizes = (kmp_int64 *)__kmp_allocate(sizeof(kmp_int64) * num_args);
    arg_types = (kmp_int64 *)__kmp_allocate(sizeof(kmp_int64) * num_args);
  }
  KG_TRACE(10, ("kmp_target_task_data::allocBuffers (end)\n"));
}

void kmp_target_task_data::deleteBuffers() {
  KG_TRACE(10, ("kmp_target_task_data::deleteBuffers (start)\n"));
  if (num_args > 0) {
    __kmp_free(args_base);
    __kmp_free(args);
    __kmp_free(arg_sizes);
    __kmp_free(arg_types);
  }
  args_base = nullptr;
  args = nullptr;
  arg_sizes = nullptr;
  arg_types = nullptr;
  KG_TRACE(10, ("kmp_target_task_data::deleteBuffers (end)\n"));
}

void kmp_target_task_data::copyBuffers(void **args_base_p, void **args_p,
                                       kmp_int64 *arg_sizes_p,
                                       kmp_int64 *arg_types_p) {
  KG_TRACE(10, ("kmp_target_task_data::copyBuffers (start)\n"));
  if (num_args > 0) {
    std::copy_n(args_base_p, num_args, args_base);
    std::copy_n(args_p, num_args, args);
    std::copy_n(arg_sizes_p, num_args, arg_sizes);
    std::copy_n(arg_types_p, num_args, arg_types);
  }
  KG_TRACE(10, ("kmp_target_task_data::copyBuffers (end)\n"));
}

// Task and graph management
// =============================================================================
void __kmpc_add_target_task_with_deps(
    const kmp_taskdata_t *taskdata, kmp_int32 ndeps,
    const kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
    const kmp_depend_info_t *dep_list_noalias) {

  std::unique_lock<std::mutex> ctx_lk(gcontext->ctx_mutex);

  assertm(taskdata != nullptr, "Invalid taskdata pointer");
  assertm((ndeps > 0) == (dep_list != nullptr), "Invalid dependencies list");
  assertm((ndeps_noalias > 0) == (dep_list_noalias != nullptr),
          "Invalid no-alias dependencies list");
  assertm(gcontext->active_tasks.count(taskdata->td_task_id) == 0,
          "Same task added multiple times to the target task graph");

  // Get general data
  kmp_info_t *thread = get_omp_thread();
  assertm(thread != nullptr,
          "Function should be called inside an OpenMP thread");

  // Set the new graph creator
  int gtid = __kmp_gtid_from_thread(thread);
  if (gcontext->graph_builder_gtid != gtid) {
    gcontext->graph_builder_gtid = gtid;
    KA_TRACE(20, ("__kmpc_add_target_task_with_deps: T#%d is the new "
                  "target-task graph builder",
                  gtid));
  }

  // Create target task structure
  const kmp_int32 task_id = taskdata->td_task_id;

  kmp_target_task_t &new_task = gcontext->active_tasks[taskdata->td_task_id];
  new_task.task_id = task_id;
  new_task.task = KMP_TASKDATA_TO_TASK(const_cast<kmp_taskdata_t *>(taskdata));
  new_task.is_target_task = (taskdata->td_target_info != nullptr);
  if (new_task.is_target_task) {
    new_task.target_data = *(taskdata->td_target_info);
  }

  // Process dependencies
  new_task.deps =
      thread_malloc<kmp_target_dep_t>(thread, ndeps + ndeps_noalias);
  new_task.num_deps = 0;

  // Filter aliased addresses and add dependencies to task
  for (int i = 0; i < ndeps; ++i) {
    const kmp_depend_info_t &dep = dep_list[i];
    bool is_alias = false;

    for (int j = 0; j < new_task.num_deps; ++j) {
      kmp_target_dep_t &prev_dep = new_task.deps[j];
      if (prev_dep.addr == dep.base_addr) {
        prev_dep.dep_type.in |= dep.flags.in;
        prev_dep.dep_type.out |= dep.flags.out;
        prev_dep.dep_type.mtx |= dep.flags.mtx;
        is_alias = true;
        break;
      }
    }

    if (!is_alias) {
      new_task.deps[new_task.num_deps++] = kmp_target_dep_t{
          .addr = dep.base_addr,
          .dep_type = {.in = static_cast<bool>(dep.flags.in),
                       .out = static_cast<bool>(dep.flags.out),
                       .mtx = static_cast<bool>(dep.flags.mtx)}};
    }
  }

  // Add no-alias dependencies
  for (int i = 0; i < ndeps_noalias; ++i) {
    const kmp_depend_info_t &dep = dep_list_noalias[i];
    new_task.deps[new_task.num_deps + i] =
        kmp_target_dep_t{.addr = dep.base_addr,
                         .dep_type = {.in = static_cast<bool>(dep.flags.in),
                                      .out = static_cast<bool>(dep.flags.out),
                                      .mtx = static_cast<bool>(dep.flags.mtx)}};
  }
  new_task.num_deps += ndeps_noalias;

  // Add task to the graph
  for (int i = 0; i < new_task.num_deps; ++i) {
    const kmp_target_dep_t &dep = new_task.deps[i];
    addr_ref_t &ref = gcontext->dep_history[dep.addr];

    if (dep.dep_type.out) {
      // If we have an out dep, link the previous in tasks as our predecessors
      // or, if no in deps is available, link the previous out dep.
      for (auto &in_task : ref.last_ins) {
        add_to_succ_list(thread, in_task, &new_task);
      }

      if (ref.last_ins.empty() && (ref.last_out != nullptr)) {
        add_to_succ_list(thread, ref.last_out, &new_task);
      }

      ref.last_ins.clear();
      ref.last_out = &new_task;
    } else if (dep.dep_type.in) {
      // If we have an in dep, add it to the ins list and link the previous out
      // dep as our predecessor.
      assert(dep.dep_type.in);

      if (ref.last_out != nullptr) {
        add_to_succ_list(thread, ref.last_out, &new_task);
      }

      ref.last_ins.push_front(&new_task);
    }

    // Note: Mtx deps are not considered when constructing the graph and should
    // be handled by the graph user.
  }

  if (new_task.num_predecessors == 0) {
    gcontext->root_tasks.emplace(task_id, &new_task);
  }
}

void __kmpc_remove_root_target_task(kmp_int32 task_id) {
  // Only one OpenMP thread can remove a task each time
  std::unique_lock<std::mutex> ctx_lk(gcontext->ctx_mutex);

  // This function can be called for tasks without dependencies. Immediately
  // return in these cases.
  if (gcontext->active_tasks.count(task_id) == 0) {
    return;
  }
  assertm(gcontext->root_tasks.count(task_id) == 1,
          "Trying to remove non-root task");

  // Get general data
  kmp_info_t *thread = get_omp_thread();
  assertm(thread != nullptr,
          "Function should be called inside an OpenMP thread");

  // Remove from root map, clear successors list and find new root tasks
  gcontext->root_tasks.erase(task_id);
  kmp_target_task_t &task = gcontext->active_tasks[task_id];
  while (task.successors != nullptr) {
    kmp_target_task_list_node_t *succ = task.successors;
    task.successors = task.successors->next;

    kmp_target_task_t *succ_task = succ->task;
    succ_task->num_predecessors--;
    if (succ_task->num_predecessors == 0) {
      gcontext->root_tasks.emplace(succ_task->task_id, succ_task);
    }

    remove_from_pred_list(thread, succ_task, &task);
    thread_free(thread, succ);
  }

  // Remove any address reference
  for (int i = 0; i < task.num_deps; ++i) {
    auto addr_ref = gcontext->dep_history.find(task.deps[i].addr);
    if (addr_ref != gcontext->dep_history.end()) {
      addr_ref_t &ref = addr_ref->second;

      if (ref.last_out == &task) {
        ref.last_out = nullptr;
      }

      ref.last_ins.remove(&task);

      // If the removed reference is empty, remove it from the history
      if (ref.last_out == nullptr && ref.last_ins.empty()) {
        gcontext->dep_history.erase(addr_ref);
      }
    }
  }
  thread_free(thread, task.deps);

  gcontext->active_tasks.erase(task_id);

  bool graph_became_empty = gcontext->active_tasks.empty();

  ctx_lk.unlock();

  if (graph_became_empty)
    __kmpc_reset_legacy_scheduler_owner();
}

// Task graph access functions
// =============================================================================

kmp_target_task_map_t *__kmpc_target_task_map_create() {
  // Init global target task context
  init_target_task_context();

  // Guarantee that no task is being removed when someone creates a task map.
  // This also synchronizes the access to the 'user_hash_map'
  std::unique_lock<std::mutex> ctx_lk(gcontext->ctx_mutex);

  if (gcontext->active_tasks.empty()) {
    return nullptr;
  }

  // Allocate user task map data
  auto *user_map = thread_malloc<kmp_target_task_map_t>(nullptr, 1);
  gcontext->user_hash_map.emplace(user_map, TaskPtrMap());

  user_map->num_tasks = gcontext->active_tasks.size();
  user_map->tasks =
      thread_malloc<kmp_target_task_t>(nullptr, user_map->num_tasks);

  user_map->num_roots = gcontext->root_tasks.size();
  user_map->roots =
      thread_malloc<kmp_target_task_t *>(nullptr, user_map->num_roots);

  // Copy tasks and set roots
  int task_idx = 0;
  int root_idx = 0;
  for (const auto &task_hash_item : gcontext->active_tasks) {
    const kmp_target_task_t &source_task = task_hash_item.second;
    assert(task_idx < user_map->num_tasks);

    // Copy task data
    kmp_target_task_t &user_task = user_map->tasks[task_idx++];
    user_task = source_task;

    user_task.successors = nullptr;
    user_task.predecessors = nullptr;

    user_task.deps =
        thread_malloc<kmp_target_dep_t>(nullptr, user_task.num_deps);
    std::copy_n(source_task.deps, source_task.num_deps, user_task.deps);

    gcontext->user_hash_map[user_map][user_task.task_id] = &user_task;

    if (gcontext->root_tasks.count(user_task.task_id) == 1) {
      assert(root_idx < user_map->num_roots);
      user_map->roots[root_idx++] = &user_task;
    }
  }

  // Copy graph structure
  for (task_idx = 0; task_idx < user_map->num_tasks; ++task_idx) {
    kmp_target_task_t &user_task = user_map->tasks[task_idx];
    assert(gcontext->active_tasks.count(user_task.task_id) == 1);
    const kmp_target_task_t &source_task =
        gcontext->active_tasks[user_task.task_id];

    kmp_target_task_list_node_t *succ = source_task.successors;
    while (succ != nullptr) {
      assert(succ->task != nullptr);
      assert(gcontext->user_hash_map[user_map].count(succ->task->task_id) == 1);

      kmp_target_task_t *succ_task =
          gcontext->user_hash_map[user_map][succ->task->task_id];
      add_to_succ_list(nullptr, &user_task, succ_task, false);

      succ = succ->next;
    }
  }

  return user_map;
}

void __kmpc_target_task_map_free(kmp_target_task_map_t *task_map) {
  // Synchronize the access to the 'user_hash_map'
  std::unique_lock<std::mutex> ctx_lk(gcontext->ctx_mutex);

  if (task_map == nullptr) {
    return;
  }

  assertm(gcontext->user_hash_map.count(task_map) == 1,
          "Invalid task map pointer");
  gcontext->user_hash_map.erase(task_map);

  for (int i = 0; i < task_map->num_tasks; ++i) {
    kmp_target_task_t &task = task_map->tasks[i];

    thread_free(nullptr, task.deps);

    while (task.successors != nullptr) {
      kmp_target_task_list_node_t *node = task.successors;
      task.successors = node->next;
      thread_free(nullptr, node);
    }
  }

  thread_free(nullptr, task_map->roots);
  thread_free(nullptr, task_map->tasks);

  thread_free(nullptr, task_map);

  // Deinit global target task context
  deinit_target_task_context();
}

kmp_target_task_t *__kmpc_target_task_map_get(kmp_target_task_map_t *task_map,
                                              kmp_int32 task_id) {
  // Synchronize the access to the 'user_hash_map'
  std::unique_lock<std::mutex> ctx_lk(gcontext->ctx_mutex);

  if (task_map == nullptr) {
    return nullptr;
  }

  if (gcontext->user_hash_map.count(task_map) == 0) {
    return nullptr;
  }

  if (gcontext->user_hash_map[task_map].count(task_id) == 0) {
    return nullptr;
  }

  return gcontext->user_hash_map[task_map][task_id];
}

void __kmpc_record_target_data_device(kmp_int32 device_id) {
  kmp_info_t *thread = get_omp_thread();
  if (!thread)
    return;

  kmp_taskdata_t *task = thread->th.th_current_task;
  if (!task || task->td_flags.tasktype != TASK_EXPLICIT ||
      !task->td_target_info)
    return;

  if (task->td_target_info->recording_device_id >= 0)
    return;

  task->td_target_info->recording_device_id = device_id;
}

struct kmp_task_io_footprint {
  // Filled by the kernels of the run in progress.
  std::vector<kmp_task_io_entry_t> pending;
  double pending_seconds = 0.0;
  kmp_uint32 pending_dropped = 0;
  // The last completed run. A replay replaces it.
  std::vector<kmp_task_io_entry_t> latest;
  double latest_seconds = 0.0;
};
static_assert(sizeof(kmp_task_io_entry_t) == 56,
              "mirrors TaskIoEntryTy in the MPI plugin");

void __kmpc_target_task_record_io(const kmp_task_io_entry_t *entries,
                                  kmp_int32 num_entries, double seconds,
                                  kmp_uint32 dropped_ops) {
#if OMPX_TASKGRAPH
  // Under DMDAS no recorded task reaches __kmp_task_io_complete, which is what
  // drains the pending list.
  if (__kmp_use_dmdas_runtime())
    return;
  const kmp_info_t *thread = get_omp_thread();
  if (thread == nullptr)
    return;
  // A blocking target inside a recorded task can run in an unrecorded child.
  kmp_taskdata_t *task = thread->th.th_current_task;
  while (task != nullptr && task->td_flags.tasktype == TASK_EXPLICIT &&
         !(task->is_taskgraph && task->tdg != nullptr))
    task = task->td_parent;
  if (task == nullptr || task->td_flags.tasktype != TASK_EXPLICIT)
    return;

  kmp_tdg_info_t *tdg = task->tdg;
  __kmp_acquire_bootstrap_lock(&tdg->graph_lock);
  kmp_node_info_t &node = tdg->record_map[task->td_tdg_task_id];
  if (node.io_footprint == nullptr)
    node.io_footprint = new kmp_task_io_footprint();
  node.io_footprint->pending.insert(node.io_footprint->pending.end(), entries,
                                    entries + num_entries);
  node.io_footprint->pending_seconds += seconds;
  node.io_footprint->pending_dropped += dropped_ops;
  __kmp_release_bootstrap_lock(&tdg->graph_lock);
#endif
}

// What the application said tasks will touch, before they run. Kept apart from
// the record map because a task is numbered at allocation, before the map grows
// to hold it, and a task with an undeferred start never reaches the code that
// fills it. Leaked: tasks can finish during shutdown.
struct DeclaredTaskIo {
  std::mutex lock;
  // Declarations waiting for the next task the declaring task creates. The
  // creator matters because the thread can run another task, whose children
  // are not the task the application meant, before it creates one.
  struct Pending {
    kmp_int32 tdg_id = -1;
    const kmp_taskdata_t *creator = nullptr;
    std::vector<kmp_task_io_entry_t> entries;
  };
  std::map<kmp_int32, Pending> pending; // by gtid
  // (graph, node) -> sorted, merged declarations.
  std::map<std::pair<kmp_int32, kmp_int32>, std::vector<kmp_task_io_entry_t>>
      nodes;
};
static DeclaredTaskIo &declaredTaskIo() {
  static DeclaredTaskIo *const declared = new DeclaredTaskIo();
  return *declared;
}

static bool taskIoLogEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("OMPCLUSTER_TASK_IO_LOG");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

kmp_int32 __kmpc_next_task_declare_io(kmp_uint64 path_key, kmp_int64 offset,
                                      kmp_uint64 extent, kmp_int32 is_write) {
#if OMPX_TASKGRAPH
  if (__kmp_use_dmdas_runtime() || __kmp_global_tdgs == nullptr)
    return 0;
  const kmp_tdg_info_t *tdg = __kmp_global_tdgs[__kmp_curr_tdg_idx];
  const kmp_int32 gtid = __kmp_get_gtid();
  if (tdg == nullptr || !__kmp_tdg_is_recording(tdg->tdg_status) || gtid < 0)
    return 0;
  const kmp_taskdata_t *creator = __kmp_threads[gtid]->th.th_current_task;
  kmp_task_io_entry_t entry = {};
  entry.path_key = path_key;
  entry.offset = offset;
  entry.extent = extent;
  entry.bytes = extent;
  entry.owner_rank = -1;
  entry.is_write = is_write != 0;
  DeclaredTaskIo &declared = declaredTaskIo();
  std::lock_guard<std::mutex> guard(declared.lock);
  DeclaredTaskIo::Pending &pending = declared.pending[gtid];
  // Anything else here was declared for a task that was never created.
  if (pending.tdg_id != tdg->tdg_id || pending.creator != creator)
    pending.entries.clear();
  pending.tdg_id = tdg->tdg_id;
  pending.creator = creator;
  pending.entries.push_back(entry);
  return 1;
#else
  return 0;
#endif
}

// Sorts by file and direction and merges touching ranges. Offset -1, the whole
// file, sorts first and absorbs the rest of its file and direction.
static void mergeDeclaredIo(std::vector<kmp_task_io_entry_t> &entries) {
  std::sort(entries.begin(), entries.end(),
            [](const kmp_task_io_entry_t &a, const kmp_task_io_entry_t &b) {
              if (a.path_key != b.path_key)
                return a.path_key < b.path_key;
              if (a.is_write != b.is_write)
                return a.is_write < b.is_write;
              return a.offset < b.offset;
            });
  std::vector<kmp_task_io_entry_t> merged;
  for (const kmp_task_io_entry_t &entry : entries) {
    if (!merged.empty()) {
      kmp_task_io_entry_t &last = merged.back();
      if (last.path_key == entry.path_key && last.is_write == entry.is_write) {
        if (last.offset < 0)
          continue;
        const kmp_uint64 last_end = last.offset + last.extent;
        if (static_cast<kmp_uint64>(entry.offset) <= last_end) {
          const kmp_uint64 end =
              std::max(last_end, static_cast<kmp_uint64>(entry.offset) +
                                     entry.extent);
          last.extent = end - last.offset;
          last.bytes = last.extent;
          continue;
        }
      }
    }
    merged.push_back(entry);
  }
  entries.swap(merged);
}

void __kmp_task_io_bind_declared(kmp_int32 gtid, kmp_tdg_info_t *tdg,
                                 kmp_int32 node, const kmp_taskdata_t *parent) {
  DeclaredTaskIo &declared = declaredTaskIo();
  std::lock_guard<std::mutex> guard(declared.lock);
  auto it = declared.pending.find(gtid);
  if (it == declared.pending.end() || it->second.creator != parent)
    return;
  if (it->second.tdg_id == tdg->tdg_id) {
    std::vector<kmp_task_io_entry_t> &entries =
        declared.nodes[{tdg->tdg_id, node}];
    entries = std::move(it->second.entries);
    mergeDeclaredIo(entries);
    if (taskIoLogEnabled()) {
      for (const kmp_task_io_entry_t &e : entries)
        fprintf(stderr,
                "[task-io] declared tdg=%d node=%d key=%016llx rw=%c "
                "offset=%lld extent=%llu\n",
                tdg->tdg_id, node, static_cast<unsigned long long>(e.path_key),
                e.is_write ? 'w' : 'r', static_cast<long long>(e.offset),
                static_cast<unsigned long long>(e.extent));
      fflush(stderr);
    }
  }
  declared.pending.erase(it);
}

// Compares a run with its declarations: observed ranges no declaration covers,
// and declarations the run did not touch.
static void compareDeclaredIo(const std::vector<kmp_task_io_entry_t> &declared,
                              const std::vector<kmp_task_io_entry_t> &observed,
                              kmp_uint32 *undeclared, kmp_uint32 *unused) {
  const auto same_data = [](const kmp_task_io_entry_t &d,
                            const kmp_task_io_entry_t &o) {
    return d.path_key == o.path_key && d.is_write == o.is_write;
  };
  *undeclared = 0;
  for (const kmp_task_io_entry_t &o : observed) {
    const bool covered =
        std::any_of(declared.begin(), declared.end(),
                    [&](const kmp_task_io_entry_t &d) {
                      if (!same_data(d, o))
                        return false;
                      return d.offset < 0 ||
                             (o.offset >= d.offset &&
                              o.offset + static_cast<kmp_int64>(o.extent) <=
                                  d.offset + static_cast<kmp_int64>(d.extent));
                    });
    *undeclared += !covered;
  }
  *unused = 0;
  for (const kmp_task_io_entry_t &d : declared) {
    const bool touched =
        std::any_of(observed.begin(), observed.end(),
                    [&](const kmp_task_io_entry_t &o) {
                      if (!same_data(d, o))
                        return false;
                      return d.offset < 0 || o.offset < 0 ||
                             (o.offset < d.offset + static_cast<kmp_int64>(d.extent) &&
                              d.offset < o.offset + static_cast<kmp_int64>(o.extent));
                    });
    *unused += !touched;
  }
}

kmp_int32 __kmp_task_io_complete(kmp_tdg_info_t *tdg, kmp_int32 node,
                                 kmp_int32 device, kmp_int32 recorded_us) {
  kmp_task_io_footprint *footprint = tdg->record_map[node].io_footprint;
  kmp_int32 io_us = 0;
  kmp_uint32 dropped = 0;
  if (footprint != nullptr) {
    io_us = static_cast<kmp_int32>(footprint->pending_seconds * 1e6);
    dropped = footprint->pending_dropped;
    footprint->latest.swap(footprint->pending);
    footprint->latest_seconds = footprint->pending_seconds;
    footprint->pending.clear();
    footprint->pending_seconds = 0.0;
    footprint->pending_dropped = 0;
  }
  const kmp_int32 charged_us = std::max(recorded_us - io_us, 0);
  if (taskIoLogEnabled()) {
    const size_t entries = footprint != nullptr ? footprint->latest.size() : 0;
    std::string lines;
    char line[256];
    snprintf(line, sizeof(line),
             "[task-io] tdg=%d node=%d device=%d recorded_us=%d io_us=%d "
             "charged_us=%d dropped=%u entries=%zu\n",
             tdg->tdg_id, node, device, recorded_us, io_us, charged_us, dropped,
             entries);
    lines += line;
    for (size_t i = 0; i < entries; ++i) {
      const kmp_task_io_entry_t &e = footprint->latest[i];
      snprintf(line, sizeof(line),
               "[task-io]   node=%d key=%016llx rw=%c owner=%d offset=%lld "
               "extent=%llu bytes=%llu ops=%u seconds=%.6f widened=%u\n",
               node, static_cast<unsigned long long>(e.path_key),
               e.is_write ? 'w' : 'r', e.owner_rank,
               static_cast<long long>(e.offset),
               static_cast<unsigned long long>(e.extent),
               static_cast<unsigned long long>(e.bytes), e.ops, e.seconds,
               e.widened);
      lines += line;
    }
    {
      DeclaredTaskIo &declared = declaredTaskIo();
      std::lock_guard<std::mutex> guard(declared.lock);
      const auto it = declared.nodes.find({tdg->tdg_id, node});
      if (it != declared.nodes.end()) {
        kmp_uint32 undeclared = 0, unused = 0;
        compareDeclaredIo(it->second,
                          footprint != nullptr
                              ? footprint->latest
                              : std::vector<kmp_task_io_entry_t>(),
                          &undeclared, &unused);
        snprintf(line, sizeof(line),
                 "[task-io]   node=%d declared=%zu undeclared=%u unused=%u\n",
                 node, it->second.size(), undeclared, unused);
        lines += line;
      }
    }
    fputs(lines.c_str(), stderr);
    fflush(stderr);
  }
  return charged_us;
}

// Mirrors ompfile_graph_write (libompfile's ompfile_graph_plan.h).
struct kmp_task_io_plan_write_t {
  kmp_uint64 path_key;
  kmp_int64 offset;
  kmp_uint64 extent;
  kmp_int32 device;
  kmp_uint32 reserved;
};
static_assert(sizeof(kmp_task_io_plan_write_t) == 32,
              "mirrors ompfile_graph_write in libompfile");

void __kmp_task_io_plan_graph(kmp_tdg_info_t *tdg) {
#if OMPX_TASKGRAPH && KMP_OS_UNIX
  using PlanFn = int (*)(const kmp_task_io_plan_write_t *, int);
  static const PlanFn plan = reinterpret_cast<PlanFn>(
      dlsym(RTLD_DEFAULT, "ompfile_graph_plan_writes"));
  if (plan == nullptr || tdg == nullptr)
    return;
  const ScheduleMap *schedule = get_schedule_map(tdg->tdg_id);
  if (schedule == nullptr)
    return;
  std::vector<kmp_task_io_plan_write_t> writes;
  size_t observed_nodes = 0, declared_nodes = 0;
  __kmp_acquire_bootstrap_lock(&tdg->graph_lock);
  {
    DeclaredTaskIo &declared = declaredTaskIo();
    std::lock_guard<std::mutex> guard(declared.lock);
    const kmp_int32 num_tasks = KMP_ATOMIC_LD_RLX(&tdg->num_tasks);
    for (kmp_int32 i = 0; i < num_tasks; ++i) {
      const kmp_node_info_t &node = tdg->record_map[i];
      if (node.task == nullptr)
        continue;
      // The device the scheduler chose, or -1 for a task on the head node (a
      // classical task, whose blocking targets pick their own device). A
      // placed write lands on the rank that issues it either way.
      kmp_int32 device = -1;
      if (node.is_target_task) {
        const auto placed =
            schedule->find(KMP_TASK_TO_TASKDATA(node.task)->td_task_id);
        if (placed != schedule->end() && placed->second.rank >= 1)
          device = placed->second.rank - 1;
      }
      // A replay plans from what the task did last time, the first run from
      // what it declared.
      const std::vector<kmp_task_io_entry_t> *ranges = nullptr;
      if (node.io_footprint != nullptr && !node.io_footprint->latest.empty()) {
        ranges = &node.io_footprint->latest;
        ++observed_nodes;
      } else {
        const auto it = declared.nodes.find({tdg->tdg_id, i});
        if (it == declared.nodes.end())
          continue;
        ranges = &it->second;
        ++declared_nodes;
      }
      for (const kmp_task_io_entry_t &e : *ranges)
        if (e.is_write && e.offset >= 0 && e.extent > 0)
          writes.push_back({e.path_key, e.offset, e.extent, device, 0});
    }
  }
  __kmp_release_bootstrap_lock(&tdg->graph_lock);
  errno = 0;
  const int sent =
      writes.empty() ? 0 : plan(writes.data(), static_cast<int>(writes.size()));
  const int plan_errno = sent < 0 ? errno : 0;
  if (taskIoLogEnabled()) {
    fprintf(stderr,
            "[task-io] plan tdg=%d writes=%zu observed_nodes=%zu "
            "declared_nodes=%zu sent=%d errno=%d\n",
            tdg->tdg_id, writes.size(), observed_nodes, declared_nodes, sent,
            plan_errno);
    fflush(stderr);
  }
#endif
}

bool __kmp_task_io_summary(const kmp_node_info_t *node, kmp_uint64 *bytes,
                           kmp_int32 *ranges, double *seconds) {
  const kmp_task_io_footprint *footprint = node->io_footprint;
  if (footprint == nullptr || footprint->latest_seconds <= 0.0)
    return false;
  *bytes = 0;
  for (const kmp_task_io_entry_t &entry : footprint->latest)
    *bytes += entry.bytes;
  *ranges = static_cast<kmp_int32>(footprint->latest.size());
  *seconds = footprint->latest_seconds;
  return true;
}

kmp_int32 __kmpc_target_task_current_task_id() {
  const kmp_info_t *thread = get_omp_thread();

  if (thread == nullptr) {
    return -1;
  }

  const kmp_taskdata_t *task = thread->th.th_current_task;

  // Only explicit tasks can be created by the user and should be accounted.
  // Implicit tasks are automatically generated by the OpenMP Runtime and should
  // not be considered in the target-task integration.
  if (task == nullptr || task->td_flags.tasktype != TASK_EXPLICIT) {
    return -1;
  }

  return thread->th.th_current_task->td_task_id;
}

bool __kmpc_is_graph_builder_tasking() {
  if (get_omp_thread() == nullptr || gcontext->graph_builder_gtid < 0) {
    return false;
  }

  const kmp_info_t *builder_thread =
      __kmp_thread_from_gtid(gcontext->graph_builder_gtid);

  if (builder_thread == nullptr) {
    return false;
  }

  return builder_thread->th.th_tasking_level;
}

bool __kmpc_has_active_tasks() {
  std::unique_lock<std::mutex> ctx_lk(gcontext->ctx_mutex);

  if (gcontext->active_tasks.size() > 0)
    return true;

  return false;
}

// Add the dependencies to a task.
void __kmpc_set_deps(kmp_node_info *task, kmp_int32 ndeps,
                     const kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
                     const kmp_depend_info_t *dep_list_noalias) {

  KG_TRACE(10, ("__kmpc_set_deps (start)\n"));
  // Get general data
  kmp_int32 omp_gtid = __kmp_get_gtid();
  kmp_info_t *thread = __kmp_thread_from_gtid(omp_gtid);
  // Process dependencies
  task->deps = thread_malloc<kmp_target_dep_t>(thread, ndeps + ndeps_noalias);
  task->num_deps = 0;

  // Filter aliased addresses and add dependencies to task
  for (int i = 0; i < ndeps; ++i) {
    const kmp_depend_info_t &dep = dep_list[i];
    bool is_alias = false;

    for (int j = 0; j < task->num_deps; ++j) {
      kmp_target_dep_t &prev_dep = task->deps[j];
      if (prev_dep.addr == dep.base_addr) {
        prev_dep.dep_type.in |= dep.flags.in;
        prev_dep.dep_type.out |= dep.flags.out;
        prev_dep.dep_type.mtx |= dep.flags.mtx;
        is_alias = true;
        break;
      }
    }

    if (!is_alias) {
      task->deps[task->num_deps++] = kmp_target_dep_t{
          .addr = dep.base_addr,
          .dep_type = {.in = static_cast<bool>(dep.flags.in),
                       .out = static_cast<bool>(dep.flags.out),
                       .mtx = static_cast<bool>(dep.flags.mtx)}};
    }
  }

  // Add no-alias dependencies
  for (int i = 0; i < ndeps_noalias; ++i) {
    const kmp_depend_info_t &dep = dep_list_noalias[i];
    task->deps[task->num_deps + i] =
        kmp_target_dep_t{.addr = dep.base_addr,
                         .dep_type = {.in = static_cast<bool>(dep.flags.in),
                                      .out = static_cast<bool>(dep.flags.out),
                                      .mtx = static_cast<bool>(dep.flags.mtx)}};
  }
  task->num_deps += ndeps_noalias;

  KG_TRACE(10, ("__kmpc_set_deps (end)\n"));
}
