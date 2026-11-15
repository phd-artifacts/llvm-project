/*
 * kmp_taskdeps.cpp
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//#define KMP_SUPPORT_GRAPH_OUTPUT 1

#include "kmp.h"
#include "kmp_debug.h"
#include "kmp_io.h"
#include "kmp_os.h"
#include "kmp_wait_release.h"
#include <cstdio>
#include "kmp_taskdeps.h"
#include "kmp_dmdas.h"
#if OMPT_SUPPORT
#include "ompt-specific.h"
#endif

// TODO: Improve memory allocation? keep a list of pre-allocated structures?
// allocate in blocks? re-use list finished list entries?
// TODO: don't use atomic ref counters for stack-allocated nodes.
// TODO: find an alternate to atomic refs for heap-allocated nodes?
// TODO: Finish graph output support
// TODO: kmp_lock_t seems a tad to big (and heavy weight) for this. Check other
// runtime locks
// TODO: Any ITT support needed?

#ifdef KMP_SUPPORT_GRAPH_OUTPUT
static std::atomic<kmp_int32> kmp_node_id_seed = 0;
#endif

static void __kmp_init_node(kmp_depnode_t *node, bool on_stack) {
  node->dn.successors = NULL;
  node->dn.task = NULL; // will point to the right task
  // once dependences have been processed
  for (int i = 0; i < MAX_MTX_DEPS; ++i)
    node->dn.mtx_locks[i] = NULL;
  node->dn.mtx_num_locks = 0;
  __kmp_init_lock(&node->dn.lock);
  // Init creates the first reference.  Bit 0 indicates that this node
  // resides on the stack.  The refcount is incremented and decremented in
  // steps of two, maintaining use of even numbers for heap nodes and odd
  // numbers for stack nodes.
  KMP_ATOMIC_ST_RLX(&node->dn.nrefs, on_stack ? 3 : 2);
#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  node->dn.id = KMP_ATOMIC_INC(&kmp_node_id_seed);
#endif
#if USE_ITT_BUILD && USE_ITT_NOTIFY
  __itt_sync_create(node, "OMP task dep node", NULL, 0);
#endif
}

static inline kmp_depnode_t *__kmp_node_ref(kmp_depnode_t *node) {
  KMP_ATOMIC_ADD(&node->dn.nrefs, 2);
  return node;
}

enum { KMP_DEPHASH_OTHER_SIZE = 97, KMP_DEPHASH_MASTER_SIZE = 997 };

size_t sizes[] = {997, 2003, 4001, 8191, 16001, 32003, 64007, 131071, 270029};
const size_t MAX_GEN = 8;

static inline size_t __kmp_dephash_hash(kmp_intptr_t addr, size_t hsize) {
  // TODO alternate to try: set = (((Addr64)(addrUsefulBits * 9.618)) %
  // m_num_sets );
  return ((addr >> 6) ^ (addr >> 2)) % hsize;
}

static kmp_dephash_t *__kmp_dephash_extend(kmp_info_t *thread,
                                           kmp_dephash_t *current_dephash) {
  kmp_dephash_t *h;

  size_t gen = current_dephash->generation + 1;
  if (gen >= MAX_GEN)
    return current_dephash;
  size_t new_size = sizes[gen];

  size_t size_to_allocate =
      new_size * sizeof(kmp_dephash_entry_t *) + sizeof(kmp_dephash_t);

#if USE_FAST_MEMORY
  h = (kmp_dephash_t *)__kmp_fast_allocate(thread, size_to_allocate);
#else
  h = (kmp_dephash_t *)__kmp_thread_malloc(thread, size_to_allocate);
#endif

  h->size = new_size;
  h->nelements = current_dephash->nelements;
  h->buckets = (kmp_dephash_entry **)(h + 1);
  h->generation = gen;
  h->nconflicts = 0;
  h->last_all = current_dephash->last_all;

  // make sure buckets are properly initialized
  for (size_t i = 0; i < new_size; i++) {
    h->buckets[i] = NULL;
  }

  // insert existing elements in the new table
  for (size_t i = 0; i < current_dephash->size; i++) {
    kmp_dephash_entry_t *next, *entry;
    for (entry = current_dephash->buckets[i]; entry; entry = next) {
      next = entry->next_in_bucket;
      // Compute the new hash using the new size, and insert the entry in
      // the new bucket.
      size_t new_bucket = __kmp_dephash_hash(entry->addr, h->size);
      entry->next_in_bucket = h->buckets[new_bucket];
      if (entry->next_in_bucket) {
        h->nconflicts++;
      }
      h->buckets[new_bucket] = entry;
    }
  }

  // Free old hash table
#if USE_FAST_MEMORY
  __kmp_fast_free(thread, current_dephash);
#else
  __kmp_thread_free(thread, current_dephash);
#endif

  return h;
}

static kmp_dephash_t *__kmp_dephash_create(kmp_info_t *thread,
                                           kmp_taskdata_t *current_task) {
  kmp_dephash_t *h;

  size_t h_size;

  if (current_task->td_flags.tasktype == TASK_IMPLICIT)
    h_size = KMP_DEPHASH_MASTER_SIZE;
  else
    h_size = KMP_DEPHASH_OTHER_SIZE;

  size_t size = h_size * sizeof(kmp_dephash_entry_t *) + sizeof(kmp_dephash_t);

#if USE_FAST_MEMORY
  h = (kmp_dephash_t *)__kmp_fast_allocate(thread, size);
#else
  h = (kmp_dephash_t *)__kmp_thread_malloc(thread, size);
#endif
  h->size = h_size;

  h->generation = 0;
  h->nelements = 0;
  h->nconflicts = 0;
  h->buckets = (kmp_dephash_entry **)(h + 1);
  h->last_all = NULL;

  for (size_t i = 0; i < h_size; i++)
    h->buckets[i] = 0;

  return h;
}

static kmp_dephash_entry *__kmp_dephash_find(kmp_info_t *thread,
                                             kmp_dephash_t **hash,
                                             kmp_intptr_t addr) {
  kmp_dephash_t *h = *hash;
  if (h->nelements != 0 && h->nconflicts / h->size >= 1) {
    *hash = __kmp_dephash_extend(thread, h);
    h = *hash;
  }
  size_t bucket = __kmp_dephash_hash(addr, h->size);

  kmp_dephash_entry_t *entry;
  for (entry = h->buckets[bucket]; entry; entry = entry->next_in_bucket)
    if (entry->addr == addr)
      break;

  if (entry == NULL) {
// create entry. This is only done by one thread so no locking required
#if USE_FAST_MEMORY
    entry = (kmp_dephash_entry_t *)__kmp_fast_allocate(
        thread, sizeof(kmp_dephash_entry_t));
#else
    entry = (kmp_dephash_entry_t *)__kmp_thread_malloc(
        thread, sizeof(kmp_dephash_entry_t));
#endif
    entry->addr = addr;
    if (!h->last_all) // no predecessor task with omp_all_memory dependence
      entry->last_out = NULL;
    else // else link the omp_all_memory depnode to the new entry
      entry->last_out = __kmp_node_ref(h->last_all);
    entry->last_set = NULL;
    entry->prev_set = NULL;
    entry->last_flag = 0;
    entry->mtx_lock = NULL;
    entry->next_in_bucket = h->buckets[bucket];
    h->buckets[bucket] = entry;
    h->nelements++;
    if (entry->next_in_bucket)
      h->nconflicts++;
  }
  return entry;
}

static kmp_depnode_list_t *__kmp_add_node(kmp_info_t *thread,
                                          kmp_depnode_list_t *list,
                                          kmp_depnode_t *node) {
  kmp_depnode_list_t *new_head;

#if USE_FAST_MEMORY
  new_head = (kmp_depnode_list_t *)__kmp_fast_allocate(
      thread, sizeof(kmp_depnode_list_t));
#else
  new_head = (kmp_depnode_list_t *)__kmp_thread_malloc(
      thread, sizeof(kmp_depnode_list_t));
#endif

  new_head->node = __kmp_node_ref(node);
  new_head->next = list;

  return new_head;
}

static bool __kmp_depnode_replace_successor(kmp_int32 gtid, kmp_info_t *thread,
                                            kmp_depnode_t *source,
                                            kmp_depnode_t *old_target,
                                            kmp_depnode_t *new_target) {
  if (!source || !old_target || !new_target)
    return false;

  KMP_ACQUIRE_DEPNODE(gtid, source);

  for (kmp_depnode_list_t *p = source->dn.successors; p != nullptr;
       p = p->next) {

    if (p->node == old_target) {
      // The list previously owned one reference to old_target.
      // Transfer this list entry to new_target.
      p->node = __kmp_node_ref(new_target);
      __kmp_node_deref(thread, old_target);

      KMP_RELEASE_DEPNODE(gtid, source);
      return true;
    }
  }

  KMP_RELEASE_DEPNODE(gtid, source);
  return false;
}

static bool __kmp_depnode_restore_successor(kmp_int32 gtid, kmp_info_t *thread,
                                            kmp_depnode_t *source,
                                            kmp_depnode_t *memory,
                                            kmp_depnode_t *target) {
  if (!source || !memory || !target)
    return false;

  KMP_ACQUIRE_DEPNODE(gtid, source);

  for (kmp_depnode_list_t *p = source->dn.successors; p != nullptr;
       p = p->next) {
    if (p->node == memory) {
      // Transfer the source successor-list reference from the temporary
      // memory node back to the original target node.
      p->node = __kmp_node_ref(target);
      __kmp_node_deref(thread, memory);

      KMP_RELEASE_DEPNODE(gtid, source);
      return true;
    }
  }

  KMP_RELEASE_DEPNODE(gtid, source);
  return false;
}

static inline void __kmp_track_dependence(kmp_int32 gtid, kmp_depnode_t *source,
                                          kmp_depnode_t *sink,
                                          kmp_task_t *sink_task) {
  KG_TRACE(30, ("__kmp_track_dependence (begin)\n"));
#if OMPX_TASKGRAPH
  kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
  kmp_taskdata_t *task_sink = KMP_TASK_TO_TASKDATA(sink_task);
  if (source->dn.task && sink_task) {
    // Not supporting dependency between two tasks that one is within the TDG
    // and the other is not
    KMP_ASSERT(task_source->is_taskgraph == task_sink->is_taskgraph);
  }
  if (task_sink->is_taskgraph &&
      __kmp_tdg_is_recording(task_sink->tdg->tdg_status)) {
    kmp_node_info_t *source_info =
        &task_sink->tdg->record_map[task_source->td_tdg_task_id];
    bool exists = false;
    for (int i = 0; i < source_info->nsuccessors; i++) {
      if (source_info->successors[i] == task_sink->td_tdg_task_id) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      if (source_info->nsuccessors >= source_info->successors_size) {
        kmp_uint old_size = source_info->successors_size;
        source_info->successors_size = 2 * source_info->successors_size;
        kmp_int32 *old_succ_ids = source_info->successors;
        kmp_int32 *new_succ_ids = (kmp_int32 *)__kmp_allocate(
            source_info->successors_size * sizeof(kmp_int32));
        KMP_MEMCPY(new_succ_ids, old_succ_ids, old_size * sizeof(kmp_int32));
        source_info->successors = new_succ_ids;
        __kmp_free(old_succ_ids);
      }

      source_info->successors[source_info->nsuccessors] =
          task_sink->td_tdg_task_id;
      source_info->nsuccessors++;

      kmp_node_info_t *sink_info =
          &(task_sink->tdg->record_map[task_sink->td_tdg_task_id]);

      if (sink_info->npredecessors + 1 >= sink_info->predecessors_size) {
        kmp_int32 old_predecessors_size = sink_info->predecessors_size;
        sink_info->predecessors_size = 2 * sink_info->predecessors_size;
        kmp_int32 *old_pred_ids = sink_info->predecessors;
        kmp_int32 *new_pred_ids = (kmp_int32 *)__kmp_allocate(
            sink_info->predecessors_size * sizeof(kmp_int32));
        KMP_MEMCPY(new_pred_ids, old_pred_ids,
                   old_predecessors_size * sizeof(kmp_int32));
        sink_info->predecessors = new_pred_ids;
        __kmp_free(old_pred_ids);
      }
      sink_info->predecessors[sink_info->npredecessors] =
          task_source->td_tdg_task_id;
      sink_info->npredecessors++;
    }
  }
#endif
#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
  // do not use sink->dn.task as that is only filled after the dependences
  // are already processed!
  kmp_taskdata_t *task_sink = KMP_TASK_TO_TASKDATA(sink_task);

  __kmp_printf("%d(%s) -> %d(%s)\n", source->dn.id,
               task_source->td_ident->psource, sink->dn.id,
               task_sink->td_ident->psource);
#endif
#if OMPT_SUPPORT && OMPT_OPTIONAL
  /* OMPT tracks dependences between task (a=source, b=sink) in which
     task a blocks the execution of b through the ompt_new_dependence_callback
     */
  if (ompt_enabled.ompt_callback_task_dependence) {
    kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
    ompt_data_t *sink_data;
    if (sink_task)
      sink_data = &(KMP_TASK_TO_TASKDATA(sink_task)->ompt_task_info.task_data);
    else
      sink_data = &__kmp_threads[gtid]->th.ompt_thread_info.task_data;

    ompt_callbacks.ompt_callback(ompt_callback_task_dependence)(
        &(task_source->ompt_task_info.task_data), sink_data);
  }
#endif /* OMPT_SUPPORT && OMPT_OPTIONAL */
  KG_TRACE(30, ("__kmp_track_dependence (end)\n"));
}

kmp_base_depnode_t *__kmpc_task_get_depnode(kmp_task_t *task) {
  kmp_taskdata_t *td = KMP_TASK_TO_TASKDATA(task);
  return td->td_depnode ? &(td->td_depnode->dn) : NULL;
}

kmp_depnode_list_t *__kmpc_task_get_successors(kmp_task_t *task) {
  kmp_taskdata_t *td = KMP_TASK_TO_TASKDATA(task);
  return td->td_depnode->dn.successors;
}

static inline kmp_int32
__kmp_depnode_link_successor(kmp_int32 gtid, kmp_info_t *thread,
                             kmp_task_t *task, kmp_depnode_t *node,
                             kmp_depnode_list_t *plist) {
  KG_TRACE(30, ("__kmp_depnode_link_successor_list (begin) \n"));
  if (!plist)
    return 0;
  kmp_int32 npredecessors = 0;
  // link node as successor of list elements
  for (kmp_depnode_list_t *p = plist; p; p = p->next) {
    kmp_depnode_t *dep = p->node;
#if OMPX_TASKGRAPH
    kmp_tdg_status tdg_status = KMP_TDG_NONE;
    if (task) {
      kmp_taskdata_t *td = KMP_TASK_TO_TASKDATA(task);
      if (td->is_taskgraph)
        tdg_status = KMP_TASK_TO_TASKDATA(task)->tdg->tdg_status;
      if (__kmp_tdg_is_recording(tdg_status))
        __kmp_track_dependence(gtid, dep, node, task);
    }
#endif
    if (dep->dn.task) {
      KMP_ACQUIRE_DEPNODE(gtid, dep);
      if (dep->dn.task) {
        if (!dep->dn.successors || dep->dn.successors->node != node) {
#if OMPX_TASKGRAPH
          if (!(__kmp_tdg_is_recording(tdg_status)) && task)
#endif
            __kmp_track_dependence(gtid, dep, node, task);
          dep->dn.successors = __kmp_add_node(thread, dep->dn.successors, node);
          KA_TRACE(40, ("__kmp_process_deps: T#%d adding dependence from %p to "
                        "%p\n",
                        gtid, KMP_TASK_TO_TASKDATA(dep->dn.task),
                        KMP_TASK_TO_TASKDATA(task)));
          npredecessors++;
        }
      }
      KMP_RELEASE_DEPNODE(gtid, dep);
    }
  }
  KG_TRACE(30, ("__kmp_depnode_link_successor_list (end) \n"));
  return npredecessors;
}

// Add the edge 'sink' -> 'source' in the task dependency graph
static inline kmp_int32 __kmp_depnode_link_successor(kmp_int32 gtid,
                                                     kmp_info_t *thread,
                                                     kmp_task_t *task,
                                                     kmp_depnode_t *source,
                                                     kmp_depnode_t *sink) {
  KG_TRACE(30, ("__kmp_depnode_link_successor (begin) \n"));
  if (!sink)
    return 0;
  kmp_int32 npredecessors = 0;
#if OMPX_TASKGRAPH
  kmp_tdg_status tdg_status = KMP_TDG_NONE;
  kmp_taskdata_t *td = KMP_TASK_TO_TASKDATA(task);
  if (task) {
    if (td->is_taskgraph)
      tdg_status = KMP_TASK_TO_TASKDATA(task)->tdg->tdg_status;
    if (__kmp_tdg_is_recording(tdg_status) && sink->dn.task)
      __kmp_track_dependence(gtid, sink, source, task);
  }
#endif
  if (sink->dn.task) {
    // synchronously add source to sink' list of successors
    KMP_ACQUIRE_DEPNODE(gtid, sink);
    if (sink->dn.task) {
      if (!sink->dn.successors || sink->dn.successors->node != source) {
#if OMPX_TASKGRAPH
        if (!(__kmp_tdg_is_recording(tdg_status)) && task)
#endif
          __kmp_track_dependence(gtid, sink, source, task);
        sink->dn.successors = __kmp_add_node(thread, sink->dn.successors, source);
        KA_TRACE(40, ("__kmp_process_deps: T#%d adding dependence from %p to "
                    "%p\n",
                    gtid, KMP_TASK_TO_TASKDATA(sink->dn.task),
                    KMP_TASK_TO_TASKDATA(task)));
#if OMPX_TASKGRAPH
        if (__kmp_tdg_is_recording(tdg_status)) {
          kmp_taskdata_t *tdd = KMP_TASK_TO_TASKDATA(sink->dn.task);
          if (tdd->is_taskgraph) {
            if (tdd->td_flags.onced)
              // decrement npredecessors if sink->dn.task belongs to a taskgraph
              // and
              //  1) the task is reset to its initial state (by kmp_free_task) or
              //  2) the task is complete but not yet reset
              npredecessors--;
          }
        }
#endif
      npredecessors++;
      }
    }
    KMP_RELEASE_DEPNODE(gtid, sink);
  }
  KG_TRACE(30, ("__kmp_depnode_link_successor (end) \n"));
  return npredecessors;
}

static inline kmp_int32
__kmp_process_dep_all(kmp_int32 gtid, kmp_depnode_t *node, kmp_dephash_t *h,
                      bool dep_barrier, kmp_task_t *task) {
  KA_TRACE(30, ("__kmp_process_dep_all: T#%d processing dep_all, "
                "dep_barrier = %d\n",
                gtid, dep_barrier));
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_int32 npredecessors = 0;

  // process previous omp_all_memory node if any
  npredecessors +=
      __kmp_depnode_link_successor(gtid, thread, task, node, h->last_all);
  __kmp_node_deref(thread, h->last_all);
  if (!dep_barrier) {
    h->last_all = __kmp_node_ref(node);
  } else {
    // if this is a sync point in the serial sequence, then the previous
    // outputs are guaranteed to be completed after the execution of this
    // task so the previous output nodes can be cleared.
    h->last_all = NULL;
  }

  // process all regular dependences
  for (size_t i = 0; i < h->size; i++) {
    kmp_dephash_entry_t *info = h->buckets[i];
    if (!info) // skip empty slots in dephash
      continue;
    for (; info; info = info->next_in_bucket) {
      // for each entry the omp_all_memory works as OUT dependence
      kmp_depnode_t *last_out = info->last_out;
      kmp_depnode_list_t *last_set = info->last_set;
      kmp_depnode_list_t *prev_set = info->prev_set;
      if (last_set) {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_set);
        __kmp_depnode_list_free(thread, last_set);
        __kmp_depnode_list_free(thread, prev_set);
        info->last_set = NULL;
        info->prev_set = NULL;
        info->last_flag = 0; // no sets in this dephash entry
      } else {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_out);
      }
      __kmp_node_deref(thread, last_out);
      if (!dep_barrier) {
        info->last_out = __kmp_node_ref(node);
      } else {
        info->last_out = NULL;
      }
    }
  }
  KA_TRACE(30, ("__kmp_process_dep_all: T#%d found %d predecessors\n", gtid,
                npredecessors));
  return npredecessors;
}

template <bool filter>
static inline kmp_int32
__kmp_process_deps(kmp_int32 gtid, kmp_depnode_t *node, kmp_dephash_t **hash,
                   bool dep_barrier, kmp_int32 ndeps,
                   kmp_depend_info_t *dep_list, kmp_task_t *task) {
  KA_TRACE(30, ("__kmp_process_deps<%d>: T#%d processing %d dependences : "
                "dep_barrier = %d\n",
                filter, gtid, ndeps, dep_barrier));

  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_int32 npredecessors = 0;
  for (kmp_int32 i = 0; i < ndeps; i++) {
    const kmp_depend_info_t *dep = &dep_list[i];

    KG_TRACE(30, ("__kmp_process_deps<%d>: Processing dep %p\n", dep));
    if (filter && dep->base_addr == 0)
      continue; // skip filtered entries

    kmp_dephash_entry_t *info =
        __kmp_dephash_find(thread, hash, dep->base_addr);
    kmp_depnode_t *last_out = info->last_out;
    kmp_depnode_list_t *last_set = info->last_set;
    kmp_depnode_list_t *prev_set = info->prev_set;

    if (dep->flags.out) { // out or inout --> clean lists if any
      if (last_set) {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_set);
        __kmp_depnode_list_free(thread, last_set);
        __kmp_depnode_list_free(thread, prev_set);
        info->last_set = NULL;
        info->prev_set = NULL;
        info->last_flag = 0; // no sets in this dephash entry
      } else {
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_out);
      }
      __kmp_node_deref(thread, last_out);
      if (!dep_barrier) {
        info->last_out = __kmp_node_ref(node);
      } else {
        // if this is a sync point in the serial sequence, then the previous
        // outputs are guaranteed to be completed after the execution of this
        // task so the previous output nodes can be cleared.
        info->last_out = NULL;
      }
    } else { // either IN or MTX or SET
      if (info->last_flag == 0 || info->last_flag == dep->flag) {
        // last_set either didn't exist or of same dep kind
        // link node as successor of the last_out if any
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_out);
        // link node as successor of all nodes in the prev_set if any
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, prev_set);
        if (dep_barrier) {
          // clean last_out and prev_set if any; don't touch last_set
          __kmp_node_deref(thread, last_out);
          info->last_out = NULL;
          __kmp_depnode_list_free(thread, prev_set);
          info->prev_set = NULL;
        }
      } else { // last_set is of different dep kind, make it prev_set
        // link node as successor of all nodes in the last_set
        npredecessors +=
            __kmp_depnode_link_successor(gtid, thread, task, node, last_set);
        // clean last_out if any
        __kmp_node_deref(thread, last_out);
        info->last_out = NULL;
        // clean prev_set if any
        __kmp_depnode_list_free(thread, prev_set);
        if (!dep_barrier) {
          // move last_set to prev_set, new last_set will be allocated
          info->prev_set = last_set;
        } else {
          info->prev_set = NULL;
          info->last_flag = 0;
        }
        info->last_set = NULL;
      }
      // for dep_barrier last_flag value should remain:
      // 0 if last_set is empty, unchanged otherwise
      if (!dep_barrier) {
        info->last_flag = dep->flag; // store dep kind of the last_set
        info->last_set = __kmp_add_node(thread, info->last_set, node);
      }
      // check if we are processing MTX dependency
      if (dep->flag == KMP_DEP_MTX) {
        if (info->mtx_lock == NULL) {
          info->mtx_lock = (kmp_lock_t *)__kmp_allocate(sizeof(kmp_lock_t));
          __kmp_init_lock(info->mtx_lock);
        }
        KMP_DEBUG_ASSERT(node->dn.mtx_num_locks < MAX_MTX_DEPS);
        kmp_int32 m;
        // Save lock in node's array
        for (m = 0; m < MAX_MTX_DEPS; ++m) {
          // sort pointers in decreasing order to avoid potential livelock
          if (node->dn.mtx_locks[m] < info->mtx_lock) {
            KMP_DEBUG_ASSERT(!node->dn.mtx_locks[node->dn.mtx_num_locks]);
            for (int n = node->dn.mtx_num_locks; n > m; --n) {
              // shift right all lesser non-NULL pointers
              KMP_DEBUG_ASSERT(node->dn.mtx_locks[n - 1] != NULL);
              node->dn.mtx_locks[n] = node->dn.mtx_locks[n - 1];
            }
            node->dn.mtx_locks[m] = info->mtx_lock;
            break;
          }
        }
        KMP_DEBUG_ASSERT(m < MAX_MTX_DEPS); // must break from loop
        node->dn.mtx_num_locks++;
      }
    }
  }
  KA_TRACE(30, ("__kmp_process_deps<%d>: T#%d found %d predecessors\n", filter,
                gtid, npredecessors));
  return npredecessors;
}

#define NO_DEP_BARRIER (false)
#define DEP_BARRIER (true)

// returns true if the task has any outstanding dependence
static bool __kmp_check_deps(kmp_int32 gtid, kmp_depnode_t *node,
                             kmp_task_t *task, kmp_dephash_t **hash,
                             bool dep_barrier, kmp_int32 ndeps,
                             kmp_depend_info_t *dep_list,
                             kmp_int32 ndeps_noalias,
                             kmp_depend_info_t *noalias_dep_list) {
  int i, n_mtxs = 0, dep_all = 0;
#if KMP_DEBUG
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
#endif
  KA_TRACE(20, ("__kmp_check_deps: T#%d checking dependences for task %p : %d "
                "possibly aliased dependences, %d non-aliased dependences : "
                "dep_barrier=%d .\n",
                gtid, taskdata, ndeps, ndeps_noalias, dep_barrier));

  // Filter deps in dep_list
  // TODO: Different algorithm for large dep_list ( > 10 ? )
  for (i = 0; i < ndeps; i++) {
    if (dep_list[i].base_addr != 0 &&
        dep_list[i].base_addr != (kmp_intptr_t)KMP_SIZE_T_MAX) {
      KMP_DEBUG_ASSERT(
          dep_list[i].flag == KMP_DEP_IN || dep_list[i].flag == KMP_DEP_OUT ||
          dep_list[i].flag == KMP_DEP_INOUT ||
          dep_list[i].flag == KMP_DEP_MTX || dep_list[i].flag == KMP_DEP_SET);
      for (int j = i + 1; j < ndeps; j++) {
        if (dep_list[i].base_addr == dep_list[j].base_addr) {
          if (dep_list[i].flag != dep_list[j].flag) {
            // two different dependences on same address work identical to OUT
            dep_list[i].flag = KMP_DEP_OUT;
          }
          dep_list[j].base_addr = 0; // Mark j element as void
        }
      }
      if (dep_list[i].flag == KMP_DEP_MTX) {
        // limit number of mtx deps to MAX_MTX_DEPS per node
        if (n_mtxs < MAX_MTX_DEPS && task != NULL) {
          ++n_mtxs;
        } else {
          dep_list[i].flag = KMP_DEP_OUT; // downgrade mutexinoutset to inout
        }
      }
    } else if (dep_list[i].flag == KMP_DEP_ALL ||
               dep_list[i].base_addr == (kmp_intptr_t)KMP_SIZE_T_MAX) {
      // omp_all_memory dependence can be marked by compiler by either
      // (addr=0 && flag=0x80) (flag KMP_DEP_ALL), or (addr=-1).
      // omp_all_memory overrides all other dependences if any
      dep_all = 1;
      break;
    }
  }

  // doesn't need to be atomic as no other thread is going to be accessing this
  // node just yet.
  // npredecessors is set -1 to ensure that none of the releasing tasks queues
  // this task before we have finished processing all the dependences
  node->dn.npredecessors = -1;

  // used to pack all npredecessors additions into a single atomic operation at
  // the end
  int npredecessors;

  if (!dep_all) { // regular dependences
    npredecessors = __kmp_process_deps<true>(gtid, node, hash, dep_barrier,
                                             ndeps, dep_list, task);
    npredecessors += __kmp_process_deps<false>(
        gtid, node, hash, dep_barrier, ndeps_noalias, noalias_dep_list, task);
  } else { // omp_all_memory dependence
    npredecessors = __kmp_process_dep_all(gtid, node, *hash, dep_barrier, task);
  }

  node->dn.task = task;
  KMP_MB();

  // Account for our initial fake value
  npredecessors++;

  // Update predecessors and obtain current value to check if there are still
  // any outstanding dependences (some tasks may have finished while we
  // processed the dependences)
  npredecessors =
      node->dn.npredecessors.fetch_add(npredecessors) + npredecessors;

  KA_TRACE(20, ("__kmp_check_deps: T#%d found %d predecessors for task %p \n",
                gtid, npredecessors, taskdata));

  // beyond this point the task could be queued (and executed) by a releasing
  // task...
  return npredecessors > 0 ? true : false;
}

typedef struct kmp_memory_task_args {
  // Common attributes of a memset operation
  void *Dst;
  const void *Src;
  int DstDevice;
  int SrcDevice;
  size_t Length;
  size_t DstOffset;
  size_t SrcOffset;

  // Original tasks connected by this temporary memory task.
  kmp_task_t *SourceTask;
  kmp_task_t *TargetTask;
} kmp_memory_task_args_t;

static void memory_task_entry(int gtid, kmp_task_t *task) {
  auto *mem_task_args =
      reinterpret_cast<kmp_memory_task_args_t *>(task->shareds);

  kmp_taskdata_t *MemoryTaskData = KMP_TASK_TO_TASKDATA(task);
  kmp_depnode_t *MemoryNode = MemoryTaskData->td_depnode;

  kmp_taskdata_t *SourceTaskData =
      mem_task_args->SourceTask
          ? KMP_TASK_TO_TASKDATA(mem_task_args->SourceTask)
          : nullptr;

  kmp_taskdata_t *TargetTaskData =
      mem_task_args->TargetTask
          ? KMP_TASK_TO_TASKDATA(mem_task_args->TargetTask)
          : nullptr;

  if (MemoryNode) {
    for (kmp_depnode_list_t *It = MemoryNode->dn.successors; It != nullptr;
         It = It->next) {
      kmp_task_t *SuccTask = It->node ? It->node->dn.task : nullptr;
      kmp_taskdata_t *SuccTD =
          SuccTask ? KMP_TASK_TO_TASKDATA(SuccTask) : nullptr;
    }
  }

  void (*tgt_target_memtask)(void *, int64_t, int64_t, int64_t);
  *(void **)(&tgt_target_memtask) = KMP_DLSYM("__tgt_target_memtask");
  KMP_ASSERT(tgt_target_memtask);
  tgt_target_memtask(mem_task_args->Dst, mem_task_args->Length,
                     mem_task_args->DstDevice, mem_task_args->SrcDevice);

  if (MemoryNode) {
    for (kmp_depnode_list_t *It = MemoryNode->dn.successors; It != nullptr;
         It = It->next) {
      kmp_task_t *SuccTask = It->node ? It->node->dn.task : nullptr;
      kmp_taskdata_t *SuccTD =
          SuccTask ? KMP_TASK_TO_TASKDATA(SuccTask) : nullptr;
    }
  }

  kmp_info_t *thread = __kmp_threads[gtid];
  // #if USE_FAST_MEMORY
  //   __kmp_fast_free(thread, mem_task_args);
  // #else
  //   __kmp_thread_free(thread, mem_task_args);
  // #endif
}

static void __kmp_print_node(kmp_int32 gtid, kmp_taskdata_t *taskdata) {
  KG_TRACE(10,
           ("T#%d: task%d has %d predecessors \n", gtid, taskdata->td_task_id,
            taskdata->td_depnode->dn.npredecessors.load()));
  KG_TRACE(10, ("T#%d: task%d successor list: ", gtid, taskdata->td_task_id));
  kmp_depnode_list_t *it = taskdata->td_depnode->dn.successors;
  while (it != nullptr) {
    KG_TRACE(10, ("%d, ", KMP_TASK_TO_TASKDATA(it->node->dn.task)->td_task_id));
    it = it->next;
  }
  KG_TRACE(10, ("\n"));
}

kmp_task_t *__kmp_alloc_memory_task(kmp_int32 gtid, int source_device,
                                    int target_device, void *dep_addr,
                                    size_t size_addr, kmp_task_t *source,
                                    kmp_task_t *target) {

  int32_t Flags = 0;
  kmp_tasking_flags_t *InputFlags =
      reinterpret_cast<kmp_tasking_flags_t *>(&Flags);
  InputFlags->hidden_helper = 1;

  kmp_info_t *thread = __kmp_threads[gtid];

#if USE_FAST_MEMORY
  auto *mem_task_args = static_cast<kmp_memory_task_args_t *>(
      __kmp_fast_allocate(thread, sizeof(kmp_memory_task_args_t)));
#else
  auto *mem_task_args = static_cast<kmp_memory_task_args_t *>(
      __kmp_thread_malloc(thread, sizeof(kmp_memory_task_args_t)));
#endif

  mem_task_args->Dst = dep_addr;
  mem_task_args->Src = dep_addr;
  mem_task_args->DstDevice = target_device;
  mem_task_args->SrcDevice = source_device;
  mem_task_args->Length = size_addr;
  mem_task_args->DstOffset = 0;
  mem_task_args->SrcOffset = 0;
  mem_task_args->SourceTask = source;
  mem_task_args->TargetTask = target;

  kmp_task_t *memory_task = __kmpc_omp_target_task_alloc(
      nullptr, gtid, Flags, sizeof(kmp_task_t), 0,
      reinterpret_cast<kmp_routine_entry_t>(memory_task_entry), -1, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr);

  if (!memory_task) {
    return nullptr;
  }

  memory_task->shareds = static_cast<void *>(mem_task_args);

  return memory_task;
}

kmp_task_t *__kmp_insert_legacy_memory_task(kmp_int32 gtid, kmp_task_t *source,
                                            kmp_task_t *target,
                                            int source_device,
                                            int target_device, void *dep_addr,
                                            size_t size_addr) {
  if (!source || !target)
    return nullptr;

  kmp_info_t *thread = __kmp_threads[gtid];

  kmp_taskdata_t *source_td = KMP_TASK_TO_TASKDATA(source);
  kmp_taskdata_t *target_td = KMP_TASK_TO_TASKDATA(target);

  if (!source_td->td_depnode || !target_td->td_depnode)
    return nullptr;

  kmp_task_t *memory_task = __kmp_alloc_memory_task(
      gtid, source_device, target_device, dep_addr, size_addr, source, target);

  if (!memory_task)
    return nullptr;

  kmp_taskdata_t *memory_td = KMP_TASK_TO_TASKDATA(memory_task);

#if USE_FAST_MEMORY
  kmp_depnode_t *node = static_cast<kmp_depnode_t *>(
      __kmp_fast_allocate(thread, sizeof(kmp_depnode_t)));
#else
  kmp_depnode_t *node = static_cast<kmp_depnode_t *>(
      __kmp_thread_malloc(thread, sizeof(kmp_depnode_t)));
#endif

  __kmp_init_node(node, /*on_stack=*/false);
  memory_td->td_depnode = node;

  // The memory task has exactly one predecessor: source. Follow the
  // __kmp_check_deps() convention of starting at -1 and adding two.
  node->dn.npredecessors = -1;
  node->dn.npredecessors.fetch_add(2);
  node->dn.task = memory_task;

  // Insert memory -> target without incrementing target's predecessor count:
  // this edge replaces source -> target.
  node->dn.successors =
      __kmp_add_node(thread, node->dn.successors, target_td->td_depnode);

  // Replace source -> target with source -> memory.
  bool replaced = __kmp_depnode_replace_successor(
      gtid, thread, source_td->td_depnode, target_td->td_depnode, node);
  KMP_ASSERT(replaced);

  KMP_MB();

  return memory_task;
}

static kmp_task_t *__kmp_insert_edge_task(kmp_int32 gtid,
                                          kmp_task_t *memory_task,
                                          kmp_task_t *source,
                                          kmp_task_t *target,
                                          bool is_taskgraph);

kmp_task_t *__kmp_create_memory_task(kmp_int32 gtid, kmp_task_t *source,
                                     kmp_task_t *target, int source_device,
                                     int target_device, void *dep_addr,
                                     size_t size_addr, bool is_taskgraph) {
  KG_TRACE(20, ("[SCHEDULER-MT] __kmp_create_memory_task (begin)\n"));
  KG_TRACE(10, ("(create_memory_task) dep_addr=%p\n", dep_addr));
  if (!KMP_TASK_TO_TASKDATA(source)->tdg)
    return nullptr;
  kmp_task_t *memory_task = __kmp_alloc_memory_task(
      gtid, source_device, target_device, dep_addr, size_addr, source, target);
  if (!memory_task)
    return nullptr;
  return __kmp_insert_edge_task(gtid, memory_task, source, target,
                                is_taskgraph);
}

// File I/O tasks the scheduler inserts on a graph edge (prefetch, stage,
// flush). Like a memory task they run on a hidden helper thread of the origin
// and block it until the proxy has done the work; libompfile does the work
// (ompfile_sched_io_task, resolved with dlsym).
typedef struct kmp_io_task_args {
  kmp_int32 kind;
  kmp_int32 device;
  kmp_uint64 path_key;
  kmp_int64 offset;
  kmp_uint64 extent;
} kmp_io_task_args_t;

static void io_task_entry(int gtid, kmp_task_t *task) {
  auto *args = reinterpret_cast<kmp_io_task_args_t *>(task->shareds);
  using IoTaskFn = int (*)(int, int, kmp_uint64, kmp_int64, kmp_uint64);
  static const IoTaskFn run =
      reinterpret_cast<IoTaskFn>(KMP_DLSYM("ompfile_sched_io_task"));
  // OMPCLUSTER_HEFT_IO_TASKS_NOOP=1 keeps the inserted tasks and their edges
  // but skips the I/O: the control for what the extra graph tasks alone do.
  static const bool noop = [] {
    const char *v = getenv("OMPCLUSTER_HEFT_IO_TASKS_NOOP");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  if (run == nullptr || noop)
    return;
  // A failed prefetch or stage only costs the read it meant to hide; a failed
  // flush leaves durability to the next commit, close or proxy shutdown.
  // libompfile counts and reports both.
  (void)run(args->kind, args->device, args->path_key, args->offset,
            args->extent);
}

kmp_task_t *__kmp_create_io_task(kmp_int32 gtid, kmp_task_t *source,
                                 kmp_task_t *target, kmp_int32 kind,
                                 kmp_int32 device, kmp_uint64 path_key,
                                 kmp_int64 offset, kmp_uint64 extent) {
  if (!KMP_TASK_TO_TASKDATA(source)->tdg)
    return nullptr;
  int32_t Flags = 0;
  kmp_tasking_flags_t *InputFlags =
      reinterpret_cast<kmp_tasking_flags_t *>(&Flags);
  InputFlags->hidden_helper = 1;
  kmp_info_t *thread = __kmp_threads[gtid];
#if USE_FAST_MEMORY
  auto *args = static_cast<kmp_io_task_args_t *>(
      __kmp_fast_allocate(thread, sizeof(kmp_io_task_args_t)));
#else
  auto *args = static_cast<kmp_io_task_args_t *>(
      __kmp_thread_malloc(thread, sizeof(kmp_io_task_args_t)));
#endif
  args->kind = kind;
  args->device = device;
  args->path_key = path_key;
  args->offset = offset;
  args->extent = extent;
  kmp_task_t *io_task = __kmpc_omp_target_task_alloc(
      nullptr, gtid, Flags, sizeof(kmp_task_t), 0,
      reinterpret_cast<kmp_routine_entry_t>(io_task_entry), -1, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr);
  if (!io_task)
    return nullptr;
  io_task->shareds = static_cast<void *>(args);
  return __kmp_insert_edge_task(gtid, io_task, source, target, true);
}

// Puts an allocated scheduler task (memory or file I/O) on the source ->
// target edge of the source's graph: source -> task -> target.
static kmp_task_t *__kmp_insert_edge_task(kmp_int32 gtid,
                                          kmp_task_t *memory_task,
                                          kmp_task_t *source,
                                          kmp_task_t *target,
                                          bool is_taskgraph) {
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *source_td = KMP_TASK_TO_TASKDATA(source);
  kmp_taskdata_t *target_td = KMP_TASK_TO_TASKDATA(target);
  kmp_tdg_info_t *tdg = source_td->tdg;
  const bool IsRecording = __kmp_tdg_is_recording(tdg->tdg_status);

  kmp_taskdata_t *memory_taskdata = KMP_TASK_TO_TASKDATA(memory_task);
  memory_taskdata->is_taskgraph = is_taskgraph;
  memory_taskdata->tdg = tdg;
  memory_taskdata->td_memory_task_has_runtime_depnode = IsRecording;

  KMP_ATOMIC_INC(&tdg->num_tasks);

  if (IsRecording) {
    __kmp_print_node(gtid, KMP_TASK_TO_TASKDATA(source));
    __kmp_print_node(gtid, KMP_TASK_TO_TASKDATA(target));
#if USE_FAST_MEMORY
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_fast_allocate(thread, sizeof(kmp_depnode_t));
#else
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_thread_malloc(thread, sizeof(kmp_depnode_t));
#endif
    __kmp_init_node(node, /*on_stack=*/false);
    memory_taskdata->td_depnode = node;
    // check_deps
    node->dn.npredecessors = -1;

    node->dn.npredecessors.fetch_add(2);
    node->dn.task = memory_task;

    node->dn.successors =
        __kmp_add_node(thread, node->dn.successors, target_td->td_depnode);

    bool replaced = __kmp_depnode_replace_successor(
        gtid, thread, source_td->td_depnode, target_td->td_depnode, node);
    KMP_ASSERT(replaced);
    KMP_MB();

    __kmp_print_node(gtid, KMP_TASK_TO_TASKDATA(source));
    __kmp_print_node(gtid, KMP_TASK_TO_TASKDATA(target));
    __kmp_print_node(gtid, memory_taskdata);
  } else {
    // Adding the memory task after the TDG has already been recorded.
    memory_taskdata->td_tdg_task_id =
        KMP_ATOMIC_LD_ACQ(&memory_taskdata->tdg->num_tasks) - 1;
  }
  KG_TRACE(10, ("MT: taskdata: %p ; memory task tdg id: %d\n", memory_taskdata,
                memory_taskdata->td_tdg_task_id));
  // extend record_map if needed
  if (memory_taskdata->td_tdg_task_id >= tdg->map_size) {
    __kmp_acquire_bootstrap_lock(&tdg->graph_lock);
    if (memory_taskdata->td_tdg_task_id >= tdg->map_size) {
      kmp_uint old_size = tdg->map_size;
      kmp_uint new_size = old_size * 2;
      kmp_node_info_t *old_record = tdg->record_map;
      kmp_node_info_t *new_record =
          (kmp_node_info_t *)__kmp_allocate(new_size * sizeof(kmp_node_info_t));
      KMP_MEMCPY(new_record, tdg->record_map,
                 old_size * sizeof(kmp_node_info_t));
      tdg->record_map = new_record;

      for (auto &map_entry : (*get_schedule_map(tdg->tdg_id))) {
        map_entry.second = TaskScheduleData{
            .state = map_entry.second.state,
            .rank = map_entry.second.rank,
            .translated_rank = map_entry.second.translated_rank,
            .task =
                &new_record[KMP_TASK_TO_TASKDATA(map_entry.second.task->task)
                                ->td_tdg_task_id],
        };
      }
      for (kmp_int i = 0; i < old_size; i++) {
        old_record[i].target_data = nullptr;
      }
      __kmp_free(old_record);

      for (kmp_int i = old_size; i < new_size; i++) {
        kmp_int32 *successorsList = (kmp_int32 *)__kmp_allocate(
            __kmp_successors_size * sizeof(kmp_int32));
        kmp_int32 *predecessorsList = (kmp_int32 *)__kmp_allocate(
            __kmp_predecessors_size * sizeof(kmp_int32));
        kmp_recorded_task_time_t *taskTimeList =
            (kmp_recorded_task_time_t *)__kmp_allocate(
                __kmpc_get_num_devices() * sizeof(kmp_recorded_task_time_t));
        new_record[i].task = nullptr;
        new_record[i].successors = successorsList;
        new_record[i].predecessors = predecessorsList;
        new_record[i].nsuccessors = 0;
        new_record[i].npredecessors = 0;
        new_record[i].successors_size = __kmp_successors_size;
        new_record[i].predecessors_size = __kmp_predecessors_size;
        new_record[i].is_target_task = 0;
        new_record[i].target_data = new kmp_target_task_data_t();
        new_record[i].task_time = taskTimeList;
        KMP_ATOMIC_ST_REL(&new_record[i].npredecessors_counter, 0);
      }
      // update the size at the end, so that we avoid other
      // threads use old_record while map_size is already updated
      tdg->map_size = new_size;
    }
    __kmp_release_bootstrap_lock(&tdg->graph_lock);
  }
  tdg->record_map[memory_taskdata->td_tdg_task_id].task = memory_task;
  tdg->record_map[memory_taskdata->td_tdg_task_id].parent_task =
      memory_taskdata->td_parent;
  tdg->record_map[memory_taskdata->td_tdg_task_id].is_target_task = 0;
  tdg->record_map[memory_taskdata->td_tdg_task_id].target_data =
      memory_taskdata->td_target_info;

  for (kmp_int32 device = 0; device < __kmpc_get_num_devices(); device++) {
    tdg->record_map[memory_taskdata->td_tdg_task_id]
        .task_time[device]
        .mean_time = -1;
    tdg->record_map[memory_taskdata->td_tdg_task_id]
        .task_time[device]
        .iteration_number = 0;
  }
  // KMP_ATOMIC_INC(&tdg->num_tasks);

  // The memory task replaces the existing source -> target TDG edge.
  // Keep source.nsuccessors and target.npredecessors unchanged.
  const kmp_int32 source_tdg_id = source_td->td_tdg_task_id;
  const kmp_int32 target_tdg_id = target_td->td_tdg_task_id;
  const kmp_int32 memory_tdg_id = memory_taskdata->td_tdg_task_id;

  kmp_node_info_t *source_info = &tdg->record_map[source_tdg_id];
  kmp_node_info_t *target_info = &tdg->record_map[target_tdg_id];
  kmp_node_info_t *memory_info = &tdg->record_map[memory_tdg_id];

  bool source_edge_replaced = false;
  for (kmp_int32 i = 0; i < source_info->nsuccessors; ++i) {
    if (source_info->successors[i] == target_tdg_id) {
      source_info->successors[i] = memory_tdg_id;
      source_edge_replaced = true;
      break;
    }
  }
  KMP_ASSERT(source_edge_replaced);

  bool target_edge_replaced = false;
  for (kmp_int32 i = 0; i < target_info->npredecessors; ++i) {
    if (target_info->predecessors[i] == source_tdg_id) {
      target_info->predecessors[i] = memory_tdg_id;
      target_edge_replaced = true;
      break;
    }
  }
  KMP_ASSERT(target_edge_replaced);

  memory_info->predecessors[0] = source_tdg_id;
  memory_info->npredecessors = 1;
  KMP_ATOMIC_ST_REL(&memory_info->npredecessors_counter, 1);

  memory_info->successors[0] = target_tdg_id;
  memory_info->nsuccessors = 1;

  KG_TRACE(50, ("(insert_edge_task) tdg = %p\n", tdg));
  KG_TRACE(10, ("__kmp_insert_edge_task (exit)\n"));
  return memory_task;
}

/*vector<kmp_task_and_pred>
__kmp_insert_memory_tasks_with_deps(kmp_int32 gtid, kmp_depnode_t *node,
                                   kmp_task_t *task, kmp_dephash_t **hash,
                                   bool dep_barrier, kmp_int32 ndeps,
                                   kmp_depend_info_t *dep_list,
                                   kmp_int32 ndeps_noalias,
                                   kmp_depend_info_t *noalias_dep_list
                                   bool is_taskgraph) {
  vector<kmp_task_and_pred> ret;
  int i, n_mtxs = 0;
  // Filter deps in dep_list
  // TODO: Different algorithm for large dep_list ( > 10 ? )
  for (i = 0; i < ndeps; i++) {
    if (dep_list[i].base_addr != 0 &&
        dep_list[i].base_addr != (kmp_intptr_t)KMP_SIZE_T_MAX) {
      KMP_DEBUG_ASSERT(
          dep_list[i].flag == KMP_DEP_IN || dep_list[i].flag == KMP_DEP_OUT ||
          dep_list[i].flag == KMP_DEP_INOUT ||
          dep_list[i].flag == KMP_DEP_MTX || dep_list[i].flag == KMP_DEP_SET);
      for (int j = i + 1; j < ndeps; j++) {
        if (dep_list[i].base_addr == dep_list[j].base_addr) {
          if (dep_list[i].flag != dep_list[j].flag) {
            // two different dependences on same address work identical to OUT
            dep_list[i].flag = KMP_DEP_OUT;
          }
          dep_list[j].base_addr = 0; // Mark j element as void
        }
      }
      if (dep_list[i].flag == KMP_DEP_MTX) {
        // limit number of mtx deps to MAX_MTX_DEPS per node
        if (n_mtxs < MAX_MTX_DEPS && task != NULL) {
          ++n_mtxs;
        } else {
          dep_list[i].flag = KMP_DEP_OUT; // downgrade mutexinoutset to inout
        }
      }
    }
  }

  kmp_info_t *thread = __kmp_threads[gtid];
  // kmp_int32 npredecessors = 0;
  for (kmp_int32 i = 0; i < ndeps; i++) {
    const kmp_depend_info_t *dep = &dep_list[i];

    kmp_dephash_entry_t *info =
        __kmp_dephash_find(thread, hash, dep->base_addr);
    kmp_depnode_t *last_out = info->last_out;
    // kmp_depnode_list_t *last_set = info->last_set;
    // kmp_depnode_list_t *prev_set = info->prev_set;

    if (dep->flags.in && last_out) {
      kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
      kmp_taskdata_t *prev_taskdata = KMP_TASK_TO_TASKDATA(last_out->dn.task);

      kmp_int64 target_device = __kmpc_get_task_device(taskdata->td_task_id);
      kmp_int64 source_device =
__kmpc_get_task_device(prev_taskdata->td_task_id);

      void **args = taskdata->td_target_info->args;
      kmp_int64 *arg_sizes = taskdata->td_target_info->arg_sizes;
      kmp_int32 num_args = taskdata->td_target_info->num_args;
      auto dep_addr = reinterpret_cast<void *>(dep->base_addr);

      for(int j = 0; j < num_args; j++) {
        if (args[j] == dep_addr) {
          // Setup the hidden helper flags
          int32_t Flags = 0;
          kmp_tasking_flags_t *InputFlags = (kmp_tasking_flags_t *)&Flags;
          InputFlags->hidden_helper = 1;

          // Lambda function that apply the task
          // TODO: update this function to do real memory transfer
          auto Fn = +[](int gtid, kmp_task_t *task) {
            auto *mem_task_args =
                 reinterpret_cast<kmp_memory_task_args_t *>(task->shareds);
            printf("Memory Task: ");
            printf("dst_device=%d, src_device=%d, hst_ptr=%p\n",
              mem_task_args->DstDevice, mem_task_args->SrcDevice,
mem_task_args->Dst);

            void (*tgt_target_memtask)(void *, int64_t , int64_t , int64_t);
            *(void **)(&tgt_target_memtask) = KMP_DLSYM("__tgt_target_memtask");

            KMP_ASSERT(tgt_target_memtask);
            tgt_target_memtask(mem_task_args->Dst, mem_task_args->Length,
              mem_task_args->DstDevice, mem_task_args->SrcDevice);

            kmp_info_t *thread = __kmp_threads[gtid];
#if USE_FAST_MEMORY
              __kmp_fast_free(thread, mem_task_args);
#else
              __kmp_thread_free(thread, mem_task_args);
#endif
          };

          kmp_info_t *thread = __kmp_threads[gtid];

#if USE_FAST_MEMORY
          auto mem_task_args = static_cast<kmp_memory_task_args_t *>(
            __kmp_fast_allocate(thread, sizeof(kmp_memory_task_args_t)));
#else
          auto mem_task_args = static_cast<kmp_memory_task_args_t *>(
            __kmp_thread_malloc(thread, sizeof(kmp_memory_task_args_t)));
#endif

          mem_task_args->Dst = dep_addr;
          mem_task_args->Src = dep_addr;
          mem_task_args->DstDevice = static_cast<int>(target_device);
          mem_task_args->SrcDevice = static_cast<int>(source_device);
          mem_task_args->Length = static_cast<size_t>(arg_sizes[j]);
          mem_task_args->DstOffset = 0;
          mem_task_args->SrcOffset = 0;

          // Alloc the memory task
          kmp_task_t *memory_task = __kmpc_omp_target_task_alloc(
            nullptr, gtid, Flags, sizeof(kmp_task_t), 0,
reinterpret_cast<kmp_routine_entry_t>(Fn), -1, nullptr, 0, nullptr, nullptr,
nullptr, nullptr
          );
          if (!memory_task) {
            return nullptr;
          }

          kmp_taskdata_t *memory_taskdata = KMP_TASK_TO_TASKDATA(memory_task);
          memory_taskdata->is_taskgraph = is_taskgraph;

          // Setup the arguments for the memory task
          memory_task->shareds = static_cast<void *>(mem_task_args);

          // Create a new depend object for the memory task
          kmp_depend_info_t mem_task_dep[1] = {
            {dep->base_addr, dep->len, {KMP_DEP_INOUT}}};

          kmp_task_and_pred temp;
          temp.pred = KMP_TASKDATA_TO_TASK(prev_taskdata);
          temp.task = memory_task;
          ret.push(temp);

          // Launch the helper task
          __kmpc_omp_task_with_deps(nullptr, gtid, memory_task, 1,
                                    mem_task_dep, 0, nullptr);
          break;
        }
      }

    }
  }

  return ret;
}
*/

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param new_task task thunk allocated by __kmp_omp_task_alloc() for the ''new
task''
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

@return Returns either TASK_CURRENT_NOT_QUEUED if the current task was not
suspended and queued, or TASK_CURRENT_QUEUED if it was suspended and queued

Schedule a non-thread-switchable task with dependences for execution
*/
kmp_int32 __kmpc_omp_task_with_deps(ident_t *loc_ref, kmp_int32 gtid,
                                    kmp_task_t *new_task, kmp_int32 ndeps,
                                    kmp_depend_info_t *dep_list,
                                    kmp_int32 ndeps_noalias,
                                    kmp_depend_info_t *noalias_dep_list) {
  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);
  KA_TRACE(10, ("__kmpc_omp_task_with_deps(enter): T#%d loc=%p task=%p\n", gtid,
                loc_ref, new_taskdata));
  __kmp_assert_valid_gtid(gtid);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

#if OMPX_TASKGRAPH
  // record TDG with deps
  if (new_taskdata->is_taskgraph &&
      __kmp_tdg_is_recording(new_taskdata->tdg->tdg_status)) {
    kmp_tdg_info_t *tdg = new_taskdata->tdg;

    // extend record_map if needed
    if (new_taskdata->td_tdg_task_id >= tdg->map_size) {
      __kmp_acquire_bootstrap_lock(&tdg->graph_lock);
      if (new_taskdata->td_tdg_task_id >= tdg->map_size) {
        kmp_uint old_size = tdg->map_size;
        kmp_uint new_size = old_size * 2;
        kmp_node_info_t *old_record = tdg->record_map;
        kmp_node_info_t *new_record = (kmp_node_info_t *)__kmp_allocate(
            new_size * sizeof(kmp_node_info_t));
        KMP_MEMCPY(new_record, tdg->record_map,
                   old_size * sizeof(kmp_node_info_t));
        tdg->record_map = new_record;
        for (kmp_int i = 0; i < old_size; i++) {
          old_record[i].target_data = nullptr;
        }

        __kmp_free(old_record);

        for (kmp_int i = old_size; i < new_size; i++) {
          kmp_int32 *successorsList = (kmp_int32 *)__kmp_allocate(
              __kmp_successors_size * sizeof(kmp_int32));
          kmp_int32 *predecessorsList = (kmp_int32 *)__kmp_allocate(
              __kmp_predecessors_size * sizeof(kmp_int32));
          kmp_recorded_task_time_t *taskTimeList =
              (kmp_recorded_task_time_t *)__kmp_allocate(
                  __kmpc_get_num_devices() * sizeof(kmp_recorded_task_time_t));
          new_record[i].task = nullptr;
          new_record[i].successors = successorsList;
          new_record[i].predecessors = predecessorsList;
          new_record[i].nsuccessors = 0;
          new_record[i].npredecessors = 0;
          new_record[i].successors_size = __kmp_successors_size;
          new_record[i].predecessors_size = __kmp_predecessors_size;
          new_record[i].is_target_task = 0;
          new_record[i].target_data = new kmp_target_task_data_t();
          new_record[i].task_time = taskTimeList;
          KMP_ATOMIC_ST_REL(&new_record[i].npredecessors_counter, 0);
        }
        // update the size at the end, so that we avoid other
        // threads use old_record while map_size is already updated
        tdg->map_size = new_size;
      }
      __kmp_release_bootstrap_lock(&tdg->graph_lock);
    }
    tdg->record_map[new_taskdata->td_tdg_task_id].task = new_task;
    tdg->record_map[new_taskdata->td_tdg_task_id].parent_task =
        new_taskdata->td_parent;
    tdg->record_map[new_taskdata->td_tdg_task_id].is_target_task =
        (new_taskdata->td_target_info != nullptr);
    tdg->record_map[new_taskdata->td_tdg_task_id].target_data =
        new_taskdata->td_target_info;

    for (kmp_int32 device = 0; device < __kmpc_get_num_devices(); device++) {
      tdg->record_map[new_taskdata->td_tdg_task_id]
          .task_time[device]
          .mean_time = -1;
      tdg->record_map[new_taskdata->td_tdg_task_id]
          .task_time[device]
          .iteration_number = 0;
    }
    __kmpc_set_deps(&tdg->record_map[new_taskdata->td_tdg_task_id], ndeps,
                    dep_list, ndeps_noalias, noalias_dep_list);

    KMP_ATOMIC_INC(&tdg->num_tasks);
  }
#endif
#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    if (!current_task->ompt_task_info.frame.enter_frame.ptr)
      current_task->ompt_task_info.frame.enter_frame.ptr =
          OMPT_GET_FRAME_ADDRESS(0);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          &(current_task->ompt_task_info.task_data),
          &(current_task->ompt_task_info.frame),
          &(new_taskdata->ompt_task_info.task_data),
          TASK_TYPE_DETAILS_FORMAT(new_taskdata), 1,
          OMPT_LOAD_OR_GET_RETURN_ADDRESS(gtid));
    }

    new_taskdata->ompt_task_info.frame.enter_frame.ptr =
        OMPT_GET_FRAME_ADDRESS(0);
  }

#if OMPT_OPTIONAL
  /* OMPT grab all dependences if requested by the tool */
  if (ndeps + ndeps_noalias > 0 && ompt_enabled.ompt_callback_dependences) {
    kmp_int32 i;

    int ompt_ndeps = ndeps + ndeps_noalias;
    ompt_dependence_t *ompt_deps = (ompt_dependence_t *)KMP_OMPT_DEPS_ALLOC(
        thread, (ndeps + ndeps_noalias) * sizeof(ompt_dependence_t));

    KMP_ASSERT(ompt_deps != NULL);

    for (i = 0; i < ndeps; i++) {
      ompt_deps[i].variable.ptr = (void *)dep_list[i].base_addr;
      if (dep_list[i].base_addr == (kmp_intptr_t)KMP_SIZE_T_MAX)
        ompt_deps[i].dependence_type = ompt_dependence_type_out_all_memory;
      else if (dep_list[i].flags.in && dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_inout;
      else if (dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_out;
      else if (dep_list[i].flags.in)
        ompt_deps[i].dependence_type = ompt_dependence_type_in;
      else if (dep_list[i].flags.mtx)
        ompt_deps[i].dependence_type = ompt_dependence_type_mutexinoutset;
      else if (dep_list[i].flags.set)
        ompt_deps[i].dependence_type = ompt_dependence_type_inoutset;
      else if (dep_list[i].flags.all)
        ompt_deps[i].dependence_type = ompt_dependence_type_out_all_memory;
    }
    for (i = 0; i < ndeps_noalias; i++) {
      ompt_deps[ndeps + i].variable.ptr = (void *)noalias_dep_list[i].base_addr;
      if (noalias_dep_list[i].base_addr == (kmp_intptr_t)KMP_SIZE_T_MAX)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_out_all_memory;
      else if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inout;
      else if (noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_out;
      else if (noalias_dep_list[i].flags.in)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_in;
      else if (noalias_dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (noalias_dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
      else if (noalias_dep_list[i].flags.all)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_out_all_memory;
    }
    ompt_callbacks.ompt_callback(ompt_callback_dependences)(
        &(new_taskdata->ompt_task_info.task_data), ompt_deps, ompt_ndeps);
    /* We can now free the allocated memory for the dependences */
    /* For OMPD we might want to delay the free until end of this function */
    KMP_OMPT_DEPS_FREE(thread, ompt_deps);
  }
#endif /* OMPT_OPTIONAL */
#endif /* OMPT_SUPPORT */

  if (__kmpc_is_legacy_scheduler()) {
    __kmpc_add_target_task_with_deps(new_taskdata, ndeps, dep_list,
                                     ndeps_noalias, noalias_dep_list);
  }

  bool serial = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;

  kmp_task_team_t *task_team = thread->th.th_task_team;

  serial = serial &&
           !(task_team && (task_team->tt.tt_found_proxy_tasks ||
                           task_team->tt.tt_hidden_helper_task_encountered));

  if (__kmp_use_dmdas_runtime()) {
    __kmp_dmdas_task_with_deps(loc_ref, gtid, new_task, ndeps, dep_list,
                               ndeps_noalias, noalias_dep_list);
  }

  if (!serial && (ndeps > 0 || ndeps_noalias > 0)) {
    /* if no dependences have been tracked yet, create the dependence hash */
    if (current_task->td_dephash == NULL)
      current_task->td_dephash = __kmp_dephash_create(thread, current_task);

#if USE_FAST_MEMORY
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_fast_allocate(thread, sizeof(kmp_depnode_t));
#else
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_thread_malloc(thread, sizeof(kmp_depnode_t));
#endif

    __kmp_init_node(node, /*on_stack=*/false);
    new_taskdata->td_depnode = node;

    bool native_blocked = __kmp_check_deps(
        gtid, node, new_task, &current_task->td_dephash, NO_DEP_BARRIER, ndeps,
        dep_list, ndeps_noalias, noalias_dep_list);

    if (native_blocked) {
      KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had blocking "
                    "dependences: "
                    "loc=%p task=%p, return: TASK_CURRENT_NOT_QUEUED\n",
                    gtid, loc_ref, new_taskdata));
#if OMPT_SUPPORT
      if (ompt_enabled.enabled) {
        current_task->ompt_task_info.frame.enter_frame = ompt_data_none;
      }
#endif

      return TASK_CURRENT_NOT_QUEUED;
    }
  } else {
    KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d ignored dependences "
                  "for task (serialized) loc=%p task=%p\n",
                  gtid, loc_ref, new_taskdata));
  }

  KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had no blocking "
                "dependences : "
                "loc=%p task=%p, transferring to __kmp_omp_task\n",
                gtid, loc_ref, new_taskdata));

  if (__kmp_use_dmdas_runtime()) {
    /*
     * __kmp_check_deps() did not find any blocking native OpenMP
     * dependency.  The task is therefore native-ready.  DMDAS may
     * submit it once its own dependency count is also zero.
     */
    __kmp_dmdas_native_ready(new_task);
    return TASK_CURRENT_NOT_QUEUED;
  }

  kmp_int32 ret = __kmp_omp_task(gtid, new_task, true);

#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    current_task->ompt_task_info.frame.enter_frame = ompt_data_none;
  }
#endif
  return ret;
}

#if OMPT_SUPPORT
void __ompt_taskwait_dep_finish(kmp_taskdata_t *current_task,
                                ompt_data_t *taskwait_task_data) {
  if (ompt_enabled.ompt_callback_task_schedule) {
    ompt_callbacks.ompt_callback(ompt_callback_task_schedule)(
        taskwait_task_data, ompt_taskwait_complete, NULL);
  }
  current_task->ompt_task_info.frame.enter_frame.ptr = NULL;
  *taskwait_task_data = ompt_data_none;
}
#endif /* OMPT_SUPPORT */

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

Blocks the current task until all specifies dependences have been fulfilled.
*/
void __kmpc_omp_wait_deps(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 ndeps,
                          kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
                          kmp_depend_info_t *noalias_dep_list) {
  __kmpc_omp_taskwait_deps_51(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                              noalias_dep_list, false);
}

/* __kmpc_omp_taskwait_deps_51 : Function for OpenMP 5.1 nowait clause.
                                 Placeholder for taskwait with nowait clause.
                                 Earlier code of __kmpc_omp_wait_deps() is now
                                 in this function.
*/
void __kmpc_omp_taskwait_deps_51(ident_t *loc_ref, kmp_int32 gtid,
                                 kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                                 kmp_int32 ndeps_noalias,
                                 kmp_depend_info_t *noalias_dep_list,
                                 kmp_int32 has_no_wait) {
  KA_TRACE(10, ("__kmpc_omp_taskwait_deps(enter): T#%d loc=%p nowait#%d\n",
                gtid, loc_ref, has_no_wait));
  if (ndeps == 0 && ndeps_noalias == 0) {
    KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d has no dependences to "
                  "wait upon : loc=%p\n",
                  gtid, loc_ref));
    return;
  }
  __kmp_assert_valid_gtid(gtid);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

#if OMPT_SUPPORT
  // this function represents a taskwait construct with depend clause
  // We signal 4 events:
  //  - creation of the taskwait task
  //  - dependences of the taskwait task
  //  - schedule and finish of the taskwait task
  ompt_data_t *taskwait_task_data = &thread->th.ompt_thread_info.task_data;
  KMP_ASSERT(taskwait_task_data->ptr == NULL);
  if (ompt_enabled.enabled) {
    if (!current_task->ompt_task_info.frame.enter_frame.ptr)
      current_task->ompt_task_info.frame.enter_frame.ptr =
          OMPT_GET_FRAME_ADDRESS(0);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          &(current_task->ompt_task_info.task_data),
          &(current_task->ompt_task_info.frame), taskwait_task_data,
          ompt_task_taskwait | ompt_task_undeferred | ompt_task_mergeable, 1,
          OMPT_LOAD_OR_GET_RETURN_ADDRESS(gtid));
    }
  }

#if OMPT_OPTIONAL
  /* OMPT grab all dependences if requested by the tool */
  if (ndeps + ndeps_noalias > 0 && ompt_enabled.ompt_callback_dependences) {
    kmp_int32 i;

    int ompt_ndeps = ndeps + ndeps_noalias;
    ompt_dependence_t *ompt_deps = (ompt_dependence_t *)KMP_OMPT_DEPS_ALLOC(
        thread, (ndeps + ndeps_noalias) * sizeof(ompt_dependence_t));

    KMP_ASSERT(ompt_deps != NULL);

    for (i = 0; i < ndeps; i++) {
      ompt_deps[i].variable.ptr = (void *)dep_list[i].base_addr;
      if (dep_list[i].flags.in && dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_inout;
      else if (dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_out;
      else if (dep_list[i].flags.in)
        ompt_deps[i].dependence_type = ompt_dependence_type_in;
      else if (dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    for (i = 0; i < ndeps_noalias; i++) {
      ompt_deps[ndeps + i].variable.ptr = (void *)noalias_dep_list[i].base_addr;
      if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inout;
      else if (noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_out;
      else if (noalias_dep_list[i].flags.in)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_in;
      else if (noalias_dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (noalias_dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    ompt_callbacks.ompt_callback(ompt_callback_dependences)(
        taskwait_task_data, ompt_deps, ompt_ndeps);
    /* We can now free the allocated memory for the dependences */
    /* For OMPD we might want to delay the free until end of this function */
    KMP_OMPT_DEPS_FREE(thread, ompt_deps);
    ompt_deps = NULL;
  }
#endif /* OMPT_OPTIONAL */
#endif /* OMPT_SUPPORT */

  // We can return immediately as:
  // - dependences are not computed in serial teams (except with proxy tasks)
  // - if the dephash is not yet created it means we have nothing to wait for
  bool ignore = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;
  ignore =
      ignore && thread->th.th_task_team != NULL &&
      thread->th.th_task_team->tt.tt_found_proxy_tasks == FALSE &&
      thread->th.th_task_team->tt.tt_hidden_helper_task_encountered == FALSE;
  ignore = ignore || current_task->td_dephash == NULL;

  if (ignore) {
    KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d has no blocking "
                  "dependences : loc=%p\n",
                  gtid, loc_ref));
#if OMPT_SUPPORT
    __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
    return;
  }

  kmp_depnode_t node = {0};
  __kmp_init_node(&node, /*on_stack=*/true);

  if (!__kmp_check_deps(gtid, &node, NULL, &current_task->td_dephash,
                        DEP_BARRIER, ndeps, dep_list, ndeps_noalias,
                        noalias_dep_list)) {
    KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d has no blocking "
                  "dependences : loc=%p\n",
                  gtid, loc_ref));
#if OMPT_SUPPORT
    __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */

    // There may still be references to this node here, due to task stealing.
    // Wait for them to be released.
    kmp_int32 nrefs;
    while ((nrefs = node.dn.nrefs) > 3) {
      KMP_DEBUG_ASSERT((nrefs & 1) == 1);
      KMP_YIELD(TRUE);
    }
    KMP_DEBUG_ASSERT(nrefs == 3);

    return;
  }

  int thread_finished = FALSE;
  kmp_flag_32<false, false> flag(
      (std::atomic<kmp_uint32> *)&node.dn.npredecessors, 0U);
  while (node.dn.npredecessors > 0) {
    flag.execute_tasks(thread, gtid, FALSE,
                       &thread_finished USE_ITT_BUILD_ARG(NULL),
                       __kmp_task_stealing_constraint);
  }

  // Wait until the last __kmp_release_deps is finished before we free the
  // current stack frame holding the "node" variable; once its nrefs count
  // reaches 3 (meaning 1, since bit zero of the refcount indicates a stack
  // rather than a heap address), we're sure nobody else can try to reference
  // it again.
  kmp_int32 nrefs;
  while ((nrefs = node.dn.nrefs) > 3) {
    KMP_DEBUG_ASSERT((nrefs & 1) == 1);
    KMP_YIELD(TRUE);
  }
  KMP_DEBUG_ASSERT(nrefs == 3);

#if OMPT_SUPPORT
  __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
  KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d finished waiting : loc=%p\
                \n",
                gtid, loc_ref));
}

void __kmpc_remove_task(kmp_node_info_t *task_tdg) {
  // Remove tasks and the related dependencies in tdg.
  // It doesn't impact depnode structure
  KG_TRACE(30, ("__kmpc_remove_task (begin)\n"));

  kmp_int gtid = __kmp_get_gtid();
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_task_t *task = task_tdg->task;
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
  kmp_depnode_t *node = taskdata->td_depnode;
  kmp_tdg_info_t *tdg = taskdata->tdg;

  KG_TRACE(30, ("__kmpc_remove_task: td_task_id=%d\n", taskdata->td_task_id));

  // A memory task represents:
  //
  //   source -> memory -> target
  //
  // Removing it restores the original TDG edge:
  //
  //   source -> target
  //
  // The insertion preserved the edge cardinalities, so the inverse
  // operation must preserve them as well.
  KMP_ASSERT(task_tdg->npredecessors == 1);
  KMP_ASSERT(task_tdg->nsuccessors == 1);

  const kmp_int32 memory_tdg_id = taskdata->td_tdg_task_id;
  const kmp_int32 source_tdg_id = task_tdg->predecessors[0];
  const kmp_int32 target_tdg_id = task_tdg->successors[0];

  kmp_node_info_t *source_tdg = &tdg->record_map[source_tdg_id];
  kmp_node_info_t *target_tdg = &tdg->record_map[target_tdg_id];

  bool source_edge_restored = false;
  for (kmp_int32 i = 0; i < source_tdg->nsuccessors; ++i) {
    if (source_tdg->successors[i] == memory_tdg_id) {
      source_tdg->successors[i] = target_tdg_id;
      source_edge_restored = true;
      break;
    }
  }
  KMP_ASSERT(source_edge_restored);

  bool target_edge_restored = false;
  for (kmp_int32 i = 0; i < target_tdg->npredecessors; ++i) {
    if (target_tdg->predecessors[i] == memory_tdg_id) {
      target_tdg->predecessors[i] = source_tdg_id;
      target_edge_restored = true;
      break;
    }
  }
  KMP_ASSERT(target_edge_restored);

  // Check mutexinoutset dependencies, release locks
  if (UNLIKELY(node && (node->dn.mtx_num_locks < 0))) {
    // negative num_locks means all locks were acquired
    node->dn.mtx_num_locks = -node->dn.mtx_num_locks;
    for (int i = node->dn.mtx_num_locks - 1; i >= 0; --i) {
      KMP_DEBUG_ASSERT(node->dn.mtx_locks[i] != NULL);
      __kmp_release_lock(node->dn.mtx_locks[i], gtid);
    }
  }
  if (taskdata->td_dephash) {
    __kmp_dephash_free(thread, taskdata->td_dephash);
    taskdata->td_dephash = NULL;
  }

  // The runtime dependency graph for this execution has already been
  // consumed by __kmp_release_deps(). Do not attempt to reconstruct its
  // source -> target edge here. Only the persistent TDG record_map edges
  // above need to be restored for the next replay.

  // kmp_depnode_list_t *next;
  // for (kmp_depnode_list_t *p = node->dn.successors; p != nullptr; p = next) {
  //   kmp_depnode_t *successor = p->node;
  //   if (successor == nullptr) continue;
  //   KMP_ATOMIC_DEC(&successor->dn.npredecessors);
  //   //We might want to launch the task if npredecessors == 0 but it won't
  //   happen hen removing memory tasks next = p->next;
  //     __kmp_node_deref(thread, p->node);
  // #if USE_FAST_MEMORY
  //     __kmp_fast_free(thread, p);
  // #else
  //     __kmp_thread_free(thread, p);
  // #endif
  // }
  // __kmp_node_deref(thread, node);

  // #if USE_FAST_MEMORY
  //   __kmp_fast_free(thread, task);
  //   __kmp_fast_free(thread, taskdata);
  // #else /* ! USE_FAST_MEMORY */
  //   __kmp_thread_free(thread, task);
  //   __kmp_thread_free(thread, taskdata);
  // #endif

  // KMP_ATOMIC_DEC(&tdg->num_tasks);
  // KMP_ATOMIC_DEC(&taskdata->td_taskgroup->count);

  kmp_int32 *successorsList =
      (kmp_int32 *)__kmp_allocate(__kmp_successors_size * sizeof(kmp_int32));
  kmp_int32 *predecessorsList =
      (kmp_int32 *)__kmp_allocate(__kmp_predecessors_size * sizeof(kmp_int32));
  kmp_recorded_task_time_t *taskTimeList =
      (kmp_recorded_task_time_t *)__kmp_allocate(
          __kmpc_get_num_devices() * sizeof(kmp_recorded_task_time_t));
  task_tdg->task = nullptr;
  task_tdg->successors = successorsList;
  task_tdg->predecessors = predecessorsList;
  task_tdg->nsuccessors = 0;
  task_tdg->npredecessors = 0;
  task_tdg->successors_size = __kmp_successors_size;
  task_tdg->predecessors_size = __kmp_predecessors_size;
  task_tdg->is_target_task = 0;
  task_tdg->target_data = new kmp_target_task_data_t();
  task_tdg->task_time = taskTimeList;
  KMP_ATOMIC_ST_REL(&task_tdg->npredecessors_counter, 0);

  KG_TRACE(30, ("__kmpc_remove_task (exit)\n"));
}
