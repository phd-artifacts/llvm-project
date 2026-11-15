#ifndef OMPFILE_GRAPH_PLAN_H
#define OMPFILE_GRAPH_PLAN_H

// The C ABI between libomp's task graph scheduler and libompfile on the
// origin. After the graph's scheduler places a recorded graph, libomp resolves
// ompfile_graph_plan_writes with dlsym and hands it the file ranges the graph's
// target tasks write, from their footprints or declarations, with the device
// each task was placed on. libomp mirrors the struct (kmp_task_io_plan_write_t
// in kmp.h); keep the two in step.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ompfile_graph_write {
  uint64_t path_key; // ompfile::keys::pathKey of the file
  int64_t offset;
  uint64_t extent;
  int32_t device; // the writing task's OpenMP device
  uint32_t reserved;
} ompfile_graph_write;

// With LIBOMPFILE_GRAPH_PLAN=1 and the coherent stage posture
// (ompfile::config::ClientConfig::graphPlanCoherent), merges the ranges per
// file and sends them to the headnode, which then places each PWRITE inside
// one on its writer's rank. Returns the number of merged ranges sent, 0 when
// the plan is off (nothing is sent), or -1 with errno set.
int ompfile_graph_plan_writes(const ompfile_graph_write *writes, int count);

// File I/O tasks libomp's HEFT scheduler inserts on graph edges
// (OMPCLUSTER_HEFT_IO_TASKS). libomp mirrors the kinds in kmp_target_heft.cpp.
enum {
  OMPFILE_IO_TASK_PREFETCH = 1, // read a range into the device's prefetch cache
  OMPFILE_IO_TASK_STAGE = 2,    // the same, on the file's owner
  OMPFILE_IO_TASK_FLUSH = 3,    // commit the file on its owner
  OMPFILE_IO_TASK_RESET = 4,    // clear every device's prefetch cache
};

// Origin: run one I/O task on `device` (-1 with RESET: every device). The path
// comes from the application's declarations and opens; returns 0, or -1 with
// errno set (ENOENT: no path known for the key).
int ompfile_sched_io_task(int kind, int device, uint64_t path_key,
                          int64_t offset, uint64_t extent);

// Proxy: what the plugin calls for an OMPFILE_IO_TASK event. Returns 0, or -1
// with errno set.
int ompfile_io_task_run(int kind, const char *path, int64_t offset,
                        uint64_t extent);

#ifdef __cplusplus
}

static_assert(sizeof(ompfile_graph_write) == 32,
              "mirrored by kmp_task_io_plan_write_t in libomp");
#endif

#endif // OMPFILE_GRAPH_PLAN_H
