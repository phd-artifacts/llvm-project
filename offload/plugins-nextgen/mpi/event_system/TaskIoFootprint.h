//===-- TaskIoFootprint.h - Per-kernel file I/O footprint -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What one target region did through libompfile, collected on the proxy and
// returned with the LAUNCH_KERNEL acknowledgement. The kernel runs to
// completion inside launch_kernel on the proxy's exec handler, so a
// thread-local collector set around that call sees exactly the region's calls;
// a libompfile call from a thread the region spawned is not attributed
// (ompfile_mpp_task_io_note counts those).
//
// TaskIoEntryTy is mirrored by kmp_task_io_entry_t in libomp (kmp.h); both
// sides assert the size.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_MPI_EVENT_SYSTEM_TASKIOFOOTPRINT_H
#define OFFLOAD_PLUGINS_NEXTGEN_MPI_EVENT_SYSTEM_TASKIOFOOTPRINT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

/// One merged range of a file, per (path key, direction, owner rank).
struct TaskIoEntryTy {
  uint64_t PathKey = 0;
  int64_t Offset = 0;  // -1 for cursor-based calls (omp_file_read/write).
  uint64_t Extent = 0; // Length of [Offset, Offset + Extent).
  uint64_t Bytes = 0;  // Requested bytes, which can exceed Extent on overlap.
  double Seconds = 0;  // Time the region spent inside these calls.
  uint32_t Ops = 0;
  int32_t OwnerRank = -1;
  uint32_t IsWrite = 0;
  uint32_t Widened = 0; // 1 when the range also covers untouched bytes.
};
static_assert(sizeof(TaskIoEntryTy) == 56, "mirrored by kmp_task_io_entry_t");

/// The fixed part of the acknowledgement.
struct TaskIoHeaderTy {
  double Seconds = 0; // All time inside libompfile, open/flush/close included.
  uint32_t NumEntries = 0;
  uint32_t DroppedOps = 0; // Ops with no room for an entry; time still counted.
};

/// Values of ompfile_mpp_task_io_note's Kind.
enum TaskIoKindTy : int { TASK_IO_READ = 0, TASK_IO_WRITE = 1, TASK_IO_META = 2 };

class TaskIoCollectorTy {
public:
  /// Enough for a task's tiles; beyond it ranges widen rather than grow.
  static constexpr size_t MaxEntries = 64;

  void note(uint64_t PathKey, int OwnerRank, int Kind, int64_t Offset,
            uint64_t Size, double Seconds) {
    Header.Seconds += Seconds;
    if (Kind == TASK_IO_META)
      return;
    const uint32_t IsWrite = Kind == TASK_IO_WRITE;
    TaskIoEntryTy *Last = nullptr;
    for (TaskIoEntryTy &E : Entries) {
      if (E.PathKey != PathKey || E.IsWrite != IsWrite ||
          E.OwnerRank != OwnerRank || (E.Offset < 0) != (Offset < 0))
        continue;
      Last = &E;
      if (Offset < 0 || touches(E, Offset, Size)) {
        merge(E, Offset, Size, Seconds);
        return;
      }
    }
    if (Entries.size() < MaxEntries) {
      TaskIoEntryTy E;
      E.PathKey = PathKey;
      E.OwnerRank = OwnerRank;
      E.IsWrite = IsWrite;
      E.Offset = Offset;
      merge(E, Offset, Size, Seconds);
      Entries.push_back(E);
      Header.NumEntries = static_cast<uint32_t>(Entries.size());
      return;
    }
    if (!Last) {
      ++Header.DroppedOps;
      return;
    }
    Last->Widened = 1;
    merge(*Last, Offset, Size, Seconds);
  }

  const TaskIoHeaderTy &header() const { return Header; }
  const TaskIoEntryTy *entries() const { return Entries.data(); }

  static TaskIoCollectorTy *&current() {
    static thread_local TaskIoCollectorTy *Current = nullptr;
    return Current;
  }

  /// Makes a collector current on this thread. Never held across a suspension
  /// point: the thread may change when a coroutine resumes.
  class ScopeTy {
  public:
    explicit ScopeTy(TaskIoCollectorTy &Collector) : Previous(current()) {
      current() = &Collector;
    }
    ~ScopeTy() { current() = Previous; }
    ScopeTy(const ScopeTy &) = delete;
    ScopeTy &operator=(const ScopeTy &) = delete;

  private:
    TaskIoCollectorTy *Previous;
  };

private:
  static bool touches(const TaskIoEntryTy &E, int64_t Offset, uint64_t Size) {
    return Offset <= E.Offset + static_cast<int64_t>(E.Extent) &&
           Offset + static_cast<int64_t>(Size) >= E.Offset;
  }
  static void merge(TaskIoEntryTy &E, int64_t Offset, uint64_t Size,
                    double Seconds) {
    if (Offset < 0) {
      E.Extent += Size;
    } else {
      const int64_t End = std::max(E.Offset + static_cast<int64_t>(E.Extent),
                                   Offset + static_cast<int64_t>(Size));
      E.Offset = std::min(E.Offset, Offset);
      E.Extent = static_cast<uint64_t>(End - E.Offset);
    }
    E.Bytes += Size;
    E.Seconds += Seconds;
    ++E.Ops;
  }

  TaskIoHeaderTy Header;
  std::vector<TaskIoEntryTy> Entries;
};

#endif // OFFLOAD_PLUGINS_NEXTGEN_MPI_EVENT_SYSTEM_TASKIOFOOTPRINT_H
