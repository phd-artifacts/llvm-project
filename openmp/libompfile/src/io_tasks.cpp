//===-- io_tasks.cpp - File I/O tasks the graph scheduler inserts ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// libomp's HEFT scheduler inserts file I/O tasks on the edges of a scheduled
// graph (OMPCLUSTER_HEFT_IO_TASKS): prefetch, stage and flush. Each runs on an
// origin helper thread and calls ompfile_sched_io_task here, which resolves
// the file's path from its key and asks one proxy to do the work
// (OMPFILE_IO_TASK through the MPI plugin). On that proxy the plugin calls
// ompfile_io_task_run, which does it with this process's client:
//
//  - prefetch / stage: read a range into the proxy's prefetch cache. The next
//    omp_file_pread on that proxy that falls inside the range is served from
//    the cache and consumes it; a write through the proxy to an overlapping
//    range drops it; a reset (sent before every scheduled run) clears the
//    whole cache, so a range never outlives the run it was planned for.
//  - flush: commit the file (on its owner, the deferred fdatasync of
//    LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=deferred).
//
// Paths: the scheduler only has path keys (footprints and declarations carry
// keys). The origin remembers the path of every file the application declared
// (omp_file_declare_next_task_io) or opened; a key it has never seen is not
// resolved and its task does nothing.
//
//===----------------------------------------------------------------------===//

#include "client_context.h"
#include "debug_log.h"
#include "mpp_shim.h"
#include "ompfile_graph_plan.h"
#include "ompfile_keys.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ompfile::io_tasks {
namespace {

struct PathRegistry {
  std::mutex Lock;
  std::unordered_map<uint64_t, std::string> Paths;
};
PathRegistry &paths() {
  static PathRegistry *const Registry = new PathRegistry();
  return *Registry;
}

struct CacheEntry {
  int64_t Offset = 0;
  std::vector<unsigned char> Data;
  uint64_t Consumed = 0;
};

struct PrefetchCache {
  std::mutex Lock;
  std::unordered_map<uint64_t, std::vector<CacheEntry>> Entries;
  uint64_t Bytes = 0;
  // Read without the lock on every pread: zero means nothing to look up.
  std::atomic<uint64_t> Count{0};
};
PrefetchCache &cache() {
  static PrefetchCache *const Cache = new PrefetchCache();
  return *Cache;
}

uint64_t cacheCapacity() {
  static const uint64_t Capacity = [] {
    const char *Env = std::getenv("LIBOMPFILE_PREFETCH_CACHE_MB");
    const long Mb = Env != nullptr ? std::atol(Env) : 4096;
    return static_cast<uint64_t>(Mb > 0 ? Mb : 0) << 20;
  }();
  return Capacity;
}

struct Stats {
  // Origin: tasks sent, by kind, and the ones that failed or had no path.
  std::atomic<uint64_t> Sent[5]{};
  std::atomic<uint64_t> Failed{0}, Unresolved{0};
  std::atomic<uint64_t> SentUs{0};
  // Proxy: cache fills, hits and the bytes behind them.
  std::atomic<uint64_t> Fills{0}, FillBytes{0}, FillUs{0}, FillSkipped{0};
  std::atomic<uint64_t> Hits{0}, HitBytes{0}, Dropped{0}, Resets{0};
  std::atomic<uint64_t> Flushes{0}, FlushUs{0}, RunFailures{0};
  ~Stats();
};
Stats &stats() {
  static Stats Instance;
  return Instance;
}

bool statsEnabled() {
  const char *Env = std::getenv("LIBOMPFILE_OPT_STATS");
  return Env != nullptr && Env[0] != '\0' && Env[0] != '0';
}

// One line per process at exit, so the origin's and every proxy's appear in
// the job log next to the other stats lines.
Stats::~Stats() {
  const uint64_t SentTotal = Sent[1] + Sent[2] + Sent[3];
  if (!statsEnabled() || (SentTotal == 0 && Fills == 0 && Hits == 0 &&
                          Flushes == 0 && Resets == 0))
    return;
  std::fprintf(stderr,
               "[omp-io] IO task stats: sent_prefetch=%llu sent_stage=%llu "
               "sent_flush=%llu sent_failed=%llu unresolved=%llu "
               "sent_us=%llu fills=%llu fill_bytes=%llu fill_us=%llu "
               "fill_skipped=%llu hits=%llu hit_bytes=%llu dropped=%llu "
               "flushes=%llu flush_us=%llu run_failures=%llu resets=%llu\n",
               static_cast<unsigned long long>(Sent[1].load()),
               static_cast<unsigned long long>(Sent[2].load()),
               static_cast<unsigned long long>(Sent[3].load()),
               static_cast<unsigned long long>(Failed.load()),
               static_cast<unsigned long long>(Unresolved.load()),
               static_cast<unsigned long long>(SentUs.load()),
               static_cast<unsigned long long>(Fills.load()),
               static_cast<unsigned long long>(FillBytes.load()),
               static_cast<unsigned long long>(FillUs.load()),
               static_cast<unsigned long long>(FillSkipped.load()),
               static_cast<unsigned long long>(Hits.load()),
               static_cast<unsigned long long>(HitBytes.load()),
               static_cast<unsigned long long>(Dropped.load()),
               static_cast<unsigned long long>(Flushes.load()),
               static_cast<unsigned long long>(FlushUs.load()),
               static_cast<unsigned long long>(RunFailures.load()),
               static_cast<unsigned long long>(Resets.load()));
  std::fflush(stderr);
}

uint64_t elapsedUs(std::chrono::steady_clock::time_point Start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - Start)
          .count());
}

void clearCache() {
  PrefetchCache &C = cache();
  std::lock_guard<std::mutex> Guard(C.Lock);
  uint64_t Left = 0;
  for (auto &KV : C.Entries)
    Left += KV.second.size();
  stats().Dropped += Left;
  C.Entries.clear();
  C.Bytes = 0;
  C.Count = 0;
}

bool insertCache(uint64_t Key, int64_t Offset,
                 std::vector<unsigned char> &&Data) {
  PrefetchCache &C = cache();
  std::lock_guard<std::mutex> Guard(C.Lock);
  if (C.Bytes + Data.size() > cacheCapacity())
    return false;
  C.Bytes += Data.size();
  CacheEntry Entry;
  Entry.Offset = Offset;
  Entry.Data = std::move(Data);
  C.Entries[Key].push_back(std::move(Entry));
  C.Count.fetch_add(1, std::memory_order_release);
  return true;
}

int numDevices() {
  using Fn = int (*)();
  static const Fn F = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT,
                                                 "omp_get_num_devices"));
  return F != nullptr ? F() : 0;
}

} // namespace

void rememberPath(const char *Path) {
  if (Path == nullptr || Path[0] == '\0')
    return;
  PathRegistry &R = paths();
  std::lock_guard<std::mutex> Guard(R.Lock);
  R.Paths.emplace(ompfile::keys::pathKey(Path), Path);
}

bool takePrefetched(int FileHandle, int64_t Offset, void *Data, size_t Size) {
  PrefetchCache &C = cache();
  if (C.Count.load(std::memory_order_acquire) == 0 || Offset < 0 || Size == 0)
    return false;
  uint64_t Key = 0;
  int Owner = -1;
  OmpFileClientContext::getInstance().taskIoTarget(FileHandle, Key, Owner);
  if (Key == 0)
    return false;
  std::lock_guard<std::mutex> Guard(C.Lock);
  auto It = C.Entries.find(Key);
  if (It == C.Entries.end())
    return false;
  auto &List = It->second;
  for (size_t I = 0; I < List.size(); ++I) {
    CacheEntry &E = List[I];
    const int64_t End = E.Offset + static_cast<int64_t>(E.Data.size());
    if (Offset < E.Offset || Offset + static_cast<int64_t>(Size) > End)
      continue;
    std::memcpy(Data, E.Data.data() + (Offset - E.Offset), Size);
    E.Consumed += Size;
    stats().Hits += 1;
    stats().HitBytes += Size;
    if (E.Consumed >= E.Data.size()) {
      C.Bytes -= E.Data.size();
      List.erase(List.begin() + static_cast<long>(I));
      C.Count.fetch_sub(1, std::memory_order_release);
      if (List.empty())
        C.Entries.erase(It);
    }
    return true;
  }
  return false;
}

void dropOverlapping(int FileHandle, int64_t Offset, size_t Size) {
  PrefetchCache &C = cache();
  if (C.Count.load(std::memory_order_acquire) == 0)
    return;
  uint64_t Key = 0;
  int Owner = -1;
  OmpFileClientContext::getInstance().taskIoTarget(FileHandle, Key, Owner);
  std::lock_guard<std::mutex> Guard(C.Lock);
  auto It = C.Entries.find(Key);
  if (It == C.Entries.end())
    return;
  auto &List = It->second;
  for (size_t I = 0; I < List.size();) {
    const CacheEntry &E = List[I];
    const int64_t End = E.Offset + static_cast<int64_t>(E.Data.size());
    const bool Overlaps =
        Offset < 0 || (Offset < End && E.Offset < Offset + static_cast<int64_t>(Size));
    if (!Overlaps) {
      ++I;
      continue;
    }
    C.Bytes -= E.Data.size();
    List.erase(List.begin() + static_cast<long>(I));
    C.Count.fetch_sub(1, std::memory_order_release);
    stats().Dropped += 1;
  }
  if (List.empty())
    C.Entries.erase(It);
}

} // namespace ompfile::io_tasks

extern "C" {

int ompfile_sched_io_task(int kind, int device, uint64_t path_key,
                          int64_t offset, uint64_t extent) {
  using namespace ompfile::io_tasks;
  const auto Start = std::chrono::steady_clock::now();
  if (kind == OMPFILE_IO_TASK_RESET) {
    const int Devices = numDevices();
    int Rc = 0;
    for (int D = 0; D < Devices; ++D) {
      int RemoteRc = 0, RemoteErrno = 0;
      if (!ompfile::mpp::ioTask(D, kind, "", 0, 0, RemoteRc, RemoteErrno))
        Rc = -1;
    }
    return Rc;
  }
  if (kind < OMPFILE_IO_TASK_PREFETCH || kind > OMPFILE_IO_TASK_FLUSH) {
    errno = EINVAL;
    return -1;
  }
  std::string Path;
  {
    auto &R = paths();
    std::lock_guard<std::mutex> Guard(R.Lock);
    auto It = R.Paths.find(path_key);
    if (It != R.Paths.end())
      Path = It->second;
  }
  if (Path.empty()) {
    stats().Unresolved += 1;
    errno = ENOENT;
    return -1;
  }
  int RemoteRc = 0, RemoteErrno = 0;
  const bool Sent = ompfile::mpp::ioTask(device, kind, Path.c_str(), offset,
                                         extent, RemoteRc, RemoteErrno);
  stats().Sent[kind] += 1;
  stats().SentUs += elapsedUs(Start);
  if (!Sent || RemoteRc != 0) {
    stats().Failed += 1;
    errno = RemoteErrno != 0 ? RemoteErrno : EIO;
    return -1;
  }
  return 0;
}

int ompfile_io_task_run(int kind, const char *path, int64_t offset,
                        uint64_t extent) {
  using namespace ompfile::io_tasks;
  if (kind == OMPFILE_IO_TASK_RESET) {
    clearCache();
    stats().Resets += 1;
    return 0;
  }
  if (path == nullptr || path[0] == '\0') {
    errno = EINVAL;
    return -1;
  }
  const auto Start = std::chrono::steady_clock::now();
  auto &Ctx = OmpFileClientContext::getInstance();
  const int Handle = Ctx.openFile(path);
  if (Handle < 0) {
    stats().RunFailures += 1;
    return -1;
  }
  int Rc = 0;
  if (kind == OMPFILE_IO_TASK_FLUSH) {
    Rc = Ctx.commitFile(Handle);
    stats().Flushes += 1;
    stats().FlushUs += elapsedUs(Start);
  } else if (extent == 0 || offset < 0 || extent > cacheCapacity()) {
    stats().FillSkipped += 1;
  } else {
    std::vector<unsigned char> Data(extent);
    Rc = Ctx.readFileAt(Handle, Data.data(), extent, offset);
    if (Rc == 0) {
      if (insertCache(ompfile::keys::pathKey(path), offset, std::move(Data))) {
        stats().Fills += 1;
        stats().FillBytes += extent;
        stats().FillUs += elapsedUs(Start);
      } else {
        stats().FillSkipped += 1;
      }
    }
  }
  const int SavedErrno = errno;
  if (Ctx.closeFile(Handle) != 0 && Rc == 0)
    Rc = -1;
  else
    errno = SavedErrno;
  if (Rc != 0)
    stats().RunFailures += 1;
  return Rc;
}

} // extern "C"
