//===-- ProxyEventStats.h - Proxy per-event-type timing ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The proxy event-stats singleton: the gate and handler threads in
// ProxyDevice.cpp record into it, and ProxyOmpFileStats.cpp prints it with
// the OMPFile stats. Every function here must stay inline with external
// linkage (no static, no anonymous namespace): otherwise each TU gets its
// own singleton and the printer reports what no recorder wrote.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_MPI_PROXY_EVENT_STATS_H
#define OFFLOAD_PLUGINS_NEXTGEN_MPI_PROXY_EVENT_STATS_H

#include "EventSystem.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// OMPTARGET_MPI_PROXY_EVENT_STATS=1: per-event-type wall time on the proxy,
// from the gate thread creating an event to the handler thread seeing it
// done, split into the wait for the first resume (queue time) and the rest
// (service time), with the number of handler resumes it took. Printed with
// the OMPFile stats as "proxy_event_stats" lines. The origin-side twin is
// OMPTARGET_MPI_EVENT_STATS in rtl.cpp.
inline bool proxyEventStatsEnabled() {
  static const bool Enabled = [] {
    const char *Raw = std::getenv("OMPTARGET_MPI_PROXY_EVENT_STATS");
    return Raw && Raw[0] == '1' && Raw[1] == '\0';
  }();
  return Enabled;
}

// OMPTARGET_MPI_PROXY_EVENT_SPIN=<n>: how many extra times a handler thread
// may resume an unfinished event in place before returning it to the queue.
// 0 (default) is the historical behaviour: one resume per queue rotation.
// Each rotation costs a mutex, a condition-variable notify and a pop, and a
// small RETRIEVE needs hundreds of them (AMD 405038: 428 resumes per event
// against 18 for a SUBMIT of the same size, because the proxy's sends
// complete only once the origin posts the matching receives). Spinning in
// place trades fairness between concurrent events for that per-poll cost,
// which is why it is bounded and opt-in.
inline unsigned proxyEventSpin() {
  static const unsigned Spin = [] {
    const char *Raw = std::getenv("OMPTARGET_MPI_PROXY_EVENT_SPIN");
    if (!Raw || !Raw[0])
      return 0u;
    char *End = nullptr;
    const long Value = std::strtol(Raw, &End, 10);
    if (!End || *End != '\0' || Value <= 0)
      return 0u;
    return static_cast<unsigned>(Value > 4096 ? 4096 : Value);
  }();
  return Spin;
}

inline uint64_t proxyNowUs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

struct ProxyEventStatsTy {
  static constexpr size_t kTypes = 96;
  std::atomic<uint64_t> Count[kTypes];
  std::atomic<uint64_t> QueueUs[kTypes];
  std::atomic<uint64_t> ServiceUs[kTypes];
  std::atomic<uint64_t> MaxTotalUs[kTypes];
  std::atomic<uint64_t> Resumes[kTypes];

  ProxyEventStatsTy() {
    for (size_t I = 0; I < kTypes; ++I) {
      Count[I] = 0;
      QueueUs[I] = 0;
      ServiceUs[I] = 0;
      MaxTotalUs[I] = 0;
      Resumes[I] = 0;
    }
  }

  void add(EventTypeTy Type, uint64_t Queue, uint64_t Service,
           uint32_t ResumeCount) {
    const size_t I = static_cast<size_t>(Type);
    if (I >= kTypes)
      return;
    Count[I].fetch_add(1, std::memory_order_relaxed);
    QueueUs[I].fetch_add(Queue, std::memory_order_relaxed);
    ServiceUs[I].fetch_add(Service, std::memory_order_relaxed);
    Resumes[I].fetch_add(ResumeCount, std::memory_order_relaxed);
    const uint64_t Total = Queue + Service;
    uint64_t Prev = MaxTotalUs[I].load(std::memory_order_relaxed);
    while (Total > Prev && !MaxTotalUs[I].compare_exchange_weak(
                               Prev, Total, std::memory_order_relaxed))
      ;
  }
};

inline ProxyEventStatsTy &proxyEventStats() {
  static ProxyEventStatsTy *Stats = new ProxyEventStatsTy();
  return *Stats;
}

#endif // OFFLOAD_PLUGINS_NEXTGEN_MPI_PROXY_EVENT_STATS_H
