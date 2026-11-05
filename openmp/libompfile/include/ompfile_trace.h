//===- ompfile_trace.h - NVTX ranges for libompfile and MPP ----*- C++ -*-===//
//
// The one place that includes NVTX. Every other file uses these wrappers, so
// the whole trace surface compiles to nothing when OMPFILE_ENABLE_NVTX is off
// (cmake/OmpFileNvtx.cmake decides, and turns it off where the header is
// missing).
//
// Two kinds of range, and the difference is load-bearing:
//
//  - Scope / push-pop: thread-local, strictly nested. Use only for work that
//    starts and ends on the same thread without suspending in between: a lock
//    wait, a blocking MPI call, a syscall, one coroutine resume.
//  - Start/end with an explicit id: not thread-local. MPP events are coroutines
//    that a handler pops, resumes, re-queues, and another handler resumes
//    later, so an event's lifetime crosses threads and must use these. The id
//    is a plain uint64_t so it can live in the coroutine promise whether or
//    not NVTX is compiled in.
//
// Collected by Nsight Systems: nsys profile --trace=mpi,nvtx ... (not osrt,
// which hangs an MPP job at startup). The recipe
// is in docs/getting-started.md.
//
//===----------------------------------------------------------------------===//

#ifndef OMPFILE_TRACE_H
#define OMPFILE_TRACE_H

#include <cstdint>

#if defined(OMPFILE_ENABLE_NVTX) && OMPFILE_ENABLE_NVTX
#include OMPFILE_NVTX_HEADER
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace ompfile::trace {

/// Which component a range belongs to. Each is its own NVTX domain, so the
/// Nsight timeline can show or hide them separately.
enum class Domain { MPP, LibOmpFile };

/// Coarse colour classes, so the timeline reads at a glance.
enum class Color : uint32_t {
  Exec = 0xFF4C72B0,   // kernel launches
  Data = 0xFF55A868,   // data movement events
  Io = 0xFFDD8452,     // file-I/O events
  Wait = 0xFFC44E52,   // anything spent waiting: locks, blocking MPI, spins
  Syscall = 0xFF8172B3, // filesystem calls
  Other = 0xFF937860,
};

#if defined(OMPFILE_ENABLE_NVTX) && OMPFILE_ENABLE_NVTX

inline nvtxDomainHandle_t domainHandle(Domain D) {
  static nvtxDomainHandle_t Mpp = nvtxDomainCreateA("MPP");
  static nvtxDomainHandle_t Lib = nvtxDomainCreateA("libompfile");
  return D == Domain::MPP ? Mpp : Lib;
}

inline nvtxEventAttributes_t attributes(const char *Name, Color C,
                                        uint64_t Payload) {
  nvtxEventAttributes_t A{};
  A.version = NVTX_VERSION;
  A.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  A.colorType = NVTX_COLOR_ARGB;
  A.color = static_cast<uint32_t>(C);
  A.messageType = NVTX_MESSAGE_TYPE_ASCII;
  A.message.ascii = Name;
  A.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  A.payload.ullValue = Payload;
  return A;
}

/// Starts a cross-thread range. Returns 0 only when NVTX is compiled out.
inline uint64_t rangeStart(Domain D, const char *Name, Color C,
                           uint64_t Payload = 0) {
  nvtxEventAttributes_t A = attributes(Name, C, Payload);
  return nvtxDomainRangeStartEx(domainHandle(D), &A);
}

inline void rangeEnd(Domain D, uint64_t Id) {
  if (Id)
    nvtxDomainRangeEnd(domainHandle(D), Id);
}

inline void push(Domain D, const char *Name, Color C, uint64_t Payload = 0) {
  nvtxEventAttributes_t A = attributes(Name, C, Payload);
  nvtxDomainRangePushEx(domainHandle(D), &A);
}

inline void pop(Domain D) { nvtxDomainRangePop(domainHandle(D)); }

inline void mark(Domain D, const char *Name, Color C, uint64_t Payload = 0) {
  nvtxEventAttributes_t A = attributes(Name, C, Payload);
  nvtxDomainMarkEx(domainHandle(D), &A);
}

/// Names the calling OS thread on the timeline (mpp-gate, mpp-io-0, ...).
inline void nameThisThread(const char *Name) {
  nvtxNameOsThreadA(static_cast<uint32_t>(::syscall(SYS_gettid)), Name);
}

constexpr bool Enabled = true;

#else

inline uint64_t rangeStart(Domain, const char *, Color, uint64_t = 0) {
  return 0;
}
inline void rangeEnd(Domain, uint64_t) {}
inline void push(Domain, const char *, Color, uint64_t = 0) {}
inline void pop(Domain) {}
inline void mark(Domain, const char *, Color, uint64_t = 0) {}
inline void nameThisThread(const char *) {}

constexpr bool Enabled = false;

#endif

/// RAII push/pop. Only for work that neither suspends nor migrates threads.
class Scope {
  Domain D;

public:
  Scope(Domain D, const char *Name, Color C, uint64_t Payload = 0) : D(D) {
    push(D, Name, C, Payload);
  }
  ~Scope() { pop(D); }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
};

} // namespace ompfile::trace

#endif // OMPFILE_TRACE_H
