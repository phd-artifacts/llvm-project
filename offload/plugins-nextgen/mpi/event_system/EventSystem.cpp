//===------ event_system.cpp - Concurrent MPI communication -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the MPI Event System used by the MPI
// target runtime for concurrent communication.
//
//===----------------------------------------------------------------------===//

#include "EventSystem.h"
#include "EventSystemOmpFile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include <mpi.h>
#include <unistd.h>

#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Shared/EnvironmentVar.h"
#include "Shared/Utils.h"
#include "omptarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include "llvm/Support/DynamicLibrary.h"

#define CHECK(expr, msg, ...)                                                  \
  if (!(expr)) {                                                               \
    REPORT(msg, ##__VA_ARGS__);                                                \
    return false;                                                              \
  }

std::string EventTypeToString(EventTypeTy eventType) {
  return eventTypeName(eventType);
}

const char *eventTypeName(EventTypeTy eventType) {
  switch (eventType) {
    case EventTypeTy::RETRIEVE_NUM_DEVICES: return "RETRIEVE_NUM_DEVICES";
    case EventTypeTy::INIT_DEVICE: return "INIT_DEVICE";
    case EventTypeTy::INIT_RECORD_REPLAY: return "INIT_RECORD_REPLAY";
    case EventTypeTy::IS_PLUGIN_COMPATIBLE: return "IS_PLUGIN_COMPATIBLE";
    case EventTypeTy::IS_DEVICE_COMPATIBLE: return "IS_DEVICE_COMPATIBLE";
    case EventTypeTy::IS_DATA_EXCHANGABLE: return "IS_DATA_EXCHANGABLE";
    case EventTypeTy::LOAD_BINARY: return "LOAD_BINARY";
    case EventTypeTy::GET_GLOBAL: return "GET_GLOBAL";
    case EventTypeTy::GET_FUNCTION: return "GET_FUNCTION";
    case EventTypeTy::SYNCHRONIZE: return "SYNCHRONIZE";
    case EventTypeTy::INIT_ASYNC_INFO: return "INIT_ASYNC_INFO";
    case EventTypeTy::INIT_DEVICE_INFO: return "INIT_DEVICE_INFO";
    case EventTypeTy::QUERY_ASYNC: return "QUERY_ASYNC";
    case EventTypeTy::PRINT_DEVICE_INFO: return "PRINT_DEVICE_INFO";
    case EventTypeTy::DATA_LOCK: return "DATA_LOCK";
    case EventTypeTy::DATA_UNLOCK: return "DATA_UNLOCK";
    case EventTypeTy::DATA_NOTIFY_MAPPED: return "DATA_NOTIFY_MAPPED";
    case EventTypeTy::DATA_NOTIFY_UNMAPPED: return "DATA_NOTIFY_UNMAPPED";
    case EventTypeTy::ALLOC: return "ALLOC";
    case EventTypeTy::DELETE: return "DELETE";
    case EventTypeTy::SUBMIT: return "SUBMIT";
    case EventTypeTy::RETRIEVE: return "RETRIEVE";
    case EventTypeTy::LOCAL_EXCHANGE: return "LOCAL_EXCHANGE";
    case EventTypeTy::EXCHANGE_SRC: return "EXCHANGE_SRC";
    case EventTypeTy::EXCHANGE_DST: return "EXCHANGE_DST";
    case EventTypeTy::LAUNCH_KERNEL: return "LAUNCH_KERNEL";
    case EventTypeTy::SYNC: return "SYNC";
    case EventTypeTy::EXIT: return "EXIT";
    default: return ompfileEventTypeName(eventType);
  }
}

namespace {

uint64_t transferNowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct RetrievePhaseStats {
  std::atomic<uint64_t> Count[2] = {};
  std::atomic<uint64_t> Time[2][3] = {};
  ~RetrievePhaseStats() {
    for (unsigned Side = 0; Side < 2; ++Side) {
      const auto N = Count[Side].load(std::memory_order_relaxed);
      if (!N)
        continue;
      fprintf(stderr, "[mpi-plugin] retrieve_phase_stats side=%s count=%llu "
                      "%s=%llu %s=%llu %s=%llu\n",
              Side ? "proxy" : "origin", (unsigned long long)N,
              Side ? "header_wait_us" : "status_wait_us",
              (unsigned long long)Time[Side][0].load(std::memory_order_relaxed),
              Side ? "device_wait_us" : "payload_post_us",
              (unsigned long long)Time[Side][1].load(std::memory_order_relaxed),
              Side ? "send_wait_us" : "payload_wait_us",
              (unsigned long long)Time[Side][2].load(std::memory_order_relaxed));
    }
  }
};

RetrievePhaseStats &retrievePhaseStats() {
  static RetrievePhaseStats Stats;
  return Stats;
}

bool ompfileTraceEnabled() {
  static const bool Enabled = []() {
    const char *Env = std::getenv("LIBOMPFILE_DEBUG_TRACE");
    return Env && Env[0] == '1' && Env[1] == '\0';
  }();
  return Enabled;
}

bool ompfileMpiErrorsReturnEnabled() {
  static const bool Enabled = []() {
    const char *Env = std::getenv("LIBOMPFILE_DEBUG_MPI_ERRORS_RETURN");
    return Env && Env[0] == '1' && Env[1] == '\0';
  }();
  return Enabled;
}

uint64_t nextTraceSeq() {
  static std::atomic<uint64_t> Seq{1};
  return Seq.fetch_add(1, std::memory_order_relaxed);
}

unsigned long threadIdHash() {
  return static_cast<unsigned long>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

void ompfileTrace(const char *Fmt, ...) {
  if (!ompfileTraceEnabled())
    return;

  va_list Args;
  va_start(Args, Fmt);
  fprintf(stderr, "[ompfile-mpi][trace][seq=%llu][pid=%d][tid=%lu] ",
          static_cast<unsigned long long>(nextTraceSeq()), getpid(),
          threadIdHash());
  vfprintf(stderr, Fmt, Args);
  va_end(Args);
}

unsigned long long requestBits(const MPI_Request &Req) {
  unsigned long long Bits = 0;
  static_assert(sizeof(Bits) >= sizeof(MPI_Request),
                "Cannot safely print MPI_Request payload.");
  std::memcpy(&Bits, &Req, sizeof(MPI_Request));
  return Bits;
}

void traceRequestSample(const llvm::SmallVector<MPI_Request> &Requests) {
  if (!ompfileTraceEnabled())
    return;

  constexpr size_t MaxSample = 6;
  char Buf[256];
  size_t Offset = 0;
  Offset += static_cast<size_t>(
      std::snprintf(Buf + Offset, sizeof(Buf) - Offset, "["));
  for (size_t I = 0; I < Requests.size() && I < MaxSample &&
                   Offset + 32 < sizeof(Buf);
       ++I) {
    Offset += static_cast<size_t>(std::snprintf(
        Buf + Offset, sizeof(Buf) - Offset, "%s0x%llx", I == 0 ? "" : ",",
        requestBits(Requests[static_cast<int>(I)])));
  }
  if (Requests.size() > MaxSample && Offset + 8 < sizeof(Buf))
    Offset += static_cast<size_t>(
        std::snprintf(Buf + Offset, sizeof(Buf) - Offset, ",..."));
  std::snprintf(Buf + Offset, sizeof(Buf) - Offset, "]");
  ompfileTrace("MPI request sample count=%zu values=%s\n", Requests.size(), Buf);
}

void configureCommErrhandlerForDebug(MPI_Comm Comm, const char *Label) {
  if (!ompfileMpiErrorsReturnEnabled() || Comm == MPI_COMM_NULL)
    return;
  const int Rc = MPI_Comm_set_errhandler(Comm, MPI_ERRORS_RETURN);
  ompfileTrace("configure errhandler label=%s rc=%d\n",
               Label ? Label : "(unknown)", Rc);
}

} // namespace

RetrieveTimingTy::RetrieveTimingTy(bool IsProxy) : Proxy(IsProxy) {
  static const bool StatsEnabled = [] {
    const char *Value = std::getenv("OMPTARGET_MPI_TRANSFER_STATS");
    return Value && std::strcmp(Value, "1") == 0;
  }();
  Enabled = StatsEnabled;
  if (Enabled) {
    (void)retrievePhaseStats();
    PreviousUs = transferNowUs();
  }
}

void RetrieveTimingTy::mark(unsigned Phase) {
  assert(Phase < 3);
  if (!Enabled)
    return;
  const uint64_t Now = transferNowUs();
  Durations[Phase] = Now - PreviousUs;
  PreviousUs = Now;
  Phases |= 1u << Phase;
}

RetrieveTimingTy::~RetrieveTimingTy() {
  if (!Enabled || Phases != 7)
    return;
  auto &Stats = retrievePhaseStats();
  Stats.Count[Proxy].fetch_add(1, std::memory_order_relaxed);
  for (unsigned Phase = 0; Phase < 3; ++Phase)
    Stats.Time[Proxy][Phase].fetch_add(Durations[Phase],
                                      std::memory_order_relaxed);
}

/// Resumes the most recent incomplete coroutine in the list.
void EventTy::resume() {
  // Acquire first handle not done.
  const CoHandleTy &RootHandle = getHandle().promise().RootHandle;
  auto &ResumableHandle = RootHandle.promise().PrevHandle;
  while (ResumableHandle.done()) {
    ResumableHandle = ResumableHandle.promise().PrevHandle;

    if (ResumableHandle == RootHandle)
      break;
  }

  if (!ResumableHandle.done())
    ResumableHandle.resume();
}

/// Wait until event completes.
void EventTy::wait() {
  // The calling thread is consumed until the event completes: on the origin
  // that is the thread that called omp_file_*, and on a proxy it is a handler
  // thread forwarding to another rank. Thread-local, so a push/pop range; its
  // width on the timeline is exactly the thread-occupancy cost.
  ompfile::trace::Scope Trace(ompfile::trace::Domain::MPP,
                              ompfile::trace::Enabled ? eventTypeName(EventType)
                                                      : "event-wait",
                              ompfile::trace::Color::Wait);
  // Advance the event progress until it is completed.
  while (!done()) {
    advance();
  }
}

/// Advance the event to the next suspension point and wait a while
void EventTy::advance() {
  resume();
  std::this_thread::sleep_for(
      std::chrono::microseconds(EventPollingRate.get()));
}

/// Check if the event has completed.
bool EventTy::done() const { return getHandle().done(); }

/// Check if it is an empty event.
bool EventTy::empty() const { return !getHandle(); }

/// Get the coroutine error from the Handle.
llvm::Error EventTy::getError() {
  auto &Error = getHandle().promise().CoroutineError;
  if (Error) {
    llvm::Error Result = std::move(*Error);
    Error.reset();
    return Result;
  }

  return llvm::Error::success();
}

///  MPI Request Manager Destructor. The Manager cannot be destroyed until all
///  the requests it manages have been completed.
MPIRequestManagerTy::~MPIRequestManagerTy() {
  if (Requests.empty())
    return;

  ompfileTrace("MPIRequestManagerTy dtor draining pending requests this=%p "
               "req_count=%zu other_rank=%d tag=%d event_type=%d\n",
               static_cast<void *>(this), Requests.size(), OtherRank, Tag,
               EventType);
  traceRequestSample(Requests);

  int MPIInitialized = 0;
  int MPIFinalized = 0;
  MPI_Initialized(&MPIInitialized);
  if (MPIInitialized)
    MPI_Finalized(&MPIFinalized);
  ompfileTrace("MPIRequestManagerTy dtor dropping pending requests because "
               "teardown reached with MPI %s this=%p req_count=%zu\n",
               (!MPIInitialized ? "uninitialized"
                                : (MPIFinalized ? "finalized" : "active")),
               static_cast<void *>(this), Requests.size());

  Requests.clear();
  PendingMPIError = MPI_SUCCESS;
}

/// Send a message to \p OtherRank asynchronously.
void MPIRequestManagerTy::send(const void *Buffer, int Size,
                               MPI_Datatype Datatype) {
  sendTagged(Buffer, Size, Datatype, 0);
}

void MPIRequestManagerTy::sendTagged(const void *Buffer, int Size,
                                     MPI_Datatype Datatype, int TagOffset) {
  MPI_Request &Req = Requests.emplace_back(MPI_REQUEST_NULL);
  int MPIError = ompfileWithMPICallLock([&]() {
    return MPI_Isend(Buffer, Size, Datatype, OtherRank, Tag + TagOffset, Comm,
                     &Req);
  });
  if (MPIError != MPI_SUCCESS) {
    PendingMPIError = MPIError;
    Req = MPI_REQUEST_NULL;
  }
}

/// Divide the \p Buffer into fragments of size \p MPIFragmentSize and send them
/// to \p OtherRank asynchronously.
void MPIRequestManagerTy::sendInBatchs(void *Buffer, int64_t Size) {
  sendInBatchsTagged(Buffer, Size, 0);
}

void MPIRequestManagerTy::sendInBatchsTagged(void *Buffer, int64_t Size,
                                             int TagOffset) {
  if (Size <= 0)
    return;

  const int64_t FragmentSize = MPIFragmentSize.get();
  if (FragmentSize <= 0) {
    PendingMPIError = MPI_ERR_COUNT;
    return;
  }

  constexpr int64_t MaxMPIChunk =
      static_cast<int64_t>(std::numeric_limits<int>::max());

  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  char *BufferByteArray = reinterpret_cast<char *>(Buffer);
  int64_t RemainingBytes = Size;
  int64_t Offset = 0;
  while (RemainingBytes > 0) {
    const int64_t Chunk = std::min({RemainingBytes, FragmentSize, MaxMPIChunk});
    assert(Chunk > 0 && "MPI fragment chunk must be positive.");
    sendTagged(&BufferByteArray[Offset], static_cast<int>(Chunk), MPI_BYTE,
               TagOffset);
    Offset += Chunk;
    RemainingBytes -= Chunk;
  }
}

/// Receive a message from \p OtherRank asynchronously.
void MPIRequestManagerTy::receive(void *Buffer, int Size,
                                  MPI_Datatype Datatype) {
  receiveTagged(Buffer, Size, Datatype, 0);
}

void MPIRequestManagerTy::receiveTagged(void *Buffer, int Size,
                                        MPI_Datatype Datatype, int TagOffset) {
  MPI_Request &Req = Requests.emplace_back(MPI_REQUEST_NULL);
  int MPIError = ompfileWithMPICallLock([&]() {
    return MPI_Irecv(Buffer, Size, Datatype, OtherRank, Tag + TagOffset, Comm,
                     &Req);
  });
  if (MPIError != MPI_SUCCESS) {
    PendingMPIError = MPIError;
    Req = MPI_REQUEST_NULL;
  }
}

/// Asynchronously receive message fragments from \p OtherRank and reconstruct
/// them into \p Buffer.
void MPIRequestManagerTy::receiveInBatchs(void *Buffer, int64_t Size) {
  receiveInBatchsTagged(Buffer, Size, 0);
}

void MPIRequestManagerTy::receiveInBatchsTagged(void *Buffer, int64_t Size,
                                                int TagOffset) {
  if (Size <= 0)
    return;

  const int64_t FragmentSize = MPIFragmentSize.get();
  if (FragmentSize <= 0) {
    PendingMPIError = MPI_ERR_COUNT;
    return;
  }

  constexpr int64_t MaxMPIChunk =
      static_cast<int64_t>(std::numeric_limits<int>::max());

  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  char *BufferByteArray = reinterpret_cast<char *>(Buffer);
  int64_t RemainingBytes = Size;
  int64_t Offset = 0;
  while (RemainingBytes > 0) {
    const int64_t Chunk = std::min({RemainingBytes, FragmentSize, MaxMPIChunk});
    assert(Chunk > 0 && "MPI fragment chunk must be positive.");
    receiveTagged(&BufferByteArray[Offset], static_cast<int>(Chunk), MPI_BYTE,
                  TagOffset);
    Offset += Chunk;
    RemainingBytes -= Chunk;
  }
}

/// Coroutine that waits until all pending requests finish.
EventTy MPIRequestManagerTy::wait() {
  if (PendingMPIError != MPI_SUCCESS) {
    ompfileTrace("MPIRequestManagerTy::wait pending mpi error this=%p "
                 "mpi_error=%d req_count=%zu\n",
                 static_cast<void *>(this), PendingMPIError, Requests.size());
    co_return createError("MPI request setup failed with code %d",
                          PendingMPIError);
  }

  if (Requests.empty())
    co_return llvm::Error::success();

  size_t CompletedRequests = 0;
  uint64_t PollIterations = 0;
  ompfileTrace("MPIRequestManagerTy::wait enter this=%p req_count=%zu "
               "other_rank=%d tag=%d event_type=%d\n",
               static_cast<void *>(this), Requests.size(), OtherRank, Tag,
               EventType);
  traceRequestSample(Requests);

  while (CompletedRequests < Requests.size()) {
    ++PollIterations;
    if (ompfileTraceEnabled() &&
        (PollIterations <= 3 || (PollIterations % 1024) == 0)) {
      ompfileTrace("MPIRequestManagerTy::wait poll this=%p iter=%llu "
                   "req_count=%zu completed=%zu\n",
                   static_cast<void *>(this),
                   static_cast<unsigned long long>(PollIterations),
                   Requests.size(), CompletedRequests);
      traceRequestSample(Requests);
    }

    int CompletedIndex = MPI_UNDEFINED;
    int RequestCompleted = false;
    int MPIError = ompfileWithMPICallLock([&]() {
      return MPI_Testany(static_cast<int>(Requests.size()), Requests.data(),
                         &CompletedIndex, &RequestCompleted,
                         MPI_STATUS_IGNORE);
    });

    if (MPIError != MPI_SUCCESS) {
      char ErrBuf[MPI_MAX_ERROR_STRING] = {0};
      int ErrLen = 0;
      MPI_Error_string(MPIError, ErrBuf, &ErrLen);
      ompfileTrace("MPIRequestManagerTy::wait MPI_Testany error this=%p "
                   "iter=%llu mpi_error=%d msg=%.*s\n",
                   static_cast<void *>(this),
                   static_cast<unsigned long long>(PollIterations), MPIError,
                   ErrLen, ErrBuf);
      traceRequestSample(Requests);
      co_return createError("Waiting of MPI requests failed with code %d",
                            MPIError);
    }

    if (RequestCompleted) {
      if (CompletedIndex == MPI_UNDEFINED) {
        // MPI reports no active requests. This can happen if a lower layer
        // completed all handles and replaced them with MPI_REQUEST_NULL; treat
        // the manager as drained rather than entering a spin loop.
        CompletedRequests = Requests.size();
      } else {
        ++CompletedRequests;
      }
      continue;
    }

    co_await std::suspend_always{};
  }

  ompfileTrace("MPIRequestManagerTy::wait success this=%p poll_iters=%llu\n",
               static_cast<void *>(this),
               static_cast<unsigned long long>(PollIterations));
  Requests.clear();

  co_return llvm::Error::success();
}

EventTy operator co_await(MPIRequestManagerTy &RequestManager) {
  return RequestManager.wait();
}

void *memAllocHost(int64_t Size) {
  void *HstPtr = nullptr;
  int MPIError = MPI_Alloc_mem(Size, MPI_INFO_NULL, &HstPtr);
  if (MPIError != MPI_SUCCESS)
    return nullptr;
  return HstPtr;
}

int memFreeHost(void *HstPtr) {
  int MPIError = MPI_Free_mem(HstPtr);
  if (MPIError != MPI_SUCCESS)
    return OFFLOAD_FAIL;
  return OFFLOAD_SUCCESS;
}

/// Event implementations on Host side.
namespace OriginEvents {

EventTy retrieveNumDevices(MPIRequestManagerTy RequestManager,
                           int32_t *NumDevices) {
  RequestManager.receive(NumDevices, 1, MPI_INT32_T);
  co_return (co_await RequestManager);
}

EventTy isPluginCompatible(MPIRequestManagerTy RequestManager,
                           __tgt_device_image *Image, bool *QueryResult) {
  uint64_t Size = utils::getPtrDiff(Image->ImageEnd, Image->ImageStart);

  void *Buffer = memAllocHost(Size);
  if (Buffer != nullptr)
    memcpy(Buffer, Image->ImageStart, Size);
  else
    Buffer = Image->ImageStart;

  RequestManager.send(&Size, 1, MPI_UINT64_T);
  RequestManager.send(Buffer, Size, MPI_BYTE);

  if (auto Err = co_await RequestManager; Err)
    co_return Err;

  if (Buffer != Image->ImageStart)
    memFreeHost(Buffer);

  RequestManager.receive(QueryResult, sizeof(bool), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy isDeviceCompatible(MPIRequestManagerTy RequestManager,
                           __tgt_device_image *Image, bool *QueryResult) {
  uint64_t Size = utils::getPtrDiff(Image->ImageEnd, Image->ImageStart);

  void *Buffer = memAllocHost(Size);
  if (Buffer != nullptr)
    memcpy(Buffer, Image->ImageStart, Size);
  else
    Buffer = Image->ImageStart;

  RequestManager.send(&Size, 1, MPI_UINT64_T);
  RequestManager.send(Buffer, Size, MPI_BYTE);

  if (auto Err = co_await RequestManager; Err)
    co_return Err;

  if (Buffer != Image->ImageStart)
    memFreeHost(Buffer);

  RequestManager.receive(QueryResult, sizeof(bool), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy initDevice(MPIRequestManagerTy RequestManager, void **DevicePtr) {
  // Wait the complete notification
  RequestManager.receive(DevicePtr, sizeof(void *), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy initRecordReplay(MPIRequestManagerTy RequestManager, int64_t MemorySize,
                         void *VAddr, bool IsRecord, bool SaveOutput,
                         uint64_t *ReqPtrArgOffset) {
  RequestManager.send(&MemorySize, 1, MPI_INT64_T);
  RequestManager.send(&VAddr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&IsRecord, sizeof(bool), MPI_BYTE);
  RequestManager.send(&SaveOutput, sizeof(bool), MPI_BYTE);
  RequestManager.receive(&ReqPtrArgOffset, 1, MPI_UINT64_T);
  co_return (co_await RequestManager);
}

EventTy isDataExchangable(MPIRequestManagerTy RequestManager,
                          int32_t DstDeviceId, bool *QueryResult) {
  RequestManager.send(&DstDeviceId, 1, MPI_INT32_T);
  RequestManager.receive(QueryResult, sizeof(bool), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy allocateBuffer(MPIRequestManagerTy RequestManager, int64_t Size,
                       int32_t Kind, void **Buffer) {
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&Kind, 1, MPI_INT32_T);

  // Event completion notification
  RequestManager.receive(Buffer, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy deleteBuffer(MPIRequestManagerTy RequestManager, void *Buffer,
                     int32_t Kind) {
  RequestManager.send(&Buffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Kind, 1, MPI_INT32_T);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy submit(MPIRequestManagerTy RequestManager, void *TgtPtr,
               void *HstPtr, int64_t Size,
               __tgt_async_info *AsyncInfoPtr) {
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  RequestManager.send(&TgtPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);

  RequestManager.sendInBatchs(HstPtr, Size);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy retrieve(MPIRequestManagerTy RequestManager, int64_t Size, void *HstPtr,
                 void *TgtPtr, __tgt_async_info *AsyncInfoPtr) {
  RetrieveTimingTy Timing(false);
  bool DeviceOpStatus = true;

  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  RequestManager.send(&TgtPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);

  RequestManager.receive(&DeviceOpStatus, sizeof(bool), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  if (!DeviceOpStatus)
    co_return createError("Failed to retrieve %p TgtPtr.", TgtPtr);

  Timing.mark(0);
  RequestManager.receiveInBatchs(HstPtr, Size);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  Timing.mark(1);
  auto Error = co_await RequestManager;
  if (!Error)
    Timing.mark(2);
  co_return Error;
}

EventTy localExchange(MPIRequestManagerTy RequestManager, void *SrcPtr,
                      int DstDeviceId, void *DstPtr, int64_t Size,
                      __tgt_async_info *AsyncInfoPtr) {
  RequestManager.send(&SrcPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&DstDeviceId, 1, MPI_INT);
  RequestManager.send(&DstPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy exchange(MPIRequestManagerTy RequestManager, int SrcRank,
                 const void *OrgBuffer, int DstRank, void *DstBuffer,
                 int64_t Size, __tgt_async_info *AsyncInfoPtr) {
  // Send data to SrcRank
  RequestManager.send(&OrgBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&DstRank, 1, MPI_INT);
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  // Send data to DstRank
  RequestManager.OtherRank = DstRank;
  RequestManager.send(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&SrcRank, 1, MPI_INT);
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  RequestManager.OtherRank = SrcRank;
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy launchKernel(MPIRequestManagerTy RequestManager, void *TgtEntryPtr,
                     EventDataHandleTy TgtArgs, EventDataHandleTy TgtOffsets,
                     EventDataHandleTy KernelArgsHandle,
                     __tgt_async_info *AsyncInfoPtr) {
  KernelArgsTy *KernelArgs =
      static_cast<KernelArgsTy *>(KernelArgsHandle.get());

  RequestManager.send(&KernelArgs->NumArgs, 1, MPI_UINT32_T);
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&TgtEntryPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(TgtArgs.get(), KernelArgs->NumArgs * sizeof(void *),
                      MPI_BYTE);
  RequestManager.send(TgtOffsets.get(), KernelArgs->NumArgs * sizeof(ptrdiff_t),
                      MPI_BYTE);

  RequestManager.send(KernelArgs, sizeof(KernelArgsTy), MPI_BYTE);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy getGlobal(MPIRequestManagerTy RequestManager,
                  __tgt_device_binary Binary, uint64_t Size, const char *Name,
                  void **DevicePtr) {
  uint32_t NameSize = strlen(Name) + 1;
  RequestManager.send(&Binary.handle, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_UINT64_T);
  RequestManager.send(&NameSize, 1, MPI_UINT32_T);
  RequestManager.send(Name, NameSize, MPI_CHAR);

  RequestManager.receive(DevicePtr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy getFunction(MPIRequestManagerTy RequestManager,
                    __tgt_device_binary Binary, const char *Name,
                    void **KernelPtr) {
  RequestManager.send(&Binary.handle, sizeof(void *), MPI_BYTE);
  uint32_t Size = strlen(Name) + 1;
  RequestManager.send(&Size, 1, MPI_UINT32_T);
  RequestManager.send(Name, Size, MPI_CHAR);

  RequestManager.receive(KernelPtr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy synchronize(MPIRequestManagerTy RequestManager,
                    __tgt_async_info *AsyncInfoPtr) {
  bool DeviceOpStatus = false;
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  RequestManager.receive(&DeviceOpStatus, sizeof(bool), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  if (!DeviceOpStatus)
    co_return createError("Failed to synchronize device.");

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy sync(EventTy Event) {
  while (!Event.done())
    co_await std::suspend_always{};

  co_return llvm::Error::success();
}

EventTy loadBinary(MPIRequestManagerTy RequestManager,
                   const __tgt_device_image *Image,
                   __tgt_device_binary *Binary) {
  auto &[ImageStart, ImageEnd, EntriesBegin, EntriesEnd] = *Image;

  // Send the target table sizes.
  size_t ImageSize = (size_t)ImageEnd - (size_t)ImageStart;
  size_t EntryCount = EntriesEnd - EntriesBegin;
  llvm::SmallVector<size_t> EntryNameSizes(EntryCount);

  for (size_t I = 0; I < EntryCount; I++) {
    // Note: +1 for the terminator.
    EntryNameSizes[I] = std::strlen(EntriesBegin[I].name) + 1;
  }

  RequestManager.send(&ImageSize, 1, MPI_UINT64_T);
  RequestManager.send(&EntryCount, 1, MPI_UINT64_T);
  RequestManager.send(EntryNameSizes.begin(), EntryCount, MPI_UINT64_T);

  void *Buffer = memAllocHost(ImageSize);
  if (Buffer != nullptr)
    memcpy(Buffer, ImageStart, ImageSize);
  else
    Buffer = ImageStart;

  // Send the image bytes and the table entries.
  RequestManager.send(Buffer, ImageSize, MPI_BYTE);

  for (size_t I = 0; I < EntryCount; I++) {
    RequestManager.send(&EntriesBegin[I].addr, 1, MPI_UINT64_T);
    RequestManager.send(EntriesBegin[I].name, EntryNameSizes[I], MPI_CHAR);
    RequestManager.send(&EntriesBegin[I].size, 1, MPI_UINT64_T);
    RequestManager.send(&EntriesBegin[I].flags, 1, MPI_INT32_T);
    RequestManager.send(&EntriesBegin[I].data, 1, MPI_INT32_T);
  }

  if (auto Err = co_await RequestManager; Err)
    co_return Err;

  if (Buffer != ImageStart)
    memFreeHost(Buffer);

  RequestManager.receive(&Binary->handle, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy queryAsync(MPIRequestManagerTy RequestManager,
                   __tgt_async_info *AsyncInfoPtr) {
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  if (auto Err = co_await RequestManager; Err)
    co_return Err;

  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy printDeviceInfo(MPIRequestManagerTy RequestManager) {
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy initAsyncInfo(MPIRequestManagerTy RequestManager,
                      __tgt_async_info **AsyncInfoPtr) {
  RequestManager.receive(AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy initDeviceInfo(MPIRequestManagerTy RequestManager,
                       __tgt_device_info *DeviceInfo) {
  RequestManager.send(DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.receive(DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy dataLock(MPIRequestManagerTy RequestManager, void *Ptr, int64_t Size,
                 void **LockedPtr) {
  RequestManager.send(&Ptr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.receive(LockedPtr, sizeof(void *), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy dataUnlock(MPIRequestManagerTy RequestManager, void *Ptr) {
  RequestManager.send(&Ptr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy dataNotifyMapped(MPIRequestManagerTy RequestManager, void *HstPtr,
                         int64_t Size) {
  RequestManager.send(&HstPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy dataNotifyUnmapped(MPIRequestManagerTy RequestManager, void *HstPtr) {
  RequestManager.send(&HstPtr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy exit(MPIRequestManagerTy RequestManager) {
  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

} // namespace OriginEvents

/// Event Queue implementation
EventQueue::EventQueue() : Queue(), QueueMtx(), CanPopCv() {}

size_t EventQueue::size() {
  std::lock_guard<std::mutex> Lock(QueueMtx);
  return Queue.size();
}

void EventQueue::push(EventTy &&Event) {
  {
    std::unique_lock<std::mutex> Lock(QueueMtx);
    Queue.emplace(Event);
  }

  // Notifies a thread possibly blocked by an empty queue.
  CanPopCv.notify_one();
}

EventTy EventQueue::pop(std::stop_token &Stop) {
  std::unique_lock<std::mutex> Lock(QueueMtx);

  // Waits for at least one item to be pushed.
  const bool HasNewEvent =
      CanPopCv.wait(Lock, Stop, [&] { return !Queue.empty(); });

  if (!HasNewEvent) {
    assert(Stop.stop_requested() && "Queue was empty while running.");
    return EventTy();
  }

  EventTy TargetEvent = std::move(Queue.front());
  Queue.pop();
  return TargetEvent;
}

/// Event System implementation
EventSystemTy::EventSystemTy()
    : EventSystemState(EventSystemStateTy::CREATED),
      NumMPIComms("OMPTARGET_NUM_MPI_COMMS", 10) {}

EventSystemTy::~EventSystemTy() {
  if (EventSystemState == EventSystemStateTy::FINALIZED)
    return;

  REPORT("Destructing internal event system before deinitializing it.\n");
  deinitialize();
}

bool EventSystemTy::initialize() {
  if (EventSystemState >= EventSystemStateTy::INITIALIZED) {
    REPORT("Trying to initialize event system twice.\n");
    return false;
  }

  if (!createLocalMPIContext())
    return false;

  EventSystemState = EventSystemStateTy::INITIALIZED;

  return true;
}

bool EventSystemTy::is_initialized() {
  return EventSystemState == EventSystemStateTy::INITIALIZED;
}

bool EventSystemTy::deinitialize() {
  if (EventSystemState == EventSystemStateTy::FINALIZED) {
    REPORT("Trying to deinitialize event system twice.\n");
    return false;
  }

  if (EventSystemState == EventSystemStateTy::RUNNING) {
    REPORT("Trying to deinitialize event system while it is running.\n");
    return false;
  }

  // Only send exit events from the origin side.
  if (isHost() && WorldSize > 1) {
    const int NumWorkers = numWorkerRanks();
    llvm::SmallVector<EventTy> ExitEvents(NumWorkers);
    for (int WorkerRank = 0; WorkerRank < NumWorkers; WorkerRank++) {
      ExitEvents[WorkerRank] =
          createEvent(OriginEvents::exit, EventTypeTy::EXIT, WorkerRank);
      ExitEvents[WorkerRank].resume();
    }

    bool SuccessfullyExited = true;
    for (int WorkerRank = 0; WorkerRank < NumWorkers; WorkerRank++) {
      ExitEvents[WorkerRank].wait();
      SuccessfullyExited &= ExitEvents[WorkerRank].done();
      auto Error = ExitEvents[WorkerRank].getError();
      if (Error)
        REPORT("Exit event failed with msg: %s\n",
               toString(std::move(Error)).data());
    }

    if (!SuccessfullyExited) {
      REPORT("Failed to stop worker processes.\n");
      return false;
    }
  }

  if (!destroyLocalMPIContext())
    return false;

  EventSystemState = EventSystemStateTy::FINALIZED;

  return true;
}

EventTy EventSystemTy::createExchangeEvent(int SrcDevice, const void *SrcBuffer,
                                           int DstDevice, void *DstBuffer,
                                           int64_t Size,
                                           __tgt_async_info *AsyncInfo) {
  const int EventTag = createNewEventTag();
  auto &EventComm = getNewEventComm(EventTag, LocalRank);
  const int MPITag = getEventMPITag(EventTag, LocalRank);

  int32_t SrcRank, SrcDeviceId, DstRank, DstDeviceId;

  std::tie(SrcRank, SrcDeviceId) = mapDeviceId(SrcDevice);
  std::tie(DstRank, DstDeviceId) = mapDeviceId(DstDevice);

  int SrcEventNotificationInfo[] = {static_cast<int>(EventTypeTy::EXCHANGE_SRC),
                                    EventTag, SrcDeviceId};

  int DstEventNotificationInfo[] = {static_cast<int>(EventTypeTy::EXCHANGE_DST),
                                    EventTag, DstDeviceId};

  MPI_Request SrcRequest = MPI_REQUEST_NULL;
  MPI_Request DstRequest = MPI_REQUEST_NULL;

  int MPIError = ompfileWithMPICallLock([&]() {
    int Rc = MPI_Isend(SrcEventNotificationInfo, 3, MPI_INT, SrcRank,
                       static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                       GateThreadComm, &SrcRequest);
    Rc &= MPI_Isend(DstEventNotificationInfo, 3, MPI_INT, DstRank,
                    static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                    GateThreadComm, &DstRequest);
    return Rc;
  });

  if (MPIError != MPI_SUCCESS)
    co_return createError(
        "MPI failed during exchange event notification with error %d",
        MPIError);

  MPIRequestManagerTy RequestManager(EventComm, MPITag, SrcRank, SrcDeviceId,
                                     {SrcRequest, DstRequest});

  co_return (co_await OriginEvents::exchange(std::move(RequestManager), SrcRank,
                                             SrcBuffer, DstRank, DstBuffer,
                                             Size, AsyncInfo));
}

/// Creates a new event tag of at least 'FIRST_EVENT' value.
/// Tag values smaller than 'FIRST_EVENT' are reserved for control
/// messages between the event systems of different MPI processes.
int EventSystemTy::createNewEventTag() {
  return EventCounter.fetch_add(1);
}

// Disambiguate events created by different processes: every process draws
// event ids from its own local counter, so the raw id sequences coincide.
// Interleaving the origin rank spreads each origin onto a disjoint logical id
// sequence, which keeps concurrent i->j and j->i events on different MPI tags.
static uint64_t logicalEventId(int EventId, int OriginRank, int WorldSize) {
  const uint64_t Span =
      WorldSize > 0 ? static_cast<uint64_t>(WorldSize) : uint64_t{1};
  const uint64_t Origin =
      OriginRank >= 0 ? static_cast<uint64_t>(OriginRank) % Span : uint64_t{0};
  return static_cast<uint64_t>(static_cast<unsigned int>(EventId)) * Span +
         Origin;
}

int EventSystemTy::getEventMPITag(int EventId, int OriginRank) const {
  constexpr int EventTagStride = 8;
  const int FirstEventTag = static_cast<int>(ControlTagsTy::FIRST_EVENT);
  if (MPITagMaxValue < FirstEventTag + EventTagStride) {
    REPORT("Invalid MPI tag upper bound (%d); using fallback tag %d.\n",
           MPITagMaxValue, FirstEventTag);
    return FirstEventTag;
  }

  const int TagSpan = MPITagMaxValue - FirstEventTag + 1;
  const int BaseSpan = std::max(1, TagSpan - EventTagStride + 1);
  const int Offset = static_cast<int>(
      (logicalEventId(EventId, OriginRank, WorldSize) * EventTagStride) %
      static_cast<uint64_t>(BaseSpan));
  return FirstEventTag + Offset;
}

MPI_Comm &EventSystemTy::getNewEventComm(int EventId, int OriginRank) {
  if (EventCommPool.empty()) {
    REPORT("Event communicator pool is empty; falling back to GateThreadComm.\n");
    assert(GateThreadComm != MPI_COMM_NULL &&
           "GateThreadComm must be valid when EventCommPool is empty");
    return GateThreadComm;
  }

  constexpr int EventTagStride = 8;
  const int FirstEventTag = static_cast<int>(ControlTagsTy::FIRST_EVENT);
  const int TagSpan = MPITagMaxValue >= FirstEventTag + EventTagStride
                          ? MPITagMaxValue - FirstEventTag + 1
                          : 1;
  const int BaseSpan = std::max(1, TagSpan - EventTagStride + 1);
  const uint64_t LogicalTagOffset =
      logicalEventId(EventId, OriginRank, WorldSize) * EventTagStride;
  const size_t CommIndex =
      (LogicalTagOffset / static_cast<uint64_t>(BaseSpan)) %
      EventCommPool.size();
  return EventCommPool[CommIndex];
}

static const char *threadLevelToString(int ThreadLevel) {
  switch (ThreadLevel) {
  case MPI_THREAD_SINGLE:
    return "MPI_THREAD_SINGLE";
  case MPI_THREAD_SERIALIZED:
    return "MPI_THREAD_SERIALIZED";
  case MPI_THREAD_FUNNELED:
    return "MPI_THREAD_FUNNELED";
  case MPI_THREAD_MULTIPLE:
    return "MPI_THREAD_MULTIPLE";
  default:
    return "unkown";
  }
}

/// Initialize the MPI context.
bool EventSystemTy::createLocalMPIContext() {
  int MPIError = MPI_SUCCESS;
  int IsMPIInitialized = 0;
  int ThreadLevel = MPI_THREAD_SINGLE;

  MPI_Initialized(&IsMPIInitialized);

  if (IsMPIInitialized)
    MPI_Query_thread(&ThreadLevel);
  else {
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &ThreadLevel);
    OwnsMPIInitialization = true;
  }

  CHECK(ThreadLevel == MPI_THREAD_MULTIPLE,
        "MPI plugin requires a MPI implementation with %s thread level. "
        "Implementation only supports up to %s.\n",
        threadLevelToString(MPI_THREAD_MULTIPLE),
        threadLevelToString(ThreadLevel));

  // Create gate thread comm.
  MPIError = MPI_Comm_dup(MPI_COMM_WORLD, &GateThreadComm);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to create gate thread MPI comm with error %d\n", MPIError);
  configureCommErrhandlerForDebug(GateThreadComm, "GateThreadComm");

  // Create event comm pool.
  int64_t NumComms = NumMPIComms.get();
  if (NumComms <= 0) {
    REPORT("Invalid OMPTARGET_NUM_MPI_COMMS=%lld; forcing 1.\n",
           static_cast<long long>(NumComms));
    NumComms = 1;
  }
  EventCommPool.resize(NumComms, MPI_COMM_NULL);
  assert(!EventCommPool.empty() && "Event communicator pool must be non-empty");
  for (auto &Comm : EventCommPool) {
    MPIError = MPI_Comm_dup(MPI_COMM_WORLD, &Comm);
    CHECK(MPIError == MPI_SUCCESS,
          "Failed to create MPI comm pool with error %d\n", MPIError);
    configureCommErrhandlerForDebug(Comm, "EventCommPool");
  }

  // Get local MPI process description.
  MPIError = MPI_Comm_rank(GateThreadComm, &LocalRank);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to acquire the local MPI rank with error %d\n", MPIError);

  MPIError = MPI_Comm_size(GateThreadComm, &WorldSize);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to acquire the world size with error %d\n", MPIError);

  // Get max value for MPI tags.
  MPI_Aint *Value = nullptr;
  int Flag = 0;
  MPIError = MPI_Comm_get_attr(GateThreadComm, MPI_TAG_UB, &Value, &Flag);
  CHECK(Flag == 1 && MPIError == MPI_SUCCESS,
        "Failed to acquire the MPI max tag value with error %d\n", MPIError);
  MPITagMaxValue = *Value;

  return true;
}

/// Destroy the MPI context.
bool EventSystemTy::destroyLocalMPIContext() {
  int MPIError = MPI_SUCCESS;

  if (GateThreadComm == MPI_COMM_NULL) {
    return true;
  }

  // Note: We don't need to assert here since application part of the program
  // was finished.
  // Free gate thread comm.
  MPIError = MPI_Comm_free(&GateThreadComm);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to destroy the gate thread MPI comm with error %d\n", MPIError);

  // Free event comm pool.
  for (auto &comm : EventCommPool) {
    MPI_Comm_free(&comm);
    CHECK(MPIError == MPI_SUCCESS,
          "Failed to destroy the event MPI comm with error %d\n", MPIError);
  }
  EventCommPool.resize(0);

  // Finalize the global MPI session.
  if (OwnsMPIInitialization) {
    int IsFinalized = false;
    MPIError = MPI_Finalized(&IsFinalized);

    if (IsFinalized) {
      DP("MPI was already finalized externally.\n");
    } else {
      MPIError = MPI_Finalize();
      CHECK(MPIError == MPI_SUCCESS, "Failed to finalize MPI with error: %d\n",
            MPIError);
    }
  }
  OwnsMPIInitialization = false;

  return true;
}

int EventSystemTy::getNumWorkers() const {
  if (isHost())
    return numWorkerRanks();
  return 0;
}

int EventSystemTy::isHost() const { return LocalRank == WorldSize - 1; };

/// Map DeviceId to the pair <RemoteRank, RemoteDeviceId>
RemoteDeviceId EventSystemTy::mapDeviceId(int32_t DeviceId) {
  int32_t NumRanks = DevicesPerRemote.size();
  for (int32_t RemoteRank = 0; RemoteRank < NumRanks; RemoteRank++) {
    if (DeviceId < DevicesPerRemote[RemoteRank])
      return {RemoteRank, DeviceId};
    DeviceId -= DevicesPerRemote[RemoteRank];
  }
  return {-1, -1};
}
