//===-- EventSystemOmpFile.cpp - OMPFile events over MPI --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The libompfile side of the MPI event system: the OMPFILE_* event names,
// the blocking MPIRequestManagerTy helpers that only the file-I/O paths
// use, and the origin-side OMPFILE_* coroutines. Part of the EventSystem
// object library, next to EventSystem.cpp.
//
//===----------------------------------------------------------------------===//

#include "EventSystemOmpFile.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "llvm/Support/Error.h"

const char *ompfileEventTypeName(EventTypeTy eventType) {
  switch (eventType) {
    case EventTypeTy::OMPFILE_PING: return "OMPFILE_PING";
    case EventTypeTy::OMPFILE_OPEN: return "OMPFILE_OPEN";
    case EventTypeTy::OMPFILE_CLOSE: return "OMPFILE_CLOSE";
    case EventTypeTy::OMPFILE_PREAD: return "OMPFILE_PREAD";
    case EventTypeTy::OMPFILE_PWRITE: return "OMPFILE_PWRITE";
    case EventTypeTy::OMPFILE_SCHED_REQUEST: return "OMPFILE_SCHED_REQUEST";
    case EventTypeTy::OMPFILE_SCHED_REQUEST_BATCH:
      return "OMPFILE_SCHED_REQUEST_BATCH";
    case EventTypeTy::OMPFILE_PREAD_NO_STAGE: return "OMPFILE_PREAD_NO_STAGE";
    case EventTypeTy::OMPFILE_FRESHNESS_QUERY:
      return "OMPFILE_FRESHNESS_QUERY";
    case EventTypeTy::OMPFILE_FRESHNESS_WRITE_COMMIT:
      return "OMPFILE_FRESHNESS_WRITE_COMMIT";
    case EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE:
      return "OMPFILE_FLUSH_DIRTY_TILE";
    case EventTypeTy::OMPFILE_DIRTY_OWNER_PREAD:
      return "OMPFILE_DIRTY_OWNER_PREAD";
    case EventTypeTy::OMPFILE_DIRTY_OWNER_QUERY:
      return "OMPFILE_DIRTY_OWNER_QUERY";
    case EventTypeTy::OMPFILE_DIRTY_OWNER_PREAD_BATCH:
      return "OMPFILE_DIRTY_OWNER_PREAD_BATCH";
    default: return "UNKNOWN_EVENT_TYPE";
  }
}

// Deliberately NOT under ompfileWithMPICallLock: a blocking MPI_Send of a
// rendezvous-sized chunk does not return until the peer posts its receive, and
// the peer's receive may be waiting on this process's lock (see the invariant
// on ompfileWithMPICallLock). Safe without it: MPI_THREAD_MULTIPLE is
// negotiated, this event owns its (communicator, tag), and the request manager
// is move-only per-event state, so there is nothing shared to protect.
llvm::Error MPIRequestManagerTy::sendBlocking(const void *Buffer, int Size,
                                              MPI_Datatype Datatype) {
  ompfile::trace::Scope Trace(ompfile::trace::Domain::MPP, "mpi-blocking-send",
                              ompfile::trace::Color::Wait,
                              static_cast<uint64_t>(Size));
  int MPIError = MPI_Send(const_cast<void *>(Buffer), Size, Datatype, OtherRank,
                          Tag, Comm);
  if (MPIError != MPI_SUCCESS)
    return createError("Blocking MPI send failed with code %d", MPIError);
  return llvm::Error::success();
}

// Not under ompfileWithMPICallLock, for the same reason as sendBlocking.
llvm::Error MPIRequestManagerTy::receiveBlocking(void *Buffer, int Size,
                                                 MPI_Datatype Datatype) {
  ompfile::trace::Scope Trace(ompfile::trace::Domain::MPP, "mpi-blocking-recv",
                              ompfile::trace::Color::Wait,
                              static_cast<uint64_t>(Size));
  int MPIError = MPI_Recv(Buffer, Size, Datatype, OtherRank, Tag, Comm,
                          MPI_STATUS_IGNORE);
  if (MPIError != MPI_SUCCESS)
    return createError("Blocking MPI receive failed with code %d", MPIError);
  return llvm::Error::success();
}

llvm::Error MPIRequestManagerTy::sendInBatchsBlocking(const void *Buffer,
                                                      uint64_t Size) {
  if (Size <= 0)
    return llvm::Error::success();
  if (!Buffer)
    return createError("Blocking batch send missing source buffer.");

  const int64_t FragmentSize = MPIFragmentSize.get();
  if (FragmentSize <= 0)
    return createError("Invalid MPI fragment size %lld for blocking send.",
                       static_cast<long long>(FragmentSize));

  constexpr int64_t MaxMPIChunk =
      static_cast<int64_t>(std::numeric_limits<int>::max());

  const char *BufferByteArray = reinterpret_cast<const char *>(Buffer);
  uint64_t RemainingBytes = Size;
  uint64_t Offset = 0;
  while (RemainingBytes > 0) {
    const int64_t Chunk = std::min(
        {static_cast<int64_t>(std::min<uint64_t>(
             RemainingBytes, static_cast<uint64_t>(MaxMPIChunk))),
         FragmentSize, MaxMPIChunk});
    assert(Chunk > 0 && "MPI fragment chunk must be positive.");
    if (auto Error = sendBlocking(&BufferByteArray[Offset],
                                  static_cast<int>(Chunk), MPI_BYTE))
      return Error;
    Offset += Chunk;
    RemainingBytes -= Chunk;
  }

  return llvm::Error::success();
}

llvm::Error MPIRequestManagerTy::receiveInBatchsBlocking(void *Buffer,
                                                         int64_t Size) {
  if (Size <= 0)
    return llvm::Error::success();
  if (!Buffer)
    return createError("Blocking batch receive missing destination buffer.");

  const int64_t FragmentSize = MPIFragmentSize.get();
  if (FragmentSize <= 0)
    return createError("Invalid MPI fragment size %lld for blocking receive.",
                       static_cast<long long>(FragmentSize));

  constexpr int64_t MaxMPIChunk =
      static_cast<int64_t>(std::numeric_limits<int>::max());

  char *BufferByteArray = reinterpret_cast<char *>(Buffer);
  int64_t RemainingBytes = Size;
  int64_t Offset = 0;
  while (RemainingBytes > 0) {
    const int64_t Chunk = std::min({RemainingBytes, FragmentSize, MaxMPIChunk});
    assert(Chunk > 0 && "MPI fragment chunk must be positive.");
    // Not under ompfileWithMPICallLock, for the same reason as sendBlocking.
    ompfile::trace::Scope Trace(ompfile::trace::Domain::MPP,
                                "mpi-blocking-recv", ompfile::trace::Color::Wait,
                                static_cast<uint64_t>(Chunk));
    int MPIError =
        MPI_Recv(&BufferByteArray[Offset], static_cast<int>(Chunk), MPI_BYTE,
                 OtherRank, Tag, Comm, MPI_STATUS_IGNORE);
    if (MPIError != MPI_SUCCESS)
      return createError("Blocking MPI receive failed with code %d", MPIError);
    Offset += Chunk;
    RemainingBytes -= Chunk;
  }

  return llvm::Error::success();
}

/// OMPFILE_* event implementations on the origin side.
namespace OriginEvents {

EventTy ompfilePing(MPIRequestManagerTy RequestManager, uint64_t Token) {
  uint64_t Response = 0;

  RequestManager.send(&Token, 1, MPI_UINT64_T);
  RequestManager.receive(&Response, 1, MPI_UINT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  if (Response != Token)
    co_return createError("OMPFile ping response mismatch.");

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy ompfileOpen(MPIRequestManagerTy RequestManager, const char *Path,
                    int Flags, int Mode, int *RemoteHandle,
                    int *RemoteErrno) {
  uint32_t PathSize = std::strlen(Path) + 1;
  RequestManager.send(&PathSize, 1, MPI_UINT32_T);
  RequestManager.send(&Flags, 1, MPI_INT);
  RequestManager.send(&Mode, 1, MPI_INT);
  RequestManager.send(Path, PathSize, MPI_CHAR);

  RequestManager.receive(RemoteHandle, 1, MPI_INT);
  RequestManager.receive(RemoteErrno, 1, MPI_INT);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy ompfileClose(MPIRequestManagerTy RequestManager, int RemoteHandle,
                     int *CloseRet, int *RemoteErrno) {
  RequestManager.send(&RemoteHandle, 1, MPI_INT);
  RequestManager.receive(CloseRet, 1, MPI_INT);
  RequestManager.receive(RemoteErrno, 1, MPI_INT);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}


EventTy ompfilePread(MPIRequestManagerTy RequestManager, int RemoteHandle,
                     int64_t Offset, void *Buffer, uint64_t Size,
                     int *IoRet, int *RemoteErrno, uint64_t *Bytes) {
  RequestManager.send(&RemoteHandle, 1, MPI_INT);
  RequestManager.send(&Offset, 1, MPI_INT64_T);
  RequestManager.send(&Size, 1, MPI_UINT64_T);

  RequestManager.receive(IoRet, 1, MPI_INT);
  RequestManager.receive(RemoteErrno, 1, MPI_INT);
  RequestManager.receive(Bytes, 1, MPI_UINT64_T);


  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Fail closed if the announced payload exceeds the caller's buffer (mirrors
  // the dirty-owner pread guard): posting a recv larger than Buffer turns a
  // corrupted/aliased size into memory corruption or an MPI truncation abort.
  if (*Bytes > Size)
    co_return createError("Pread reply payload too large: %llu > %llu",
                          static_cast<unsigned long long>(*Bytes),
                          static_cast<unsigned long long>(Size));

  if (*Bytes > 0) {
    RequestManager.receiveInBatchs(Buffer, *Bytes);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;
  }

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy ompfileDirtyOwnerPread(MPIRequestManagerTy RequestManager,
                               int RemoteHandle, int64_t Offset,
                               void *Buffer, uint64_t Size,
                               uint64_t ExpectedVersion, int *IoRet,
                               int *RemoteErrno, uint64_t *Bytes) {
  RequestManager.send(&RemoteHandle, 1, MPI_INT);
  RequestManager.send(&Offset, 1, MPI_INT64_T);
  RequestManager.send(&Size, 1, MPI_UINT64_T);
  RequestManager.send(&ExpectedVersion, 1, MPI_UINT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  OmpFileDirtyOwnerPreadReplyFrame Reply{};
  RequestManager.receiveTagged(&Reply, sizeof(Reply), MPI_BYTE,
                               /*TagOffset=*/1);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;
  if (Reply.AbiVersion != OMPFILE_DIRTY_OWNER_PREAD_ABI_VERSION ||
      Reply.Magic != OMPFILE_DIRTY_OWNER_PREAD_REPLY_MAGIC ||
      Reply.Offset != static_cast<uint64_t>(Offset) ||
      Reply.ExpectedVersion != ExpectedVersion) {
    co_return createError("Invalid dirty-owner pread reply frame: abi=%u "
                          "magic=%u offset=%llu expected_offset=%llu "
                          "version=%llu expected_version=%llu",
                          Reply.AbiVersion, Reply.Magic,
                          static_cast<unsigned long long>(Reply.Offset),
                          static_cast<unsigned long long>(Offset),
                          static_cast<unsigned long long>(Reply.ExpectedVersion),
                          static_cast<unsigned long long>(ExpectedVersion));
  }

  *IoRet = Reply.Ret;
  *RemoteErrno = Reply.Errno;
  *Bytes = Reply.PayloadBytes;
  if (*Bytes > Size)
    co_return createError("Dirty-owner pread reply payload too large: %llu > %llu",
                          static_cast<unsigned long long>(*Bytes),
                          static_cast<unsigned long long>(Size));

  if (*Bytes > 0) {
    RequestManager.receiveInBatchsTagged(Buffer, *Bytes, /*TagOffset=*/2);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;
  }

  RequestManager.receiveTagged(nullptr, 0, MPI_BYTE, /*TagOffset=*/3);
  co_return (co_await RequestManager);
}

EventTy ompfileDirtyOwnerPreadBatch(
    MPIRequestManagerTy RequestManager, int RemoteHandle,
    const OmpFileDirtyOwnerPreadBatchSegment *Segments, uint64_t SegmentCount,
    void *const *Buffers, int *IoRets, int *RemoteErrnos, uint64_t *Bytes) {
  if (!Segments || !Buffers || !IoRets || !RemoteErrnos || !Bytes)
    co_return createError("OMPFile dirty-owner batch missing buffers.");
  if (SegmentCount == 0)
    co_return createError("OMPFile dirty-owner batch has no segments.");

  RequestManager.send(&RemoteHandle, 1, MPI_INT);
  RequestManager.send(&SegmentCount, 1, MPI_UINT64_T);
  RequestManager.send(Segments,
                      SegmentCount * sizeof(OmpFileDirtyOwnerPreadBatchSegment),
                      MPI_BYTE);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  OmpFileDirtyOwnerPreadBatchReplyHeader Header{};
  RequestManager.receiveTagged(&Header, sizeof(Header), MPI_BYTE,
                               /*TagOffset=*/1);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;
  if (Header.AbiVersion != OMPFILE_DIRTY_OWNER_PREAD_ABI_VERSION ||
      Header.Magic != OMPFILE_DIRTY_OWNER_PREAD_BATCH_REPLY_MAGIC ||
      Header.SegmentCount != SegmentCount)
    co_return createError("Invalid dirty-owner batch header: abi=%u magic=%u "
                          "segments=%llu expected=%llu",
                          Header.AbiVersion, Header.Magic,
                          static_cast<unsigned long long>(Header.SegmentCount),
                          static_cast<unsigned long long>(SegmentCount));

  std::vector<OmpFileDirtyOwnerPreadBatchReplySegment> Replies(SegmentCount);
  RequestManager.receiveTagged(Replies.data(),
                               Replies.size() * sizeof(Replies.front()),
                               MPI_BYTE, /*TagOffset=*/2);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  std::vector<bool> SeenSegment(SegmentCount, false);
  for (uint64_t I = 0; I < SegmentCount; ++I) {
    const auto &Reply = Replies[I];
    if (Reply.ClientSegmentId >= SegmentCount)
      co_return createError("Invalid dirty-owner batch segment id: %llu >= %llu",
                            static_cast<unsigned long long>(
                                Reply.ClientSegmentId),
                            static_cast<unsigned long long>(SegmentCount));
    if (SeenSegment[Reply.ClientSegmentId])
      co_return createError("Duplicate dirty-owner batch segment id: %llu",
                            static_cast<unsigned long long>(
                                Reply.ClientSegmentId));
    SeenSegment[Reply.ClientSegmentId] = true;
    const auto &Segment = Segments[Reply.ClientSegmentId];
    if (Reply.Offset != static_cast<uint64_t>(Segment.Offset) ||
        Reply.ExpectedVersion != Segment.ExpectedVersion ||
        Reply.Bytes > Segment.Size)
      co_return createError("Invalid dirty-owner batch segment reply: id=%llu "
                            "offset=%llu expected_offset=%llu bytes=%llu "
                            "size=%llu version=%llu expected_version=%llu",
                            static_cast<unsigned long long>(
                                Reply.ClientSegmentId),
                            static_cast<unsigned long long>(Reply.Offset),
                            static_cast<unsigned long long>(Segment.Offset),
                            static_cast<unsigned long long>(Reply.Bytes),
                            static_cast<unsigned long long>(Segment.Size),
                            static_cast<unsigned long long>(
                                Reply.ExpectedVersion),
                            static_cast<unsigned long long>(
                                Segment.ExpectedVersion));
    IoRets[Reply.ClientSegmentId] = Reply.Ret;
    RemoteErrnos[Reply.ClientSegmentId] = Reply.Errno;
    Bytes[Reply.ClientSegmentId] = Reply.Bytes;
  }

  // Payloads share TagOffset 3 and are matched purely by send/receive order,
  // so consume them in the proxy's REPLY order (which is how it sends them),
  // not in ascending client segment order — the two differ whenever the proxy
  // reorders segments, and a mismatch desyncs sizes vs buffers on the shared
  // tag stream (MPI truncation or silent corruption).
  for (uint64_t I = 0; I < SegmentCount; ++I) {
    const uint64_t SegId = Replies[I].ClientSegmentId;
    if (Bytes[SegId] == 0)
      continue;
    RequestManager.receiveInBatchsTagged(Buffers[SegId], Bytes[SegId],
                                         /*TagOffset=*/3);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;
  }

  RequestManager.receiveTagged(nullptr, 0, MPI_BYTE, /*TagOffset=*/4);
  co_return (co_await RequestManager);
}

EventTy ompfileDirtyOwnerQuery(MPIRequestManagerTy RequestManager,
                               int RemoteHandle, int64_t Offset,
                               uint64_t Size, uint64_t ExpectedVersion,
                               int *IoRet, int *RemoteErrno, int *State) {
  RequestManager.send(&RemoteHandle, 1, MPI_INT);
  RequestManager.send(&Offset, 1, MPI_INT64_T);
  RequestManager.send(&Size, 1, MPI_UINT64_T);
  RequestManager.send(&ExpectedVersion, 1, MPI_UINT64_T);

  RequestManager.receive(IoRet, 1, MPI_INT);
  RequestManager.receive(RemoteErrno, 1, MPI_INT);
  RequestManager.receive(State, 1, MPI_INT);

  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy ompfilePwrite(MPIRequestManagerTy RequestManager, int RemoteHandle,
                      int64_t Offset, const void *Buffer, uint64_t Size,
                      int *IoRet, int *RemoteErrno, uint64_t *Bytes) {
  if (!IoRet || !RemoteErrno || !Bytes)
    co_return createError("OMPFile pwrite missing output buffers.");
  if (Size > 0 && !Buffer)
    co_return createError("OMPFile pwrite missing payload buffer.");
  *Bytes = 0;

  RequestManager.send(&RemoteHandle, 1, MPI_INT);
  RequestManager.send(&Offset, 1, MPI_INT64_T);
  RequestManager.send(&Size, 1, MPI_UINT64_T);

  // Complete local ownership of the control sends before the payload. The
  // proxy receives the control header before posting its payload receive, so
  // the payload never shares a wait with the header.
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  if (Size > 0) {
    if (auto Error = RequestManager.sendInBatchsBlocking(Buffer, Size))
      co_return Error;
  }

  RequestManager.receive(IoRet, 1, MPI_INT);
  RequestManager.receive(RemoteErrno, 1, MPI_INT);
  RequestManager.receive(Bytes, 1, MPI_UINT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy ompfileFreshnessQuery(MPIRequestManagerTy RequestManager,
                              const OmpFileFreshnessQueryRequest *Request,
                              OmpFileFreshnessQueryReply *Reply) {
  if (!Request || !Reply)
    co_return createError("OMPFile freshness query missing buffers.");

  RequestManager.send(Request, sizeof(*Request), MPI_BYTE);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.receive(Reply, sizeof(*Reply), MPI_BYTE);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy ompfileFreshnessWriteCommit(
    MPIRequestManagerTy RequestManager,
    const OmpFileFreshnessWriteCommitRequest *Request,
    OmpFileFreshnessWriteCommitReply *Reply) {
  if (!Request || !Reply)
    co_return createError("OMPFile freshness write commit missing buffers.");

  RequestManager.send(Request, sizeof(*Request), MPI_BYTE);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.receive(Reply, sizeof(*Reply), MPI_BYTE);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy ompfileFlushDirtyTile(MPIRequestManagerTy RequestManager,
                              const OmpFileFlushDirtyTileRequest *Request,
                              OmpFileFlushDirtyTileReply *Reply) {
  if (!Request || !Reply)
    co_return createError("OMPFile flush-dirty-tile missing buffers.");

  RequestManager.send(Request, sizeof(*Request), MPI_BYTE);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.receive(Reply, sizeof(*Reply), MPI_BYTE);
  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy ompfileSchedRequest(MPIRequestManagerTy RequestManager,
                            const OmpFileIORequest *Request,
                            const char *Path, OmpFileIOPlan *Plan) {
  if (!Request || !Plan)
    co_return createError("OMPFile sched request missing buffers.");

  if (Request->PathSize > 0 && !Path)
    co_return createError("OMPFile sched request missing path.");

  RequestManager.send(Request, sizeof(*Request), MPI_BYTE);
  if (Request->PathSize > 0)
    RequestManager.send(Path, Request->PathSize, MPI_CHAR);

  RequestManager.receive(Plan, sizeof(*Plan), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy ompfileSchedBatchRequest(MPIRequestManagerTy RequestManager,
                                 const OmpFileIOBatchRequest *Request,
                                 const void *RequestPayload,
                                 uint64_t RequestPayloadBytes,
                                 OmpFileIOBatchPlan *Plan, void *PlanPayload,
                                 uint64_t PlanPayloadCapBytes,
                                 uint64_t *PlanPayloadOutBytes) {
  if (!Request || !Plan || !PlanPayloadOutBytes)
    co_return createError("OMPFile sched batch request missing buffers.");
  if (RequestPayloadBytes > 0 && !RequestPayload)
    co_return createError("OMPFile sched batch request missing payload.");

  RequestManager.send(Request, sizeof(*Request), MPI_BYTE);
  RequestManager.send(&RequestPayloadBytes, 1, MPI_UINT64_T);
  if (RequestPayloadBytes > 0)
    RequestManager.sendInBatchs(const_cast<void *>(RequestPayload),
                                RequestPayloadBytes);

  RequestManager.receive(Plan, sizeof(*Plan), MPI_BYTE);
  RequestManager.receive(PlanPayloadOutBytes, 1, MPI_UINT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  if (*PlanPayloadOutBytes > PlanPayloadCapBytes) {
    std::vector<uint8_t> Scratch(static_cast<size_t>(*PlanPayloadOutBytes));
    if (*PlanPayloadOutBytes > 0)
      RequestManager.receiveInBatchs(Scratch.data(), *PlanPayloadOutBytes);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;
    RequestManager.receive(nullptr, 0, MPI_BYTE);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;
    co_return createError("OMPFile sched batch response payload too large.");
  }

  if (*PlanPayloadOutBytes > 0) {
    if (!PlanPayload)
      co_return createError("OMPFile sched batch response missing payload "
                            "buffer.");
    RequestManager.receiveInBatchs(PlanPayload, *PlanPayloadOutBytes);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;
  }

  // Event completion notification.
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

} // namespace OriginEvents
