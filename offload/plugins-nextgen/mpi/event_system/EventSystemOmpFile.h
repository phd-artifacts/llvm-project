//===-- EventSystemOmpFile.h - OMPFile events and wire types ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The libompfile side of the MPI event system: which event types are file
// I/O (isFileIoEvent), the OMPFILE_* wire types, ABI constants and layout
// checks shared by the origin plugin, the proxy and the headnode manager,
// and the origin-side OMPFILE_* coroutines. The EventTypeTy enum and its
// pinned wire values stay in EventSystem.h.
//
//===----------------------------------------------------------------------===//

#ifndef _MPI_PROXY_EVENT_SYSTEM_OMPFILE_H_
#define _MPI_PROXY_EVENT_SYSTEM_OMPFILE_H_

#include "EventSystem.h"

#include "ompfile_mpp_abi.h"
#include "ompfile_sched.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

/// True for the event types the proxy serves on its file-I/O queue
/// (EventSystemTy::IoEventQueue): file operations, the HEADNODE scheduler's
/// control traffic, and the write-back coherence events. The scheduler events
/// go with the I/O they plan, so plan-before-use ordering stays inside one
/// queue. This list is the one place that classification lives; a new
/// OMPFILE_* event must be added here or it lands on the data queue.
constexpr bool isFileIoEvent(EventTypeTy Type) {
  switch (Type) {
  case EventTypeTy::OMPFILE_PING:
  case EventTypeTy::OMPFILE_OPEN:
  case EventTypeTy::OMPFILE_CLOSE:
  case EventTypeTy::OMPFILE_PREAD:
  case EventTypeTy::OMPFILE_PREAD_NO_STAGE:
  case EventTypeTy::OMPFILE_PWRITE:
  case EventTypeTy::OMPFILE_SCHED_REQUEST:
  case EventTypeTy::OMPFILE_SCHED_REQUEST_BATCH:
  case EventTypeTy::OMPFILE_FRESHNESS_QUERY:
  case EventTypeTy::OMPFILE_FRESHNESS_WRITE_COMMIT:
  case EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE:
  case EventTypeTy::OMPFILE_DIRTY_OWNER_PREAD:
  case EventTypeTy::OMPFILE_DIRTY_OWNER_QUERY:
  case EventTypeTy::OMPFILE_DIRTY_OWNER_PREAD_BATCH:
    return true;
  default:
    return false;
  }
}

using OmpFileFreshnessDecision = ompfile::OmpFileFreshnessDecision;
using OmpFileFreshnessQueryRequest = ompfile::OmpFileFreshnessQueryRequest;
using OmpFileFreshnessQueryReply = ompfile::OmpFileFreshnessQueryReply;

constexpr uint32_t OMPFILE_FRESHNESS_QUERY_ABI_VERSION =
    ompfile::OMPFILE_FRESHNESS_QUERY_ABI_VERSION;

struct OmpFileFreshnessWriteCommitRequest {
  uint32_t AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
  uint32_t WriteThroughMode = 0;
  uint64_t PathKey = 0;
  uint64_t TileId = 0;
  int32_t WriterRank = -1;
  int32_t Reserved0 = 0;
};

struct OmpFileFreshnessWriteCommitReply {
  uint32_t AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
  int32_t Status = 0;
  int32_t Errno = 0;
  int32_t Reserved0 = 0;
  uint64_t CommittedVersion = 0;
};

struct OmpFileFlushDirtyTileRequest {
  uint32_t AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
  uint32_t Action = 0; // 0=prepare, 1=complete
  uint64_t PathKey = 0;
  int32_t SourceRank = -1;
  int32_t Success = 0;
  uint64_t Version = 0;
};

struct OmpFileFlushDirtyTileReply {
  uint32_t AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
  int32_t Status = 0;
  int32_t Errno = 0;
  int32_t SourceRank = -1;
  uint64_t FlushedVersion = 0;
};

constexpr uint32_t OMPFILE_DIRTY_OWNER_PREAD_ABI_VERSION = 1u;
constexpr uint32_t OMPFILE_DIRTY_OWNER_PREAD_REPLY_MAGIC = 0x44525052u;

struct OmpFileDirtyOwnerPreadReplyFrame {
  uint32_t AbiVersion = OMPFILE_DIRTY_OWNER_PREAD_ABI_VERSION;
  uint32_t Magic = OMPFILE_DIRTY_OWNER_PREAD_REPLY_MAGIC;
  int32_t Ret = -1;
  int32_t Errno = 0;
  uint64_t PayloadBytes = 0;
  uint64_t ExpectedVersion = 0;
  uint64_t Offset = 0;
};

constexpr uint32_t OMPFILE_DIRTY_OWNER_PREAD_BATCH_REPLY_MAGIC = 0x44525042u;

// Shared with the shim through ompfile_mpp_abi.h (it is a bridge parameter).
using ompfile::OmpFileDirtyOwnerPreadBatchSegment;

struct OmpFileDirtyOwnerPreadBatchReplyHeader {
  uint32_t AbiVersion = OMPFILE_DIRTY_OWNER_PREAD_ABI_VERSION;
  uint32_t Magic = OMPFILE_DIRTY_OWNER_PREAD_BATCH_REPLY_MAGIC;
  uint64_t SegmentCount = 0;
  uint64_t PayloadBytes = 0;
};

struct OmpFileDirtyOwnerPreadBatchReplySegment {
  uint64_t ClientSegmentId = 0;
  int32_t Ret = -1;
  int32_t Errno = 0;
  uint64_t Bytes = 0;
  uint64_t PayloadOffset = 0;
  uint64_t ExpectedVersion = 0;
  uint64_t Offset = 0;
};

// The scheduler wire types are libompfile's (ompfile_sched.h). The plugin
// used to carry its own copies of these thirteen, and they had already
// drifted (OmpFileIOHint gained Role on the client side only). They are
// aliased into the global namespace so every existing unqualified use in the
// plugin keeps compiling against the one definition.
using ompfile::OmpFileIOOp;
using ompfile::OMPFILE_IO_HINT_ABI_VERSION;
using ompfile::OmpFileIOHintFlags;
using ompfile::OMPFILE_IO_HINT_HAS_EPOCH;
using ompfile::OMPFILE_IO_HINT_HAS_STREAM;
using ompfile::OMPFILE_IO_HINT_HAS_TILE;
using ompfile::OMPFILE_IO_HINT_HAS_ROLE;
using ompfile::OmpFileIOHint;
using ompfile::OmpFileIORequestFlags;
using ompfile::OMPFILE_IO_REQUEST_PLACEMENT_ONLY;
using ompfile::OmpFileIORequest;
using ompfile::OmpFileIOPlan;
using ompfile::OMPFILE_SCHED_BATCH_ABI_VERSION;
using ompfile::OmpFileBatchRequestFlags;
using ompfile::OMPFILE_BATCH_REQ_FAIL_ON_ANY_ERROR;
using ompfile::OMPFILE_BATCH_REQ_DISABLE_SCALAR_FALLBACK;
using ompfile::OmpFileBatchPlanFlags;
using ompfile::OMPFILE_BATCH_PLAN_BATCH_API;
using ompfile::OMPFILE_BATCH_PLAN_SCALAR_FALLBACK;
using ompfile::OMPFILE_BATCH_PLAN_FILE_AFFINITY;
using ompfile::OMPFILE_BATCH_PLAN_REBALANCED;
using ompfile::OmpFileIOBatchSegment;
using ompfile::OmpFileIOBatchRequest;
using ompfile::OmpFileIOBatchPlanEntry;
using ompfile::OmpFileIOBatchPlan;

static_assert(std::is_standard_layout_v<OmpFileIORequest>);
static_assert(std::is_standard_layout_v<OmpFileIOPlan>);
static_assert(std::is_standard_layout_v<OmpFileIOHint>);
static_assert(std::is_standard_layout_v<OmpFileIOBatchSegment>);
static_assert(std::is_standard_layout_v<OmpFileIOBatchRequest>);
static_assert(std::is_standard_layout_v<OmpFileIOBatchPlanEntry>);
static_assert(std::is_standard_layout_v<OmpFileIOBatchPlan>);
static_assert(std::is_standard_layout_v<OmpFileFreshnessQueryRequest>);
static_assert(std::is_standard_layout_v<OmpFileFreshnessQueryReply>);
// Wire types cross the MPI transport by memcpy: trivially copyable, always.
static_assert(std::is_trivially_copyable_v<OmpFileIORequest>);
static_assert(std::is_trivially_copyable_v<OmpFileIOPlan>);
static_assert(std::is_trivially_copyable_v<OmpFileIOBatchSegment>);
static_assert(std::is_trivially_copyable_v<OmpFileIOBatchRequest>);
static_assert(std::is_trivially_copyable_v<OmpFileIOBatchPlanEntry>);
static_assert(std::is_trivially_copyable_v<OmpFileIOBatchPlan>);
static_assert(std::is_trivially_copyable_v<OmpFileFreshnessQueryRequest>);
static_assert(std::is_trivially_copyable_v<OmpFileFreshnessQueryReply>);
static_assert(std::is_trivially_copyable_v<OmpFileDirtyOwnerPreadBatchSegment>);
static_assert(sizeof(OmpFileFreshnessQueryRequest) == 32);
static_assert(offsetof(OmpFileFreshnessQueryRequest, AbiVersion) == 0);
static_assert(offsetof(OmpFileFreshnessQueryRequest, PathKey) == 8);
static_assert(offsetof(OmpFileFreshnessQueryRequest, LocalVersion) == 16);
static_assert(offsetof(OmpFileFreshnessQueryRequest, RequesterRank) == 24);
static_assert(sizeof(OmpFileFreshnessQueryReply) == 32);
static_assert(offsetof(OmpFileFreshnessQueryReply, AbiVersion) == 0);
static_assert(offsetof(OmpFileFreshnessQueryReply, Decision) == 4);
static_assert(offsetof(OmpFileFreshnessQueryReply, SourceRank) == 8);
static_assert(offsetof(OmpFileFreshnessQueryReply, Status) == 12);
static_assert(offsetof(OmpFileFreshnessQueryReply, Errno) == 16);
static_assert(offsetof(OmpFileFreshnessQueryReply, SelectedVersion) == 24);

// Name of an OMPFILE_* event type; UNKNOWN_EVENT_TYPE for any other.
const char *ompfileEventTypeName(EventTypeTy eventType);

/// OMPFILE_* coroutine events created at the origin rank of the event.
namespace OriginEvents {

EventTy ompfilePing(MPIRequestManagerTy RequestManager, uint64_t Token);
EventTy ompfileOpen(MPIRequestManagerTy RequestManager, const char *Path,
                    int Flags, int Mode, int *RemoteHandle,
                    int *RemoteErrno);
EventTy ompfileClose(MPIRequestManagerTy RequestManager, int RemoteHandle,
                     int *CloseRet, int *RemoteErrno);
EventTy ompfilePread(MPIRequestManagerTy RequestManager, int RemoteHandle,
                     int64_t Offset, void *Buffer, uint64_t Size,
                     int *IoRet, int *RemoteErrno, uint64_t *Bytes);
EventTy ompfileDirtyOwnerPread(MPIRequestManagerTy RequestManager,
                               int RemoteHandle, int64_t Offset,
                               void *Buffer, uint64_t Size,
                               uint64_t ExpectedVersion, int *IoRet,
                               int *RemoteErrno, uint64_t *Bytes);
EventTy ompfileDirtyOwnerPreadBatch(
    MPIRequestManagerTy RequestManager, int RemoteHandle,
    const OmpFileDirtyOwnerPreadBatchSegment *Segments, uint64_t SegmentCount,
    void *const *Buffers, int *IoRets, int *RemoteErrnos, uint64_t *Bytes);
EventTy ompfileDirtyOwnerQuery(MPIRequestManagerTy RequestManager,
                               int RemoteHandle, int64_t Offset,
                               uint64_t Size, uint64_t ExpectedVersion,
                               int *IoRet, int *RemoteErrno, int *State);
EventTy ompfilePwrite(MPIRequestManagerTy RequestManager, int RemoteHandle,
                      int64_t Offset, const void *Buffer, uint64_t Size,
                      int *IoRet, int *RemoteErrno, uint64_t *Bytes);
EventTy ompfileFreshnessQuery(MPIRequestManagerTy RequestManager,
                               const OmpFileFreshnessQueryRequest *Request,
                               OmpFileFreshnessQueryReply *Reply);
EventTy ompfileFreshnessWriteCommit(
    MPIRequestManagerTy RequestManager,
    const OmpFileFreshnessWriteCommitRequest *Request,
    OmpFileFreshnessWriteCommitReply *Reply);
EventTy ompfileFlushDirtyTile(MPIRequestManagerTy RequestManager,
                              const OmpFileFlushDirtyTileRequest *Request,
                              OmpFileFlushDirtyTileReply *Reply);
EventTy ompfileSchedRequest(MPIRequestManagerTy RequestManager,
                              const OmpFileIORequest *Request, const char *Path,
                              OmpFileIOPlan *Plan);
EventTy ompfileSchedBatchRequest(MPIRequestManagerTy RequestManager,
                                 const OmpFileIOBatchRequest *Request,
                                 const void *RequestPayload,
                                 uint64_t RequestPayloadBytes,
                                 OmpFileIOBatchPlan *Plan, void *PlanPayload,
                                 uint64_t PlanPayloadCapBytes,
                                 uint64_t *PlanPayloadOutBytes);

} // namespace OriginEvents

#endif // _MPI_PROXY_EVENT_SYSTEM_OMPFILE_H_
