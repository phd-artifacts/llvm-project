//===-- ProxyOmpFileBridge.cpp - Proxy ompfile_mpp_* exports ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The proxy rank's side of the libompfile MPP ABI (ompfile_mpp_abi.h): the
// extern C ompfile_mpp_* exports that libompfile resolves with dlsym when a
// target region runs on this proxy, and the registry of the ProxyOmpFile
// they forward to. The tool links with --export-dynamic so dlsym sees them.
//
//===----------------------------------------------------------------------===//

#include "ProxyOmpFile.h"

#include <cerrno>
#include <mutex>

namespace {
std::mutex ActiveProxyOmpFileMutex;
ProxyOmpFile *ActiveProxyOmpFile = nullptr;
} // namespace

ProxyOmpFile *getActiveProxyOmpFile() {
  const std::lock_guard<std::mutex> Lock(ActiveProxyOmpFileMutex);
  return ActiveProxyOmpFile;
}

void setActiveProxyOmpFile(ProxyOmpFile *Instance) {
  const std::lock_guard<std::mutex> Lock(ActiveProxyOmpFileMutex);
  ActiveProxyOmpFile = Instance;
}

void clearActiveProxyOmpFile(ProxyOmpFile *Instance) {
  const std::lock_guard<std::mutex> Lock(ActiveProxyOmpFileMutex);
  if (ActiveProxyOmpFile == Instance)
    ActiveProxyOmpFile = nullptr;
}

extern "C" {
int ompfile_mpp_init() {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppInit();
}

int ompfile_mpp_submit(uint64_t Token) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppSubmit(Token);
}

int ompfile_mpp_open(const char *Path, int Flags, int Mode, int *Handle) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppOpen(Path, Flags, Mode, Handle);
}

int ompfile_mpp_open_on_rank(const char *Path, int Flags, int Mode, int Rank,
                             int *Handle) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppOpenOnRank(Path, Flags, Mode, Rank, Handle);
}

int ompfile_mpp_sched_request(const OmpFileIORequest *Request, const char *Path,
                              OmpFileIOPlan *Plan) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppSchedRequest(Request, Path, Plan);
}

int ompfile_mpp_sched_request_batch(
    const OmpFileIOBatchRequest *Request, const void *RequestPayload,
    uint64_t RequestPayloadBytes, OmpFileIOBatchPlan *Plan, void *PlanPayload,
    uint64_t PlanPayloadCapBytes, uint64_t *PlanPayloadOutBytes) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppSchedBatchRequest(Request, RequestPayload, RequestPayloadBytes,
                                  Plan, PlanPayload, PlanPayloadCapBytes,
                                  PlanPayloadOutBytes);
}

int ompfile_mpp_close(int Handle) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppClose(Handle);
}

int ompfile_mpp_handle_owner_rank(int Handle, int *RankOut) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppHandleOwnerRank(Handle, RankOut);
}

int ompfile_mpp_pread_ex(int Handle, int64_t Offset, void *Buffer,
                          uint64_t Size, uint64_t *BytesRead) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPreadEx(Handle, Offset, Buffer, Size, BytesRead);
}

int ompfile_mpp_pread_no_stage_ex(int Handle, int64_t Offset, void *Buffer,
                                  uint64_t Size, uint64_t *BytesRead) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPreadNoStageEx(Handle, Offset, Buffer, Size, BytesRead);
}

int ompfile_mpp_dirty_owner_pread_ex(int Handle, int SourceRank,
                                     uint64_t ExpectedVersion, int64_t Offset,
                                     void *Buffer, uint64_t Size,
                                     uint64_t *BytesRead) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppDirtyOwnerPreadEx(Handle, SourceRank, ExpectedVersion, Offset,
                                  Buffer, Size, BytesRead);
}

int ompfile_mpp_dirty_owner_pread_batch_ex(
    int Handle, int SourceRank,
    const OmpFileDirtyOwnerPreadBatchSegment *Segments, uint64_t SegmentCount,
    void *const *Buffers, uint64_t *BytesRead, int *Statuses, int *Errnos) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppDirtyOwnerPreadBatchEx(Handle, SourceRank, Segments,
                                       SegmentCount, Buffers, BytesRead,
                                       Statuses, Errnos);
}

int ompfile_mpp_dirty_owner_query_ex(int Handle, int SourceRank,
                                     uint64_t ExpectedVersion, int64_t Offset,
                                     uint64_t Size, int *StateOut) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppDirtyOwnerQueryEx(Handle, SourceRank, ExpectedVersion, Offset,
                                  Size, StateOut);
}

int ompfile_mpp_pwrite_ex(int Handle, int64_t Offset, const void *Buffer,
                          uint64_t Size, uint64_t *BytesWritten) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPwriteEx(Handle, Offset, Buffer, Size, BytesWritten);
}

int ompfile_mpp_freshness_query(const OmpFileFreshnessQueryRequest *Request,
                                OmpFileFreshnessQueryReply *Reply) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppFreshnessQuery(Request, Reply);
}

int ompfile_mpp_freshness_write_commit(uint64_t PathKey, int WriterRank,
                                       uint64_t TileId, int WriteThroughMode,
                                       uint64_t *CommittedVersion) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppFreshnessWriteCommit(PathKey, WriterRank, TileId,
                                     WriteThroughMode, CommittedVersion);
}

int ompfile_mpp_flush_dirty_tile(uint64_t PathKey, int *SourceRank,
                                 uint64_t *FlushedVersion) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return EHOSTDOWN;
  return PD->mppFlushDirtyTile(PathKey, SourceRank, FlushedVersion);
}

// Proxy-side counterpart of the origin export in OriginOmpFileBridge.cpp.
// libompfile resolves whichever definition its own process provides; this
// is the path a commit issued from inside a target region takes, because
// that region executes on the proxy. It flushes this process and
// broadcasts Action 3 to the peer proxies (commitPathKeyOnAllWorkers).
// Until Sep 2026 it flushed only this process, on the assumption that the
// proxy running the region owns the file; the HEADNODE can place it on a
// peer, and then the commit reached nothing.
int ompfile_mpp_commit_stage_path_key(uint64_t PathKey) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return EHOSTDOWN;
  if (PathKey == 0)
    return EINVAL;
  errno = 0;
  if (!PD->commitPathKeyOnAllWorkers(PathKey))
    return errno != 0 ? errno : EIO;
  return 0;
}

int ompfile_mpp_poll(uint64_t Token, int *Done) {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPoll(Token, Done);
}

int ompfile_mpp_finalize() {
  ProxyOmpFile *PD = getActiveProxyOmpFile();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppFinalize();
}
} // extern "C"
