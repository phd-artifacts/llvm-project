//===-- OriginOmpFileBridge.cpp - Origin ompfile_mpp_* bridges --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The origin rank's side of the libompfile MPP ABI (ompfile_mpp_abi.h): the
// extern "C" ompfile_mpp_* bridges that libompfile's shim resolves with dlsym,
// the origin handle table (local handle -> proxy rank and remote handle) and
// the token -> event map behind submit/poll. Part of omptarget.rtl.mpi, next
// to rtl.cpp, which publishes the active plugin through
// ompfilePublishMPIPlugin.
//
//===----------------------------------------------------------------------===//

#include "OriginOmpFileBridge.h"

#include "EventSystemOmpFile.h"

#include "Shared/Debug.h"
#include "omptarget.h"

#include "llvm/Support/Error.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace llvm::omp::target::plugin {

static std::atomic<MPIPluginTy *> ActiveMPIPlugin{nullptr};

void ompfilePublishMPIPlugin(MPIPluginTy *Plugin) {
  ActiveMPIPlugin.store(Plugin);
}

void ompfileRetractMPIPlugin(MPIPluginTy *Plugin) {
  MPIPluginTy *Expected = Plugin;
  ActiveMPIPlugin.compare_exchange_strong(Expected, nullptr);
}

static std::mutex &getOmpfileMppMutex() {
  static std::mutex *Mutex = new std::mutex();
  return *Mutex;
}

static std::unordered_map<uint64_t, EventTy> &getOmpfileMppEvents() {
  static auto *Events = new std::unordered_map<uint64_t, EventTy>();
  return *Events;
}

struct OmpfileMppHandleEntry {
  int Rank = -1;
  int RemoteHandle = -1;
};

static std::unordered_map<int, OmpfileMppHandleEntry> &getOmpfileMppHandles() {
  static auto *Handles = new std::unordered_map<int, OmpfileMppHandleEntry>();
  return *Handles;
}

static int registerOmpfileMppHandle(int Rank, int RemoteHandle) {
  static std::atomic<int> NextHandle{1};
  auto &Handles = getOmpfileMppHandles();
  const int LocalHandle = NextHandle.fetch_add(1, std::memory_order_relaxed);
  Handles[LocalHandle] = {Rank, RemoteHandle};
  return LocalHandle;
}

static bool findOmpfileMppHandle(int LocalHandle,
                                 OmpfileMppHandleEntry &Entry) {
  auto &Handles = getOmpfileMppHandles();
  auto It = Handles.find(LocalHandle);
  if (It == Handles.end()) {
    errno = EBADF;
    return false;
  }
  Entry = It->second;
  return true;
}

static bool eraseOmpfileMppHandle(int LocalHandle,
                                  OmpfileMppHandleEntry &Entry) {
  auto &Handles = getOmpfileMppHandles();
  auto It = Handles.find(LocalHandle);
  if (It == Handles.end()) {
    errno = EBADF;
    return false;
  }
  Entry = It->second;
  Handles.erase(It);
  return true;
}

} // namespace llvm::omp::target::plugin

extern "C" {
int ompfile_mpp_init() {
  using namespace llvm::omp::target::plugin;
  ompfileDiagLog("ompfile_mpp_init_enter", -1, -1, -1, 0, 0);
  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin) {
    DP("ompfile_mpp_init: ActiveMPIPlugin is null.\n");
    return OFFLOAD_FAIL;
  }
  if (auto Err = ompfileInitMPIPlugin(*Plugin)) {
    DP("ompfile_mpp_init: Plugin->init failed: %s\n",
       llvm::toString(std::move(Err)).c_str());
    return OFFLOAD_FAIL;
  }
  if (!ompfileEnsureEventSystem(*Plugin)) {
    DP("ompfile_mpp_init: ensureEventSystemInitializedForOmpFile failed.\n");
    return OFFLOAD_FAIL;
  }
  DP("ompfile_mpp_init: success.\n");
  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_submit(uint64_t Token) {
  using namespace llvm::omp::target::plugin;
  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return OFFLOAD_FAIL;

  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return OFFLOAD_FAIL;

  if (!ompfileEnsureEventSystem(*Plugin))
    return OFFLOAD_FAIL;

  EventTy Event = ompfileEventSystem(*Plugin).createEvent(
      OriginEvents::ompfilePing, EventTypeTy::OMPFILE_PING, /*DstDeviceID=*/0,
      Token);
  if (Event.empty())
    return OFFLOAD_FAIL;

  auto &Events = getOmpfileMppEvents();
  const std::lock_guard<std::mutex> Lock(getOmpfileMppMutex());
  if (Events.count(Token))
    return OFFLOAD_FAIL;
  Events.emplace(Token, std::move(Event));
  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_open(const char *Path, int Flags, int Mode, int *Handle) {
  using namespace llvm::omp::target::plugin;
  if (!Path || !Handle)
    return OFFLOAD_FAIL;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return OFFLOAD_FAIL;

  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return OFFLOAD_FAIL;

  if (!ompfileEnsureEventSystem(*Plugin))
    return OFFLOAD_FAIL;

  int RemoteHandle = -1;
  int RemoteErrno = 0;
  EventTy Event = ompfileEventSystem(*Plugin).createEvent(
      OriginEvents::ompfileOpen, EventTypeTy::OMPFILE_OPEN,
      /*DstDeviceID=*/0, Path, Flags, Mode, &RemoteHandle, &RemoteErrno);
  if (Event.empty())
    return OFFLOAD_FAIL;

  Event.wait();
  if (auto Error = Event.getError())
    return OFFLOAD_FAIL;

  if (RemoteHandle < 0) {
    errno = RemoteErrno;
    return OFFLOAD_FAIL;
  }

  const std::lock_guard<std::mutex> Lock(getOmpfileMppMutex());
  *Handle = registerOmpfileMppHandle(/*Rank=*/0, RemoteHandle);
  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_open_on_rank(const char *Path, int Flags, int Mode, int Rank,
                             int *Handle) {
  using namespace llvm::omp::target::plugin;
  if (!Path || !Handle)
    return OFFLOAD_FAIL;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return OFFLOAD_FAIL;

  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return OFFLOAD_FAIL;

  if (!ompfileEnsureEventSystem(*Plugin))
    return OFFLOAD_FAIL;

  int RemoteHandle = -1;
  int RemoteErrno = 0;
  EventTy Event = ompfileEventSystem(*Plugin).createEvent(
      OriginEvents::ompfileOpen, EventTypeTy::OMPFILE_OPEN,
      /*DstDeviceID=*/Rank, Path, Flags, Mode, &RemoteHandle, &RemoteErrno);
  if (Event.empty())
    return OFFLOAD_FAIL;

  Event.wait();
  if (auto Error = Event.getError())
    return OFFLOAD_FAIL;

  if (RemoteHandle < 0) {
    errno = RemoteErrno;
    return OFFLOAD_FAIL;
  }

  const std::lock_guard<std::mutex> Lock(getOmpfileMppMutex());
  *Handle = registerOmpfileMppHandle(Rank, RemoteHandle);
  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_sched_request(const OmpFileIORequest *Request, const char *Path,
                              OmpFileIOPlan *Plan) {
  using namespace llvm::omp::target::plugin;
  if (!Request || !Plan) {
    DP("ompfile_mpp_sched_request: missing Request/Plan buffers.\n");
    return OFFLOAD_FAIL;
  }
  if (Request->PathSize > 0 && !Path) {
    DP("ompfile_mpp_sched_request: missing path for request %llu.\n",
       static_cast<unsigned long long>(Request->RequestId));
    return OFFLOAD_FAIL;
  }

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin) {
    DP("ompfile_mpp_sched_request: ActiveMPIPlugin is null.\n");
    return OFFLOAD_FAIL;
  }

  if (auto Err = ompfileInitMPIPlugin(*Plugin)) {
    DP("ompfile_mpp_sched_request: Plugin->init failed: %s\n",
       llvm::toString(std::move(Err)).c_str());
    return OFFLOAD_FAIL;
  }

  if (!ompfileEnsureEventSystem(*Plugin)) {
    DP("ompfile_mpp_sched_request: ensureEventSystemInitializedForOmpFile "
       "failed.\n");
    return OFFLOAD_FAIL;
  }

  EventTy Event = ompfileEventSystem(*Plugin).createEvent(
      OriginEvents::ompfileSchedRequest, EventTypeTy::OMPFILE_SCHED_REQUEST,
      /*DstDeviceID=*/0, Request, Path, Plan);
  if (Event.empty()) {
    DP("ompfile_mpp_sched_request: failed to create event.\n");
    return OFFLOAD_FAIL;
  }

  Event.wait();
  if (auto Error = Event.getError()) {
    DP("ompfile_mpp_sched_request: event returned error: %s\n",
       llvm::toString(std::move(Error)).c_str());
    return OFFLOAD_FAIL;
  }

  if (Plan->Status != 0) {
    errno = Plan->Errno;
    DP("ompfile_mpp_sched_request: non-zero plan status=%d errno=%d.\n",
       Plan->Status, Plan->Errno);
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_sched_request_batch(const OmpFileIOBatchRequest *Request,
                                    const void *RequestPayload,
                                    uint64_t RequestPayloadBytes,
                                    OmpFileIOBatchPlan *Plan, void *PlanPayload,
                                    uint64_t PlanPayloadCapBytes,
                                    uint64_t *PlanPayloadOutBytes) {
  using namespace llvm::omp::target::plugin;
  if (!Request || !Plan || !PlanPayloadOutBytes) {
    DP("ompfile_mpp_sched_request_batch: missing Request/Plan buffers.\n");
    return OFFLOAD_FAIL;
  }
  if (RequestPayloadBytes > 0 && !RequestPayload) {
    DP("ompfile_mpp_sched_request_batch: missing request payload.\n");
    return OFFLOAD_FAIL;
  }
  if (Request->PayloadBytes != RequestPayloadBytes) {
    DP("ompfile_mpp_sched_request_batch: payload size mismatch request=%u "
       "arg=%llu.\n",
       Request->PayloadBytes,
       static_cast<unsigned long long>(RequestPayloadBytes));
    return OFFLOAD_FAIL;
  }
  if (PlanPayloadCapBytes > 0 && !PlanPayload) {
    DP("ompfile_mpp_sched_request_batch: missing plan payload buffer.\n");
    return OFFLOAD_FAIL;
  }

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin) {
    DP("ompfile_mpp_sched_request_batch: ActiveMPIPlugin is null.\n");
    return OFFLOAD_FAIL;
  }

  if (auto Err = ompfileInitMPIPlugin(*Plugin)) {
    DP("ompfile_mpp_sched_request_batch: Plugin->init failed: %s\n",
       llvm::toString(std::move(Err)).c_str());
    return OFFLOAD_FAIL;
  }

  if (!ompfileEnsureEventSystem(*Plugin)) {
    DP("ompfile_mpp_sched_request_batch: "
       "ensureEventSystemInitializedForOmpFile failed.\n");
    return OFFLOAD_FAIL;
  }

  EventTy Event = ompfileEventSystem(*Plugin).createEvent(
      OriginEvents::ompfileSchedBatchRequest,
      EventTypeTy::OMPFILE_SCHED_REQUEST_BATCH, /*DstDeviceID=*/0, Request,
      RequestPayload, RequestPayloadBytes, Plan, PlanPayload,
      PlanPayloadCapBytes, PlanPayloadOutBytes);
  if (Event.empty()) {
    DP("ompfile_mpp_sched_request_batch: failed to create event.\n");
    return OFFLOAD_FAIL;
  }

  Event.wait();
  if (auto Error = Event.getError()) {
    DP("ompfile_mpp_sched_request_batch: event returned error: %s\n",
       llvm::toString(std::move(Error)).c_str());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_close(int Handle) {
  using namespace llvm::omp::target::plugin;
  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return OFFLOAD_FAIL;

  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return OFFLOAD_FAIL;

  if (!ompfileEnsureEventSystem(*Plugin))
    return OFFLOAD_FAIL;

  OmpfileMppHandleEntry Entry{};
  {
    const std::lock_guard<std::mutex> Lock(getOmpfileMppMutex());
    if (!eraseOmpfileMppHandle(Handle, Entry))
      return OFFLOAD_FAIL;
  }

  int CloseRet = -1;
  int RemoteErrno = 0;
  EventTy Event = ompfileEventSystem(*Plugin).createEvent(
      OriginEvents::ompfileClose, EventTypeTy::OMPFILE_CLOSE,
      /*DstDeviceID=*/Entry.Rank, Entry.RemoteHandle, &CloseRet, &RemoteErrno);
  if (Event.empty())
    return OFFLOAD_FAIL;

  Event.wait();
  if (auto Error = Event.getError())
    return OFFLOAD_FAIL;

  if (CloseRet != 0) {
    errno = RemoteErrno;
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_pread_ex(int Handle, int64_t Offset, void *Buffer,
                         uint64_t Size, uint64_t *BytesRead) {
  using namespace llvm::omp::target::plugin;
  if (!Buffer && Size > 0)
    return OFFLOAD_FAIL;
  if (!BytesRead)
    return OFFLOAD_FAIL;
  *BytesRead = 0;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return OFFLOAD_FAIL;

  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return OFFLOAD_FAIL;

  if (!ompfileEnsureEventSystem(*Plugin))
    return OFFLOAD_FAIL;

  OmpfileMppHandleEntry Entry{};
  {
    const std::lock_guard<std::mutex> Lock(getOmpfileMppMutex());
    if (!findOmpfileMppHandle(Handle, Entry))
      return OFFLOAD_FAIL;
  }

  int IoRet = -1;
  int RemoteErrno = 0;
  uint64_t Bytes = 0;
  EventTy Event = ompfileEventSystem(*Plugin).createEvent(
      OriginEvents::ompfilePread, EventTypeTy::OMPFILE_PREAD,
      /*DstDeviceID=*/Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
      &IoRet, &RemoteErrno, &Bytes);
  if (Event.empty())
    return OFFLOAD_FAIL;

  Event.wait();
  if (auto Error = Event.getError())
    return OFFLOAD_FAIL;

  if (IoRet != 0) {
    errno = RemoteErrno;
    return OFFLOAD_FAIL;
  }

  if (Bytes > Size) {
    errno = EPROTO;
    return OFFLOAD_FAIL;
  }
  if (Bytes < Size)
    std::memset(static_cast<char *>(Buffer) + Bytes, 0,
                static_cast<size_t>(Size - Bytes));
  *BytesRead = Bytes;

  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_pwrite_ex(int Handle, int64_t Offset, const void *Buffer,
                          uint64_t Size, uint64_t *BytesWritten) {
  using namespace llvm::omp::target::plugin;
  if (!Buffer && Size > 0)
    return OFFLOAD_FAIL;
  if (!BytesWritten)
    return OFFLOAD_FAIL;
  *BytesWritten = 0;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return OFFLOAD_FAIL;

  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return OFFLOAD_FAIL;

  if (!ompfileEnsureEventSystem(*Plugin))
    return OFFLOAD_FAIL;

  OmpfileMppHandleEntry Entry{};
  {
    const std::lock_guard<std::mutex> Lock(getOmpfileMppMutex());
    if (!findOmpfileMppHandle(Handle, Entry))
      return OFFLOAD_FAIL;
  }

  int IoRet = -1;
  int RemoteErrno = 0;
  uint64_t Bytes = 0;
  EventTy Event = ompfileEventSystem(*Plugin).createEvent(
      OriginEvents::ompfilePwrite, EventTypeTy::OMPFILE_PWRITE,
      /*DstDeviceID=*/Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
      &IoRet, &RemoteErrno, &Bytes);
  if (Event.empty())
    return OFFLOAD_FAIL;

  Event.wait();
  if (auto Error = Event.getError())
    return OFFLOAD_FAIL;

  if (IoRet != 0) {
    errno = RemoteErrno;
    return OFFLOAD_FAIL;
  }
  if (Bytes > Size) {
    errno = EPROTO;
    return OFFLOAD_FAIL;
  }
  *BytesWritten = Bytes;

  return OFFLOAD_SUCCESS;
}

// Commit: push every proxy's dirty stage for this file to the source now.
//
// Deliberately not the flush_dirty_tile protocol above. That one asks the
// headnode which rank holds the freshest tile and carries a freshness version,
// which only exists once a write has been freshness-committed - so it cannot
// serve a commit issued in the middle of a write wave. This broadcasts a
// version-free stage flush to every worker instead; a rank holding nothing for
// the file reports success without work.
int ompfile_mpp_commit_stage_path_key(uint64_t PathKey) {
  using namespace llvm::omp::target::plugin;
  if (PathKey == 0)
    return EINVAL;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return EHOSTDOWN;
  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return EHOSTDOWN;
  if (!ompfileEnsureEventSystem(*Plugin))
    return EHOSTDOWN;

  OmpFileFlushDirtyTileRequest Request{};
  Request.AbiVersion = OMPFILE_FRESHNESS_QUERY_ABI_VERSION;
  Request.Action = 3;
  Request.PathKey = PathKey;

  EventSystemTy &EventSystem = ompfileEventSystem(*Plugin);
  int FirstErrno = 0;
  for (int Rank = 0; Rank < EventSystem.getNumWorkers(); ++Rank) {
    OmpFileFlushDirtyTileReply Reply{};
    EventTy Event = EventSystem.createEvent(
        OriginEvents::ompfileFlushDirtyTile,
        EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE, /*DstDeviceID=*/Rank, &Request,
        &Reply);
    if (Event.empty()) {
      if (FirstErrno == 0)
        FirstErrno = EIO;
      continue;
    }
    Event.wait();
    if (auto Error = Event.getError()) {
      if (FirstErrno == 0)
        FirstErrno = EIO;
      continue;
    }
    if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION) {
      if (FirstErrno == 0)
        FirstErrno = EPROTO;
      continue;
    }
    if (Reply.Status != 0 && FirstErrno == 0)
      FirstErrno = Reply.Errno != 0 ? Reply.Errno : EIO;
  }

  return FirstErrno;
}

int ompfile_mpp_freshness_query(const OmpFileFreshnessQueryRequest *Request,
                                OmpFileFreshnessQueryReply *Reply) {
  using namespace llvm::omp::target::plugin;
  if (!Request || !Reply)
    return EINVAL;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return EHOSTDOWN;
  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return EHOSTDOWN;
  if (!ompfileEnsureEventSystem(*Plugin))
    return EHOSTDOWN;

  EventSystemTy &EventSystem = ompfileEventSystem(*Plugin);
  EventTy Event = EventSystem.createEvent(
      OriginEvents::ompfileFreshnessQuery,
      EventTypeTy::OMPFILE_FRESHNESS_QUERY, /*DstDeviceID=*/0, Request, Reply);
  if (Event.empty())
    return EIO;
  Event.wait();
  if (auto Error = Event.getError())
    return EIO;
  if (Reply->AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION)
    return EPROTO;
  if (Reply->Status != 0)
    return Reply->Errno != 0 ? Reply->Errno : EIO;
  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_freshness_write_commit(uint64_t PathKey, int WriterRank,
                                       uint64_t TileId, int WriteThroughMode,
                                       uint64_t *CommittedVersion) {
  using namespace llvm::omp::target::plugin;
  if (PathKey == 0 || WriterRank < 0 || !CommittedVersion)
    return EINVAL;
  *CommittedVersion = 0;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return EHOSTDOWN;
  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return EHOSTDOWN;
  if (!ompfileEnsureEventSystem(*Plugin))
    return EHOSTDOWN;

  OmpFileFreshnessWriteCommitRequest Request{};
  Request.PathKey = PathKey;
  Request.TileId = TileId;
  Request.WriterRank = WriterRank;
  Request.WriteThroughMode = WriteThroughMode != 0 ? 1u : 0u;
  OmpFileFreshnessWriteCommitReply Reply{};

  EventSystemTy &EventSystem = ompfileEventSystem(*Plugin);
  EventTy Event = EventSystem.createEvent(
      OriginEvents::ompfileFreshnessWriteCommit,
      EventTypeTy::OMPFILE_FRESHNESS_WRITE_COMMIT, /*DstDeviceID=*/0, &Request,
      &Reply);
  if (Event.empty())
    return EIO;
  Event.wait();
  if (auto Error = Event.getError())
    return EIO;
  if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION)
    return EPROTO;
  if (Reply.Status != 0)
    return Reply.Errno != 0 ? Reply.Errno : EIO;
  *CommittedVersion = Reply.CommittedVersion;
  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_flush_dirty_tile(uint64_t PathKey, int *SourceRank,
                                 uint64_t *FlushedVersion) {
  using namespace llvm::omp::target::plugin;
  if (PathKey == 0 || !SourceRank || !FlushedVersion)
    return EINVAL;
  *SourceRank = -1;
  *FlushedVersion = 0;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return EHOSTDOWN;
  if (auto Err = ompfileInitMPIPlugin(*Plugin))
    return EHOSTDOWN;
  if (!ompfileEnsureEventSystem(*Plugin))
    return EHOSTDOWN;

  EventSystemTy &EventSystem = ompfileEventSystem(*Plugin);

  OmpFileFlushDirtyTileRequest Request{};
  Request.Action = 0;
  Request.PathKey = PathKey;
  OmpFileFlushDirtyTileReply Reply{};
  EventTy Event = EventSystem.createEvent(
      OriginEvents::ompfileFlushDirtyTile, EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE,
      /*DstDeviceID=*/0, &Request, &Reply);
  if (Event.empty())
    return EIO;
  Event.wait();
  if (auto Error = Event.getError())
    return EIO;
  if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION)
    return EPROTO;
  if (Reply.Status != 0)
    return Reply.Errno != 0 ? Reply.Errno : EIO;

  if (Reply.SourceRank >= 0) {
    OmpFileFlushDirtyTileRequest WritebackRequest{};
    WritebackRequest.Action = 2;
    WritebackRequest.PathKey = PathKey;
    WritebackRequest.SourceRank = Reply.SourceRank;
    WritebackRequest.Version = Reply.FlushedVersion;
    WritebackRequest.Success = 1;
    OmpFileFlushDirtyTileReply WritebackReply{};
    EventTy WritebackEvent = EventSystem.createEvent(
        OriginEvents::ompfileFlushDirtyTile,
        EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE,
        /*DstDeviceID=*/Reply.SourceRank, &WritebackRequest, &WritebackReply);
    if (WritebackEvent.empty())
      return EIO;
    WritebackEvent.wait();
    if (auto Error = WritebackEvent.getError())
      return EIO;
    if (WritebackReply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION)
      return EPROTO;
    if (WritebackReply.Status != 0)
      return WritebackReply.Errno != 0 ? WritebackReply.Errno : EIO;

    OmpFileFlushDirtyTileRequest CompleteRequest{};
    CompleteRequest.Action = 1;
    CompleteRequest.PathKey = PathKey;
    CompleteRequest.SourceRank = Reply.SourceRank;
    CompleteRequest.Version = Reply.FlushedVersion;
    CompleteRequest.Success = 1;
    OmpFileFlushDirtyTileReply CompleteReply{};
    EventTy CompleteEvent = EventSystem.createEvent(
        OriginEvents::ompfileFlushDirtyTile,
        EventTypeTy::OMPFILE_FLUSH_DIRTY_TILE, /*DstDeviceID=*/0,
        &CompleteRequest, &CompleteReply);
    if (CompleteEvent.empty())
      return EIO;
    CompleteEvent.wait();
    if (auto Error = CompleteEvent.getError())
      return EIO;
    if (CompleteReply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION)
      return EPROTO;
    if (CompleteReply.Status != 0)
      return CompleteReply.Errno != 0 ? CompleteReply.Errno : EIO;
  }

  *SourceRank = Reply.SourceRank;
  *FlushedVersion = Reply.FlushedVersion;
  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_poll(uint64_t Token, int *Done) {
  using namespace llvm::omp::target::plugin;
  if (!Done)
    return OFFLOAD_FAIL;

  auto &Events = getOmpfileMppEvents();
  const std::lock_guard<std::mutex> Lock(getOmpfileMppMutex());
  auto It = Events.find(Token);
  if (It == Events.end())
    return OFFLOAD_FAIL;

  It->second.advance();
  if (!It->second.done()) {
    *Done = 0;
    return OFFLOAD_SUCCESS;
  }

  *Done = 1;
  auto Error = It->second.getError();
  Events.erase(It);
  if (Error)
    return OFFLOAD_FAIL;

  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_finalize() {
  using namespace llvm::omp::target::plugin;
  auto &Events = getOmpfileMppEvents();
  auto &Handles = getOmpfileMppHandles();
  const std::lock_guard<std::mutex> Lock(getOmpfileMppMutex());
  Events.clear();
  Handles.clear();
  return OFFLOAD_SUCCESS;
}
}
