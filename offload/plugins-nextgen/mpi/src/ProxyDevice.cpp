#include <chrono>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "EventSystem.h"
#include "OmpFileHeadnodeManager.h"
#include "RemotePluginManager.h"
#include "Shared/APITypes.h"
#include "mpi.h"
#include "omptarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#ifdef OMPT_SUPPORT
#include "OpenMP/OMPT/Callback.h"
#include "omp-tools.h"
extern void llvm::omp::target::ompt::connectLibrary();
#endif

struct ProxyDevice;

namespace {
std::mutex ActiveProxyDeviceMutex;
ProxyDevice *ActiveProxyDevice = nullptr;

bool ompfileProxyTraceEnabled() {
  static const bool Enabled = []() {
    const char *Env = std::getenv("LIBOMPFILE_DEBUG_TRACE");
    return Env && Env[0] == '1' && Env[1] == '\0';
  }();
  return Enabled;
}

uint64_t nextProxyTraceSeq() {
  static std::atomic<uint64_t> Seq{1};
  return Seq.fetch_add(1, std::memory_order_relaxed);
}

unsigned long proxyThreadIdHash() {
  return static_cast<unsigned long>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
}
}

/// Class that holds a stage pointer for data transfer between host and remote
/// device (RAII)
struct PluginDataHandle {
  void *HstPtr;
  uint32_t Plugin;
  uint32_t Device;
  RemotePluginManager *PM;
  PluginDataHandle(RemotePluginManager *PluginManager, uint32_t PluginId,
                   uint32_t DeviceId, uint64_t Size) {
    Device = DeviceId;
    Plugin = PluginId;
    PM = PluginManager;
    // HstPtr = PM->Plugins[Plugin]->data_alloc(Device, Size, nullptr,
    //                                          TARGET_ALLOC_HOST);
    HstPtr = malloc(Size);
  }
  ~PluginDataHandle() {
    // PM->Plugins[Plugin]->data_delete(Device, HstPtr, TARGET_ALLOC_HOST);
    free(HstPtr);
  }
};

struct AsyncInfoHandle {
  std::unique_ptr<__tgt_async_info> AsyncInfoPtr;
  std::mutex AsyncInfoMutex;
  AsyncInfoHandle() { AsyncInfoPtr = std::make_unique<__tgt_async_info>(); }
  AsyncInfoHandle(AsyncInfoHandle &&Other) {
    AsyncInfoPtr = std::move(Other.AsyncInfoPtr);
    Other.AsyncInfoPtr = nullptr;
  }
};

/// Event Implementations on Device side.
struct ProxyDevice {
  struct OmpFileHandleEntry {
    int Rank = -1;
    int RemoteHandle = -1;
  };

  struct OmpFileOpenCacheEntry {
    int Fd = -1;
    uint64_t RefCount = 0;
  };

  void traceOmpFile(const char *Fmt, ...) const {
    if (!ompfileProxyTraceEnabled())
      return;

    va_list Args;
    va_start(Args, Fmt);
    fprintf(stderr,
            "[ompfile-proxy][trace][seq=%llu][pid=%d][tid=%lu][rank=%d] ",
            static_cast<unsigned long long>(nextProxyTraceSeq()), getpid(),
            proxyThreadIdHash(), EventSystem.LocalRank);
    vfprintf(stderr, Fmt, Args);
    va_end(Args);
  }

  ProxyDevice()
      : NumExecEventHandlers("OMPTARGET_NUM_EXEC_EVENT_HANDLERS", 1),
        NumDataEventHandlers("OMPTARGET_NUM_DATA_EVENT_HANDLERS", 1),
        EventPollingRate("OMPTARGET_EVENT_POLLING_RATE", 1),
        OmpFileOpenCacheEnable("LIBOMPFILE_OPT_OPEN_CACHE", false),
        OmpFileOpenCacheKeepOpen("LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN", true),
        OmpFileOptStats("LIBOMPFILE_OPT_STATS", false) {
#ifdef OMPT_SUPPORT
    // Initialize OMPT first
    llvm::omp::target::ompt::connectLibrary();
#endif

    EventSystem.initialize();
    PluginManager.init();
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      OmpFileHeadnodeManager::instance().initialize(EventSystem.WorldSize,
                                                    getHeadnodeRank());
      DP("Initialized global OMPFile headnode manager on rank %d\n",
         EventSystem.LocalRank);
    }
    if (OmpFileOptStats || OmpFileOpenCacheEnable) {
      fprintf(stderr,
              "MPIProxyDevice --> OMPFile cache config rank=%d enabled=%d "
              "keep_open=%d stats=%d\n",
              EventSystem.LocalRank, (int)OmpFileOpenCacheEnable.get(),
              (int)OmpFileOpenCacheKeepOpen.get(), (int)OmpFileOptStats.get());
    }
    {
      const std::lock_guard<std::mutex> Lock(ActiveProxyDeviceMutex);
      ActiveProxyDevice = this;
    }
  }

  ~ProxyDevice() {
    drainOmpFileOpenCache();
    reportOmpFileStats("proxy-dtor");
    {
      const std::lock_guard<std::mutex> Lock(ActiveProxyDeviceMutex);
      if (ActiveProxyDevice == this)
        ActiveProxyDevice = nullptr;
    }
    EventSystem.deinitialize();
    PluginManager.deinit();
  }

  void mapDevicesPerRemote() {
    EventSystem.DevicesPerRemote = {};
    for (int PluginId = 0; PluginId < PluginManager.getNumUsedPlugins();
         PluginId++) {
      EventSystem.DevicesPerRemote.emplace_back(
          PluginManager.getNumDevices(PluginId));
    }
  }

  AsyncInfoHandle *MapAsyncInfo(void *HostAsyncInfoPtr) {
    const std::lock_guard<std::mutex> Lock(TableMutex);
    AsyncInfoHandle *AsyncInfoHandlerPtr = nullptr;

    if (AsyncInfoTable[HostAsyncInfoPtr])
      AsyncInfoHandlerPtr = AsyncInfoTable[HostAsyncInfoPtr];
    else {
      AsyncInfoHandlerPtr =
          AsyncInfoList.emplace_back(std::make_unique<AsyncInfoHandle>()).get();
      AsyncInfoTable[HostAsyncInfoPtr] = AsyncInfoHandlerPtr;
    }

    return AsyncInfoHandlerPtr;
  }

  EventTy waitAsyncOpEnd(int32_t PluginId, int32_t DeviceId,
                         void *AsyncInfoPtr) {
    auto *TgtAsyncInfo = MapAsyncInfo(AsyncInfoPtr)->AsyncInfoPtr.get();
    auto *RPCServer =
        PluginManager.Plugins[PluginId]->getDevice(DeviceId).getRPCServer();

    while (TgtAsyncInfo->Queue != nullptr) {
      if (PluginManager.Plugins[PluginId]->query_async(
              DeviceId, TgtAsyncInfo) == OFFLOAD_FAIL)
        co_return createError("Failure to wait AsyncOp\n");

      if (RPCServer)
        if (auto Err = RPCServer->runServer(
                PluginManager.Plugins[PluginId]->getDevice(DeviceId)))
          co_return Err;
      co_await std::suspend_always{};
    }

    co_return llvm::Error::success();
  }

  EventTy retrieveNumDevices(MPIRequestManagerTy RequestManager) {
    int32_t NumDevices = PluginManager.getNumDevices();
    RequestManager.send(&NumDevices, 1, MPI_INT);

    co_return (co_await RequestManager);
  }

  EventTy isPluginCompatible(MPIRequestManagerTy RequestManager) {
    __tgt_device_image Image;
    bool QueryResult = false;

    uint64_t Size = 0;

    RequestManager.receive(&Size, 1, MPI_UINT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageStart = memAllocHost(Size);
    RequestManager.receive(Image.ImageStart, Size, MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageEnd = (void *)((ptrdiff_t)(Image.ImageStart) + Size);

    llvm::SmallVector<std::unique_ptr<GenericPluginTy>> UsedPlugins;

    for (auto &Plugin : PluginManager.Plugins) {
      QueryResult = Plugin->is_plugin_compatible(&Image);
      if (QueryResult) {
        UsedPlugins.emplace_back(std::move(Plugin));
        break;
      }
    }

    for (auto &Plugin : PluginManager.Plugins) {
      if (Plugin)
        UsedPlugins.emplace_back(std::move(Plugin));
    }

    PluginManager.Plugins = std::move(UsedPlugins);
    mapDevicesPerRemote();

    memFreeHost(Image.ImageStart);
    RequestManager.send(&QueryResult, sizeof(bool), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy isDeviceCompatible(MPIRequestManagerTy RequestManager) {
    __tgt_device_image Image;
    bool QueryResult = false;

    uint64_t Size = 0;

    RequestManager.receive(&Size, 1, MPI_UINT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageStart = memAllocHost(Size);
    RequestManager.receive(Image.ImageStart, Size, MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageEnd = (void *)((ptrdiff_t)(Image.ImageStart) + Size);

    int32_t DeviceId, PluginId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    QueryResult =
        PluginManager.Plugins[PluginId]->is_device_compatible(DeviceId, &Image);

    memFreeHost(Image.ImageStart);
    RequestManager.send(&QueryResult, sizeof(bool), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy initDevice(MPIRequestManagerTy RequestManager) {
    int32_t DeviceId, PluginId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->init_device(DeviceId);

    auto *DevicePtr = &PluginManager.Plugins[PluginId]->getDevice(DeviceId);

    // Event completion notification
    RequestManager.send(&DevicePtr, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy initRecordReplay(MPIRequestManagerTy RequestManager) {
    int64_t MemorySize = 0;
    void *VAddr = nullptr;
    bool IsRecord = false, SaveOutput = false;
    uint64_t ReqPtrArgOffset = 0;

    RequestManager.receive(&MemorySize, 1, MPI_INT64_T);
    RequestManager.receive(&VAddr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&IsRecord, sizeof(bool), MPI_BYTE);
    RequestManager.receive(&SaveOutput, sizeof(bool), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t DeviceId, PluginId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->initialize_record_replay(
        DeviceId, MemorySize, VAddr, IsRecord, SaveOutput, ReqPtrArgOffset);

    RequestManager.send(&ReqPtrArgOffset, 1, MPI_UINT64_T);
    co_return (co_await RequestManager);
  }

  EventTy isDataExchangable(MPIRequestManagerTy RequestManager) {
    int32_t DstDeviceId = 0;
    bool QueryResult = false;
    RequestManager.receive(&DstDeviceId, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t SrcDeviceId, PluginId;

    std::tie(PluginId, SrcDeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    QueryResult = PluginManager.Plugins[PluginId]->isDataExchangable(
        SrcDeviceId, DstDeviceId);

    RequestManager.send(&QueryResult, sizeof(bool), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy allocateBuffer(MPIRequestManagerTy RequestManager) {
    int64_t Size = 0;
    int32_t Kind = 0;
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&Kind, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    void *Buffer = PluginManager.Plugins[PluginId]->data_alloc(DeviceId, Size,
                                                               nullptr, Kind);

    RequestManager.send(&Buffer, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy deleteBuffer(MPIRequestManagerTy RequestManager) {
    void *Buffer = nullptr;
    int32_t Kind = 0;

    RequestManager.receive(&Buffer, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Kind, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_delete(DeviceId, Buffer, Kind);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy submit(MPIRequestManagerTy RequestManager) {
    void *TgtPtr = nullptr, *HstAsyncInfoPtr = nullptr;
    int64_t Size = 0;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    RequestManager.receive(&TgtPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginDataHandle DataHandler(&PluginManager, PluginId, DeviceId, Size);
    RequestManager.receiveInBatchs(DataHandler.HstPtr, Size);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    // Issues data transfer on the device
    // Only one call at a time to avoid the
    // risk of overwriting device queues
    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_submit_async(
          DeviceId, TgtPtr, DataHandler.HstPtr, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy retrieve(MPIRequestManagerTy RequestManager) {
    void *TgtPtr = nullptr, *HstAsyncInfoPtr = nullptr;
    int64_t Size = 0;
    bool DeviceOpStatus = true;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    RequestManager.receive(&TgtPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginDataHandle DataHandler(&PluginManager, PluginId, DeviceId, Size);

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_retrieve_async(
          DeviceId, DataHandler.HstPtr, TgtPtr, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    if (auto Error =
            co_await waitAsyncOpEnd(PluginId, DeviceId, HstAsyncInfoPtr);
        Error)
      REPORT("Retrieve event failed with msg: %s\n",
             toString(std::move(Error)).data());

    RequestManager.send(&DeviceOpStatus, sizeof(bool), MPI_BYTE);

    if (!DeviceOpStatus)
      co_return (co_await RequestManager);

    RequestManager.sendInBatchs(DataHandler.HstPtr, Size);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy exchange(MPIRequestManagerTy RequestManager) {
    void *SrcPtr = nullptr, *DstPtr = nullptr;
    int DstDeviceId = 0;
    int64_t Size = 0;
    void *HstAsyncInfoPtr = nullptr;

    RequestManager.receive(&SrcPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&DstDeviceId, 1, MPI_INT);
    RequestManager.receive(&DstPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, SrcDeviceId;

    std::tie(PluginId, SrcDeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_exchange_async(
          SrcDeviceId, SrcPtr, DstDeviceId, DstPtr, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy exchangeSrc(MPIRequestManagerTy RequestManager) {
    void *SrcBuffer, *HstAsyncInfoPtr = nullptr;
    int64_t Size;
    int DstRank;

    // Save head node rank
    int HeadNodeRank = RequestManager.OtherRank;

    RequestManager.receive(&SrcBuffer, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&DstRank, 1, MPI_INT);
    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginDataHandle DataHandler(&PluginManager, PluginId, DeviceId, Size);

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_retrieve_async(
          DeviceId, DataHandler.HstPtr, SrcBuffer, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    if (auto Error =
            co_await waitAsyncOpEnd(PluginId, DeviceId, HstAsyncInfoPtr);
        Error)
      co_return Error;

    // Set the Destination Rank in RequestManager
    RequestManager.OtherRank = DstRank;

    // Send buffer to target device
    RequestManager.sendInBatchs(DataHandler.HstPtr, Size);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // Set the HeadNode Rank to send the final notificatin
    RequestManager.OtherRank = HeadNodeRank;

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy exchangeDst(MPIRequestManagerTy RequestManager) {
    void *DstBuffer, *HstAsyncInfoPtr = nullptr;
    int64_t Size;
    // Save head node rank
    int SrcRank, HeadNodeRank = RequestManager.OtherRank;

    RequestManager.receive(&DstBuffer, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&SrcRank, 1, MPI_INT);
    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginDataHandle DataHandler(&PluginManager, PluginId, DeviceId, Size);

    // Set the Source Rank in RequestManager
    RequestManager.OtherRank = SrcRank;

    // Receive buffer from the Source device
    RequestManager.receiveInBatchs(DataHandler.HstPtr, Size);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    {
      std::lock_guard<std::mutex> Lock(TgtAsyncInfo->AsyncInfoMutex);
      PluginManager.Plugins[PluginId]->data_submit_async(
          DeviceId, DstBuffer, DataHandler.HstPtr, Size,
          TgtAsyncInfo->AsyncInfoPtr.get());
    }

    if (auto Error =
            co_await waitAsyncOpEnd(PluginId, DeviceId, HstAsyncInfoPtr);
        Error)
      co_return Error;

    // Set the HeadNode Rank to send the final notificatin
    RequestManager.OtherRank = HeadNodeRank;

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy launchKernel(MPIRequestManagerTy RequestManager) {
    void *TgtEntryPtr = nullptr, *HostAsyncInfoPtr = nullptr;
    KernelArgsTy KernelArgs;

    llvm::SmallVector<void *> TgtArgs;
    llvm::SmallVector<ptrdiff_t> TgtOffsets;

    uint32_t NumArgs = 0;

    RequestManager.receive(&NumArgs, 1, MPI_UINT32_T);
    RequestManager.receive(&HostAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    TgtArgs.resize(NumArgs);
    TgtOffsets.resize(NumArgs);

    RequestManager.receive(&TgtEntryPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(TgtArgs.data(), NumArgs * sizeof(void *), MPI_BYTE);
    RequestManager.receive(TgtOffsets.data(), NumArgs * sizeof(ptrdiff_t),
                           MPI_BYTE);

    RequestManager.receive(&KernelArgs, sizeof(KernelArgsTy), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    auto *TgtAsyncInfo = MapAsyncInfo(HostAsyncInfoPtr)->AsyncInfoPtr.get();

    PluginManager.Plugins[PluginId]->launch_kernel(
        DeviceId, TgtEntryPtr, TgtArgs.data(), TgtOffsets.data(), &KernelArgs,
        TgtAsyncInfo);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy loadBinary(MPIRequestManagerTy RequestManager) {
    // Receive the target table sizes.
    size_t ImageSize = 0;
    size_t EntryCount = 0;
    RequestManager.receive(&ImageSize, 1, MPI_UINT64_T);
    RequestManager.receive(&EntryCount, 1, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    llvm::SmallVector<size_t> EntryNameSizes(EntryCount);

    RequestManager.receive(EntryNameSizes.begin(), EntryCount, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // Create the device name with the appropriate sizes and receive its
    // content.
    DeviceImage *Image =
        RemoteImages
            .emplace_back(std::make_unique<DeviceImage>(ImageSize, EntryCount))
            .get();

    Image->setImageEntries(EntryNameSizes);

    // Received the image bytes and the table entries.
    RequestManager.receive(Image->ImageStart, ImageSize, MPI_BYTE);

    for (size_t I = 0; I < EntryCount; I++) {
      RequestManager.receive(&Image->Entries[I].addr, 1, MPI_UINT64_T);
      RequestManager.receive(Image->Entries[I].name, EntryNameSizes[I],
                             MPI_CHAR);
      RequestManager.receive(&Image->Entries[I].size, 1, MPI_UINT64_T);
      RequestManager.receive(&Image->Entries[I].flags, 1, MPI_INT32_T);
      RequestManager.receive(&Image->Entries[I].data, 1, MPI_INT32_T);
    }

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    __tgt_device_binary Binary;

    PluginManager.Plugins[PluginId]->load_binary(DeviceId, Image, &Binary);

    RequestManager.send(&Binary.handle, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy getGlobal(MPIRequestManagerTy RequestManager) {
    __tgt_device_binary Binary;
    uint64_t Size = 0;
    llvm::SmallVector<char> Name;
    void *DevicePtr = nullptr;
    uint32_t NameSize = 0;

    RequestManager.receive(&Binary.handle, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_UINT64_T);
    RequestManager.receive(&NameSize, 1, MPI_UINT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    Name.resize(NameSize);
    RequestManager.receive(Name.data(), NameSize, MPI_CHAR);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->get_global(Binary, Size, Name.data(),
                                                &DevicePtr);

    RequestManager.send(&DevicePtr, sizeof(void *), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy getFunction(MPIRequestManagerTy RequestManager) {
    __tgt_device_binary Binary;
    uint32_t Size = 0;
    llvm::SmallVector<char> Name;
    void *KernelPtr = nullptr;

    RequestManager.receive(&Binary.handle, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_UINT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    Name.resize(Size);
    RequestManager.receive(Name.data(), Size, MPI_CHAR);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->get_function(Binary, Name.data(),
                                                  &KernelPtr);

    RequestManager.send(&KernelPtr, sizeof(void *), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy synchronize(MPIRequestManagerTy RequestManager) {
    void *HstAsyncInfoPtr = nullptr;
    bool DeviceOpStatus = true;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    if (auto Error =
            co_await waitAsyncOpEnd(PluginId, DeviceId, HstAsyncInfoPtr);
        Error)
      REPORT("Synchronize event failed with msg: %s\n",
             toString(std::move(Error)).data());

    RequestManager.send(&DeviceOpStatus, sizeof(bool), MPI_BYTE);

    if (!DeviceOpStatus)
      co_return (co_await RequestManager);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy ompfilePing(MPIRequestManagerTy RequestManager) {
    uint64_t Token = 0;

    RequestManager.receive(&Token, 1, MPI_UINT64_T);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    RequestManager.send(&Token, 1, MPI_UINT64_T);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  bool canUseOmpFileOpenCache(int Flags) const {
    if (!OmpFileOpenCacheEnable)
      return false;
    if (Flags & (O_CREAT | O_EXCL | O_TRUNC))
      return false;
    const int AccessMode = Flags & O_ACCMODE;
    if (AccessMode != O_RDONLY && AccessMode != O_WRONLY &&
        AccessMode != O_RDWR)
      return false;
    return true;
  }

  std::string getOmpFileOpenCacheKey(const char *Path, int Flags, int Mode) const {
    std::string Key = Path ? Path : "";
    Key.push_back('\x1f');
    Key += std::to_string(Flags);
    Key.push_back('\x1f');
    Key += std::to_string(Mode);
    return Key;
  }

  int openWithOptionalCache(const char *Path, int Flags, int Mode, int &ErrnoOut) {
    ErrnoOut = 0;
    OmpFileStatsOpenRequests.fetch_add(1, std::memory_order_relaxed);

    if (!Path) {
      ErrnoOut = EINVAL;
      return -1;
    }

    if (!canUseOmpFileOpenCache(Flags)) {
      OmpFileStatsOpenSyscalls.fetch_add(1, std::memory_order_relaxed);
      int Fd = ::open(Path, Flags, static_cast<mode_t>(Mode));
      if (Fd < 0)
        ErrnoOut = errno;
      return Fd;
    }

    const std::string Key = getOmpFileOpenCacheKey(Path, Flags, Mode);
    const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
    auto It = OmpFileOpenCacheByKey.find(Key);
    if (It != OmpFileOpenCacheByKey.end()) {
      It->second.RefCount += 1;
      OmpFileStatsOpenCacheHits.fetch_add(1, std::memory_order_relaxed);
      return It->second.Fd;
    }

    OmpFileStatsOpenSyscalls.fetch_add(1, std::memory_order_relaxed);
    int Fd = ::open(Path, Flags, static_cast<mode_t>(Mode));
    if (Fd < 0) {
      ErrnoOut = errno;
      return -1;
    }

    OmpFileOpenCacheByKey.emplace(Key, OmpFileOpenCacheEntry{Fd, 1});
    OmpFileOpenCacheFdToKey[Fd] = Key;
    return Fd;
  }

  int closeWithOptionalCache(int Fd, int &ErrnoOut) {
    ErrnoOut = 0;
    OmpFileStatsCloseRequests.fetch_add(1, std::memory_order_relaxed);

    if (!OmpFileOpenCacheEnable) {
      OmpFileStatsCloseSyscalls.fetch_add(1, std::memory_order_relaxed);
      int Ret = ::close(Fd);
      if (Ret != 0)
        ErrnoOut = errno;
      return Ret;
    }

    bool NeedSysClose = false;
    {
      const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
      auto FdIt = OmpFileOpenCacheFdToKey.find(Fd);
      if (FdIt != OmpFileOpenCacheFdToKey.end()) {
        auto KeyIt = OmpFileOpenCacheByKey.find(FdIt->second);
        if (KeyIt != OmpFileOpenCacheByKey.end()) {
          OmpFileOpenCacheEntry &Entry = KeyIt->second;
          if (Entry.RefCount == 0) {
            ErrnoOut = EBADF;
            return -1;
          }

          Entry.RefCount -= 1;
          if (Entry.RefCount > 0 || OmpFileOpenCacheKeepOpen) {
            OmpFileStatsCloseDeferred.fetch_add(1, std::memory_order_relaxed);
            return 0;
          }

          OmpFileOpenCacheByKey.erase(KeyIt);
          OmpFileOpenCacheFdToKey.erase(FdIt);
          NeedSysClose = true;
        } else {
          OmpFileOpenCacheFdToKey.erase(FdIt);
          NeedSysClose = true;
        }
      } else {
        NeedSysClose = true;
      }
    }

    if (!NeedSysClose)
      return 0;

    OmpFileStatsCloseSyscalls.fetch_add(1, std::memory_order_relaxed);
    int Ret = ::close(Fd);
    if (Ret != 0)
      ErrnoOut = errno;
    return Ret;
  }

  void drainOmpFileOpenCache() {
    std::vector<int> FdsToClose;
    {
      const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
      if (OmpFileOpenCacheByKey.empty() && OmpFileOpenCacheFdToKey.empty())
        return;

      FdsToClose.reserve(OmpFileOpenCacheByKey.size());
      for (const auto &It : OmpFileOpenCacheByKey)
        FdsToClose.push_back(It.second.Fd);
      OmpFileOpenCacheByKey.clear();
      OmpFileOpenCacheFdToKey.clear();
    }

    for (int Fd : FdsToClose) {
      OmpFileStatsCloseSyscalls.fetch_add(1, std::memory_order_relaxed);
      (void)::close(Fd);
    }
  }

  void reportOmpFileStats(const char *Scope) {
    if (!OmpFileOptStats)
      return;

    uint64_t OpenReq = OmpFileStatsOpenRequests.load(std::memory_order_relaxed);
    uint64_t OpenSys = OmpFileStatsOpenSyscalls.load(std::memory_order_relaxed);
    uint64_t OpenHits =
        OmpFileStatsOpenCacheHits.load(std::memory_order_relaxed);
    uint64_t CloseReq =
        OmpFileStatsCloseRequests.load(std::memory_order_relaxed);
    uint64_t CloseSys =
        OmpFileStatsCloseSyscalls.load(std::memory_order_relaxed);
    uint64_t CloseDeferred =
        OmpFileStatsCloseDeferred.load(std::memory_order_relaxed);
    size_t CacheEntries = 0;

    {
      const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
      CacheEntries = OmpFileOpenCacheByKey.size();
    }

    fprintf(stderr,
            "MPIProxyDevice --> OMPFile stats [%s] rank=%d "
            "open_req=%llu open_sys=%llu open_hits=%llu "
            "close_req=%llu close_sys=%llu close_deferred=%llu "
            "cache_entries=%zu\n",
            Scope ? Scope : "unknown", EventSystem.LocalRank,
            static_cast<unsigned long long>(OpenReq),
            static_cast<unsigned long long>(OpenSys),
            static_cast<unsigned long long>(OpenHits),
            static_cast<unsigned long long>(CloseReq),
            static_cast<unsigned long long>(CloseSys),
            static_cast<unsigned long long>(CloseDeferred), CacheEntries);
  }

  EventTy ompfileOpen(MPIRequestManagerTy RequestManager) {
    uint32_t PathSize = 0;
    int Flags = 0;
    int Mode = 0;

    RequestManager.receive(&PathSize, 1, MPI_UINT32_T);
    RequestManager.receive(&Flags, 1, MPI_INT);
    RequestManager.receive(&Mode, 1, MPI_INT);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::string Path;
    if (PathSize > 0) {
      Path.resize(PathSize, '\0');
      RequestManager.receive(Path.data(), PathSize, MPI_CHAR);
      if (auto Error = co_await RequestManager; Error)
        co_return Error;
    }

    if (!Path.empty() && Path.back() == '\0')
      Path.pop_back();

    int Errno = 0;
    int Fd = openWithOptionalCache(Path.c_str(), Flags, Mode, Errno);

    RequestManager.send(&Fd, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileClose(MPIRequestManagerTy RequestManager) {
    int Fd = -1;
    RequestManager.receive(&Fd, 1, MPI_INT);
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int Errno = 0;
    int Ret = closeWithOptionalCache(Fd, Errno);

    RequestManager.send(&Ret, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfilePread(MPIRequestManagerTy RequestManager) {
    int Fd = -1;
    int64_t Offset = 0;
    uint64_t Size = 0;

    RequestManager.receive(&Fd, 1, MPI_INT);
    RequestManager.receive(&Offset, 1, MPI_INT64_T);
    RequestManager.receive(&Size, 1, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::vector<char> Buffer;
    if (Size > 0)
      Buffer.resize(Size);

    ssize_t BytesRead = 0;
    int Errno = 0;
    if (Size > 0) {
      BytesRead = ::pread(Fd, Buffer.data(), Size, static_cast<off_t>(Offset));
      if (BytesRead < 0)
        Errno = errno;
    }

    int Ret = (BytesRead < 0) ? -1 : 0;
    uint64_t PayloadSize = (BytesRead > 0) ? static_cast<uint64_t>(BytesRead) : 0;

    RequestManager.send(&Ret, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(&PayloadSize, 1, MPI_UINT64_T);
    if (PayloadSize > 0)
      RequestManager.sendInBatchs(Buffer.data(), PayloadSize);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfilePwrite(MPIRequestManagerTy RequestManager) {
    int Fd = -1;
    int64_t Offset = 0;
    uint64_t Size = 0;

    RequestManager.receive(&Fd, 1, MPI_INT);
    RequestManager.receive(&Offset, 1, MPI_INT64_T);
    RequestManager.receive(&Size, 1, MPI_UINT64_T);

    std::vector<char> Buffer;
    if (Size > 0) {
      Buffer.resize(Size);
      RequestManager.receiveInBatchs(Buffer.data(), Size);
    }

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    ssize_t BytesWritten = 0;
    int Errno = 0;
    if (Size > 0) {
      BytesWritten = ::pwrite(Fd, Buffer.data(), Size, static_cast<off_t>(Offset));
      if (BytesWritten < 0)
        Errno = errno;
    }

    int Ret = (BytesWritten < 0) ? -1 : 0;
    RequestManager.send(&Ret, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }



  bool isWorkerRank(int Rank) const {
    return Rank >= 0 && Rank < EventSystem.WorldSize - 1;
  }

  int getHeadnodeRank() const { return 0; }

  OmpFileIOPlan buildSchedulePlan(const OmpFileIORequest &Request,
                                  const char *Path) {
    OmpFileHeadnodeManager &Manager = OmpFileHeadnodeManager::instance();
    Manager.initialize(EventSystem.WorldSize, getHeadnodeRank());
    return Manager.planRequest(Request, Path, EventSystem.LocalRank);
  }

  bool buildScheduleBatchPlan(const OmpFileIOBatchRequest &Request,
                              const void *Payload, uint64_t PayloadBytes,
                              OmpFileIOBatchPlan &Plan,
                              std::vector<uint8_t> &PlanPayload) {
    Plan = {};
    PlanPayload.clear();
    Plan.AbiVersion = OMPFILE_SCHED_BATCH_ABI_VERSION;
    Plan.SegmentCount = Request.SegmentCount;
    Plan.BatchId = Request.BatchId;
    Plan.PlanFlags = OMPFILE_BATCH_PLAN_BATCH_API;

    if (Request.AbiVersion != OMPFILE_SCHED_BATCH_ABI_VERSION) {
      Plan.Status = -1;
      Plan.Errno = EPROTO;
      return true;
    }

    if (Request.PayloadBytes != PayloadBytes) {
      Plan.Status = -1;
      Plan.Errno = EINVAL;
      return true;
    }

    const uint64_t SegmentCount = static_cast<uint64_t>(Request.SegmentCount);
    const uint64_t SegmentBytes =
        SegmentCount * static_cast<uint64_t>(sizeof(OmpFileIOBatchSegment));
    if (Request.SegmentCount != 0 &&
        SegmentBytes / sizeof(OmpFileIOBatchSegment) != SegmentCount) {
      Plan.Status = -1;
      Plan.Errno = EOVERFLOW;
      return true;
    }

    if (SegmentBytes != PayloadBytes) {
      Plan.Status = -1;
      Plan.Errno = EINVAL;
      return true;
    }

    if (PayloadBytes > 0 && !Payload) {
      Plan.Status = -1;
      Plan.Errno = EINVAL;
      return true;
    }

    if (Request.SegmentCount == 0) {
      Plan.PayloadBytes = 0;
      return true;
    }

    const auto *Segments =
        static_cast<const OmpFileIOBatchSegment *>(Payload);
    std::vector<OmpFileIOBatchPlanEntry> Entries(Request.SegmentCount);
    const bool FailOnAnyError =
        (Request.RequestFlags & OMPFILE_BATCH_REQ_FAIL_ON_ANY_ERROR) != 0;

    bool SawError = false;
    bool StopScheduling = false;
    int FirstErrno = 0;
    for (uint32_t I = 0; I < Request.SegmentCount; ++I) {
      const OmpFileIOBatchSegment &Segment = Segments[I];
      OmpFileIOBatchPlanEntry &Entry = Entries[I];
      Entry.SegmentId = Segment.SegmentId;
      Entry.Offset = Segment.Offset;
      Entry.Size = Segment.Size;
      Entry.PlanFlags = OMPFILE_BATCH_PLAN_BATCH_API;

      if (StopScheduling) {
        Entry.Status = -1;
        Entry.Errno = ECANCELED;
        continue;
      }

      OmpFileIORequest ScalarRequest{};
      ScalarRequest.RequestId =
          Segment.SegmentId != 0
              ? Segment.SegmentId
              : (Request.BatchId + static_cast<uint64_t>(I) + 1);
      ScalarRequest.Op = OmpFileIOOp::PREAD;
      ScalarRequest.FileHandle = Segment.FileHandle;
      ScalarRequest.ClientRank = Segment.ClientRank;
      ScalarRequest.Offset = Segment.Offset;
      ScalarRequest.Size = Segment.Size;

      OmpFileIOPlan ScalarPlan = buildSchedulePlan(ScalarRequest, nullptr);
      Entry.AggregatorRank = ScalarPlan.AggregatorRank;
      Entry.RemoteHandle = ScalarPlan.RemoteHandle;
      Entry.Status = ScalarPlan.Status;
      Entry.Errno = ScalarPlan.Errno;

      if (Entry.Status != 0) {
        SawError = true;
        if (FirstErrno == 0)
          FirstErrno = Entry.Errno;
        if (FailOnAnyError)
          StopScheduling = true;
      }
    }

    const uint64_t EntryBytes =
        static_cast<uint64_t>(Entries.size()) *
        static_cast<uint64_t>(sizeof(OmpFileIOBatchPlanEntry));
    if (EntryBytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      Plan.Status = -1;
      Plan.Errno = EOVERFLOW;
      return true;
    }

    PlanPayload.resize(static_cast<size_t>(EntryBytes));
    if (!PlanPayload.empty())
      std::memcpy(PlanPayload.data(), Entries.data(), PlanPayload.size());
    Plan.PayloadBytes = static_cast<uint32_t>(PlanPayload.size());

    if (SawError) {
      Plan.Status = -1;
      Plan.Errno = FirstErrno != 0 ? FirstErrno : EIO;
    }

    return true;
  }

  template <class EventFuncTy, typename... ArgsTy>
    requires std::invocable<EventFuncTy, MPIRequestManagerTy, ArgsTy...>
  EventTy createRankEvent(EventFuncTy EventFunc, EventTypeTy EventType,
                          int TargetRank, int TargetDeviceId, ArgsTy... Args) {
    traceOmpFile("createRankEvent enter type=%s target_rank=%d target_dev=%d\n",
                 EventTypeToString(EventType).c_str(), TargetRank,
                 TargetDeviceId);
    if (!isWorkerRank(TargetRank)) {
      REPORT("Invalid OMPFile target rank %d for event %s.\n", TargetRank,
             EventTypeToString(EventType).c_str());
      traceOmpFile("createRankEvent invalid target type=%s rank=%d\n",
                   EventTypeToString(EventType).c_str(), TargetRank);
      return EventTy{};
    }

    const int EventTag = EventSystem.createNewEventTag();
    auto &EventComm = EventSystem.getNewEventComm(EventTag);
    int EventNotificationInfo[] = {static_cast<int>(EventType), EventTag,
                                   TargetDeviceId};
    MPI_Request NotificationRequest = MPI_REQUEST_NULL;
    const int MPIError =
        MPI_Isend(EventNotificationInfo, 3, MPI_INT, TargetRank,
                  static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                  EventSystem.GateThreadComm, &NotificationRequest);

    if (MPIError != MPI_SUCCESS) {
      REPORT("Failed to notify rank %d for OMPFile event %s. MPI error=%d\n",
             TargetRank, EventTypeToString(EventType).c_str(), MPIError);
      traceOmpFile("createRankEvent notify fail type=%s rank=%d mpi_err=%d\n",
                   EventTypeToString(EventType).c_str(), TargetRank, MPIError);
      return EventTy{};
    }

    MPIRequestManagerTy RequestManager(EventComm, EventTag, TargetRank,
                                       TargetDeviceId, {NotificationRequest});
    RequestManager.EventType = EventNotificationInfo[0];

    auto Event = EventFunc(std::move(RequestManager), Args...);
    Event.setEventType(EventType);
    traceOmpFile("createRankEvent exit type=%s tag=%d target_rank=%d\n",
                 EventTypeToString(EventType).c_str(), EventTag, TargetRank);
    return Event;
  }

  bool waitForEvent(EventTy &Event, const char *OpName) {
    traceOmpFile("waitForEvent enter op=%s empty=%d\n", OpName ? OpName : "?",
                 static_cast<int>(Event.empty()));
    if (Event.empty()) {
      errno = EIO;
      REPORT("OMPFile %s failed to create event.\n", OpName);
      traceOmpFile("waitForEvent empty op=%s\n", OpName ? OpName : "?");
      return false;
    }

    Event.wait();
    if (auto Error = Event.getError()) {
      errno = EIO;
      REPORT("OMPFile %s event failed: %s\n", OpName,
             toString(std::move(Error)).c_str());
      traceOmpFile("waitForEvent error op=%s\n", OpName ? OpName : "?");
      return false;
    }
    traceOmpFile("waitForEvent success op=%s\n", OpName ? OpName : "?");
    return true;
  }

  bool dispatchSchedRequest(const OmpFileIORequest &Request, const char *Path,
                            OmpFileIOPlan &Plan) {
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      Plan = buildSchedulePlan(Request, Path);
    } else {
      EventTy Event = createRankEvent(
          OriginEvents::ompfileSchedRequest, EventTypeTy::OMPFILE_SCHED_REQUEST,
          getHeadnodeRank(), /*TargetDeviceId=*/0, &Request, Path, &Plan);
      if (!waitForEvent(Event, "sched_request"))
        return false;
    }

    if (Plan.Status != 0) {
      errno = Plan.Errno;
      return false;
    }

    return true;
  }

  bool dispatchSchedBatchRequest(const OmpFileIOBatchRequest &Request,
                                 const void *RequestPayload,
                                 uint64_t RequestPayloadBytes,
                                 OmpFileIOBatchPlan &Plan, void *PlanPayload,
                                 uint64_t PlanPayloadCapBytes,
                                 uint64_t &PlanPayloadOutBytes) {
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      std::vector<uint8_t> LocalPayload;
      if (!buildScheduleBatchPlan(Request, RequestPayload, RequestPayloadBytes,
                                  Plan, LocalPayload))
        return false;

      PlanPayloadOutBytes = LocalPayload.size();
      if (PlanPayloadOutBytes > PlanPayloadCapBytes) {
        errno = ENOBUFS;
        return false;
      }

      if (PlanPayloadOutBytes > 0) {
        if (!PlanPayload) {
          errno = EINVAL;
          return false;
        }
        std::memcpy(PlanPayload, LocalPayload.data(),
                    static_cast<size_t>(PlanPayloadOutBytes));
      }

      return true;
    }

    EventTy Event = createRankEvent(
        OriginEvents::ompfileSchedBatchRequest,
        EventTypeTy::OMPFILE_SCHED_REQUEST_BATCH, getHeadnodeRank(),
        /*TargetDeviceId=*/0, &Request, RequestPayload, RequestPayloadBytes,
        &Plan, PlanPayload, PlanPayloadCapBytes, &PlanPayloadOutBytes);
    if (!waitForEvent(Event, "sched_request_batch"))
      return false;

    return true;
  }

  int registerRemoteHandle(int Rank, int RemoteHandle) {
    const int LocalHandle =
        NextOmpFileHandle.fetch_add(1, std::memory_order_relaxed);
    const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
    OmpFileHandles[LocalHandle] = {Rank, RemoteHandle};
    traceOmpFile("registerRemoteHandle local=%d rank=%d remote=%d map_size=%zu\n",
                 LocalHandle, Rank, RemoteHandle, OmpFileHandles.size());
    return LocalHandle;
  }

  bool findRemoteHandle(int LocalHandle, OmpFileHandleEntry &Entry) {
    const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
    auto It = OmpFileHandles.find(LocalHandle);
    if (It == OmpFileHandles.end()) {
      errno = EBADF;
      traceOmpFile("findRemoteHandle miss local=%d map_size=%zu\n", LocalHandle,
                   OmpFileHandles.size());
      return false;
    }
    Entry = It->second;
    traceOmpFile("findRemoteHandle hit local=%d rank=%d remote=%d map_size=%zu\n",
                 LocalHandle, Entry.Rank, Entry.RemoteHandle,
                 OmpFileHandles.size());
    return true;
  }

  bool eraseRemoteHandle(int LocalHandle, OmpFileHandleEntry &Entry) {
    const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
    auto It = OmpFileHandles.find(LocalHandle);
    if (It == OmpFileHandles.end()) {
      errno = EBADF;
      traceOmpFile("eraseRemoteHandle miss local=%d map_size=%zu\n", LocalHandle,
                   OmpFileHandles.size());
      return false;
    }
    Entry = It->second;
    OmpFileHandles.erase(It);
    traceOmpFile(
        "eraseRemoteHandle hit local=%d rank=%d remote=%d map_size=%zu\n",
        LocalHandle, Entry.Rank, Entry.RemoteHandle, OmpFileHandles.size());
    return true;
  }

  bool openOnRank(int Rank, const char *Path, int Flags, int Mode,
                  int &RemoteHandle) {
    if (!Path) {
      errno = EINVAL;
      return false;
    }

    if (Rank == EventSystem.LocalRank) {
      int OpenErrno = 0;
      int Fd = openWithOptionalCache(Path, Flags, Mode, OpenErrno);
      if (Fd < 0)
        errno = OpenErrno;
      if (Fd < 0)
        return false;
      RemoteHandle = Fd;
      return true;
    }

    int RemoteErrno = 0;
    EventTy Event =
        createRankEvent(OriginEvents::ompfileOpen, EventTypeTy::OMPFILE_OPEN,
                        Rank, /*TargetDeviceId=*/0, Path, Flags, Mode,
                        &RemoteHandle, &RemoteErrno);
    if (!waitForEvent(Event, "open"))
      return false;

    if (RemoteHandle < 0) {
      errno = RemoteErrno;
      return false;
    }
    return true;
  }

  bool closeOnRank(int Rank, int RemoteHandle) {
    if (Rank == EventSystem.LocalRank) {
      int CloseErrno = 0;
      if (closeWithOptionalCache(RemoteHandle, CloseErrno) != 0) {
        errno = CloseErrno;
        return false;
      }
      return true;
    }

    int CloseRet = -1;
    int RemoteErrno = 0;
    EventTy Event = createRankEvent(
        OriginEvents::ompfileClose, EventTypeTy::OMPFILE_CLOSE, Rank,
        /*TargetDeviceId=*/0, RemoteHandle, &CloseRet, &RemoteErrno);
    if (!waitForEvent(Event, "close"))
      return false;
    if (CloseRet != 0) {
      errno = RemoteErrno;
      return false;
    }
    return true;
  }

  bool preadOnRank(int Rank, int RemoteHandle, int64_t Offset, void *Buffer,
                   uint64_t Size) {
    if (!Buffer && Size > 0) {
      errno = EINVAL;
      return false;
    }

    if (Rank == EventSystem.LocalRank) {
      const ssize_t BytesRead =
          ::pread(RemoteHandle, Buffer, Size, static_cast<off_t>(Offset));
      if (BytesRead < 0)
        return false;
      if (static_cast<uint64_t>(BytesRead) < Size) {
        errno = EIO;
        return false;
      }
      return true;
    }

    int IoRet = -1;
    int RemoteErrno = 0;
    uint64_t Bytes = 0;
    EventTy Event = createRankEvent(
        OriginEvents::ompfilePread, EventTypeTy::OMPFILE_PREAD, Rank,
        /*TargetDeviceId=*/0, RemoteHandle, Offset, Buffer, Size, &IoRet,
        &RemoteErrno, &Bytes);
    if (!waitForEvent(Event, "pread"))
      return false;
    if (IoRet != 0) {
      errno = RemoteErrno;
      return false;
    }
    if (Bytes < Size) {
      errno = EIO;
      return false;
    }
    return true;
  }

  bool pwriteOnRank(int Rank, int RemoteHandle, int64_t Offset,
                    const void *Buffer, uint64_t Size) {
    if (!Buffer && Size > 0) {
      errno = EINVAL;
      return false;
    }

    if (Rank == EventSystem.LocalRank) {
      const ssize_t BytesWritten =
          ::pwrite(RemoteHandle, Buffer, Size, static_cast<off_t>(Offset));
      if (BytesWritten < 0)
        return false;
      if (static_cast<uint64_t>(BytesWritten) < Size) {
        errno = EIO;
        return false;
      }
      return true;
    }

    int IoRet = -1;
    int RemoteErrno = 0;
    EventTy Event = createRankEvent(
        OriginEvents::ompfilePwrite, EventTypeTy::OMPFILE_PWRITE, Rank,
        /*TargetDeviceId=*/0, RemoteHandle, Offset, Buffer, Size, &IoRet,
        &RemoteErrno);
    if (!waitForEvent(Event, "pwrite"))
      return false;
    if (IoRet != 0) {
      errno = RemoteErrno;
      return false;
    }
    return true;
  }

  int selectAggregatorRankForOpen(const char *Path, int Flags, int Mode) {
    int Rank = isWorkerRank(EventSystem.LocalRank) ? EventSystem.LocalRank : 0;
    const char *SchedulerEnv = std::getenv("LIBOMPFILE_SCHEDULER");
    if (!SchedulerEnv || std::strcmp(SchedulerEnv, "HEADNODE") != 0)
      return Rank;

    OmpFileIORequest Request{};
    Request.RequestId =
        NextSchedRequestId.fetch_add(1, std::memory_order_relaxed);
    Request.Op = OmpFileIOOp::OPEN;
    Request.Flags = Flags;
    Request.Mode = Mode;
    Request.ClientRank = EventSystem.LocalRank;
    Request.PathSize = Path ? static_cast<uint32_t>(std::strlen(Path) + 1) : 0;

    OmpFileIOPlan Plan{};
    if (!dispatchSchedRequest(Request, Path, Plan)) {
      DP("OMPFile scheduler request failed in proxy rank %d; using local rank\n",
         EventSystem.LocalRank);
      return Rank;
    }

    if (isWorkerRank(Plan.AggregatorRank))
      return Plan.AggregatorRank;
    return Rank;
  }

  int mppInit() {
    DP("ompfile_mpp_init via proxy runtime (rank=%d, world=%d)\n",
       EventSystem.LocalRank, EventSystem.WorldSize);
    if (EventSystem.LocalRank == getHeadnodeRank())
      OmpFileHeadnodeManager::instance().initialize(EventSystem.WorldSize,
                                                    getHeadnodeRank());
    const auto State = EventSystem.EventSystemState.load();
    if (State == EventSystemStateTy::RUNNING ||
        State == EventSystemStateTy::INITIALIZED)
      return OFFLOAD_SUCCESS;
    return OFFLOAD_FAIL;
  }

  int mppSubmit(uint64_t Token) {
    if (EventSystem.LocalRank == getHeadnodeRank()) {
      const std::lock_guard<std::mutex> Lock(MppEventMutex);
      CompletedMppTokens.insert(Token);
      return OFFLOAD_SUCCESS;
    }

    EventTy Event = createRankEvent(OriginEvents::ompfilePing,
                                    EventTypeTy::OMPFILE_PING,
                                    getHeadnodeRank(), /*TargetDeviceId=*/0,
                                    Token);
    if (Event.empty())
      return OFFLOAD_FAIL;

    const std::lock_guard<std::mutex> Lock(MppEventMutex);
    if (MppEvents.count(Token))
      return OFFLOAD_FAIL;
    MppEvents.emplace(Token, std::move(Event));
    return OFFLOAD_SUCCESS;
  }

  int mppPoll(uint64_t Token, int *Done) {
    if (!Done)
      return OFFLOAD_FAIL;

    const std::lock_guard<std::mutex> Lock(MppEventMutex);
    auto CompletedIt = CompletedMppTokens.find(Token);
    if (CompletedIt != CompletedMppTokens.end()) {
      *Done = 1;
      CompletedMppTokens.erase(CompletedIt);
      return OFFLOAD_SUCCESS;
    }

    auto It = MppEvents.find(Token);
    if (It == MppEvents.end())
      return OFFLOAD_FAIL;

    It->second.advance();
    if (!It->second.done()) {
      *Done = 0;
      return OFFLOAD_SUCCESS;
    }

    *Done = 1;
    auto Error = It->second.getError();
    MppEvents.erase(It);
    if (Error)
      return OFFLOAD_FAIL;
    return OFFLOAD_SUCCESS;
  }

  int mppOpen(const char *Path, int Flags, int Mode, int *Handle) {
    if (!Path || !Handle)
      return OFFLOAD_FAIL;

    traceOmpFile("mppOpen enter path=%s flags=0x%x mode=%o\n", Path, Flags,
                 Mode);
    const int Rank = selectAggregatorRankForOpen(Path, Flags, Mode);
    int RemoteHandle = -1;
    if (!openOnRank(Rank, Path, Flags, Mode, RemoteHandle))
      return OFFLOAD_FAIL;

    *Handle = registerRemoteHandle(Rank, RemoteHandle);
    traceOmpFile("mppOpen exit local=%d rank=%d remote=%d\n", *Handle, Rank,
                 RemoteHandle);
    return OFFLOAD_SUCCESS;
  }

  int mppClose(int Handle) {
    traceOmpFile("mppClose enter local=%d\n", Handle);
    OmpFileHandleEntry Entry{};
    if (!eraseRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (!closeOnRank(Entry.Rank, Entry.RemoteHandle))
      return OFFLOAD_FAIL;
    traceOmpFile("mppClose exit local=%d rank=%d remote=%d\n", Handle,
                 Entry.Rank, Entry.RemoteHandle);
    return OFFLOAD_SUCCESS;
  }

  int mppPread(int Handle, int64_t Offset, void *Buffer, uint64_t Size) {
    traceOmpFile("mppPread enter local=%d offset=%lld size=%llu\n", Handle,
                 static_cast<long long>(Offset),
                 static_cast<unsigned long long>(Size));
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (!preadOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size))
      return OFFLOAD_FAIL;
    traceOmpFile("mppPread exit local=%d rank=%d remote=%d\n", Handle,
                 Entry.Rank, Entry.RemoteHandle);
    return OFFLOAD_SUCCESS;
  }

  int mppPwrite(int Handle, int64_t Offset, const void *Buffer, uint64_t Size) {
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (!pwriteOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size))
      return OFFLOAD_FAIL;
    return OFFLOAD_SUCCESS;
  }

  int mppSchedRequest(const OmpFileIORequest *Request, const char *Path,
                      OmpFileIOPlan *Plan) {
    if (!Request || !Plan)
      return OFFLOAD_FAIL;
    if (Request->PathSize > 0 && !Path)
      return OFFLOAD_FAIL;
    if (!dispatchSchedRequest(*Request, Path, *Plan)) {
      DP("ompfile_mpp_sched_request failed in proxy runtime (rank=%d)\n",
         EventSystem.LocalRank);
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  int mppSchedBatchRequest(const OmpFileIOBatchRequest *Request,
                           const void *RequestPayload,
                           uint64_t RequestPayloadBytes,
                           OmpFileIOBatchPlan *Plan, void *PlanPayload,
                           uint64_t PlanPayloadCapBytes,
                           uint64_t *PlanPayloadOutBytes) {
    if (!Request || !Plan || !PlanPayloadOutBytes)
      return OFFLOAD_FAIL;
    if (RequestPayloadBytes > 0 && !RequestPayload)
      return OFFLOAD_FAIL;
    if (PlanPayloadCapBytes > 0 && !PlanPayload)
      return OFFLOAD_FAIL;

    if (!dispatchSchedBatchRequest(*Request, RequestPayload,
                                   RequestPayloadBytes, *Plan, PlanPayload,
                                   PlanPayloadCapBytes, *PlanPayloadOutBytes)) {
      DP("ompfile_mpp_sched_request_batch failed in proxy runtime (rank=%d)\n",
         EventSystem.LocalRank);
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int mppFinalize() {
    size_t HandleCount = 0;
    {
      const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
      HandleCount = OmpFileHandles.size();
    }
    traceOmpFile("mppFinalize enter handle_count=%zu\n", HandleCount);
    reportOmpFileStats("mpp-finalize-before-drain");
    drainOmpFileOpenCache();
    {
      const std::lock_guard<std::mutex> Lock(MppEventMutex);
      MppEvents.clear();
      CompletedMppTokens.clear();
    }
    {
      const std::lock_guard<std::mutex> Lock(OmpFileHandleMutex);
      OmpFileHandles.clear();
    }
    reportOmpFileStats("mpp-finalize-after-drain");
    traceOmpFile("mppFinalize exit\n");
    return OFFLOAD_SUCCESS;
  }

  EventTy ompfileSchedRequest(MPIRequestManagerTy RequestManager) {
    OmpFileIORequest Request{};
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::string Path;
    if (Request.PathSize > 0) {
      Path.resize(Request.PathSize, '\0');
      RequestManager.receive(Path.data(), Request.PathSize, MPI_CHAR);
      if (auto Error = co_await RequestManager; Error)
        co_return Error;
    }

    OmpFileIOPlan Plan =
        buildSchedulePlan(Request, Request.PathSize > 0 ? Path.c_str() : nullptr);

    RequestManager.send(&Plan, sizeof(Plan), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileSchedBatchRequest(MPIRequestManagerTy RequestManager) {
    OmpFileIOBatchRequest Request{};
    uint64_t RequestPayloadBytes = 0;
    RequestManager.receive(&Request, sizeof(Request), MPI_BYTE);
    RequestManager.receive(&RequestPayloadBytes, 1, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::vector<uint8_t> RequestPayload;
    if (RequestPayloadBytes > 0) {
      RequestPayload.resize(static_cast<size_t>(RequestPayloadBytes));
      RequestManager.receiveInBatchs(RequestPayload.data(), RequestPayloadBytes);
      if (auto Error = co_await RequestManager; Error)
        co_return Error;
    }

    OmpFileIOBatchPlan Plan{};
    std::vector<uint8_t> PlanPayload;
    buildScheduleBatchPlan(Request,
                           RequestPayload.empty() ? nullptr
                                                  : RequestPayload.data(),
                           RequestPayloadBytes, Plan, PlanPayload);

    uint64_t PlanPayloadBytes = PlanPayload.size();
    RequestManager.send(&Plan, sizeof(Plan), MPI_BYTE);
    RequestManager.send(&PlanPayloadBytes, 1, MPI_UINT64_T);
    if (PlanPayloadBytes > 0)
      RequestManager.sendInBatchs(PlanPayload.data(), PlanPayloadBytes);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy ompfileSchedPlan(MPIRequestManagerTy RequestManager) {
    OmpFileIOPlan Plan{};
    RequestManager.receive(&Plan, sizeof(Plan), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    OmpFileIOCompletion Completion{};
    Completion.RequestId = Plan.RequestId;
    Completion.Status = 0;
    Completion.Errno = 0;
    Completion.Bytes = 0;

    OmpFileHeadnodeManager::instance().completeRequest(Plan);

    RequestManager.send(&Completion, sizeof(Completion), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy initAsyncInfo(MPIRequestManagerTy RequestManager) {
    __tgt_async_info *TgtAsyncInfoPtr = nullptr;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->init_async_info(DeviceId,
                                                     &TgtAsyncInfoPtr);

    RequestManager.send(&TgtAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy initDeviceInfo(MPIRequestManagerTy RequestManager) {
    __tgt_device_info DeviceInfo;
    const char *ErrStr = nullptr;

    RequestManager.receive(&DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->init_device_info(DeviceId, &DeviceInfo,
                                                      &ErrStr);

    RequestManager.send(&DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy queryAsync(MPIRequestManagerTy RequestManager) {
    void *HstAsyncInfoPtr = nullptr;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    auto *TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr)->AsyncInfoPtr.get();

    PluginManager.Plugins[PluginId]->query_async(DeviceId, TgtAsyncInfo);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy printDeviceInfo(MPIRequestManagerTy RequestManager) {
    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->print_device_info(DeviceId);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataLock(MPIRequestManagerTy RequestManager) {
    void *Ptr = nullptr;
    int64_t Size = 0;
    void *LockedPtr = nullptr;

    RequestManager.receive(&Ptr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_lock(DeviceId, Ptr, Size, &LockedPtr);

    RequestManager.send(&LockedPtr, sizeof(void *), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataUnlock(MPIRequestManagerTy RequestManager) {
    void *Ptr = nullptr;
    RequestManager.receive(&Ptr, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_unlock(DeviceId, Ptr);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataNotifyMapped(MPIRequestManagerTy RequestManager) {
    void *HstPtr = nullptr;
    int64_t Size = 0;
    RequestManager.receive(&HstPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_notify_mapped(DeviceId, HstPtr, Size);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataNotifyUnmapped(MPIRequestManagerTy RequestManager) {
    void *HstPtr = nullptr;
    RequestManager.receive(&HstPtr, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_notify_unmapped(DeviceId, HstPtr);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy exit(MPIRequestManagerTy RequestManager,
               std::atomic<EventSystemStateTy> &EventSystemState) {
    EventSystemStateTy OldState =
        EventSystemState.exchange(EventSystemStateTy::EXITED);
    assert(OldState != EventSystemStateTy::EXITED &&
           "Exit event received multiple times");

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  /// Function executed by the event handler threads.
  void runEventHandler(std::stop_token Stop, EventQueue &Queue) {
    while (EventSystem.EventSystemState == EventSystemStateTy::RUNNING ||
           Queue.size() > 0) {
      EventTy Event = Queue.pop(Stop);

      // Re-checks the stop condition when no event was found.
      if (Event.empty()) {
        continue;
      }

      Event.resume();

      if (!Event.done()) {
        Queue.push(std::move(Event));
        continue;
      }

      auto Error = Event.getError();
      if (Error)
        REPORT("Internal event failed with msg: %s\n",
               toString(std::move(Error)).data());
    }
  }

  /// Gate thread procedure.
  ///
  /// Caller thread will spawn the event handlers, execute the gate logic and
  /// wait until the event system receive an Exit event.
  void runGateThread() {
    // Device image to be used by this gate thread.
    // DeviceImage Image;

    // Updates the event state and
    EventSystem.EventSystemState = EventSystemStateTy::RUNNING;

    // Spawns the event handlers.
    llvm::SmallVector<std::jthread> EventHandlers;
    EventHandlers.resize(NumExecEventHandlers.get() +
                         NumDataEventHandlers.get());
    int EventHandlersSize = EventHandlers.size();
    auto HandlerFunction = std::bind_front(&ProxyDevice::runEventHandler, this);
    for (int Idx = 0; Idx < EventHandlersSize; Idx++) {
      EventHandlers[Idx] = std::jthread(
          HandlerFunction, std::ref(Idx < NumExecEventHandlers.get()
                                        ? EventSystem.ExecEventQueue
                                        : EventSystem.DataEventQueue));
    }

    // Executes the gate thread logic
    while (EventSystem.EventSystemState == EventSystemStateTy::RUNNING) {
      // Checks for new incoming event requests.
      MPI_Message EventReqMsg;
      MPI_Status EventStatus;
      int HasReceived = false;
      MPI_Improbe(MPI_ANY_SOURCE,
                  static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                  EventSystem.GateThreadComm, &HasReceived, &EventReqMsg,
                  MPI_STATUS_IGNORE);

      // If none was received, wait for `EVENT_POLLING_RATE`us for the next
      // check.
      if (!HasReceived) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(EventPollingRate.get()));
        continue;
      }

      // Acquires the event information from the received request, which are:
      // - Event type
      // - Event tag
      // - Target comm
      // - Event source rank
      int EventInfo[3];
      MPI_Mrecv(EventInfo, 3, MPI_INT, &EventReqMsg, &EventStatus);
      const auto NewEventType = static_cast<EventTypeTy>(EventInfo[0]);
      MPIRequestManagerTy RequestManager(
          EventSystem.getNewEventComm(EventInfo[1]), EventInfo[1],
          EventStatus.MPI_SOURCE, EventInfo[2]);

      // Creates a new receive event of 'event_type' type.
      using enum EventTypeTy;
      EventTy NewEvent;
      switch (NewEventType) {
      case RETRIEVE_NUM_DEVICES:
        NewEvent = retrieveNumDevices(std::move(RequestManager));
        break;
      case IS_PLUGIN_COMPATIBLE:
        NewEvent = isPluginCompatible(std::move(RequestManager));
        break;
      case IS_DEVICE_COMPATIBLE:
        NewEvent = isDeviceCompatible(std::move(RequestManager));
        break;
      case INIT_DEVICE:
        NewEvent = initDevice(std::move(RequestManager));
        break;
      case INIT_RECORD_REPLAY:
        NewEvent = initRecordReplay(std::move(RequestManager));
        break;
      case IS_DATA_EXCHANGABLE:
        NewEvent = isDataExchangable(std::move(RequestManager));
        break;
      case ALLOC:
        NewEvent = allocateBuffer(std::move(RequestManager));
        break;
      case DELETE:
        NewEvent = deleteBuffer(std::move(RequestManager));
        break;
      case SUBMIT:
        NewEvent = submit(std::move(RequestManager));
        break;
      case RETRIEVE:
        NewEvent = retrieve(std::move(RequestManager));
        break;
      case LOCAL_EXCHANGE:
        NewEvent = exchange(std::move(RequestManager));
        break;
      case EXCHANGE_SRC:
        NewEvent = exchangeSrc(std::move(RequestManager));
        break;
      case EXCHANGE_DST:
        NewEvent = exchangeDst(std::move(RequestManager));
        break;
      case EXIT:
        NewEvent =
            exit(std::move(RequestManager), EventSystem.EventSystemState);
        break;
      case LOAD_BINARY:
        NewEvent = loadBinary(std::move(RequestManager));
        break;
      case GET_GLOBAL:
        NewEvent = getGlobal(std::move(RequestManager));
        break;
      case GET_FUNCTION:
        NewEvent = getFunction(std::move(RequestManager));
        break;
      case LAUNCH_KERNEL:
        NewEvent = launchKernel(std::move(RequestManager));
        break;
      case SYNCHRONIZE:
        NewEvent = synchronize(std::move(RequestManager));
        break;
      case OMPFILE_PING:
        NewEvent = ompfilePing(std::move(RequestManager));
        break;
      case OMPFILE_OPEN:
        NewEvent = ompfileOpen(std::move(RequestManager));
        break;
      case OMPFILE_CLOSE:
        NewEvent = ompfileClose(std::move(RequestManager));
        break;
      case OMPFILE_PREAD:
        NewEvent = ompfilePread(std::move(RequestManager));
        break;
      case OMPFILE_PWRITE:
        NewEvent = ompfilePwrite(std::move(RequestManager));
        break;
      case OMPFILE_SCHED_REQUEST:
        NewEvent = ompfileSchedRequest(std::move(RequestManager));
        break;
      case OMPFILE_SCHED_REQUEST_BATCH:
        NewEvent = ompfileSchedBatchRequest(std::move(RequestManager));
        break;
      case OMPFILE_SCHED_PLAN:
        NewEvent = ompfileSchedPlan(std::move(RequestManager));
        break;
      case INIT_ASYNC_INFO:
        NewEvent = initAsyncInfo(std::move(RequestManager));
        break;
      case INIT_DEVICE_INFO:
        NewEvent = initDeviceInfo(std::move(RequestManager));
        break;
      case QUERY_ASYNC:
        NewEvent = queryAsync(std::move(RequestManager));
        break;
      case PRINT_DEVICE_INFO:
        NewEvent = printDeviceInfo(std::move(RequestManager));
        break;
      case DATA_LOCK:
        NewEvent = dataLock(std::move(RequestManager));
        break;
      case DATA_UNLOCK:
        NewEvent = dataUnlock(std::move(RequestManager));
        break;
      case DATA_NOTIFY_MAPPED:
        NewEvent = dataNotifyMapped(std::move(RequestManager));
        break;
      case DATA_NOTIFY_UNMAPPED:
        NewEvent = dataNotifyUnmapped(std::move(RequestManager));
        break;
      case SYNC:
        assert(false && "Trying to create a local event on a remote node");
      }

      if (NewEventType == LAUNCH_KERNEL) {
        EventSystem.ExecEventQueue.push(std::move(NewEvent));
      } else {
        EventSystem.DataEventQueue.push(std::move(NewEvent));
      }
    }

    assert(EventSystem.EventSystemState == EventSystemStateTy::EXITED &&
           "Event State should be EXITED after receiving an Exit event");
  }

private:
  llvm::SmallVector<std::unique_ptr<AsyncInfoHandle>> AsyncInfoList;
  llvm::SmallVector<std::unique_ptr<DeviceImage>> RemoteImages;
  llvm::DenseMap<void *, AsyncInfoHandle *> AsyncInfoTable;
  RemotePluginManager PluginManager;
  EventSystemTy EventSystem;
  /// Number of execute event handlers to spawn.
  IntEnvar NumExecEventHandlers;
  /// Number of data event handlers to spawn.
  IntEnvar NumDataEventHandlers;
  /// Polling rate period (us) used by event handlers.
  IntEnvar EventPollingRate;
  BoolEnvar OmpFileOpenCacheEnable;
  BoolEnvar OmpFileOpenCacheKeepOpen;
  BoolEnvar OmpFileOptStats;
  // Mutex for AsyncInfoTable
  std::mutex TableMutex;
  std::atomic<uint64_t> NextSchedRequestId{1};
  std::mutex OmpFileHandleMutex;
  std::unordered_map<int, OmpFileHandleEntry> OmpFileHandles;
  std::atomic<int> NextOmpFileHandle{1};
  std::mutex OmpFileOpenCacheMutex;
  std::unordered_map<std::string, OmpFileOpenCacheEntry> OmpFileOpenCacheByKey;
  std::unordered_map<int, std::string> OmpFileOpenCacheFdToKey;
  std::atomic<uint64_t> OmpFileStatsOpenRequests{0};
  std::atomic<uint64_t> OmpFileStatsOpenSyscalls{0};
  std::atomic<uint64_t> OmpFileStatsOpenCacheHits{0};
  std::atomic<uint64_t> OmpFileStatsCloseRequests{0};
  std::atomic<uint64_t> OmpFileStatsCloseSyscalls{0};
  std::atomic<uint64_t> OmpFileStatsCloseDeferred{0};
  std::mutex MppEventMutex;
  std::unordered_map<uint64_t, EventTy> MppEvents;
  std::unordered_set<uint64_t> CompletedMppTokens;
};

static ProxyDevice *getActiveProxyDevice() {
  const std::lock_guard<std::mutex> Lock(ActiveProxyDeviceMutex);
  return ActiveProxyDevice;
}

extern "C" {
int ompfile_mpp_init() {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppInit();
}

int ompfile_mpp_submit(uint64_t Token) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppSubmit(Token);
}

int ompfile_mpp_open(const char *Path, int Flags, int Mode, int *Handle) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppOpen(Path, Flags, Mode, Handle);
}

int ompfile_mpp_sched_request(const OmpFileIORequest *Request, const char *Path,
                              OmpFileIOPlan *Plan) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppSchedRequest(Request, Path, Plan);
}

int ompfile_mpp_sched_request_batch(
    const OmpFileIOBatchRequest *Request, const void *RequestPayload,
    uint64_t RequestPayloadBytes, OmpFileIOBatchPlan *Plan, void *PlanPayload,
    uint64_t PlanPayloadCapBytes, uint64_t *PlanPayloadOutBytes) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppSchedBatchRequest(Request, RequestPayload, RequestPayloadBytes,
                                  Plan, PlanPayload, PlanPayloadCapBytes,
                                  PlanPayloadOutBytes);
}

int ompfile_mpp_close(int Handle) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppClose(Handle);
}

int ompfile_mpp_pread(int Handle, int64_t Offset, void *Buffer, uint64_t Size) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPread(Handle, Offset, Buffer, Size);
}

int ompfile_mpp_pwrite(int Handle, int64_t Offset, const void *Buffer,
                       uint64_t Size) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPwrite(Handle, Offset, Buffer, Size);
}

int ompfile_mpp_poll(uint64_t Token, int *Done) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPoll(Token, Done);
}

int ompfile_mpp_finalize() {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppFinalize();
}
} // extern "C"

int main(int argc, char **argv) {
  ProxyDevice PD;
  PD.runGateThread();
  return 0;
}
