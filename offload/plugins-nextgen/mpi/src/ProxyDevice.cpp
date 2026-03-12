#include <chrono>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdarg>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <limits.h>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>

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

std::string trimWhitespace(std::string Value) {
  const size_t Begin = Value.find_first_not_of(" \t\r\n");
  if (Begin == std::string::npos)
    return {};
  const size_t End = Value.find_last_not_of(" \t\r\n");
  return Value.substr(Begin, End - Begin + 1);
}

std::string shortHostname(std::string Host) {
  Host = trimWhitespace(std::move(Host));
  const size_t DotPos = Host.find('.');
  if (DotPos != std::string::npos)
    Host.resize(DotPos);
  return Host;
}

std::string getLocalShortHostname() {
  char Buffer[HOST_NAME_MAX + 1] = {};
  if (gethostname(Buffer, sizeof(Buffer) - 1) != 0)
    return "unknown";
  return shortHostname(Buffer);
}

std::string envStringOrDefault(const char *Name, const char *DefaultValue) {
  const char *Value = std::getenv(Name);
  if (!Value || Value[0] == '\0')
    return DefaultValue ? std::string(DefaultValue) : std::string();
  return Value;
}

uint64_t envUint64OrDefault(const char *Name, uint64_t DefaultValue) {
  const char *Value = std::getenv(Name);
  if (!Value || Value[0] == '\0')
    return DefaultValue;
  char *End = nullptr;
  errno = 0;
  unsigned long long Parsed = std::strtoull(Value, &End, 10);
  if (errno != 0 || End == Value || (End && *End != '\0'))
    return DefaultValue;
  return static_cast<uint64_t>(Parsed);
}

bool hostsMatch(const std::string &Expected, const std::string &Observed) {
  return shortHostname(Expected) == shortHostname(Observed);
}

bool resolveTopologyStageRoot(const std::string &TopologyFile,
                              const std::string &LocalHost,
                              std::string &StageRoot, uint64_t &EntryCount,
                              std::string &Error) {
  StageRoot.clear();
  EntryCount = 0;
  Error.clear();
  if (TopologyFile.empty()) {
    Error = "topology-unset";
    return false;
  }

  std::ifstream Input(TopologyFile);
  if (!Input) {
    Error = "open-failed";
    return false;
  }

  std::string Line;
  while (std::getline(Input, Line)) {
    Line = trimWhitespace(std::move(Line));
    if (Line.empty() || Line[0] == '#')
      continue;

    std::istringstream Stream(Line);
    std::string Host;
    std::string Path;
    if (!(Stream >> Host >> Path))
      continue;

    ++EntryCount;
    if (hostsMatch(Host, LocalHost))
      StageRoot = Path;
  }

  if (StageRoot.empty()) {
    Error = EntryCount == 0 ? "empty-topology" : "host-missing";
    return false;
  }

  return true;
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
        EventPollingRate("OMPTARGET_EVENT_POLLING_RATE", 0),
        OmpFileOpenCacheEnable("LIBOMPFILE_OPT_OPEN_CACHE", false),
        OmpFileOpenCacheKeepOpen("LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN", true),
        OmpFileOptStats("LIBOMPFILE_OPT_STATS", false),
        OmpFileForceBlockingPwrite("LIBOMPFILE_MPP_FORCE_BLOCKING_PWRITE",
                                   false),
        OmpFileMPIFragmentSize("OMPTARGET_MPI_FRAGMENT_SIZE", 100e6),
        OmpFileStageMode(envStringOrDefault("LIBOMPFILE_STAGE_MODE", "off")),
        OmpFileStageRootPolicy(
            envStringOrDefault("LIBOMPFILE_STAGE_ROOT_POLICY", "topology")),
        OmpFileTopologyFile(
            envStringOrDefault("LIBOMPFILE_TOPOLOGY_FILE", "")),
        OmpFileStorageEnvironment(
            envStringOrDefault("LIBOMPFILE_STORAGE_ENVIRONMENT", "unspecified")),
        OmpFileStageMinFreeBytes(
            envUint64OrDefault("LIBOMPFILE_STAGE_MIN_FREE_BYTES", 0)),
        OmpFileStageLocalHost(getLocalShortHostname()) {
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
    OmpFileTopologyLoaded = resolveTopologyStageRoot(
        OmpFileTopologyFile, OmpFileStageLocalHost, OmpFileStageRoot,
        OmpFileTopologyEntries, OmpFileTopologyLoadError);
    if (OmpFileOptStats || OmpFileOpenCacheEnable) {
      fprintf(stderr,
              "MPIProxyDevice --> OMPFile cache config rank=%d enabled=%d "
              "keep_open=%d stats=%d blocking_pwrite=%d fragment_size=%lld\n",
              EventSystem.LocalRank, (int)OmpFileOpenCacheEnable.get(),
              (int)OmpFileOpenCacheKeepOpen.get(), (int)OmpFileOptStats.get(),
              (int)OmpFileForceBlockingPwrite.get(),
              static_cast<long long>(OmpFileMPIFragmentSize.get()));
    }
    if (OmpFileOptStats || OmpFileOpenCacheEnable ||
        OmpFileStageMode != "off" || !OmpFileTopologyFile.empty()) {
      fprintf(stderr,
              "MPIProxyDevice --> OMPFile stage config rank=%d "
              "stage_mode=%s stage_root_policy=%s topology_file_set=%d "
              "topology_file=%s topology_loaded=%d topology_entries=%llu "
              "stage_root=%s storage_environment=%s "
              "stage_min_free_bytes=%llu local_host=%s load_error=%s\n",
              EventSystem.LocalRank, OmpFileStageMode.c_str(),
              OmpFileStageRootPolicy.c_str(),
              static_cast<int>(!OmpFileTopologyFile.empty()),
              OmpFileTopologyFile.empty() ? "(unset)"
                                          : OmpFileTopologyFile.c_str(),
              static_cast<int>(OmpFileTopologyLoaded),
              static_cast<unsigned long long>(OmpFileTopologyEntries),
              OmpFileStageRoot.empty() ? "(unset)" : OmpFileStageRoot.c_str(),
              OmpFileStorageEnvironment.c_str(),
              static_cast<unsigned long long>(OmpFileStageMinFreeBytes),
              OmpFileStageLocalHost.c_str(),
              OmpFileTopologyLoadError.empty()
                  ? "(none)"
                  : OmpFileTopologyLoadError.c_str());
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
      traceOmpFile("openWithOptionalCache nocache path=%s flags=0x%x mode=%o "
                   "fd=%d errno=%d\n",
                   Path, Flags, Mode, Fd, ErrnoOut);
      return Fd;
    }

    const std::string Key = getOmpFileOpenCacheKey(Path, Flags, Mode);
    const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
    auto It = OmpFileOpenCacheByKey.find(Key);
    if (It != OmpFileOpenCacheByKey.end()) {
      It->second.RefCount += 1;
      OmpFileStatsOpenCacheHits.fetch_add(1, std::memory_order_relaxed);
      traceOmpFile("openWithOptionalCache hit path=%s flags=0x%x mode=%o "
                   "fd=%d refcount=%llu\n",
                   Path, Flags, Mode, It->second.Fd,
                   static_cast<unsigned long long>(It->second.RefCount));
      return It->second.Fd;
    }

    OmpFileStatsOpenSyscalls.fetch_add(1, std::memory_order_relaxed);
    int Fd = ::open(Path, Flags, static_cast<mode_t>(Mode));
    if (Fd < 0) {
      ErrnoOut = errno;
      traceOmpFile("openWithOptionalCache miss-fail path=%s flags=0x%x mode=%o "
                   "errno=%d\n",
                   Path, Flags, Mode, ErrnoOut);
      return -1;
    }

    OmpFileOpenCacheByKey.emplace(Key, OmpFileOpenCacheEntry{Fd, 1});
    OmpFileOpenCacheFdToKey[Fd] = Key;
    traceOmpFile("openWithOptionalCache miss-open path=%s flags=0x%x mode=%o "
                 "fd=%d refcount=1\n",
                 Path, Flags, Mode, Fd);
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
      traceOmpFile("closeWithOptionalCache nocache fd=%d ret=%d errno=%d\n", Fd,
                   Ret, ErrnoOut);
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
            traceOmpFile("closeWithOptionalCache bad-ref fd=%d\n", Fd);
            return -1;
          }

          Entry.RefCount -= 1;
          if (Entry.RefCount > 0 || OmpFileOpenCacheKeepOpen) {
            OmpFileStatsCloseDeferred.fetch_add(1, std::memory_order_relaxed);
            traceOmpFile("closeWithOptionalCache defer fd=%d refcount=%llu "
                         "keep_open=%d\n",
                         Fd, static_cast<unsigned long long>(Entry.RefCount),
                         static_cast<int>(OmpFileOpenCacheKeepOpen.get()));
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
    traceOmpFile("closeWithOptionalCache close fd=%d ret=%d errno=%d\n", Fd,
                 Ret, ErrnoOut);
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
    uint64_t PwriteAsyncEvents =
        OmpFileStatsPwriteAsyncEvents.load(std::memory_order_relaxed);
    uint64_t PwriteAsyncFragments =
        OmpFileStatsPwriteAsyncFragments.load(std::memory_order_relaxed);
    uint64_t PwriteBlockingFallbacks =
        OmpFileStatsPwriteBlockingFallbacks.load(std::memory_order_relaxed);
    uint64_t PwritePayloadBytes =
        OmpFileStatsPwritePayloadBytes.load(std::memory_order_relaxed);
    uint64_t PwriteFailures =
        OmpFileStatsPwriteFailures.load(std::memory_order_relaxed);
    uint64_t StagedReadHits =
        OmpFileStatsStagedReadHits.load(std::memory_order_relaxed);
    uint64_t StagedReadBytes =
        OmpFileStatsStagedReadBytes.load(std::memory_order_relaxed);
    uint64_t StagingInvalidations =
        OmpFileStatsStagingInvalidations.load(std::memory_order_relaxed);
    uint64_t StagingWriteBypass =
        OmpFileStatsStagingWriteBypassCount.load(std::memory_order_relaxed);
    uint64_t StagingEvictions =
        OmpFileStatsStagingEvictions.load(std::memory_order_relaxed);
    size_t CacheEntries = 0;

    {
      const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
      CacheEntries = OmpFileOpenCacheByKey.size();
    }

    fprintf(stderr,
            "MPIProxyDevice --> OMPFile stats [%s] rank=%d "
            "open_req=%llu open_sys=%llu open_hits=%llu "
            "close_req=%llu close_sys=%llu close_deferred=%llu "
            "cache_entries=%zu pwrite_async_events=%llu "
            "pwrite_async_fragments=%llu pwrite_blocking_fallbacks=%llu "
            "pwrite_payload_bytes=%llu pwrite_failures=%llu "
            "topology_file_set=%d topology_loaded=%d topology_entries=%llu "
            "stage_mode=%s stage_root_policy=%s stage_root=%s "
            "storage_environment=%s stage_min_free_bytes=%llu "
            "staged_read_hits=%llu staged_read_bytes=%llu "
            "staging_invalidations=%llu staging_write_bypass_count=%llu "
            "staging_evictions=%llu\n",
            Scope ? Scope : "unknown", EventSystem.LocalRank,
            static_cast<unsigned long long>(OpenReq),
            static_cast<unsigned long long>(OpenSys),
            static_cast<unsigned long long>(OpenHits),
            static_cast<unsigned long long>(CloseReq),
            static_cast<unsigned long long>(CloseSys),
            static_cast<unsigned long long>(CloseDeferred), CacheEntries,
            static_cast<unsigned long long>(PwriteAsyncEvents),
            static_cast<unsigned long long>(PwriteAsyncFragments),
            static_cast<unsigned long long>(PwriteBlockingFallbacks),
            static_cast<unsigned long long>(PwritePayloadBytes),
            static_cast<unsigned long long>(PwriteFailures),
            static_cast<int>(!OmpFileTopologyFile.empty()),
            static_cast<int>(OmpFileTopologyLoaded),
            static_cast<unsigned long long>(OmpFileTopologyEntries),
            OmpFileStageMode.c_str(), OmpFileStageRootPolicy.c_str(),
            OmpFileStageRoot.empty() ? "(unset)" : OmpFileStageRoot.c_str(),
            OmpFileStorageEnvironment.c_str(),
            static_cast<unsigned long long>(OmpFileStageMinFreeBytes),
            static_cast<unsigned long long>(StagedReadHits),
            static_cast<unsigned long long>(StagedReadBytes),
            static_cast<unsigned long long>(StagingInvalidations),
            static_cast<unsigned long long>(StagingWriteBypass),
            static_cast<unsigned long long>(StagingEvictions));
  }

  uint64_t computePwriteFragmentCount(uint64_t Size) const {
    if (Size == 0)
      return 0;
    const int64_t FragmentSize = OmpFileMPIFragmentSize.get();
    if (FragmentSize <= 0)
      return 0;
    return (Size + static_cast<uint64_t>(FragmentSize) - 1) /
           static_cast<uint64_t>(FragmentSize);
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
    traceOmpFile("event ompfilePread fd=%d offset=%lld size=%llu buf=%p "
                 "ret=%d errno=%d bytes=%llu\n",
                 Fd, static_cast<long long>(Offset),
                 static_cast<unsigned long long>(Size),
                 static_cast<void *>(Buffer.data()), Ret, Errno,
                 static_cast<unsigned long long>(PayloadSize));

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
    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    if (OmpFileStageMode != "off")
      OmpFileStatsStagingWriteBypassCount.fetch_add(1,
                                                    std::memory_order_relaxed);

    std::vector<char> Buffer;
    if (Size > 0) {
      Buffer.resize(Size);
      OmpFileStatsPwritePayloadBytes.fetch_add(Size, std::memory_order_relaxed);
      if (OmpFileForceBlockingPwrite.get()) {
        OmpFileStatsPwriteBlockingFallbacks.fetch_add(
            1, std::memory_order_relaxed);
        if (auto Error =
                RequestManager.receiveInBatchsBlocking(Buffer.data(), Size);
            Error) {
          OmpFileStatsPwriteFailures.fetch_add(1, std::memory_order_relaxed);
          co_return Error;
        }
      } else {
        OmpFileStatsPwriteAsyncEvents.fetch_add(1, std::memory_order_relaxed);
        OmpFileStatsPwriteAsyncFragments.fetch_add(
            computePwriteFragmentCount(Size), std::memory_order_relaxed);
        RequestManager.receiveInBatchs(Buffer.data(), Size);
        if (auto Error = co_await RequestManager; Error) {
          OmpFileStatsPwriteFailures.fetch_add(1, std::memory_order_relaxed);
          co_return Error;
        }
      }
    }

    ssize_t BytesWritten = 0;
    uint64_t BytesWrittenOut = 0;
    int Errno = 0;
    if (Size > 0) {
      BytesWritten = ::pwrite(Fd, Buffer.data(), Size, static_cast<off_t>(Offset));
      if (BytesWritten < 0) {
        Errno = errno;
      } else {
        BytesWrittenOut = static_cast<uint64_t>(BytesWritten);
        if (BytesWrittenOut < Size)
          Errno = EIO;
      }
    }

    int Ret = (BytesWritten < 0 || BytesWrittenOut < Size) ? -1 : 0;
    if (Ret != 0)
      OmpFileStatsPwriteFailures.fetch_add(1, std::memory_order_relaxed);
    RequestManager.send(&Ret, 1, MPI_INT);
    RequestManager.send(&Errno, 1, MPI_INT);
    RequestManager.send(&BytesWrittenOut, 1, MPI_UINT64_T);
    if (auto Error = co_await RequestManager; Error) {
      OmpFileStatsPwriteFailures.fetch_add(1, std::memory_order_relaxed);
      co_return Error;
    }
    traceOmpFile("event ompfilePwrite fd=%d offset=%lld size=%llu buf=%p "
                 "ret=%d errno=%d bytes=%llu transport=%s fragments=%llu\n",
                 Fd, static_cast<long long>(Offset),
                 static_cast<unsigned long long>(Size),
                 static_cast<void *>(Buffer.data()), Ret, Errno,
                 static_cast<unsigned long long>(BytesWrittenOut),
                 OmpFileForceBlockingPwrite.get() ? "blocking_debug" : "async",
                 static_cast<unsigned long long>(
                     computePwriteFragmentCount(Size)));
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

    const auto *Segments = static_cast<const OmpFileIOBatchSegment *>(Payload);
    std::vector<OmpFileIOBatchPlanEntry> Entries;
    OmpFileHeadnodeManager &Manager = OmpFileHeadnodeManager::instance();
    Manager.initialize(EventSystem.WorldSize, getHeadnodeRank());
    if (!Manager.planBatchRequest(Request, Segments, Plan, Entries,
                                  EventSystem.LocalRank)) {
      Plan.Status = -1;
      Plan.Errno = EIO;
      return true;
    }
    if (Entries.size() != Request.SegmentCount) {
      Plan.Status = -1;
      Plan.Errno = EPROTO;
      return true;
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
    Plan.PlanFlags |= OMPFILE_BATCH_PLAN_BATCH_API;

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
    const int MPIError = ompfileWithMPICallLock([&]() {
      return MPI_Isend(EventNotificationInfo, 3, MPI_INT, TargetRank,
                       static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                       EventSystem.GateThreadComm, &NotificationRequest);
    });

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
                   uint64_t Size, uint64_t *BytesReadOut = nullptr) {
    if (!Buffer && Size > 0) {
      errno = EINVAL;
      return false;
    }
    if (BytesReadOut)
      *BytesReadOut = 0;

    if (Rank == EventSystem.LocalRank) {
      const ssize_t BytesRead =
          ::pread(RemoteHandle, Buffer, Size, static_cast<off_t>(Offset));
      if (BytesRead < 0)
        return false;
      const uint64_t Bytes = static_cast<uint64_t>(BytesRead);
      if (Bytes < Size)
        std::memset(static_cast<char *>(Buffer) + Bytes, 0,
                    static_cast<size_t>(Size - Bytes));
      if (BytesReadOut)
        *BytesReadOut = Bytes;
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
    if (Bytes > Size) {
      errno = EPROTO;
      return false;
    }
    if (Bytes < Size)
      std::memset(static_cast<char *>(Buffer) + Bytes, 0,
                  static_cast<size_t>(Size - Bytes));
    if (BytesReadOut)
      *BytesReadOut = Bytes;
    return true;
  }

  bool pwriteOnRank(int Rank, int RemoteHandle, int64_t Offset,
                    const void *Buffer, uint64_t Size,
                    uint64_t *BytesWrittenOut) {
    if (!Buffer && Size > 0) {
      errno = EINVAL;
      return false;
    }
    if (BytesWrittenOut)
      *BytesWrittenOut = 0;

    if (Rank == EventSystem.LocalRank) {
      const ssize_t BytesWritten =
          ::pwrite(RemoteHandle, Buffer, Size, static_cast<off_t>(Offset));
      if (BytesWritten < 0)
        return false;
      const uint64_t Bytes = static_cast<uint64_t>(BytesWritten);
      if (BytesWrittenOut)
        *BytesWrittenOut = Bytes;
      if (Bytes < Size) {
        errno = EIO;
        return false;
      }
      return true;
    }

    int IoRet = -1;
    int RemoteErrno = 0;
    uint64_t Bytes = 0;
    EventTy Event = createRankEvent(
        OriginEvents::ompfilePwrite, EventTypeTy::OMPFILE_PWRITE, Rank,
        /*TargetDeviceId=*/0, RemoteHandle, Offset, Buffer, Size, &IoRet,
        &RemoteErrno, &Bytes);
    if (!waitForEvent(Event, "pwrite"))
      return false;
    if (IoRet != 0) {
      errno = RemoteErrno;
      return false;
    }
    if (Bytes > Size) {
      errno = EPROTO;
      return false;
    }
    if (BytesWrittenOut)
      *BytesWrittenOut = Bytes;
    if (Bytes < Size) {
      errno = EIO;
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
    uint64_t BytesRead = 0;
    if (!preadOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
                     &BytesRead)) {
      return OFFLOAD_FAIL;
    }
    traceOmpFile("mppPread exit local=%d rank=%d remote=%d bytes=%llu\n",
                 Handle, Entry.Rank, Entry.RemoteHandle,
                 static_cast<unsigned long long>(BytesRead));
    return OFFLOAD_SUCCESS;
  }

  int mppPreadEx(int Handle, int64_t Offset, void *Buffer, uint64_t Size,
                 uint64_t *BytesRead) {
    if (!BytesRead)
      return OFFLOAD_FAIL;
    *BytesRead = 0;
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (!preadOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
                     BytesRead))
      return OFFLOAD_FAIL;
    return OFFLOAD_SUCCESS;
  }

  int mppPwriteEx(int Handle, int64_t Offset, const void *Buffer, uint64_t Size,
                  uint64_t *BytesWritten) {
    if (!BytesWritten)
      return OFFLOAD_FAIL;
    *BytesWritten = 0;
    OmpFileHandleEntry Entry{};
    if (!findRemoteHandle(Handle, Entry))
      return OFFLOAD_FAIL;
    if (!pwriteOnRank(Entry.Rank, Entry.RemoteHandle, Offset, Buffer, Size,
                      BytesWritten)) {
      traceOmpFile("mppPwriteEx fail local=%d rank=%d remote=%d offset=%lld "
                   "size=%llu errno=%d\n",
                   Handle, Entry.Rank, Entry.RemoteHandle,
                   static_cast<long long>(Offset),
                   static_cast<unsigned long long>(Size), errno);
      return OFFLOAD_FAIL;
    }
    traceOmpFile("mppPwriteEx success local=%d rank=%d remote=%d offset=%lld "
                 "size=%llu bytes=%llu\n",
                 Handle, Entry.Rank, Entry.RemoteHandle,
                 static_cast<long long>(Offset),
                 static_cast<unsigned long long>(Size),
                 static_cast<unsigned long long>(*BytesWritten));
    return OFFLOAD_SUCCESS;
  }

  int mppPwrite(int Handle, int64_t Offset, const void *Buffer, uint64_t Size) {
    uint64_t BytesWritten = 0;
    if (mppPwriteEx(Handle, Offset, Buffer, Size, &BytesWritten) !=
        OFFLOAD_SUCCESS)
      return OFFLOAD_FAIL;
    if (BytesWritten != Size) {
      errno = EIO;
      return OFFLOAD_FAIL;
    }
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
      ompfileWithMPICallLock([&]() {
        return MPI_Improbe(MPI_ANY_SOURCE,
                           static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                           EventSystem.GateThreadComm, &HasReceived,
                           &EventReqMsg, MPI_STATUS_IGNORE);
      });

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
      ompfileWithMPICallLock([&]() {
        return MPI_Mrecv(EventInfo, 3, MPI_INT, &EventReqMsg, &EventStatus);
      });
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
  BoolEnvar OmpFileForceBlockingPwrite;
  Int64Envar OmpFileMPIFragmentSize;
  std::string OmpFileStageMode;
  std::string OmpFileStageRootPolicy;
  std::string OmpFileTopologyFile;
  std::string OmpFileStorageEnvironment;
  uint64_t OmpFileStageMinFreeBytes = 0;
  bool OmpFileTopologyLoaded = false;
  uint64_t OmpFileTopologyEntries = 0;
  std::string OmpFileStageRoot;
  std::string OmpFileStageLocalHost;
  std::string OmpFileTopologyLoadError;
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
  std::atomic<uint64_t> OmpFileStatsPwriteAsyncEvents{0};
  std::atomic<uint64_t> OmpFileStatsPwriteAsyncFragments{0};
  std::atomic<uint64_t> OmpFileStatsPwriteBlockingFallbacks{0};
  std::atomic<uint64_t> OmpFileStatsPwritePayloadBytes{0};
  std::atomic<uint64_t> OmpFileStatsPwriteFailures{0};
  std::atomic<uint64_t> OmpFileStatsStagedReadHits{0};
  std::atomic<uint64_t> OmpFileStatsStagedReadBytes{0};
  std::atomic<uint64_t> OmpFileStatsStagingInvalidations{0};
  std::atomic<uint64_t> OmpFileStatsStagingWriteBypassCount{0};
  std::atomic<uint64_t> OmpFileStatsStagingEvictions{0};
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

int ompfile_mpp_pread_ex(int Handle, int64_t Offset, void *Buffer,
                         uint64_t Size, uint64_t *BytesRead) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPreadEx(Handle, Offset, Buffer, Size, BytesRead);
}

int ompfile_mpp_pwrite(int Handle, int64_t Offset, const void *Buffer,
                       uint64_t Size) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPwrite(Handle, Offset, Buffer, Size);
}

int ompfile_mpp_pwrite_ex(int Handle, int64_t Offset, const void *Buffer,
                          uint64_t Size, uint64_t *BytesWritten) {
  ProxyDevice *PD = getActiveProxyDevice();
  if (!PD)
    return OFFLOAD_FAIL;
  return PD->mppPwriteEx(Handle, Offset, Buffer, Size, BytesWritten);
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
