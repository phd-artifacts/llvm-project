//===------RTLs/mpi/src/rtl.cpp - Target RTLs Implementation - C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for MPI applications
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>

#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Utils/ELF.h"

#include "EventSystem.h"
#include "GlobalHandler.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginInterface.h"
#include "omptarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"
#include "llvm/Support/Error.h"

#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) ||           \
    !defined(__ORDER_BIG_ENDIAN__)
#error "Missing preprocessor definitions for endianness detection."
#endif

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define LITTLEENDIAN_CPU
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define BIGENDIAN_CPU
#endif

namespace llvm::omp::target::plugin {

/// Forward declarations for all specialized data structures.
struct MPIPluginTy;
struct MPIDeviceTy;
struct MPIDeviceImageTy;
struct MPIKernelTy;
class MPIGlobalHandlerTy;

static std::atomic<MPIPluginTy *> ActiveMPIPlugin{nullptr};

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

static bool ompfileDiagEnabled() {
  static const bool Enabled = []() {
    const char *Raw = std::getenv("OMPFILE_MPP_DEVICE_DIAG");
    return Raw && Raw[0] && std::strcmp(Raw, "0") != 0;
  }();
  return Enabled;
}

// Extra fields are phase-specific:
// - plugin_init_count: extra_a=world_size, extra_b=device_slots_before_publish
// - plugin_init_publish: extra_a=world_size, extra_b=computed_device_count
// - is_device_compatible: extra_a=compat_result_or_rc, extra_b=diag_reason
//   diag_reason: 0=normal, 1=event_empty, 2=event_error
// - is_device_initialized: extra_a=initialized(0|1), extra_b=device_ptr_is_null
// - init_device: extra_a=init_result_or_rc, extra_b=device_ptr_is_null
static void ompfileDiagLog(const char *Phase, int Rank, int DeviceCount,
                           int DeviceId, int ExtraA, int ExtraB) {
  if (!ompfileDiagEnabled())
    return;
  REPORT("[ompfile-mpi-diag] phase=%s rank=%d device_count=%d device_id=%d "
         "extra_a=%d extra_b=%d\n",
         Phase ? Phase : "unknown", Rank, DeviceCount, DeviceId, ExtraA,
         ExtraB);
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

// TODO: Should this be defined inside the EventSystem?
using MPIEventQueue = std::list<EventTy>;
using MPIEventQueuePtr = MPIEventQueue *;

/// Class implementing the MPI device images properties.
struct MPIDeviceImageTy : public DeviceImageTy {
  /// Create the MPI image with the id and the target image pointer.
  MPIDeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                   const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, Device, TgtImage), DeviceImageAddrs(getSize()) {}

  llvm::SmallVector<void *> DeviceImageAddrs;
};

class MPIGlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  Error getGlobalMetadataFromDevice(GenericDeviceTy &GenericDevice,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    const char *GlobalName = DeviceGlobal.getName().data();
    MPIDeviceImageTy &MPIImage = static_cast<MPIDeviceImageTy &>(Image);

    if (GlobalName == nullptr) {
      return Plugin::error("Failed to get name for global %p", &DeviceGlobal);
    }

    void *EntryAddress = nullptr;

    __tgt_offload_entry *Begin = MPIImage.getTgtImage()->EntriesBegin;
    __tgt_offload_entry *End = MPIImage.getTgtImage()->EntriesEnd;

    int I = 0;
    for (auto &Entry = Begin; Entry < End; ++Entry) {
      if (!strcmp(Entry->name, GlobalName)) {
        EntryAddress = MPIImage.DeviceImageAddrs[I];
        break;
      }
      I++;
    }

    if (EntryAddress == nullptr) {
      return Plugin::error("Failed to find global %s", GlobalName);
    }

    // Save the pointer to the symbol.
    DeviceGlobal.setPtr(EntryAddress);

    return Plugin::success();
  }
};

struct MPIKernelTy : public GenericKernelTy {
  /// Construct the kernel with a name and an execution mode.
  MPIKernelTy(const char *Name) : GenericKernelTy(Name), Func(nullptr) {}

  /// Initialize the kernel.
  Error initImpl(GenericDeviceTy &Device, DeviceImageTy &Image) override {
    // Functions have zero size.
    GlobalTy Global(getName(), 0);

    // Get the metadata (address) of the kernel function.
    GenericGlobalHandlerTy &GHandler = Device.Plugin.getGlobalHandler();
    if (auto Err = GHandler.getGlobalMetadataFromDevice(Device, Image, Global))
      return Err;

    // Check that the function pointer is valid.
    if (!Global.getPtr())
      return Plugin::error("Invalid function for kernel %s", getName());

    // Save the function pointer.
    Func = (void (*)())Global.getPtr();

    // TODO: Check which settings are appropriate for the mpi plugin
    // for now we are using the Elf64 plugin configuration
    KernelEnvironment.Configuration.ExecMode = OMP_TGT_EXEC_MODE_GENERIC;
    KernelEnvironment.Configuration.MayUseNestedParallelism = /* Unknown */ 2;
    KernelEnvironment.Configuration.UseGenericStateMachine = /* Unknown */ 2;

    // Set the maximum number of threads to a single.
    MaxNumThreads = 1;
    return Plugin::success();
  }

  /// Launch the kernel.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, KernelArgsTy &KernelArgs,
                   KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

private:
  /// The kernel function to execute.
  void (*Func)(void);
};

/// MPI resource reference and queue. These are the objects handled by the
/// MPIQueue Manager for the MPI plugin.
template <typename ResourceTy>
struct MPIResourceRef final : public GenericDeviceResourceRef {

  /// The underlying handler type for the resource.
  using HandleTy = ResourceTy *;

  /// Create a empty reference to an invalid resource.
  MPIResourceRef() : Resource(nullptr) {}

  /// Create a reference to an existing resource.
  MPIResourceRef(HandleTy Queue) : Resource(Queue) {}

  /// Create a new resource and save the reference.
  Error create(GenericDeviceTy &Device) override {
    if (Resource)
      return Plugin::error("Recreating an existing resource");

    Resource = new ResourceTy;
    if (!Resource)
      return Plugin::error("Failed to allocated a new resource");

    return Plugin::success();
  }

  /// Destroy the resource and invalidate the reference.
  Error destroy(GenericDeviceTy &Device) override {
    if (!Resource)
      return Plugin::error("Destroying an invalid resource");

    delete Resource;
    Resource = nullptr;

    return Plugin::success();
  }

  operator HandleTy() const { return Resource; }

private:
  HandleTy Resource;
};

/// Class implementing the device functionalities for remote x86_64 processes.
struct MPIDeviceTy : public GenericDeviceTy {
  /// Create a MPI Device with a device id and the default MPI grid values.
  MPIDeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, MPIGridValues),
        MPIEventQueueManager(*this), MPIEventManager(*this) {}

  /// Initialize the device, its resources and get its properties.
  Error initImpl(GenericPluginTy &Plugin) override {
    if (auto Err = MPIEventQueueManager.init(OMPX_InitialNumStreams))
      return Err;

    if (auto Err = MPIEventManager.init(OMPX_InitialNumEvents))
      return Err;

    return Plugin::success();
  }

  /// Deinitizalize the device and release its resources.
  Error deinitImpl() override {
    if (auto Err = MPIEventQueueManager.deinit())
      return Err;

    if (auto Err = MPIEventManager.deinit())
      return Err;

    return Plugin::success();
  }

  Error setContext() override { return Plugin::success(); }

  /// Load the binary image into the device and allocate an image object.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    // Allocate and initialize the image object.
    MPIDeviceImageTy *Image = Plugin.allocate<MPIDeviceImageTy>();
    new (Image) MPIDeviceImageTy(ImageId, *this, TgtImage);
    return Image;
  }

  /// Allocate memory on the device or related to the device.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    return nullptr;
  }

  /// Deallocate memory on the device or related to the device.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    return OFFLOAD_SUCCESS;
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }

  /// Exchange data between two devices directly. In the MPI plugin, this
  /// function will create an event for the host to tell the devices about the
  /// exchange. Then, the devices will do the transfer themselves and let the
  /// host know when it's done.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }

  /// Allocate and construct a MPI kernel.
  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    // Allocate and construct the kernel.
    MPIKernelTy *MPIKernel = Plugin.allocate<MPIKernelTy>();

    if (!MPIKernel)
      return Plugin::error("Failed to allocate memory for MPI kernel");

    new (MPIKernel) MPIKernelTy(Name);

    return *MPIKernel;
  }

  /// Create an event.
  Error createEventImpl(void **EventStoragePtr) override {
    return Plugin::success();
  }

  /// Destroy a previously created event.
  Error destroyEventImpl(void *Event) override {
    return MPIEventManager.returnResource(reinterpret_cast<EventTy *>(Event));
  }

  /// Record the event.
  Error recordEventImpl(void *Event,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }

  /// Make the queue wait on the event.
  Error waitEventImpl(void *Event,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }

  /// Synchronize the current thread with the event
  Error syncEventImpl(void *Event) override { return Plugin::success(); }

  /// Synchronize current thread with the pending operations on the async info.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    return Plugin::success();
  }

  /// Query for the completion of the pending operations on the async info.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override {
    return Plugin::success();
  }

  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    return HstPtr;
  }

  /// Indicate that the buffer is not pinned.
  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    return false;
  }

  Error dataUnlockImpl(void *HstPtr) override { return Plugin::success(); }

  /// This plugin should not setup the device environment or memory pool.
  virtual bool shouldSetupDeviceEnvironment() const override { return false; };
  virtual bool shouldSetupDeviceMemoryPool() const override { return false; };

  /// Device memory limits are currently not applicable to the MPI plugin.
  Error getDeviceStackSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }

  Error setDeviceStackSize(uint64_t Value) override {
    return Plugin::success();
  }

  Error getDeviceHeapSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }

  Error setDeviceHeapSize(uint64_t Value) override { return Plugin::success(); }

  /// Device interoperability. Not supported by MPI right now.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::error("initAsyncInfoImpl not supported");
  }

  /// This plugin does not support interoperability.
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    return Plugin::error("initDeviceInfoImpl not supported");
  }

  /// Print information about the device.
  Error obtainInfoImpl(InfoQueueTy &Info) override {
    // TODO: Add more information about the device.
    Info.add("MPI plugin");
    Info.add("MPI OpenMP Device Number", DeviceId);

    return Plugin::success();
  }

  Error getQueue(AsyncInfoWrapperTy &AsyncInfoWrapper,
                 MPIEventQueuePtr &Queue) {
    return Plugin::success();
  }

private:
  using MPIEventQueueManagerTy =
      GenericDeviceResourceManagerTy<MPIResourceRef<MPIEventQueue>>;
  using MPIEventManagerTy =
      GenericDeviceResourceManagerTy<MPIResourceRef<EventTy>>;

  MPIEventQueueManagerTy MPIEventQueueManager;
  MPIEventManagerTy MPIEventManager;

  /// Grid values for the MPI plugin.
  static constexpr GV MPIGridValues = {
      1, // GV_Slot_Size
      1, // GV_Warp_Size
      1, // GV_Max_Teams
      1, // GV_Default_Num_Teams
      1, // GV_SimpleBufferSize
      1, // GV_Max_WG_Size
      1, // GV_Default_WG_Size
  };
};

Error MPIKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                              uint32_t NumThreads, uint64_t NumBlocks,
                              KernelArgsTy &KernelArgs,
                              KernelLaunchParamsTy LaunchParams,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  return Plugin::success();
}

/// Class implementing the MPI plugin.
struct MPIPluginTy : public GenericPluginTy {
  MPIPluginTy() : GenericPluginTy(getTripleArch()) {}
  ~MPIPluginTy() override {
    MPIPluginTy *Expected = this;
    ActiveMPIPlugin.compare_exchange_strong(Expected, nullptr);
  }

  /// This class should not be copied.
  MPIPluginTy(const MPIPluginTy &) = delete;
  MPIPluginTy(MPIPluginTy &&) = delete;

  EventSystemTy &getEventSystemForOmpFile() { return EventSystem; }

  bool ensureEventSystemInitializedForOmpFile() {
    if (!EventSystem.is_initialized())
      return EventSystem.initialize();
    return true;
  }

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override {
    ompfileDiagLog("plugin_init_enter", -1, -1, -1, 0, 0);
    if (!EventSystem.is_initialized() && !EventSystem.initialize()) {
      DP("Failed to initialize EventSystem during plugin init\n");
      ompfileDiagLog("plugin_init_eventsystem_fail", -1, -1, -1,
                     OFFLOAD_FAIL, 0);
      return Plugin::error("Failed to initialize EventSystem during plugin "
                           "init");
    }
    int32_t NumWorkers = EventSystem.getNumWorkers();
    int32_t DeviceSlotsBeforePublish = static_cast<int32_t>(RemoteDevices.size());
    int32_t NumRemoteDevices = getNumRemoteDevices();
    ompfileDiagLog("plugin_init_count", -1, NumRemoteDevices, -1, NumWorkers,
                   DeviceSlotsBeforePublish);
    assert(RemoteDevices.size() == 0 && "MPI Plugin already initialized");
    RemoteDevices.resize(NumRemoteDevices, nullptr);
    RemoteDevicesInitialized.resize(NumRemoteDevices, false);
    ompfileDiagLog("plugin_init_publish", -1,
                   static_cast<int>(RemoteDevices.size()), -1, NumWorkers,
                   NumRemoteDevices);

    return NumRemoteDevices;
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override {
    EventSystem.deinitialize();
    return Plugin::success();
  }

  /// Creates a MPI device.
  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override {
    return new MPIDeviceTy(Plugin, DeviceId, NumDevices);
  }

  /// Creates a MPI global handler.
  GenericGlobalHandlerTy *createGlobalHandler() override {
    return new MPIGlobalHandlerTy();
  }

  /// Get the ELF code to recognize the compatible binary images.
  uint16_t getMagicElfBits() const override {
    return utils::elf::getTargetMachine();
  }

  /// All images (ELF-compatible) should be compatible with this plugin.
  Expected<bool> isELFCompatible(uint32_t DeviceID,
                                 StringRef Image) const override {
    return true;
  }

  Triple::ArchType getTripleArch() const override {
#if defined(__x86_64__)
    return llvm::Triple::x86_64;
#elif defined(__s390x__)
    return llvm::Triple::systemz;
#elif defined(__aarch64__)
#ifdef LITTLEENDIAN_CPU
    return llvm::Triple::aarch64;
#else
    return llvm::Triple::aarch64_be;
#endif
#elif defined(__powerpc64__)
#ifdef LITTLEENDIAN_CPU
    return llvm::Triple::ppc64le;
#else
    return llvm::Triple::ppc64;
#endif
#else
    return llvm::Triple::UnknownArch;
#endif
  }

  Error getQueue(__tgt_async_info *AsyncInfoPtr, MPIEventQueuePtr &Queue) {
    const std::lock_guard<std::mutex> Lock(MPIQueueMutex);
    Queue = static_cast<MPIEventQueuePtr>(AsyncInfoPtr->Queue);
    if (!Queue) {
      Queue = new MPIEventQueue;
      if (Queue == nullptr)
        return Plugin::error("Failed to get Queue from AsyncInfoPtr %p\n",
                             AsyncInfoPtr);
      // Modify the AsyncInfoWrapper to hold the new queue.
      AsyncInfoPtr->Queue = Queue;
    }
    return Plugin::success();
  }

  Error returnQueue(MPIEventQueuePtr &Queue) {
    const std::lock_guard<std::mutex> Lock(MPIQueueMutex);
    if (Queue == nullptr)
      return Plugin::error("Failed to return Queue: invalid Queue ptr");

    delete Queue;

    return Plugin::success();
  }

  const char *getName() const override { return GETNAME(TARGET_NAME); }

  /// This plugin does not support exchanging data between two devices.
  bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) override {
    bool QueryResult = false;

    int32_t SrcRank = -1, SrcDevId, DstRank = -1, DstDevId;

    std::tie(SrcRank, SrcDevId) = EventSystem.mapDeviceId(SrcDeviceId);
    std::tie(DstRank, DstDevId) = EventSystem.mapDeviceId(DstDeviceId);

    // If the exchange is between different mpi processes, it is possible to
    // perform the operation without consulting the devices
    if ((SrcRank != -1) && (DstRank != -1) && (SrcRank != DstRank))
      return true;

    EventTy Event = EventSystem.createEvent(
        OriginEvents::isDataExchangable, EventTypeTy::IS_DATA_EXCHANGABLE,
        SrcDeviceId, DstDevId, &QueryResult);

    if (Event.empty()) {
      DP("Failed to create isDataExchangeble event in %d SrcDevice\n",
         SrcDeviceId);
      return false;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      DP("Failed to query isDataExchangeble from device %d SrcDevice: %s\n",
         SrcDeviceId, toString(std::move(Error)).c_str());
      return false;
    }

    return QueryResult;
  }

  /// Get the number of devices considering all devices per rank
  int32_t getNumRemoteDevices() {
    int32_t NumRemoteDevices = 0;
    int32_t NumRanks = EventSystem.getNumWorkers();

    for (int32_t RemoteRank = 0; RemoteRank < NumRanks; RemoteRank++) {
      auto Event = EventSystem.createEvent(
          OriginEvents::retrieveNumDevices, EventTypeTy::RETRIEVE_NUM_DEVICES,
          RemoteRank, &EventSystem.DevicesPerRemote.emplace_back(0));

      if (Event.empty()) {
        DP("Error retrieving Num Devices from rank %d\n", RemoteRank);
        return 0;
      }

      Event.wait();
      if (auto Err = Event.getError())
        DP("Error retrieving Num Devices from rank %d: %s\n", RemoteRank,
           toString(std::move(Err)).c_str());

      NumRemoteDevices += EventSystem.DevicesPerRemote[RemoteRank];
    }

    return NumRemoteDevices;
  }

  int32_t is_plugin_compatible(__tgt_device_image *Image) override {
    if (!EventSystem.is_initialized() && !EventSystem.initialize()) {
      DP("Failed to initialize EventSystem in is_plugin_compatible\n");
      ompfileDiagLog("is_plugin_compatible_eventsystem_fail", -1,
                     static_cast<int>(RemoteDevices.size()), -1, OFFLOAD_FAIL,
                     0);
      return false;
    }

    int NumRanks = EventSystem.getNumWorkers();
    llvm::SmallVector<bool> QueryResults{};
    bool QueryResult = true;
    for (int RemoteRank = 0; RemoteRank < NumRanks; RemoteRank++) {
      EventTy Event = EventSystem.createEvent(
          OriginEvents::isPluginCompatible, EventTypeTy::IS_PLUGIN_COMPATIBLE,
          RemoteRank, Image, &QueryResults.emplace_back(false));

      if (Event.empty()) {
        DP("Failed to create isPluginCompatible on Rank %d\n", RemoteRank);
        QueryResults[RemoteRank] = false;
        QueryResult = false;
        continue;
      }

      Event.wait();
      if (auto Err = Event.getError()) {
        DP("Error querying the binary compability on Rank %d\n", RemoteRank);
        QueryResults[RemoteRank] = false;
      }

      QueryResult &= QueryResults[RemoteRank];
    }

    return QueryResult;
  }

  int32_t is_device_compatible(int32_t DeviceId,
                               __tgt_device_image *Image) override {
    bool QueryResult = true;
    int CompatDiagA = static_cast<int>(QueryResult);
    int CompatDiagB = 0;
    ompfileDiagLog("is_device_compatible_enter", -1,
                   static_cast<int>(RemoteDevices.size()), DeviceId, 0, 0);

    EventTy Event = EventSystem.createEvent(OriginEvents::isDeviceCompatible,
                                            EventTypeTy::IS_DEVICE_COMPATIBLE,
                                            DeviceId, Image, &QueryResult);

    if (Event.empty()) {
      DP("Failed to create isDeviceCompatible on Device %d\n", DeviceId);
      CompatDiagA = OFFLOAD_FAIL;
      CompatDiagB = 1;
      ompfileDiagLog("is_device_compatible", -1,
                     static_cast<int>(RemoteDevices.size()), DeviceId,
                     CompatDiagA, CompatDiagB);
      return false;
    }

    Event.wait();
    if (auto Err = Event.getError()) {
      DP("Error querying the binary compability on Device %d\n", DeviceId);
      CompatDiagA = OFFLOAD_FAIL;
      CompatDiagB = 2;
      ompfileDiagLog("is_device_compatible", -1,
                     static_cast<int>(RemoteDevices.size()), DeviceId,
                     CompatDiagA, CompatDiagB);
      return false;
    }

    CompatDiagA = static_cast<int>(QueryResult);
    ompfileDiagLog("is_device_compatible", -1,
                   static_cast<int>(RemoteDevices.size()), DeviceId, CompatDiagA,
                   CompatDiagB);

    return QueryResult;
  }

  int32_t is_device_initialized(int32_t DeviceId) const override {
    bool IsValid = isValidDeviceId(DeviceId);
    void *DevicePtr = IsValid ? RemoteDevices[DeviceId] : nullptr;
    int32_t IsInitialized =
        IsValid ? static_cast<int32_t>(RemoteDevicesInitialized[DeviceId]) : 0;
    ompfileDiagLog("is_device_initialized", -1,
                   static_cast<int>(RemoteDevices.size()), DeviceId,
                   IsInitialized, DevicePtr == nullptr ? 1 : 0);
    return IsInitialized;
  }

  int32_t init_device(int32_t DeviceId) override {
    void *DevicePtr = nullptr;
    ompfileDiagLog("init_device_enter", -1, static_cast<int>(RemoteDevices.size()),
                   DeviceId, 0, 0);

    EventTy Event =
        EventSystem.createEvent(OriginEvents::initDevice,
                                EventTypeTy::INIT_DEVICE, DeviceId, &DevicePtr);

    if (Event.empty()) {
      REPORT("Error to create InitDevice Event for device %d\n", DeviceId);
      if (isValidDeviceId(DeviceId))
        RemoteDevicesInitialized[DeviceId] = false;
      ompfileDiagLog("init_device", -1, static_cast<int>(RemoteDevices.size()),
                     DeviceId, OFFLOAD_FAIL, 1);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      REPORT("Failure to initialize device %d: %s\n", DeviceId,
             toString(std::move(Error)).data());
      if (isValidDeviceId(DeviceId))
        RemoteDevicesInitialized[DeviceId] = false;
      int DevicePtrIsNull = DevicePtr == nullptr ? 1 : 0;
      ompfileDiagLog("init_device", -1, static_cast<int>(RemoteDevices.size()),
                     DeviceId, OFFLOAD_FAIL, DevicePtrIsNull);
      return OFFLOAD_FAIL;
    }

    if (!isValidDeviceId(DeviceId)) {
      int DevicePtrIsNull = DevicePtr == nullptr ? 1 : 0;
      ompfileDiagLog("init_device", -1, static_cast<int>(RemoteDevices.size()),
                     DeviceId, OFFLOAD_FAIL, DevicePtrIsNull);
      return OFFLOAD_FAIL;
    }

    if (DevicePtr == nullptr) {
      RemoteDevicesInitialized[DeviceId] = false;
      ompfileDiagLog("init_device", -1, static_cast<int>(RemoteDevices.size()),
                     DeviceId, OFFLOAD_FAIL, 1);
      return OFFLOAD_FAIL;
    }

    RemoteDevices[DeviceId] = DevicePtr;
    RemoteDevicesInitialized[DeviceId] = true;
    int DevicePtrIsNull = DevicePtr == nullptr ? 1 : 0;
    ompfileDiagLog("init_device", -1, static_cast<int>(RemoteDevices.size()),
                   DeviceId, OFFLOAD_SUCCESS, DevicePtrIsNull);

    return OFFLOAD_SUCCESS;
  }

  int32_t initialize_record_replay(int32_t DeviceId, int64_t MemorySize,
                                   void *VAddr, bool isRecord, bool SaveOutput,
                                   uint64_t &ReqPtrArgOffset) override {
    EventTy Event = EventSystem.createEvent(
        OriginEvents::initRecordReplay, EventTypeTy::INIT_RECORD_REPLAY,
        DeviceId, MemorySize, VAddr, isRecord, SaveOutput, &ReqPtrArgOffset);

    if (Event.empty()) {
      REPORT("Error to create initRecordReplay Event for device %d\n",
             DeviceId);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      REPORT("WARNING RR did not intialize RR-properly with %lu bytes"
             "(Error: %s)\n",
             MemorySize, toString(std::move(Error)).data());
      if (!isRecord) {
        return OFFLOAD_FAIL;
      }
    }
    return OFFLOAD_SUCCESS;
  }

  int32_t load_binary(int32_t DeviceId, __tgt_device_image *TgtImage,
                      __tgt_device_binary *Binary) override {
    EventTy Event = EventSystem.createEvent(OriginEvents::loadBinary,
                                            EventTypeTy::LOAD_BINARY, DeviceId,
                                            TgtImage, Binary);

    if (Event.empty()) {
      REPORT("Failed to create loadBinary event for image %p", TgtImage);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Error = Event.getError(); Error) {
      REPORT("Event failed during loadBinary. %s\n",
             toString(std::move(Error)).c_str());
      return OFFLOAD_FAIL;
    }

    DeviceImgPtrToDeviceId[Binary->handle] = DeviceId;

    return OFFLOAD_SUCCESS;
  }

  void *data_alloc(int32_t DeviceId, int64_t Size, void *HostPtr,
                   int32_t Kind) override {
    if (Size == 0)
      return nullptr;

    void *TgtPtr = nullptr;
    std::optional<Error> Err = std::nullopt;
    EventTy Event;

    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
      Event = EventSystem.createEvent(OriginEvents::allocateBuffer,
                                      EventTypeTy::ALLOC, DeviceId, Size, Kind,
                                      &TgtPtr);

      if (Event.empty()) {
        Err = Plugin::error("Failed to create alloc event with size %z", Size);
        break;
      }

      Event.wait();
      Err = Event.getError();
      break;
    case TARGET_ALLOC_HOST:
      TgtPtr = memAllocHost(Size);
      Err = Plugin::check(TgtPtr == nullptr, "Failed to allocate host memory");
      break;
    case TARGET_ALLOC_SHARED:
      Err = Plugin::error("Incompatible memory type %d", Kind);
      break;
    }

    if (*Err) {
      REPORT("Failed to allocate data for HostPtr %p: %s\n", HostPtr,
             toString(std::move(*Err)).c_str());
      return nullptr;
    }

    return TgtPtr;
  }

  int32_t data_delete(int32_t DeviceId, void *TgtPtr, int32_t Kind) override {
    if (TgtPtr == nullptr)
      return OFFLOAD_SUCCESS;

    std::optional<Error> Err = std::nullopt;
    EventTy Event;

    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
      Event =
          EventSystem.createEvent(OriginEvents::deleteBuffer,
                                  EventTypeTy::DELETE, DeviceId, TgtPtr, Kind);

      if (Event.empty()) {
        Err = Plugin::error("Failed to create data delete event for %p TgtPtr",
                            TgtPtr);
        break;
      }

      Event.wait();
      Err = Event.getError();
      break;
    case TARGET_ALLOC_HOST:
      Err = Plugin::check(memFreeHost(TgtPtr), "Failed to free host memory");
      break;
    case TARGET_ALLOC_SHARED:
      Err = createStringError(inconvertibleErrorCode(),
                              "Incompatible memory type %d", Kind);
      break;
    }

    if (*Err) {
      REPORT("Failed delete data at %p TgtPtr: %s\n", TgtPtr,
             toString(std::move(*Err)).c_str());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t data_lock(int32_t DeviceId, void *Ptr, int64_t Size,
                    void **LockedPtr) override {
    EventTy Event =
        EventSystem.createEvent(OriginEvents::dataLock, EventTypeTy::DATA_LOCK,
                                DeviceId, Ptr, Size, LockedPtr);

    if (Event.empty()) {
      REPORT("Failed to create data lock event on device %d\n", DeviceId);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      REPORT("Failure to lock memory %p: %s\n", Ptr,
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    if (!(*LockedPtr)) {
      REPORT("Failure to lock memory %p: obtained a null locked pointer\n",
             Ptr);
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t data_unlock(int32_t DeviceId, void *Ptr) override {
    EventTy Event = EventSystem.createEvent(
        OriginEvents::dataUnlock, EventTypeTy::DATA_UNLOCK, DeviceId, Ptr);

    if (Event.empty()) {
      REPORT("Failed to create data unlock event on device %d\n", DeviceId);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      REPORT("Failure to unlock memory %p: %s\n", Ptr,
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t data_notify_mapped(int32_t DeviceId, void *HstPtr,
                             int64_t Size) override {
    EventTy Event = EventSystem.createEvent(OriginEvents::dataNotifyMapped,
                                            EventTypeTy::DATA_NOTIFY_MAPPED,
                                            DeviceId, HstPtr, Size);

    if (Event.empty()) {
      REPORT("Failed to create data notify mapped event on device %d\n",
             DeviceId);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      REPORT("Failure to notify data mapped %p: %s\n", HstPtr,
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t data_notify_unmapped(int32_t DeviceId, void *HstPtr) override {
    EventTy Event = EventSystem.createEvent(OriginEvents::dataNotifyUnmapped,
                                            EventTypeTy::DATA_NOTIFY_UNMAPPED,
                                            DeviceId, HstPtr);

    if (Event.empty()) {
      REPORT("Failed to create data notify unmapped event on device %d\n",
             DeviceId);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      REPORT("Failure to notify data unmapped %p: %s\n", HstPtr,
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t data_submit_async(int32_t DeviceId, void *TgtPtr, void *HstPtr,
                            int64_t Size,
                            __tgt_async_info *AsyncInfoPtr) override {
    MPIEventQueuePtr Queue = nullptr;
    if (auto Error = getQueue(AsyncInfoPtr, Queue)) {
      REPORT("Failed to get async Queue: %s\n",
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    EventTy Event =
        EventSystem.createEvent(OriginEvents::submit, EventTypeTy::SUBMIT,
                                DeviceId, TgtPtr, HstPtr, Size, AsyncInfoPtr);

    if (Event.empty()) {
      REPORT("Failed to create dataSubmit event from %p HstPtr to %p TgtPtr\n",
             HstPtr, TgtPtr);
      return OFFLOAD_FAIL;
    }

    Queue->push_back(Event);

    return OFFLOAD_SUCCESS;
  }

  int32_t data_retrieve_async(int32_t DeviceId, void *HstPtr, void *TgtPtr,
                              int64_t Size,
                              __tgt_async_info *AsyncInfoPtr) override {
    MPIEventQueuePtr Queue = nullptr;
    if (auto Error = getQueue(AsyncInfoPtr, Queue)) {
      REPORT("Failed to get async Queue: %s\n",
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    EventTy Event =
        EventSystem.createEvent(OriginEvents::retrieve, EventTypeTy::RETRIEVE,
                                DeviceId, Size, HstPtr, TgtPtr, AsyncInfoPtr);

    if (Event.empty()) {
      REPORT(
          "Failed to create dataRetrieve event from %p TgtPtr to %p HstPtr\n",
          TgtPtr, HstPtr);
      return OFFLOAD_FAIL;
    }

    Queue->push_back(Event);

    return OFFLOAD_SUCCESS;
  }

  int32_t data_exchange_async(int32_t SrcDeviceId, void *SrcPtr,
                              int DstDeviceId, void *DstPtr, int64_t Size,
                              __tgt_async_info *AsyncInfo) override {
    MPIEventQueuePtr Queue = nullptr;
    if (auto Error = getQueue(AsyncInfo, Queue)) {
      REPORT("Failed to get async Queue: %s\n",
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    int32_t SrcRank, SrcDevId, DstRank, DstDevId;
    EventTy Event;

    std::tie(SrcRank, SrcDevId) = EventSystem.mapDeviceId(SrcDeviceId);
    std::tie(DstRank, DstDevId) = EventSystem.mapDeviceId(DstDeviceId);

    if (SrcRank == DstRank) {
      Event = EventSystem.createEvent(
          OriginEvents::localExchange, EventTypeTy::LOCAL_EXCHANGE, SrcDeviceId,
          SrcPtr, DstDevId, DstPtr, Size, AsyncInfo);
    }

    else {
      Event = EventSystem.createExchangeEvent(SrcDeviceId, SrcPtr, DstDeviceId,
                                              DstPtr, Size, AsyncInfo);
    }

    if (Event.empty()) {
      REPORT("Failed to create data exchange event from %d SrcDeviceId to %d "
             "DstDeviceId\n",
             SrcDeviceId, DstDeviceId);
      return OFFLOAD_FAIL;
    }

    Queue->push_back(Event);

    return OFFLOAD_SUCCESS;
  }

  int32_t launch_kernel(int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs,
                        ptrdiff_t *TgtOffsets, KernelArgsTy *KernelArgs,
                        __tgt_async_info *AsyncInfoPtr) override {
    MPIEventQueuePtr Queue = nullptr;
    if (auto Error = getQueue(AsyncInfoPtr, Queue)) {
      REPORT("Failed to get async Queue: %s\n",
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    uint32_t NumArgs = KernelArgs->NumArgs;

    void *Args = memAllocHost(sizeof(void *) * NumArgs);
    std::memcpy(Args, TgtArgs, sizeof(void *) * NumArgs);
    EventDataHandleTy ArgsHandle(Args, &memFreeHost);

    void *Offsets = memAllocHost(sizeof(ptrdiff_t) * NumArgs);
    std::memcpy(Offsets, TgtOffsets, sizeof(ptrdiff_t) * NumArgs);
    EventDataHandleTy OffsetsHandle(Offsets, &memFreeHost);

    void *KernelArgsPtr = memAllocHost(sizeof(KernelArgsTy));
    std::memcpy(KernelArgsPtr, KernelArgs, sizeof(KernelArgsTy));
    EventDataHandleTy KernelArgsHandle(KernelArgsPtr, &memFreeHost);

    EventTy Event = EventSystem.createEvent(
        OriginEvents::launchKernel, EventTypeTy::LAUNCH_KERNEL, DeviceId,
        TgtEntryPtr, ArgsHandle, OffsetsHandle, KernelArgsHandle, AsyncInfoPtr);

    if (Event.empty()) {
      REPORT("Failed to create launchKernel event on device %d\n", DeviceId);
      return OFFLOAD_FAIL;
    }

    Queue->push_back(Event);

    return OFFLOAD_SUCCESS;
  }

  int32_t synchronize(int32_t DeviceId,
                      __tgt_async_info *AsyncInfoPtr) override {
    MPIEventQueuePtr Queue =
        reinterpret_cast<MPIEventQueuePtr>(AsyncInfoPtr->Queue);

    EventTy Event = EventSystem.createEvent(OriginEvents::synchronize,
                                            EventTypeTy::SYNCHRONIZE, DeviceId,
                                            AsyncInfoPtr);

    if (Event.empty()) {
      REPORT("Failed to create synchronize event on device %d\n", DeviceId);
      return OFFLOAD_FAIL;
    }

    Queue->push_back(Event);

    auto EventIt = Queue->begin();
    EventTypeTy CurrentEventType;

    while (EventIt != Queue->end()) {
      CurrentEventType = EventIt->getEventType();

      // Find the first event that differs in type from the current Event
      auto EventRangeEnd = std::find_if(
          EventIt, Queue->end(), [CurrentEventType](const EventTy &Event) {
            return Event.getEventType() != CurrentEventType;
          });

      std::list<MPIEventQueue::iterator> PendingEvents;
      for (auto It = EventIt; It != EventRangeEnd; ++It) {
        PendingEvents.push_back(It);
      }

      // Progress all the events in the range simultaneously
      while (!PendingEvents.empty()) {
        auto Event = PendingEvents.front();
        PendingEvents.pop_front();
        Event->resume();

        if (!Event->done()) {
          PendingEvents.push_back(Event);
          continue;
        }

        if (auto Error = Event->getError(); Error) {
          REPORT("Event failed during synchronization. %s\n",
                 toString(std::move(Error)).c_str());
          return OFFLOAD_FAIL;
        }
      }

      EventIt = EventRangeEnd;
    }

    // Once the queue is synchronized, return it to the pool and reset the
    // AsyncInfo. This is to make sure that the synchronization only works
    // for its own tasks.
    AsyncInfoPtr->Queue = nullptr;
    if (auto Error = returnQueue(Queue)) {
      REPORT("Failed to return async Queue: %s\n",
             toString(std::move(Error)).c_str());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t query_async(int32_t DeviceId,
                      __tgt_async_info *AsyncInfoPtr) override {
    auto *Queue = reinterpret_cast<MPIEventQueue *>(AsyncInfoPtr->Queue);

    EventTypeTy CurrentEventType;

    // Returns success when there are pending operations in AsyncInfo, moving
    // forward through the events on the queue until it is fully completed.
    for (auto EventIt = Queue->begin(); EventIt != Queue->end();) {
      CurrentEventType = EventIt->getEventType();
      EventIt->resume();
      auto NextEventIt = std::next(EventIt);
      if (!EventIt->done() &&
          ((CurrentEventType != EventTypeTy::SUBMIT &&
            CurrentEventType != EventTypeTy::RETRIEVE) ||
           (NextEventIt != Queue->end() &&
            NextEventIt->getEventType() != CurrentEventType))) {
        return OFFLOAD_SUCCESS;
      }

      if (EventIt->done()) {
        if (auto Error = EventIt->getError(); Error) {
          REPORT("Event failed during query. %s\n",
                 toString(std::move(Error)).c_str());
          return OFFLOAD_FAIL;
        }
        EventIt = Queue->erase(EventIt);
        continue;
      }

      ++EventIt;
    }

    if (!Queue->empty())
      return OFFLOAD_SUCCESS;

    // Once the queue is synchronized, return it to the pool and reset the
    // AsyncInfo. This is to make sure that the synchronization only works
    // for its own tasks.
    AsyncInfoPtr->Queue = nullptr;
    if (auto Error = returnQueue(Queue)) {
      REPORT("Failed to return async Queue: %s\n",
             toString(std::move(Error)).c_str());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  void print_device_info(int32_t DeviceId) override {
    EventTy Event =
        EventSystem.createEvent(OriginEvents::printDeviceInfo,
                                EventTypeTy::PRINT_DEVICE_INFO, DeviceId);

    if (Event.empty()) {
      REPORT("Failed to create printDeviceInfo event on device %d\n", DeviceId);
      return;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      REPORT("Failure to print device %d info: %s\n", DeviceId,
             toString(std::move(Error)).data());
    }
  }

  int32_t create_event(int32_t DeviceId, void **EventPtr) override {
    if (!EventPtr) {
      REPORT("Failure to record event: Received invalid event pointer\n");
      return OFFLOAD_FAIL;
    }

    EventTy *NewEvent = new EventTy;

    if (NewEvent == nullptr) {
      REPORT("Failed to createEvent\n");
      return OFFLOAD_FAIL;
    }

    *EventPtr = reinterpret_cast<void *>(NewEvent);

    return OFFLOAD_SUCCESS;
  }

  int32_t record_event(int32_t DeviceId, void *EventPtr,
                       __tgt_async_info *AsyncInfoPtr) override {
    if (!EventPtr) {
      REPORT("Failure to record event: Received invalid event pointer\n");
      return OFFLOAD_FAIL;
    }

    MPIEventQueuePtr Queue = nullptr;
    if (auto Error = getQueue(AsyncInfoPtr, Queue)) {
      REPORT("Failed to get async Queue: %s\n",
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    if (Queue->empty())
      return OFFLOAD_SUCCESS;

    auto &RecordedEvent = *reinterpret_cast<EventTy *>(EventPtr);
    RecordedEvent = Queue->back();

    return OFFLOAD_SUCCESS;
  }

  int32_t wait_event(int32_t DeviceId, void *EventPtr,
                     __tgt_async_info *AsyncInfoPtr) override {
    if (!EventPtr) {
      REPORT("Failure to wait event: Received invalid event pointer\n");
      return OFFLOAD_FAIL;
    }

    auto &RecordedEvent = *reinterpret_cast<EventTy *>(EventPtr);
    auto SyncEvent = OriginEvents::sync(RecordedEvent);

    MPIEventQueuePtr Queue = nullptr;
    if (auto Error = getQueue(AsyncInfoPtr, Queue)) {
      REPORT("Failed to get async Queue: %s\n",
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    Queue->push_back(SyncEvent);

    return OFFLOAD_SUCCESS;
  }

  int32_t sync_event(int32_t DeviceId, void *EventPtr) override {
    if (!EventPtr) {
      REPORT("Failure to wait event: Received invalid event pointer\n");
      return OFFLOAD_FAIL;
    }

    auto &RecordedEvent = *reinterpret_cast<EventTy *>(EventPtr);
    auto SyncEvent = OriginEvents::sync(RecordedEvent);

    SyncEvent.wait();

    if (auto Err = SyncEvent.getError()) {
      REPORT("Failure to synchronize event %p: %s\n", EventPtr,
             toString(std::move(Err)).data());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t destroy_event(int32_t DeviceId, void *EventPtr) override {

    if (!EventPtr) {
      REPORT("Failure to destroy event: Received invalid event pointer\n");
      return OFFLOAD_FAIL;
    }

    EventTy *MPIEventPtr = reinterpret_cast<EventTy *>(EventPtr);

    delete MPIEventPtr;

    return OFFLOAD_SUCCESS;
  }

  int32_t init_async_info(int32_t DeviceId,
                          __tgt_async_info **AsyncInfoPtr) override {
    assert(AsyncInfoPtr && "Invalid async info");

    EventTy Event = EventSystem.createEvent(OriginEvents::initAsyncInfo,
                                            EventTypeTy::INIT_ASYNC_INFO,
                                            DeviceId, AsyncInfoPtr);

    if (Event.empty()) {
      REPORT("Failed to create initAsyncInfo on device %d\n", DeviceId);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Err = Event.getError()) {
      REPORT("Failure to initialize async info at " DPxMOD
             " on device %d: %s\n",
             DPxPTR(*AsyncInfoPtr), DeviceId, toString(std::move(Err)).data());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t init_device_info(int32_t DeviceId, __tgt_device_info *DeviceInfo,
                           const char **ErrStr) override {
    *ErrStr = "";

    EventTy Event = EventSystem.createEvent(OriginEvents::initDeviceInfo,
                                            EventTypeTy::INIT_DEVICE_INFO,
                                            DeviceId, DeviceInfo);

    if (Event.empty()) {
      REPORT("Failed to create initDeviceInfo on device %d\n", DeviceId);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Err = Event.getError()) {
      REPORT("Failure to initialize device info at " DPxMOD
             " on device %d: %s\n",
             DPxPTR(DeviceInfo), DeviceId, toString(std::move(Err)).data());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t use_auto_zero_copy(int32_t DeviceId) override { return false; }

  int32_t get_global(__tgt_device_binary Binary, uint64_t Size,
                     const char *Name, void **DevicePtr) override {
    int32_t DeviceId = DeviceImgPtrToDeviceId[Binary.handle];

    EventTy Event = EventSystem.createEvent(OriginEvents::getGlobal,
                                            EventTypeTy::GET_GLOBAL, DeviceId,
                                            Binary, Size, Name, DevicePtr);
    if (Event.empty()) {
      REPORT("Failed to create getGlobal event on device %d\n", 0);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      REPORT("Failed to get Global on device %d: %s\n", 0,
             toString(std::move(Error)).c_str());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t get_function(__tgt_device_binary Binary, const char *Name,
                       void **KernelPtr) override {

    int32_t DeviceId = DeviceImgPtrToDeviceId[Binary.handle];

    EventTy Event = EventSystem.createEvent(OriginEvents::getFunction,
                                            EventTypeTy::GET_FUNCTION, DeviceId,
                                            Binary, Name, KernelPtr);
    if (Event.empty()) {
      REPORT("Failed to create getFunction event on device %d\n", 0);
      return OFFLOAD_FAIL;
    }

    Event.wait();

    if (auto Error = Event.getError()) {
      REPORT("Failed to get function on device %d: %s\n", 0,
             toString(std::move(Error)).c_str());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

private:
  std::mutex MPIQueueMutex;
  llvm::DenseMap<uintptr_t, int32_t> DeviceImgPtrToDeviceId;
  llvm::SmallVector<void *> RemoteDevices;
  llvm::SmallVector<bool> RemoteDevicesInitialized;
  EventSystemTy EventSystem;
};

template <typename... ArgsTy>
static Error Plugin::check(int32_t ErrorCode, const char *ErrFmt,
                           ArgsTy... Args) {
  if (ErrorCode == OFFLOAD_SUCCESS)
    return Error::success();

  return createStringError<ArgsTy..., const char *>(
      inconvertibleErrorCode(), ErrFmt, Args...,
      std::to_string(ErrorCode).data());
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
  if (auto Err = Plugin->init()) {
    DP("ompfile_mpp_init: Plugin->init failed: %s\n",
       llvm::toString(std::move(Err)).c_str());
    return OFFLOAD_FAIL;
  }
  if (!Plugin->ensureEventSystemInitializedForOmpFile()) {
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

  if (auto Err = Plugin->init())
    return OFFLOAD_FAIL;

  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return OFFLOAD_FAIL;

  EventTy Event = Plugin->getEventSystemForOmpFile().createEvent(
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

  if (auto Err = Plugin->init())
    return OFFLOAD_FAIL;

  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return OFFLOAD_FAIL;

  int RemoteHandle = -1;
  int RemoteErrno = 0;
  EventTy Event = Plugin->getEventSystemForOmpFile().createEvent(
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

  if (auto Err = Plugin->init())
    return OFFLOAD_FAIL;

  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return OFFLOAD_FAIL;

  int RemoteHandle = -1;
  int RemoteErrno = 0;
  EventTy Event = Plugin->getEventSystemForOmpFile().createEvent(
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

  if (auto Err = Plugin->init()) {
    DP("ompfile_mpp_sched_request: Plugin->init failed: %s\n",
       llvm::toString(std::move(Err)).c_str());
    return OFFLOAD_FAIL;
  }

  if (!Plugin->ensureEventSystemInitializedForOmpFile()) {
    DP("ompfile_mpp_sched_request: ensureEventSystemInitializedForOmpFile "
       "failed.\n");
    return OFFLOAD_FAIL;
  }

  EventTy Event = Plugin->getEventSystemForOmpFile().createEvent(
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

  if (auto Err = Plugin->init()) {
    DP("ompfile_mpp_sched_request_batch: Plugin->init failed: %s\n",
       llvm::toString(std::move(Err)).c_str());
    return OFFLOAD_FAIL;
  }

  if (!Plugin->ensureEventSystemInitializedForOmpFile()) {
    DP("ompfile_mpp_sched_request_batch: "
       "ensureEventSystemInitializedForOmpFile failed.\n");
    return OFFLOAD_FAIL;
  }

  EventTy Event = Plugin->getEventSystemForOmpFile().createEvent(
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

  if (auto Err = Plugin->init())
    return OFFLOAD_FAIL;

  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return OFFLOAD_FAIL;

  OmpfileMppHandleEntry Entry{};
  {
    const std::lock_guard<std::mutex> Lock(getOmpfileMppMutex());
    if (!eraseOmpfileMppHandle(Handle, Entry))
      return OFFLOAD_FAIL;
  }

  int CloseRet = -1;
  int RemoteErrno = 0;
  EventTy Event = Plugin->getEventSystemForOmpFile().createEvent(
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

  if (auto Err = Plugin->init())
    return OFFLOAD_FAIL;

  if (!Plugin->ensureEventSystemInitializedForOmpFile())
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
  EventTy Event = Plugin->getEventSystemForOmpFile().createEvent(
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

int ompfile_mpp_pread(int Handle, int64_t Offset, void *Buffer, uint64_t Size) {
  uint64_t BytesRead = 0;
  return ompfile_mpp_pread_ex(Handle, Offset, Buffer, Size, &BytesRead);
}

int ompfile_mpp_pread_no_stage_ex(int Handle, int64_t Offset, void *Buffer,
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

  if (auto Err = Plugin->init())
    return OFFLOAD_FAIL;

  if (!Plugin->ensureEventSystemInitializedForOmpFile())
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
  EventTy Event = Plugin->getEventSystemForOmpFile().createEvent(
      OriginEvents::ompfilePread, EventTypeTy::OMPFILE_PREAD_NO_STAGE,
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

  if (auto Err = Plugin->init())
    return OFFLOAD_FAIL;

  if (!Plugin->ensureEventSystemInitializedForOmpFile())
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
  EventTy Event = Plugin->getEventSystemForOmpFile().createEvent(
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

int ompfile_mpp_pwrite(int Handle, int64_t Offset, const void *Buffer,
                       uint64_t Size) {
  uint64_t BytesWritten = 0;
  if (ompfile_mpp_pwrite_ex(Handle, Offset, Buffer, Size, &BytesWritten) !=
      OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;
  if (BytesWritten != Size) {
    errno = EIO;
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_stage_invalidate_path_key(uint64_t PathKey,
                                          uint64_t Generation,
                                          const char *Path) {
  using namespace llvm::omp::target::plugin;
  if (PathKey == 0 || !Path || Path[0] == '\0')
    return ENOKEY;

  const size_t PathSize = std::strlen(Path) + 1;
  if (PathSize > static_cast<size_t>(INT_MAX) ||
      PathSize > static_cast<size_t>(PATH_MAX) + 1)
    return EOVERFLOW;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return EHOSTDOWN;

  if (auto Err = Plugin->init())
    return EHOSTDOWN;

  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return EHOSTDOWN;

  OmpFileStageInvalidateRequest Request{};
  Request.AbiVersion = OMPFILE_STAGE_INVALIDATE_ABI_VERSION;
  Request.PathSize = static_cast<uint32_t>(PathSize);
  Request.PathKey = PathKey;
  Request.Generation = Generation;

  EventSystemTy &EventSystem = Plugin->getEventSystemForOmpFile();
  int FirstErrno = 0;
  for (int Rank = 0; Rank < EventSystem.getNumWorkers(); ++Rank) {
    OmpFileStageInvalidateReply Reply{};
    EventTy Event = EventSystem.createEvent(
        OriginEvents::ompfileStageInvalidate,
        EventTypeTy::OMPFILE_STAGE_INVALIDATE, /*DstDeviceID=*/Rank, &Request,
        Path, &Reply);
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

    if (Reply.AbiVersion != OMPFILE_STAGE_INVALIDATE_ABI_VERSION) {
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
  if (auto Err = Plugin->init())
    return EHOSTDOWN;
  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return EHOSTDOWN;

  EventSystemTy &EventSystem = Plugin->getEventSystemForOmpFile();
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
  if (auto Err = Plugin->init())
    return EHOSTDOWN;
  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return EHOSTDOWN;

  OmpFileFreshnessWriteCommitRequest Request{};
  Request.PathKey = PathKey;
  Request.TileId = TileId;
  Request.WriterRank = WriterRank;
  Request.WriteThroughMode = WriteThroughMode != 0 ? 1u : 0u;
  OmpFileFreshnessWriteCommitReply Reply{};

  EventSystemTy &EventSystem = Plugin->getEventSystemForOmpFile();
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

int ompfile_mpp_proxy_copy_tile(uint64_t PathKey, uint64_t TileId,
                                int SourceRank, int DestRank,
                                uint64_t Version) {
  using namespace llvm::omp::target::plugin;
  if (PathKey == 0 || SourceRank < 0 || DestRank < 0 || Version == 0)
    return EINVAL;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return EHOSTDOWN;
  if (auto Err = Plugin->init())
    return EHOSTDOWN;
  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return EHOSTDOWN;

  EventSystemTy &EventSystem = Plugin->getEventSystemForOmpFile();
  if (SourceRank >= EventSystem.getNumWorkers() ||
      DestRank >= EventSystem.getNumWorkers())
    return EINVAL;

  OmpFileProxyCopyTileRequest Request{};
  Request.PathKey = PathKey;
  Request.TileId = TileId;
  Request.SourceRank = SourceRank;
  Request.DestRank = DestRank;
  Request.Version = Version;

  OmpFileProxyCopyTileReply HeadnodeReply{};
  EventTy HeadnodeEvent = EventSystem.createEvent(
      OriginEvents::ompfileProxyCopyTile, EventTypeTy::OMPFILE_PROXY_COPY_TILE,
      /*DstDeviceID=*/0, &Request, &HeadnodeReply);
  if (HeadnodeEvent.empty())
    return EIO;
  HeadnodeEvent.wait();
  if (auto Error = HeadnodeEvent.getError())
    return EIO;
  if (HeadnodeReply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION)
    return EPROTO;
  if (HeadnodeReply.Status != 0)
    return HeadnodeReply.Errno != 0 ? HeadnodeReply.Errno : EIO;

  if (SourceRank == DestRank)
    return OFFLOAD_SUCCESS;

  OmpFileProxyCopyTileReply Reply{};
  EventTy Event = EventSystem.createEvent(
      OriginEvents::ompfileProxyCopyTile, EventTypeTy::OMPFILE_PROXY_COPY_TILE,
      /*DstDeviceID=*/DestRank, &Request, &Reply);
  if (Event.empty())
    return EIO;
  Event.wait();
  if (auto Error = Event.getError())
    return EIO;
  if (Reply.AbiVersion != OMPFILE_FRESHNESS_QUERY_ABI_VERSION)
    return EPROTO;
  if (Reply.Status != 0)
    return Reply.Errno != 0 ? Reply.Errno : EIO;
  return OFFLOAD_SUCCESS;
}

int ompfile_mpp_freshness_mark_fresh(uint64_t PathKey, int Rank,
                                     uint64_t Version) {
  using namespace llvm::omp::target::plugin;
  if (PathKey == 0 || Rank < 0 || Version == 0)
    return EINVAL;

  MPIPluginTy *Plugin = ActiveMPIPlugin.load();
  if (!Plugin)
    return EHOSTDOWN;
  if (auto Err = Plugin->init())
    return EHOSTDOWN;
  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return EHOSTDOWN;

  OmpFileFreshnessMarkFreshRequest Request{};
  Request.PathKey = PathKey;
  Request.Rank = Rank;
  Request.Version = Version;
  OmpFileFreshnessMarkFreshReply Reply{};

  EventSystemTy &EventSystem = Plugin->getEventSystemForOmpFile();
  EventTy Event = EventSystem.createEvent(
      OriginEvents::ompfileFreshnessMarkFresh,
      EventTypeTy::OMPFILE_FRESHNESS_MARK_FRESH, /*DstDeviceID=*/0, &Request,
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
  if (auto Err = Plugin->init())
    return EHOSTDOWN;
  if (!Plugin->ensureEventSystemInitializedForOmpFile())
    return EHOSTDOWN;

  EventSystemTy &EventSystem = Plugin->getEventSystemForOmpFile();

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

llvm::omp::target::plugin::GenericPluginTy *createPlugin_mpi() {
  auto *Plugin = new llvm::omp::target::plugin::MPIPluginTy();
  llvm::omp::target::plugin::ActiveMPIPlugin.store(Plugin);
  return Plugin;
}
}
