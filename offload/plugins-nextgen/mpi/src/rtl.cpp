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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Utils/ELF.h"

#include "EventSystem.h"
#include "OriginOmpFileBridge.h"
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

// Two opt-in knobs for the origin (application) side of a mapped variable.
// Every mapped variable of a target region costs the origin four synchronous
// round trips to the proxy (ALLOC, DATA_NOTIFY_MAPPED, DATA_NOTIFY_UNMAPPED,
// DELETE) on top of the queued SUBMIT/RETRIEVE, and on a slow interconnect
// those round trips are the per-record floor of an I/O-bound region. Both
// default off so every existing lane keeps its behaviour.
//
// OMPTARGET_MPI_SKIP_DATA_NOTIFY=1: answer notifyDataMapped/Unmapped locally
// without a round trip. The proxy forwards them to its host plugin, for which
// they are no-ops (they exist so GPU plugins can pin host memory), so nothing
// observable changes on the proxy.
//
// OMPTARGET_MPI_ALLOC_POOL_BYTES=<n>: keep up to n bytes of freed device
// buffers per origin instead of sending DELETE, and serve a later ALLOC of the
// same (device, size, kind) from that pool without a round trip. Device memory
// contents are unspecified after an allocation either way, so the reuse is
// invisible to the program; pooled buffers are released when the plugin
// deinitializes.
static bool mpiSkipDataNotifyEnabled() {
  static const bool Enabled = [] {
    const char *Raw = std::getenv("OMPTARGET_MPI_SKIP_DATA_NOTIFY");
    return Raw && Raw[0] == '1' && Raw[1] == '\0';
  }();
  return Enabled;
}

static int64_t mpiAllocPoolBytes() {
  static const int64_t Bytes = [] {
    const char *Raw = std::getenv("OMPTARGET_MPI_ALLOC_POOL_BYTES");
    if (!Raw || !Raw[0])
      return static_cast<int64_t>(0);
    char *End = nullptr;
    const long long Value = std::strtoll(Raw, &End, 10);
    if (!End || *End != '\0' || Value <= 0)
      return static_cast<int64_t>(0);
    return static_cast<int64_t>(Value);
  }();
  return Bytes;
}

// OMPTARGET_MPI_EVENT_STATS=1: origin-side wall time per event type. The
// synchronous waits (ALLOC, DELETE, DATA_NOTIFY_*) are timed one event at a
// time; the queued types (SUBMIT, RETRIEVE, LAUNCH_KERNEL, SYNCHRONIZE) are
// timed per same-type range as synchronize() progresses them together, so a
// range's wall time is what one target region actually pays for that type.
// Printed at plugin deinit as "[mpi-plugin] origin_event_stats ..." lines.
static bool mpiEventStatsEnabled() {
  static const bool Enabled = [] {
    const char *Raw = std::getenv("OMPTARGET_MPI_EVENT_STATS");
    return Raw && Raw[0] == '1' && Raw[1] == '\0';
  }();
  return Enabled;
}

static inline uint64_t mpiNowUs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

struct MPIOriginEventStatsTy {
  static constexpr size_t kTypes = 96;
  std::atomic<uint64_t> Count[kTypes];
  std::atomic<uint64_t> Ranges[kTypes];
  std::atomic<uint64_t> TotalUs[kTypes];
  std::atomic<uint64_t> MaxUs[kTypes];

  MPIOriginEventStatsTy() {
    for (size_t I = 0; I < kTypes; ++I) {
      Count[I] = 0;
      Ranges[I] = 0;
      TotalUs[I] = 0;
      MaxUs[I] = 0;
    }
  }

  void add(EventTypeTy Type, uint64_t Us, uint64_t Events) {
    const size_t I = static_cast<size_t>(Type);
    if (I >= kTypes)
      return;
    Count[I].fetch_add(Events, std::memory_order_relaxed);
    Ranges[I].fetch_add(1, std::memory_order_relaxed);
    TotalUs[I].fetch_add(Us, std::memory_order_relaxed);
    uint64_t Prev = MaxUs[I].load(std::memory_order_relaxed);
    while (Us > Prev &&
           !MaxUs[I].compare_exchange_weak(Prev, Us, std::memory_order_relaxed))
      ;
  }

  void print() const {
    for (size_t I = 0; I < kTypes; ++I) {
      const uint64_t N = Count[I].load(std::memory_order_relaxed);
      if (N == 0)
        continue;
      const uint64_t R = Ranges[I].load(std::memory_order_relaxed);
      const uint64_t T = TotalUs[I].load(std::memory_order_relaxed);
      fprintf(stderr,
              "[mpi-plugin] origin_event_stats type=%s events=%llu ranges=%llu "
              "total_us=%llu max_range_us=%llu avg_range_us=%.1f\n",
              EventTypeToString(static_cast<EventTypeTy>(I)).c_str(),
              static_cast<unsigned long long>(N),
              static_cast<unsigned long long>(R),
              static_cast<unsigned long long>(T),
              static_cast<unsigned long long>(
                  MaxUs[I].load(std::memory_order_relaxed)),
              R ? static_cast<double>(T) / static_cast<double>(R) : 0.0);
    }
  }
};

static MPIOriginEventStatsTy &originEventStats() {
  static MPIOriginEventStatsTy *Stats = new MPIOriginEventStatsTy();
  return *Stats;
}

// OMPTARGET_MPI_MERGE_SATISFIED_SYNC=1: let a SYNC event that is already
// satisfied join the range synchronize() is progressing instead of ending
// it. libomptarget records an event on every new mapping after its submit
// and, at region end, waits on that event (wait_event -> a SYNC entry)
// right before the variable's retrieve, so a region's queue reads
// SYNC, RETRIEVE, SYNC, RETRIEVE, ... and each retrieve is progressed alone,
// one proxy round trip at a time (AMD 404114: 804 retrieves in 804 ranges
// at 5 ms each, the largest term of a legacy record). Ranges complete in
// order, so by the time such a SYNC is reached its recorded event has
// completed; a SYNC that is not satisfied after one step still ends the
// range, which keeps the ordering it exists for.
static bool mpiMergeSatisfiedSyncEnabled() {
  static const bool Enabled = [] {
    const char *Raw = std::getenv("OMPTARGET_MPI_MERGE_SATISFIED_SYNC");
    return Raw && Raw[0] == '1' && Raw[1] == '\0';
  }();
  return Enabled;
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
void ompfileDiagLog(const char *Phase, int Rank, int DeviceCount,
                    int DeviceId, int ExtraA, int ExtraB) {
  if (!ompfileDiagEnabled())
    return;
  REPORT("[ompfile-mpi-diag] phase=%s rank=%d device_count=%d device_id=%d "
         "extra_a=%d extra_b=%d\n",
         Phase ? Phase : "unknown", Rank, DeviceCount, DeviceId, ExtraA,
         ExtraB);
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
  ~MPIPluginTy() override { ompfileRetractMPIPlugin(this); }

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
    allocPoolRelease();
    if (mpiEventStatsEnabled())
      originEventStats().print();
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
      if (void *Pooled = allocPoolTake(DeviceId, Size, Kind))
        return Pooled;

      Event = EventSystem.createEvent(OriginEvents::allocateBuffer,
                                      EventTypeTy::ALLOC, DeviceId, Size, Kind,
                                      &TgtPtr);

      if (Event.empty()) {
        Err = Plugin::error("Failed to create alloc event with size %z", Size);
        break;
      }

      {
        const uint64_t T0 = mpiEventStatsEnabled() ? mpiNowUs() : 0;
        Event.wait();
        if (T0)
          originEventStats().add(EventTypeTy::ALLOC, mpiNowUs() - T0, 1);
      }
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

    allocPoolRecord(DeviceId, TgtPtr, Size, Kind);
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
      if (allocPoolGive(DeviceId, TgtPtr, Kind))
        return OFFLOAD_SUCCESS;

      Event =
          EventSystem.createEvent(OriginEvents::deleteBuffer,
                                  EventTypeTy::DELETE, DeviceId, TgtPtr, Kind);

      if (Event.empty()) {
        Err = Plugin::error("Failed to create data delete event for %p TgtPtr",
                            TgtPtr);
        break;
      }

      {
        const uint64_t T0 = mpiEventStatsEnabled() ? mpiNowUs() : 0;
        Event.wait();
        if (T0)
          originEventStats().add(EventTypeTy::DELETE, mpiNowUs() - T0, 1);
      }
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
    if (mpiSkipDataNotifyEnabled())
      return OFFLOAD_SUCCESS;

    EventTy Event = EventSystem.createEvent(OriginEvents::dataNotifyMapped,
                                            EventTypeTy::DATA_NOTIFY_MAPPED,
                                            DeviceId, HstPtr, Size);

    if (Event.empty()) {
      REPORT("Failed to create data notify mapped event on device %d\n",
             DeviceId);
      return OFFLOAD_FAIL;
    }

    {
      const uint64_t T0 = mpiEventStatsEnabled() ? mpiNowUs() : 0;
      Event.wait();
      if (T0)
        originEventStats().add(EventTypeTy::DATA_NOTIFY_MAPPED,
                               mpiNowUs() - T0, 1);
    }

    if (auto Error = Event.getError()) {
      REPORT("Failure to notify data mapped %p: %s\n", HstPtr,
             toString(std::move(Error)).data());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  int32_t data_notify_unmapped(int32_t DeviceId, void *HstPtr) override {
    if (mpiSkipDataNotifyEnabled())
      return OFFLOAD_SUCCESS;

    EventTy Event = EventSystem.createEvent(OriginEvents::dataNotifyUnmapped,
                                            EventTypeTy::DATA_NOTIFY_UNMAPPED,
                                            DeviceId, HstPtr);

    if (Event.empty()) {
      REPORT("Failed to create data notify unmapped event on device %d\n",
             DeviceId);
      return OFFLOAD_FAIL;
    }

    {
      const uint64_t T0 = mpiEventStatsEnabled() ? mpiNowUs() : 0;
      Event.wait();
      if (T0)
        originEventStats().add(EventTypeTy::DATA_NOTIFY_UNMAPPED,
                               mpiNowUs() - T0, 1);
    }

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

      // Find the first event that differs in type from the current Event.
      // With OMPTARGET_MPI_MERGE_SATISFIED_SYNC a SYNC whose recorded event
      // has already completed is stepped once here and, once done, skipped
      // over so the range continues with the next event of the same type.
      auto EventRangeEnd = EventIt;
      const bool MergeSync = mpiMergeSatisfiedSyncEnabled() &&
                             CurrentEventType != EventTypeTy::SYNC;
      while (EventRangeEnd != Queue->end()) {
        const EventTypeTy Type = EventRangeEnd->getEventType();
        if (Type == CurrentEventType) {
          ++EventRangeEnd;
          continue;
        }
        if (MergeSync && Type == EventTypeTy::SYNC) {
          if (!EventRangeEnd->done())
            EventRangeEnd->resume();
          if (EventRangeEnd->done()) {
            if (auto Error = EventRangeEnd->getError(); Error) {
              REPORT("Event failed during synchronization. %s\n",
                     toString(std::move(Error)).c_str());
              return OFFLOAD_FAIL;
            }
            ++EventRangeEnd;
            continue;
          }
        }
        break;
      }

      std::list<MPIEventQueue::iterator> PendingEvents;
      for (auto It = EventIt; It != EventRangeEnd; ++It) {
        if (!It->done())
          PendingEvents.push_back(It);
      }
      const uint64_t RangeEvents = PendingEvents.size();
      const uint64_t RangeT0 = mpiEventStatsEnabled() ? mpiNowUs() : 0;

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
      if (RangeT0)
        originEventStats().add(CurrentEventType, mpiNowUs() - RangeT0,
                               RangeEvents);

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
  // Origin-side pool of device buffers (OMPTARGET_MPI_ALLOC_POOL_BYTES).
  // Keyed by (device, size, kind); AllocSizes remembers the size of every
  // buffer this origin allocated while the pool is enabled, because
  // data_delete is not told it.
  using AllocPoolKeyTy = std::tuple<int32_t, int64_t, int32_t>;
  struct AllocPoolKeyHash {
    size_t operator()(const AllocPoolKeyTy &Key) const {
      return std::hash<int64_t>()(std::get<1>(Key)) ^
             (std::hash<int32_t>()(std::get<0>(Key)) << 1) ^
             (std::hash<int32_t>()(std::get<2>(Key)) << 2);
    }
  };

  void *allocPoolTake(int32_t DeviceId, int64_t Size, int32_t Kind) {
    if (mpiAllocPoolBytes() <= 0)
      return nullptr;
    const std::lock_guard<std::mutex> Lock(AllocPoolMutex);
    auto It = AllocPool.find(AllocPoolKeyTy{DeviceId, Size, Kind});
    if (It == AllocPool.end() || It->second.empty())
      return nullptr;
    void *TgtPtr = It->second.back();
    It->second.pop_back();
    AllocPoolBytesHeld -= Size;
    return TgtPtr;
  }

  void allocPoolRecord(int32_t DeviceId, void *TgtPtr, int64_t Size,
                       int32_t Kind) {
    if (mpiAllocPoolBytes() <= 0 || !TgtPtr)
      return;
    const std::lock_guard<std::mutex> Lock(AllocPoolMutex);
    AllocSizes[TgtPtr] = std::make_pair(Size, DeviceId);
    (void)Kind;
  }

  // True when the buffer was kept for reuse and no DELETE is needed.
  bool allocPoolGive(int32_t DeviceId, void *TgtPtr, int32_t Kind) {
    const int64_t Cap = mpiAllocPoolBytes();
    if (Cap <= 0)
      return false;
    const std::lock_guard<std::mutex> Lock(AllocPoolMutex);
    auto SizeIt = AllocSizes.find(TgtPtr);
    if (SizeIt == AllocSizes.end() || SizeIt->second.second != DeviceId)
      return false;
    const int64_t Size = SizeIt->second.first;
    if (AllocPoolBytesHeld + Size > Cap) {
      AllocSizes.erase(SizeIt);
      return false;
    }
    AllocPool[AllocPoolKeyTy{DeviceId, Size, Kind}].push_back(TgtPtr);
    AllocPoolBytesHeld += Size;
    return true;
  }

  void allocPoolRelease() {
    std::vector<std::tuple<int32_t, void *, int32_t>> Pending;
    {
      const std::lock_guard<std::mutex> Lock(AllocPoolMutex);
      for (auto &Entry : AllocPool)
        for (void *TgtPtr : Entry.second)
          Pending.emplace_back(std::get<0>(Entry.first), TgtPtr,
                               std::get<2>(Entry.first));
      AllocPool.clear();
      AllocSizes.clear();
      AllocPoolBytesHeld = 0;
    }
    for (auto &[DeviceId, TgtPtr, Kind] : Pending) {
      EventTy Event =
          EventSystem.createEvent(OriginEvents::deleteBuffer,
                                  EventTypeTy::DELETE, DeviceId, TgtPtr, Kind);
      if (Event.empty())
        continue;
      Event.wait();
      if (auto Error = Event.getError())
        REPORT("Failed to release pooled buffer %p: %s\n", TgtPtr,
               toString(std::move(Error)).c_str());
    }
  }

  std::mutex AllocPoolMutex;
  std::unordered_map<AllocPoolKeyTy, std::vector<void *>, AllocPoolKeyHash>
      AllocPool;
  std::unordered_map<void *, std::pair<int64_t, int32_t>> AllocSizes;
  int64_t AllocPoolBytesHeld = 0;

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

// Accessors for OriginOmpFileBridge.cpp, which sees MPIPluginTy incomplete.
llvm::Error ompfileInitMPIPlugin(MPIPluginTy &Plugin) { return Plugin.init(); }
bool ompfileEnsureEventSystem(MPIPluginTy &Plugin) {
  return Plugin.ensureEventSystemInitializedForOmpFile();
}
EventSystemTy &ompfileEventSystem(MPIPluginTy &Plugin) {
  return Plugin.getEventSystemForOmpFile();
}

} // namespace llvm::omp::target::plugin

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_mpi() {
  auto *Plugin = new llvm::omp::target::plugin::MPIPluginTy();
  llvm::omp::target::plugin::ompfilePublishMPIPlugin(Plugin);
  return Plugin;
}
}
