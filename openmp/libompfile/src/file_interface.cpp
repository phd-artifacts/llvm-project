#include "abstract_backend.h"
#include "debug_log.h"
#include "mpp_shim.h"
#include "ompfile_sched.h"
#include "mpi.h"
#include "mpi_io_backend.h"
#include "posix_backend.h"
#include "io_uring_io_backend.h"
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <string>

enum class IOSchedulerTy {
  LOCAL,
  HEADNODE,
};

class IOScheduler {
public:
  virtual ~IOScheduler() {}
  virtual int open(const char *filename) = 0;
  virtual int write(int file_handle, const void *data, size_t size) = 0;
  virtual int read(int file_handle, void *data, size_t size) = 0;
  virtual int close(int file_handle) = 0;
  virtual int seek(int file_handle, long offset) = 0;
  virtual int readAt(int file_handle, long offset, void *data, size_t size) = 0;
  virtual int writeAt(int file_handle, long offset, const void *data,
                      size_t size) = 0;
};

class LocalScheduler final : public IOScheduler {
public:
  explicit LocalScheduler(IOBackend &backend) : backend(backend) {}

  int open(const char *filename) override { return backend.open(filename); }
  int write(int file_handle, const void *data, size_t size) override {
    return backend.write(file_handle, data, size);
  }
  int read(int file_handle, void *data, size_t size) override {
    return backend.read(file_handle, data, size);
  }
  int close(int file_handle) override { return backend.close(file_handle); }
  int seek(int file_handle, long offset) override {
    return backend.seek(file_handle, offset);
  }
  int readAt(int file_handle, long offset, void *data, size_t size) override {
    return backend.readAt(file_handle, offset, data, size);
  }
  int writeAt(int file_handle, long offset, const void *data,
              size_t size) override {
    return backend.writeAt(file_handle, offset, data, size);
  }

private:
  IOBackend &backend;
};

class HeadnodeScheduler final : public IOScheduler {
public:
  explicit HeadnodeScheduler(IOBackend &backend)
      : backend(backend), client_rank(resolveClientRank()) {}

  int open(const char *filename) override {
    scheduleOpen(filename);
    const int file_handle = backend.open(filename);
    if (file_handle >= 0 && filename)
      rememberOpenPath(file_handle, filename);
    return file_handle;
  }

  int write(int file_handle, const void *data, size_t size) override {
    scheduleWrite(file_handle, /*offset=*/0, size);
    return backend.write(file_handle, data, size);
  }

  int read(int file_handle, void *data, size_t size) override {
    scheduleRead(file_handle, /*offset=*/0, size);
    return backend.read(file_handle, data, size);
  }

  int close(int file_handle) override {
    scheduleClose(file_handle);
    const int rc = backend.close(file_handle);
    if (rc == 0)
      forgetOpenPath(file_handle);
    return rc;
  }

  int seek(int file_handle, long offset) override {
    return backend.seek(file_handle, offset);
  }

  int readAt(int file_handle, long offset, void *data, size_t size) override {
    ompfile::OmpFileReadRequestContext context{};
    buildReadContext(file_handle, offset, size, context);
    return backend.readAtWithContext(context, data, size);
  }

  int writeAt(int file_handle, long offset, const void *data,
              size_t size) override {
    scheduleWrite(file_handle, offset, size);
    return backend.writeAt(file_handle, offset, data, size);
  }

private:
  struct TrackedFileMetadata {
    std::string Path;
    uint64_t PathKey = 0;
  };

  IOBackend &backend;
  int client_rank = -1;
  std::atomic<uint64_t> request_id{1};
  std::mutex tracked_file_mutex;
  std::unordered_map<int, TrackedFileMetadata> tracked_file_map;

  static int resolveClientRank() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
      return -1;
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  static uint64_t computePathKey(const char *path) {
    // 64-bit FNV-1a hash for a stable per-path key in scheduler context.
    constexpr uint64_t offset_basis = 1469598103934665603ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset_basis;
    if (!path)
      return hash;

    for (const unsigned char *p =
             reinterpret_cast<const unsigned char *>(path);
         *p != '\0'; ++p) {
      hash ^= static_cast<uint64_t>(*p);
      hash *= prime;
    }
    return hash;
  }

  void rememberOpenPath(int file_handle, const char *path) {
    if (!path)
      return;
    TrackedFileMetadata metadata{};
    metadata.Path = path;
    metadata.PathKey = computePathKey(path);
    const std::lock_guard<std::mutex> lock(tracked_file_mutex);
    tracked_file_map[file_handle] = std::move(metadata);
  }

  void forgetOpenPath(int file_handle) {
    const std::lock_guard<std::mutex> lock(tracked_file_mutex);
    tracked_file_map.erase(file_handle);
  }

  bool getTrackedPath(int file_handle, std::string &path_out,
                      uint64_t &path_key_out) {
    const std::lock_guard<std::mutex> lock(tracked_file_mutex);
    auto it = tracked_file_map.find(file_handle);
    if (it == tracked_file_map.end())
      return false;
    path_out = it->second.Path;
    path_key_out = it->second.PathKey;
    return true;
  }

  bool schedRequest(const ompfile::OmpFileIORequest &req,
                    const char *path, ompfile::OmpFileIOPlan *plan_out) {
    ompfile::OmpFileIOPlan plan{};
    if (!ompfile::mpp::schedRequest(req, path, plan)) {
      io_log("HEADNODE scheduler request failed (global manager unavailable); "
             "falling back to LOCAL.\n");
      return false;
    }

    if (plan.Status != 0) {
      errno = plan.Errno;
      return false;
    }

    if (plan_out)
      *plan_out = plan;
    return true;
  }

  void scheduleOpen(const char *path) {
    if (!path)
      return;
    ompfile::OmpFileIORequest req{};
    req.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    req.Op = ompfile::OmpFileIOOp::OPEN;
    req.ClientRank = client_rank;
    req.PathSize = static_cast<uint32_t>(std::strlen(path) + 1);
    (void)schedRequest(req, path, nullptr);
  }

  void scheduleClose(int file_handle) {
    ompfile::OmpFileIORequest req{};
    req.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    req.Op = ompfile::OmpFileIOOp::CLOSE;
    req.ClientRank = client_rank;
    req.FileHandle = file_handle;
    (void)schedRequest(req, nullptr, nullptr);
  }

  void buildReadContext(int file_handle, long offset, size_t size,
                        ompfile::OmpFileReadRequestContext &context) {
    context = {};
    context.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    context.FileHandle = file_handle;
    context.ClientRank = client_rank;
    context.Offset = static_cast<int64_t>(offset);
    context.Size = static_cast<uint64_t>(size);
    context.PathKey = static_cast<uint64_t>(static_cast<uint32_t>(file_handle));

    std::string tracked_path;
    if (getTrackedPath(file_handle, tracked_path, context.PathKey))
      context.ContextFlags |= ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY;

    ompfile::OmpFileIORequest req{};
    req.RequestId = context.RequestId;
    req.Op = ompfile::OmpFileIOOp::PREAD;
    req.ClientRank = client_rank;
    req.FileHandle = file_handle;
    req.Offset = static_cast<int64_t>(offset);
    req.Size = static_cast<uint64_t>(size);

    ompfile::OmpFileIOPlan plan{};
    if (schedRequest(req, nullptr, &plan)) {
      context.ContextFlags |= ompfile::OMPFILE_READ_CTX_HAS_PLAN;
      context.Plan = plan;
    }
  }

  void scheduleRead(int file_handle, long offset, size_t size) {
    ompfile::OmpFileReadRequestContext context{};
    buildReadContext(file_handle, offset, size, context);
  }

  void scheduleWrite(int file_handle, long offset, size_t size) {
    ompfile::OmpFileIORequest req{};
    req.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    req.Op = ompfile::OmpFileIOOp::PWRITE;
    req.ClientRank = client_rank;
    req.FileHandle = file_handle;
    req.Offset = static_cast<int64_t>(offset);
    req.Size = static_cast<uint64_t>(size);
    (void)schedRequest(req, nullptr, nullptr);
  }
};

static IOSchedulerTy getSchedulerType() {
  const char *env = std::getenv("LIBOMPFILE_SCHEDULER");
  if (!env) {
    return IOSchedulerTy::LOCAL;
  }

  std::string envStr(env);
  if (envStr == "LOCAL") {
    return IOSchedulerTy::LOCAL;
  }
  if (envStr == "HEADNODE") {
    return IOSchedulerTy::HEADNODE;
  }

  io_log("Unknown LIBOMPFILE_SCHEDULER '%s', defaulting to LOCAL\n", env);
  return IOSchedulerTy::LOCAL;
}

class OmpFileClientContext {

private:
  static OmpFileClientContext *instance;
  std::unique_ptr<IOBackend> io_backend;
  std::unique_ptr<IOScheduler> io_scheduler;
  std::atomic<int> io_resource_token;

  // RAII guard for IO resource token
  class IOResourceGuard {
    std::atomic<int> &token;
    std::chrono::milliseconds delay{1};
    static constexpr std::chrono::milliseconds max_delay{100};
  public:
    IOResourceGuard(std::atomic<int> &tok) : token(tok) {
      // Acquire a token with exponential backoff when none available
      while (true) {
        int current = token.load(std::memory_order_relaxed);
        if (current > 0) {
          if (token.compare_exchange_strong(current, current - 1,
                                            std::memory_order_acquire)) {
            break;
          }
        } else {
          std::this_thread::sleep_for(delay);
          // Exponential backoff capped at max_delay
          delay = std::min(delay * 2, max_delay);
        }
      }
    }

    ~IOResourceGuard() {
      token.fetch_add(1, std::memory_order_release);
    }

    // disable copying and moving
    IOResourceGuard(const IOResourceGuard &) = delete;
    IOResourceGuard &operator=(const IOResourceGuard &) = delete;
  };

public:
  OmpFileClientContext(IOBackendTy backend_type)
    : io_resource_token([](){
        const char* env = std::getenv("LIBOMPFILE_IO_TOKENS");
        if (env) {
          try {
            int val = std::stoi(env);
            if (val > 0) return val;
          } catch (...) {}
        }
        // default tokens
        return 4;
      }()) {
    io_log("IO resource slots set to %d\n", io_resource_token.load());

    switch (backend_type) {
    case IOBackendTy::MPI:
      io_log("MPI backend selected\n");
      io_backend = std::make_unique<MPIIOBackend>();
      break;
    case IOBackendTy::POSIX:
      io_log("POSIX backend selected\n");
      io_backend = std::make_unique<POSIXIOBackend>();
      break;
    case IOBackendTy::IO_URING:
      io_log("IO_URING selected\n");
      io_backend = std::make_unique<IoUringIOBackend>();
      break;
    case IOBackendTy::HDF5:
      io_log("HDF5 backend not implemented yet\n");
      break;
    }

    IOSchedulerTy scheduler_type = getSchedulerType();
    if (scheduler_type == IOSchedulerTy::HEADNODE) {
      io_log("HEADNODE scheduler selected (client mode)\n");
      io_scheduler = std::make_unique<HeadnodeScheduler>(*io_backend);
    } else {
      io_log("LOCAL scheduler selected (client mode)\n");
      io_scheduler = std::make_unique<LocalScheduler>(*io_backend);
    }

    io_log("OmpFileClientContext constructor called\n");
  }

  ~OmpFileClientContext() { io_log("Destroying OmpFileClientContext\n"); }

  static OmpFileClientContext &getInstance() {
    static std::once_flag init_once;
    std::call_once(init_once, []() {
      io_log("Creating new libompfile client instance\n");

      IOBackendTy backend = IOBackendTy::MPI; // Default
      const char *env = std::getenv("LIBOMPFILE_BACKEND");
      if (env) {
        std::string envStr(env);
        if (envStr == "MPI") {
          backend = IOBackendTy::MPI;
        } else if (envStr == "POSIX") {
          backend = IOBackendTy::POSIX;
        } else if (envStr == "IO_URING") {
          backend = IOBackendTy::IO_URING;
        } else if (envStr == "HDF5") {
          backend = IOBackendTy::HDF5;
        } else {
          io_log("Unknown LIBOMPFILE_BACKEND '%s', defaulting to MPI\n", env);
        }
      } else {
        io_log("LIBOMPFILE_BACKEND not set, defaulting to MPI\n");
      }

      instance = new OmpFileClientContext(backend);
      std::atexit(&OmpFileClientContext::finalize);
    });

    return *instance;
  }

  int openFile(const char *filename) {
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->open(filename);
  }

  int writeFile(int file_handle, const void *data, size_t size) {
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->write(file_handle, data, size);
  }

  int readFile(int file_handle, void *data, size_t size) {
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->read(file_handle, data, size);
  }

  int closeFile(int file_handle) {
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->close(file_handle);
  }

  int seekFile(int file_handle, long offset) {
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->seek(file_handle, offset);
  }

  int writeFileAt(int file_handle, const void *data, size_t size, long offset) {
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->writeAt(file_handle, offset, data, size);
  }

  int readFileAt(int file_handle, void *data, size_t size, long offset) {
    IOResourceGuard guard(io_resource_token);
    return io_scheduler->readAt(file_handle, offset, data, size);
  }

  int getFileHandle(int file_handle) { return file_handle; }

  static void finalize() {
    if (instance != nullptr) {
      io_log("Finalizing OmpFileClientContext\n");
      delete instance;
      instance = nullptr;
    }
  }
};

OmpFileClientContext *OmpFileClientContext::instance = nullptr;

extern "C" {

inline int acquire_async_not_supported(int async) {
  if (async) {
    io_log("Error: Asynchronous IO not supported yet\n");
    return -1;
  }
  return 0;
}

int omp_file_open(const char *filename) {
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.openFile(filename);
}

int omp_file_write(int file_handle, const void *data, size_t size, int async) {
  if (acquire_async_not_supported(async)) return -1;
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.writeFile(file_handle, data, size);
}

int omp_file_pwrite(int file_handle, long offset, const void *data, size_t size,
                    int async) {
  if (acquire_async_not_supported(async)) return -1;
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.writeFileAt(file_handle, data, size, offset);
}

int omp_file_pread(int file_handle, long offset, void *data, size_t size,
                   int async) {
  if (acquire_async_not_supported(async)) return -1;
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.readFileAt(file_handle, data, size, offset);
}

int omp_file_read(int file_handle, void *data, size_t size, int async) {
  if (acquire_async_not_supported(async)) return -1;
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.readFile(file_handle, data, size);
}

int omp_file_close(int file_handle) {
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.closeFile(file_handle);
}

int omp_file_seek(int file_handle, long offset) {
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.seekFile(file_handle, offset);
}

} // extern "C"
