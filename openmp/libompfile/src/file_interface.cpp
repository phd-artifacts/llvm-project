#include "abstract_backend.h"
#include "debug_log.h"
#include "mpp_shim.h"
#include "ompfile_sched.h"
#include "mpi.h"
#include "mpi_io_backend.h"
#include "posix_backend.h"
#include "io_uring_io_backend.h"
#include <algorithm>
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

static const char *schedulerOpToString(ompfile::OmpFileIOOp op) {
  switch (op) {
  case ompfile::OmpFileIOOp::OPEN:
    return "OPEN";
  case ompfile::OmpFileIOOp::CLOSE:
    return "CLOSE";
  case ompfile::OmpFileIOOp::PREAD:
    return "PREAD";
  case ompfile::OmpFileIOOp::PWRITE:
    return "PWRITE";
  case ompfile::OmpFileIOOp::PREFETCH:
    return "PREFETCH";
  }
  return "UNKNOWN";
}

static const char *backendTypeToString(IOBackendTy backend_type) {
  switch (backend_type) {
  case IOBackendTy::MPI:
    return "MPI";
  case IOBackendTy::POSIX:
    return "POSIX";
  case IOBackendTy::IO_URING:
    return "IO_URING";
  case IOBackendTy::HDF5:
    return "HDF5";
  }
  return "UNKNOWN";
}

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
  std::mutex sched_request_mutex;
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
    io_trace("HeadnodeScheduler::schedRequest enter scheduler=%p req_id=%llu "
             "op=%s client_rank=%d file=%d offset=%lld size=%llu path_size=%u "
             "path=%s\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(req.RequestId),
             schedulerOpToString(req.Op), req.ClientRank, req.FileHandle,
             static_cast<long long>(req.Offset),
             static_cast<unsigned long long>(req.Size), req.PathSize,
             path ? path : "(null)");
    {
      const std::lock_guard<std::mutex> lock(sched_request_mutex);
      if (!ompfile::mpp::schedRequest(req, path, plan)) {
        io_log("HEADNODE scheduler request failed (global manager unavailable); "
               "falling back to LOCAL.\n");
        io_trace("HeadnodeScheduler::schedRequest mpp call failed req_id=%llu\n",
                 static_cast<unsigned long long>(req.RequestId));
        return false;
      }
    }

    if (plan.Status != 0) {
      errno = plan.Errno;
      io_trace("HeadnodeScheduler::schedRequest plan error req_id=%llu "
               "status=%d errno=%d\n",
               static_cast<unsigned long long>(req.RequestId), plan.Status,
               plan.Errno);
      return false;
    }

    if (plan_out)
      *plan_out = plan;
    io_trace("HeadnodeScheduler::schedRequest success req_id=%llu "
             "aggregator_rank=%d remote_handle=%d plan_flags=0x%x\n",
             static_cast<unsigned long long>(req.RequestId),
             plan.AggregatorRank, plan.RemoteHandle, plan.PlanFlags);
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
    io_trace("HeadnodeScheduler::scheduleOpen scheduler=%p req_id=%llu path=%s\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(req.RequestId), path);
    (void)schedRequest(req, path, nullptr);
  }

  void scheduleClose(int file_handle) {
    ompfile::OmpFileIORequest req{};
    req.RequestId = request_id.fetch_add(1, std::memory_order_relaxed);
    req.Op = ompfile::OmpFileIOOp::CLOSE;
    req.ClientRank = client_rank;
    req.FileHandle = file_handle;
    io_trace("HeadnodeScheduler::scheduleClose scheduler=%p req_id=%llu "
             "file=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(req.RequestId), file_handle);
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
    io_trace("HeadnodeScheduler::buildReadContext scheduler=%p req_id=%llu "
             "file=%d offset=%lld size=%llu has_path_key=%d path_key=%llu\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(context.RequestId), file_handle,
             static_cast<long long>(offset),
             static_cast<unsigned long long>(size),
             (context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY) !=
                 0,
             static_cast<unsigned long long>(context.PathKey));

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

static int resolveDefaultIOTokens() {
  const char *env = std::getenv("LIBOMPFILE_IO_TOKENS");
  if (env) {
    try {
      const int parsed = std::stoi(env);
      if (parsed > 0)
        return parsed;
    } catch (...) {
    }
    io_log("Invalid LIBOMPFILE_IO_TOKENS='%s'; using runtime default.\n", env);
  }

  int tokens = 4;
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized)
    return tokens;

  int thread_level = MPI_THREAD_SINGLE;
  MPI_Query_thread(&thread_level);
  if (thread_level < MPI_THREAD_MULTIPLE) {
    tokens = 1;
    io_log("MPI thread level=%d; defaulting LIBOMPFILE_IO_TOKENS to %d for "
           "safe scheduler serialization (override with LIBOMPFILE_IO_TOKENS).\n",
           thread_level, tokens);
  }

  return tokens;
}

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

static std::mutex &getClientContextInitMutex() {
  static std::mutex init_mutex;
  return init_mutex;
}

class OmpFileClientContext {

private:
  static OmpFileClientContext *instance;
  std::unique_ptr<IOBackend> io_backend;
  std::unique_ptr<IOScheduler> io_scheduler;
  std::atomic<int> io_resource_token;
  std::atomic<uint64_t> api_call_id{1};

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
      : io_resource_token(resolveDefaultIOTokens()) {
    io_log("IO resource slots set to %d\n", io_resource_token.load());
    io_trace("OmpFileClientContext ctor enter this=%p backend=%s tokens=%d\n",
             static_cast<void *>(this),
             backendTypeToString(backend_type), io_resource_token.load());

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
    io_trace("OmpFileClientContext ctor done this=%p io_backend=%p "
             "io_scheduler=%p\n",
             static_cast<void *>(this), static_cast<void *>(io_backend.get()),
             static_cast<void *>(io_scheduler.get()));
  }

  ~OmpFileClientContext() {
    io_trace("OmpFileClientContext dtor this=%p io_backend=%p io_scheduler=%p\n",
             static_cast<void *>(this), static_cast<void *>(io_backend.get()),
             static_cast<void *>(io_scheduler.get()));
    io_log("Destroying OmpFileClientContext\n");
  }

  static OmpFileClientContext &getInstance() {
    io_trace("OmpFileClientContext::getInstance enter instance=%p\n",
             static_cast<void *>(instance));
    const std::lock_guard<std::mutex> lock(getClientContextInitMutex());
    if (instance == nullptr) {
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
      io_trace_symbol_owner(
          "OmpFileClientContext::getInstance",
          reinterpret_cast<const void *>(&OmpFileClientContext::getInstance));
      io_trace("OmpFileClientContext::getInstance created instance=%p\n",
               static_cast<void *>(instance));
      std::atexit(&OmpFileClientContext::finalize);
    } else {
      io_trace("OmpFileClientContext::getInstance reuse instance=%p\n",
               static_cast<void *>(instance));
    }

    return *instance;
  }

  int openFile(const char *filename) {
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu openFile enter filename=%s tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id),
             filename ? filename : "(null)", io_resource_token.load());
    IOResourceGuard guard(io_resource_token);
    const int rc = io_scheduler->open(filename);
    io_trace("ctx=%p call=%llu openFile exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    return rc;
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
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu closeFile enter file_handle=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), file_handle,
             io_resource_token.load());
    IOResourceGuard guard(io_resource_token);
    const int rc = io_scheduler->close(file_handle);
    io_trace("ctx=%p call=%llu closeFile exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    return rc;
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
    const uint64_t call_id =
        api_call_id.fetch_add(1, std::memory_order_relaxed);
    io_trace("ctx=%p call=%llu readFileAt enter file_handle=%d offset=%ld "
             "size=%zu tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), file_handle, offset,
             size, io_resource_token.load());
    IOResourceGuard guard(io_resource_token);
    const int rc = io_scheduler->readAt(file_handle, offset, data, size);
    io_trace("ctx=%p call=%llu readFileAt exit rc=%d tokens=%d\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), rc,
             io_resource_token.load());
    return rc;
  }

  int getFileHandle(int file_handle) { return file_handle; }

  static void finalize() {
    io_trace("OmpFileClientContext::finalize enter instance=%p\n",
             static_cast<void *>(instance));
    if (instance != nullptr) {
      io_log("Finalizing OmpFileClientContext\n");
      delete instance;
      instance = nullptr;
    }
    io_trace("OmpFileClientContext::finalize exit instance=%p\n",
             static_cast<void *>(instance));
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
  io_trace("omp_file_open api enter filename=%s\n",
           filename ? filename : "(null)");
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.openFile(filename);
  io_trace("omp_file_open api exit rc=%d\n", rc);
  return rc;
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
  io_trace("omp_file_pread api enter file_handle=%d offset=%ld size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.readFileAt(file_handle, data, size, offset);
  io_trace("omp_file_pread api exit rc=%d\n", rc);
  return rc;
}

int omp_file_read(int file_handle, void *data, size_t size, int async) {
  if (acquire_async_not_supported(async)) return -1;
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.readFile(file_handle, data, size);
}

int omp_file_close(int file_handle) {
  io_trace("omp_file_close api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.closeFile(file_handle);
  io_trace("omp_file_close api exit rc=%d\n", rc);
  return rc;
}

int omp_file_seek(int file_handle, long offset) {
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.seekFile(file_handle, offset);
}

} // extern "C"
