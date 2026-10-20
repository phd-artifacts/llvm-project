#include "client_context.h"
#include "debug_log.h"
#include "mpi.h"
#include "mpp_backend.h"
#include "ompfile_config.h"
#include "ompfile_sched.h"
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

enum class IOSchedulerTy {
  LOCAL,
  HEADNODE,
};

static void applyIoHint(ompfile::OmpFileIOHint &dst,
                        const omp_file_io_hint_v1 *src) {
  dst = {};
  if (!src)
    return;
  if (src->abi_version != OMPFILE_IO_HINT_ABI_VERSION)
    return;
  dst.AbiVersion = src->abi_version;
  dst.HintFlags = src->hint_flags;
  dst.EpochId = src->epoch_id;
  dst.StreamId = src->stream_id;
  dst.TileId = src->tile_id;
  dst.Role = src->role;
}

static int resolveDefaultIOTokens(const ompfile::config::ClientConfig &config) {
  if (config.IoTokensSet) {
    const char *env = config.IoTokensRaw.c_str();
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

static IOSchedulerTy getSchedulerType(
    const ompfile::config::ClientConfig &config) {
  if (config.Shared.SchedulerUnknown)
    io_log("Unknown LIBOMPFILE_SCHEDULER '%s', defaulting to HEADNODE\n",
           config.Shared.SchedulerRaw.c_str());
  return config.Shared.SchedulerHeadnode ? IOSchedulerTy::HEADNODE
                                         : IOSchedulerTy::LOCAL;
}

static std::mutex &getClientContextInitMutex() {
  static std::mutex init_mutex;
  return init_mutex;
}

OmpFileClientContext *OmpFileClientContext::instance = nullptr;

std::vector<int> OmpFileClientContext::sameFileHandles(int file_handle) {
  std::lock_guard<std::mutex> lock(open_handles_mutex);
  std::vector<int> out;
  auto it = handle_paths.find(file_handle);
  if (it == handle_paths.end()) {
    out.push_back(file_handle);
    return out;
  }
  for (const auto &entry : handle_paths)
    if (entry.second == it->second)
      out.push_back(entry.first);
  return out;
}

void OmpFileClientContext::waitForQueuedWrites(int file_handle, long offset,
                                               size_t size) {
  if (!async_engine.active())
    return;
  if (config_.AsyncReadWaitRange)
    async_engine.waitForRange(sameFileHandles(file_handle), offset, size);
  else
    async_engine.drain();
}

OmpFileClientContext::OmpFileClientContext()
    : config_(ompfile::config::ClientConfig::fromEnvironment()),
      io_resource_token(resolveDefaultIOTokens(config_)) {
  io_log("IO resource slots set to %d\n", io_resource_token.load());
  io_trace("OmpFileClientContext ctor enter this=%p tokens=%d\n",
           static_cast<void *>(this), io_resource_token.load());

  io_log("MPI backend selected\n");
  io_backend = std::make_unique<MppBackend>(config_);

  IOSchedulerTy scheduler_type = getSchedulerType(config_);
  if (scheduler_type == IOSchedulerTy::HEADNODE) {
    io_log("HEADNODE scheduler selected (client mode)\n");
    io_scheduler = createHeadnodeScheduler(*io_backend, config_);
  } else {
    io_log("LOCAL scheduler selected (client mode)\n");
    io_scheduler = createLocalScheduler(*io_backend);
  }

  // Wire the async engine to drain queued writes through the same scheduler
  // path (and token semaphore) a synchronous write would take.
  async_engine.configure([this](const AsyncWriteEngine::Task &task) -> int {
    IOResourceGuard guard(io_resource_token);
    const void *bytes = task.payload();
    const size_t size = task.payloadSize();
    if (task.has_offset) {
      if (task.has_hint)
        return io_scheduler->writeAtHint(task.handle, task.offset, bytes,
                                         size, &task.hint);
      return io_scheduler->writeAt(task.handle, task.offset, bytes, size);
    }
    return io_scheduler->write(task.handle, bytes, size);
  }, config_.AsyncQueueDepth, config_.OptStats.Value);

  io_log("OmpFileClientContext constructor called\n");
  io_trace("OmpFileClientContext ctor done this=%p io_backend=%p "
           "io_scheduler=%p\n",
           static_cast<void *>(this), static_cast<void *>(io_backend.get()),
           static_cast<void *>(io_scheduler.get()));
}

OmpFileClientContext::~OmpFileClientContext() {
  io_trace("OmpFileClientContext dtor this=%p io_backend=%p io_scheduler=%p\n",
           static_cast<void *>(this), static_cast<void *>(io_backend.get()),
           static_cast<void *>(io_scheduler.get()));
  // Queue must already be drained by the last close/read; join the idle
  // worker before tearing down the scheduler it calls into.
  async_engine.shutdown();
  io_log("Destroying OmpFileClientContext\n");
}

OmpFileClientContext &OmpFileClientContext::getInstance() {
  io_trace("OmpFileClientContext::getInstance enter instance=%p\n",
           static_cast<void *>(instance));
  const std::lock_guard<std::mutex> lock(getClientContextInitMutex());
  if (instance == nullptr) {
    io_log("Creating new libompfile client instance\n");

    instance = new OmpFileClientContext();
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

int OmpFileClientContext::openFile(const char *filename) {
  const uint64_t call_id =
      api_call_id.fetch_add(1, std::memory_order_relaxed);
  io_trace("ctx=%p call=%llu openFile enter filename=%s tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id),
           filename ? filename : "(null)", io_resource_token.load());
  if (async_engine.active())
    async_engine.drain();
  IOResourceGuard guard(io_resource_token);
  const int rc = io_scheduler->open(filename);
  if (rc >= 0) {
    {
      std::lock_guard<std::mutex> lock(open_handles_mutex);
      open_handles.insert(rc);
      handle_paths[rc] = filename ? filename : "";
    }
    // A recycled file id must not inherit the previous owner's async rc.
    async_engine.forgetHandle(rc);
  }
  io_trace("ctx=%p call=%llu openFile exit rc=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), rc,
           io_resource_token.load());
  return rc;
}

int OmpFileClientContext::writeFile(int file_handle, const void *data,
                                    size_t size) {
  IOResourceGuard guard(io_resource_token);
  return io_scheduler->write(file_handle, data, size);
}

int OmpFileClientContext::readFile(int file_handle, void *data, size_t size) {
  if (async_engine.active())
    async_engine.drain();
  IOResourceGuard guard(io_resource_token);
  return io_scheduler->read(file_handle, data, size);
}

int OmpFileClientContext::closeFile(int file_handle) {
  const uint64_t call_id =
      api_call_id.fetch_add(1, std::memory_order_relaxed);
  io_trace("ctx=%p call=%llu closeFile enter file_handle=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), file_handle,
           io_resource_token.load());
  // Every queued async write must physically complete before the file is
  // closed; this is the pipeline's join point. Draining all (not just this
  // handle) also keeps scheduler/headnode state single-threaded. Always drain
  // (even if the queue already emptied) so a sticky async-write failure is
  // surfaced here rather than silently swallowed.
  const int async_rc = async_engine.drain();
  if (async_rc != 0)
    io_log("closeFile: a queued async write failed (rc=%d) before close of "
           "handle %d\n",
           async_rc, file_handle);
  IOResourceGuard guard(io_resource_token);
  const int rc = io_scheduler->close(file_handle);
  if (rc == 0) {
    {
      std::lock_guard<std::mutex> lock(open_handles_mutex);
      open_handles.erase(file_handle);
      handle_paths.erase(file_handle);
    }
    async_engine.forgetHandle(file_handle);
  }
  io_trace("ctx=%p call=%llu closeFile exit rc=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), rc,
           io_resource_token.load());
  if (rc == 0 && async_rc != 0)
    return -1; // surface the async write failure the close would otherwise hide
  return rc;
}

int OmpFileClientContext::seekFile(int file_handle, long offset) {
  if (async_engine.active())
    async_engine.drain();
  IOResourceGuard guard(io_resource_token);
  return io_scheduler->seek(file_handle, offset);
}

bool OmpFileClientContext::isOpenHandle(int file_handle) {
  if (file_handle < 0)
    return false;
  std::lock_guard<std::mutex> lock(open_handles_mutex);
  return open_handles.count(file_handle) != 0;
}

int OmpFileClientContext::flushFile(int file_handle) {
  const uint64_t call_id =
      api_call_id.fetch_add(1, std::memory_order_relaxed);
  io_trace("ctx=%p call=%llu flushFile enter file_handle=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), file_handle,
           io_resource_token.load());
  if (!isOpenHandle(file_handle)) {
    io_log("flushFile: invalid file handle %d\n", file_handle);
    io_trace("ctx=%p call=%llu flushFile exit rc=-1 reason=bad-handle\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id));
    errno = EBADF;
    return -1;
  }
  const int rc = async_engine.flushHandle(file_handle);
  if (rc != 0)
    io_log("flushFile: a queued async write failed (rc=%d) for handle %d\n",
           rc, file_handle);
  io_trace("ctx=%p call=%llu flushFile exit rc=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), rc,
           io_resource_token.load());
  return rc;
}

int OmpFileClientContext::flushFileEpoch(int file_handle, uint64_t epoch) {
  const uint64_t call_id =
      api_call_id.fetch_add(1, std::memory_order_relaxed);
  io_trace("ctx=%p call=%llu flushFileEpoch enter file_handle=%d epoch=%llu "
           "tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), file_handle,
           static_cast<unsigned long long>(epoch),
           io_resource_token.load());
  if (!isOpenHandle(file_handle)) {
    io_log("flushFileEpoch: invalid file handle %d\n", file_handle);
    io_trace("ctx=%p call=%llu flushFileEpoch exit rc=-1 reason=bad-handle\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id));
    errno = EBADF;
    return -1;
  }
  const int rc = async_engine.flushEpoch(file_handle, epoch);
  if (rc != 0)
    io_log("flushFileEpoch: a queued async write failed (rc=%d) for handle "
           "%d epoch %llu\n",
           rc, file_handle, static_cast<unsigned long long>(epoch));
  io_trace("ctx=%p call=%llu flushFileEpoch exit rc=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), rc,
           io_resource_token.load());
  return rc;
}

int OmpFileClientContext::commitFile(int file_handle) {
  const uint64_t call_id =
      api_call_id.fetch_add(1, std::memory_order_relaxed);
  io_trace("ctx=%p call=%llu commitFile enter file_handle=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), file_handle,
           io_resource_token.load());
  if (!isOpenHandle(file_handle)) {
    io_log("commitFile: invalid file handle %d\n", file_handle);
    io_trace("ctx=%p call=%llu commitFile exit rc=-1 reason=bad-handle\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id));
    errno = EBADF;
    return -1;
  }
  // Queued async writes are part of what the caller means by "what I wrote",
  // so they have to land before the stage is pushed to the source. A failure
  // here means the data is incomplete, and committing it would be a lie.
  const int flush_rc = async_engine.flushHandle(file_handle);
  if (flush_rc != 0) {
    io_log("commitFile: a queued async write failed (rc=%d) for handle %d\n",
           flush_rc, file_handle);
    io_trace("ctx=%p call=%llu commitFile exit rc=%d reason=queued-write\n",
             static_cast<void *>(this),
             static_cast<unsigned long long>(call_id), flush_rc);
    return flush_rc;
  }
  IOResourceGuard guard(io_resource_token);
  const int rc = io_scheduler->commit(file_handle);
  io_trace("ctx=%p call=%llu commitFile exit rc=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), rc,
           io_resource_token.load());
  return rc;
}

int OmpFileClientContext::writeFileAt(int file_handle, const void *data,
                                      size_t size, long offset) {
  IOResourceGuard guard(io_resource_token);
  return io_scheduler->writeAt(file_handle, offset, data, size);
}

int OmpFileClientContext::writeFileAtHint(int file_handle, const void *data,
                                          size_t size, long offset,
                                          const omp_file_io_hint_v1 *hint) {
  ompfile::OmpFileIOHint internal_hint{};
  applyIoHint(internal_hint, hint);
  IOResourceGuard guard(io_resource_token);
  return io_scheduler->writeAtHint(file_handle, offset, data, size,
                                   hint ? &internal_hint : nullptr);
}

int OmpFileClientContext::submitWrite(int file_handle, long offset,
                                      bool has_offset, const void *data,
                                      size_t size,
                                      const omp_file_io_hint_v1 *hint,
                                      bool async) {
  if (async && async_engine.available()) {
    AsyncWriteEngine::Task task;
    task.handle = file_handle;
    task.offset = offset;
    task.has_offset = has_offset;
    if (hint) {
      task.has_hint = true;
      applyIoHint(task.hint, hint);
    }
    const char *bytes = static_cast<const char *>(data);
    task.data.assign(bytes, bytes + size);
    return async_engine.enqueue(std::move(task));
  }
  if (has_offset) {
    // A synchronous pwrite behind queued async ones must land after any of
    // them that cover its range; the rest may keep draining.
    waitForQueuedWrites(file_handle, offset, size);
    if (hint)
      return writeFileAtHint(file_handle, data, size, offset, hint);
    return writeFileAt(file_handle, data, size, offset);
  }
  if (async_engine.active())
    async_engine.drain();
  return writeFile(file_handle, data, size);
}

int OmpFileClientContext::submitWriteOwned(int file_handle, long offset,
                                           void *data, size_t size,
                                           void (*release)(void *),
                                           const omp_file_io_hint_v1 *hint) {
  assert(release && "submitWriteOwned expects a resolved release callback");
  if (!isOpenHandle(file_handle)) {
    io_log("submitWriteOwned: invalid file handle %d\n", file_handle);
    errno = EBADF;
    return -1;
  }
  if (async_engine.available()) {
    AsyncWriteEngine::Task task;
    task.handle = file_handle;
    task.offset = offset;
    task.has_offset = true;
    if (hint) {
      task.has_hint = true;
      applyIoHint(task.hint, hint);
    }
    task.owned = data;
    task.owned_size = size;
    task.release = release;
    // A refused task is never moved into the queue, so `data` is untouched
    // and stays the caller's, as the contract above promises.
    return async_engine.enqueue(std::move(task));
  }
  if (async_engine.active())
    async_engine.drain();
  const int rc = hint ? writeFileAtHint(file_handle, data, size, offset, hint)
                      : writeFileAt(file_handle, data, size, offset);
  if (rc == 0)
    release(data);
  return rc;
}

int OmpFileClientContext::readFileAt(int file_handle, void *data, size_t size,
                                     long offset) {
  const uint64_t call_id =
      api_call_id.fetch_add(1, std::memory_order_relaxed);
  io_trace("ctx=%p call=%llu readFileAt enter file_handle=%d offset=%ld "
           "size=%zu tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), file_handle, offset,
           size, io_resource_token.load());
  waitForQueuedWrites(file_handle, offset, size);
  IOResourceGuard guard(io_resource_token);
  const int rc = io_scheduler->readAt(file_handle, offset, data, size);
  io_trace("ctx=%p call=%llu readFileAt exit rc=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), rc,
           io_resource_token.load());
  return rc;
}

int OmpFileClientContext::readFileAtHint(int file_handle, void *data,
                                         size_t size, long offset,
                                         const omp_file_io_hint_v1 *hint) {
  const uint64_t call_id =
      api_call_id.fetch_add(1, std::memory_order_relaxed);
  io_trace("ctx=%p call=%llu readFileAtHint enter file_handle=%d offset=%ld "
           "size=%zu tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), file_handle, offset,
           size, io_resource_token.load());
  ompfile::OmpFileIOHint internal_hint{};
  applyIoHint(internal_hint, hint);
  waitForQueuedWrites(file_handle, offset, size);
  IOResourceGuard guard(io_resource_token);
  const int rc = io_scheduler->readAtHint(file_handle, offset, data, size,
                                          hint ? &internal_hint : nullptr);
  io_trace("ctx=%p call=%llu readFileAtHint exit rc=%d tokens=%d\n",
           static_cast<void *>(this),
           static_cast<unsigned long long>(call_id), rc,
           io_resource_token.load());
  return rc;
}

void OmpFileClientContext::finalize() {
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
