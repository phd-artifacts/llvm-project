#ifndef OMPFILE_CLIENT_CONTEXT_H
#define OMPFILE_CLIENT_CONTEXT_H

#include "abstract_backend.h"
#include "async_write_engine.h"
#include "io_scheduler.h"
#include "ompfile_config.h"
#include "file_interface.h"
#include "ompfile_test_hooks.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class OmpFileClientContext {
  // The test placement hook (ompfile_test_hooks.h, test_hooks.cpp) reaches
  // the backend through the context's own queue and token discipline.
  friend int(::ompfile_test_pwrite_on_rank)(int, long, const void *, size_t,
                                           int, const omp_file_io_hint_v1 *);
  friend int(::ompfile_test_open_relative)(const char *, int, int *, int *);

private:
  static OmpFileClientContext *instance;
  // Every knob, read once when the context is created (ompfile_config.h).
  // Declared first: the initializers below read it.
  const ompfile::config::ClientConfig config_;
  std::unique_ptr<IOBackend> io_backend;
  std::unique_ptr<IOScheduler> io_scheduler;
  std::atomic<int> io_resource_token;
  std::atomic<uint64_t> api_call_id{1};
  AsyncWriteEngine async_engine;
  // Handles handed out by a successful open and not yet closed. The scheduler
  // owns handle lifetime, but flush has no scheduler round trip of its own, so
  // the client keeps this set to reject unknown/stale ids instead of reporting
  // "all writes completed" for a file that was never open.
  std::mutex open_handles_mutex;
  std::unordered_set<int> open_handles;

  // The wait a read or synchronous pwrite owes the async queue: the whole
  // queue (LIBOMPFILE_ASYNC_READ_WAIT=range went in Oct 2026; Phase E keys
  // the wait on task-graph edges instead).
  void waitForQueuedWrites();

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
  OmpFileClientContext();

  ~OmpFileClientContext();

  static OmpFileClientContext &getInstance();

  int openFile(const char *filename);

  int writeFile(int file_handle, const void *data, size_t size);

  int readFile(int file_handle, void *data, size_t size);

  int closeFile(int file_handle);

  int seekFile(int file_handle, long offset);

  bool isOpenHandle(int file_handle);

  // Completion boundary for async writes queued on one handle. This drains the
  // client-side queue only: it does not fsync, and under proxy write-back
  // staging the bytes may still sit in the proxy stage when it returns.
  int flushFile(int file_handle);

  // Same boundary as flushFile, narrowed to the writes tagged with `epoch`.
  // Same contract: client-side queue completion only, no fsync, no stage
  // flush. Writes issued without an epoch hint are not waited on here.
  int flushFileEpoch(int file_handle, uint64_t epoch);

  // Visibility boundary, as opposed to flushFile's queue boundary: when this
  // returns 0, everything written on this handle has reached the source
  // filesystem and another reader can observe it. It is deliberately the
  // expensive one - it drains the handle's queue *and* pushes the proxy stage
  // to the source - so it belongs at a phase boundary, not inside a write wave.
  int commitFile(int file_handle);

  int writeFileAt(int file_handle, const void *data, size_t size, long offset);

  int writeFileAtHint(int file_handle, const void *data, size_t size,
                      long offset, const omp_file_io_hint_v1 *hint);

  // Unified write entry for the C API. When async is requested and the worker
  // path is available, the payload is copied and queued so the caller can
  // pipeline the next issue; otherwise it runs synchronously after draining any
  // in-flight async writes (preserving program order and single-threaded
  // scheduler access). Returns 0/queued-accept or a negative error.
  int submitWrite(int file_handle, long offset, bool has_offset,
                  const void *data, size_t size,
                  const omp_file_io_hint_v1 *hint, bool async);

  // The owned form of submitWrite: no payload copy — the engine writes from
  // the caller's buffer and hands it to `release` once the write has run.
  // Ownership follows the return value and nothing else: zero means the
  // runtime has the buffer (released after the write, whatever its rc, which
  // then surfaces through flush/close like any queued write); non-zero means
  // nothing was queued, `release` was not called and the buffer is still the
  // caller's. Without an async worker the write runs inline under the same
  // rule: released on success, left with the caller on failure.
  int submitWriteOwned(int file_handle, long offset, void *data, size_t size,
                       void (*release)(void *),
                       const omp_file_io_hint_v1 *hint);

  int readFileAt(int file_handle, void *data, size_t size, long offset);

  int readFileAtHint(int file_handle, void *data, size_t size, long offset,
                     const omp_file_io_hint_v1 *hint);

  static void finalize();
};

#endif // OMPFILE_CLIENT_CONTEXT_H
