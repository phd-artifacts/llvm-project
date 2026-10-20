#ifndef OMPFILE_ASYNC_WRITE_ENGINE_H
#define OMPFILE_ASYNC_WRITE_ENGINE_H

#include "debug_log.h"
#include "mpi.h"
#include "ompfile_sched.h"
#include "ompfile_trace.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Background write engine that turns async-issued writes into a producer/
// consumer pipeline: the issuing (target-region) thread copies the payload,
// enqueues it, and returns immediately, so it can dispatch the next write while
// a single worker thread drains the previous one through the normal scheduler
// path. This attacks overhead #3 ("no pipelining") — issue of write N+1 now
// overlaps the drain of write N instead of stalling behind it.
//
// Safety: the proxy initializes MPI at MPI_THREAD_MULTIPLE, so the worker may
// call MPI concurrently. Even so, we keep exactly one worker and never let the
// issuing thread touch the scheduler while writes are queued: it only enqueues
// during a write wave and drains the engine at every sync point (open/close/
// read/seek). That keeps scheduler/headnode state single-threaded in practice
// while still overlapping issue with drain. If the runtime is ever initialized
// below MPI_THREAD_SERIALIZED, async falls back to synchronous execution.
class AsyncWriteEngine {
public:
  struct Task {
    int handle = -1;
    long offset = 0;
    bool has_offset = false; // pwrite when true, sequential write when false
    bool has_hint = false;
    ompfile::OmpFileIOHint hint{};
    // The payload, one of two ways. `data` is the copy `enqueue` takes on the
    // omp_file_pwrite path. `owned` is the caller's own buffer on the
    // omp_file_pwrite_owned path: written from in place, then handed to
    // `release` once the write has run, so the issuing thread never pays for
    // a memcpy of the payload. Exactly one of the two carries the bytes.
    std::vector<char> data;
    void *owned = nullptr;
    size_t owned_size = 0;
    void (*release)(void *) = nullptr;

    const void *payload() const { return owned ? owned : data.data(); }
    size_t payloadSize() const { return owned ? owned_size : data.size(); }

    // Hands an owned buffer back. Called by the worker after the executor
    // returns (success or failure alike), never under the engine mutex: the
    // callback is the application's. Not called from the destructor, because
    // a task that was refused by `enqueue` is still the caller's buffer.
    void releaseOwned() {
      if (!owned)
        return;
      assert(release && "owned task without a release callback");
      release(owned);
      owned = nullptr;
      owned_size = 0;
    }
  };

  // Executes one queued task synchronously; returns the backend rc.
  using Executor = std::function<int(const Task &)>;

  // A completion unit the caller can wait on: one handle plus one epoch id
  // taken from the write's hint. Writes issued without an epoch belong to the
  // handle alone and are waited on by flushHandle, never by flushEpoch.
  using EpochKey = std::pair<int, uint64_t>;

  static bool taskHasEpoch(const Task &task) {
    return task.has_hint &&
           (task.hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_EPOCH) != 0;
  }

  static EpochKey taskEpochKey(const Task &task) {
    return EpochKey{task.handle, task.hint.EpochId};
  }

  AsyncWriteEngine() = default;
  ~AsyncWriteEngine() { shutdown(); }

  AsyncWriteEngine(const AsyncWriteEngine &) = delete;
  AsyncWriteEngine &operator=(const AsyncWriteEngine &) = delete;

  void configure(Executor exec, size_t max_depth, bool report_stats) {
    executor_ = std::move(exec);
    max_depth_ = max_depth;
    report_stats_ = report_stats;
  }

  // Lazily decides (once) whether the async worker path is usable. When it
  // returns false the caller must run the write synchronously.
  bool available() {
    std::call_once(available_once_, [this] { enabled_ = resolveEnabled(); });
    return enabled_;
  }

  // Copies and queues a write. Returns 0 on accept, or the rc of an earlier
  // failure *within this write's own completion unit* — its epoch when the
  // write carries one, otherwise its handle — so the caller can stop early
  // without an unrelated branch's failure stopping it.
  int enqueue(Task &&task) {
    std::unique_lock<std::mutex> lock(mtx_);
    // Refuse only writes whose own completion unit already failed: an epoched
    // write is blocked by its epoch, an un-epoched one by its handle (that is
    // the scope it can be waited on at). A failure on an unrelated branch must
    // not stop this one; `drain()` at close still reports engine-wide.
    const bool has_epoch = taskHasEpoch(task);
    if (has_epoch) {
      auto it = epoch_error_.find(taskEpochKey(task));
      if (it != epoch_error_.end() && it->second != 0)
        return it->second;
    } else {
      auto it = handle_error_.find(task.handle);
      if (it != handle_error_.end() && it->second != 0)
        return it->second;
    }
    startWorkerLocked();
    if (queue_.size() >= max_depth_ && !stop_) {
      // Back-pressure: the issuing thread blocks here when the bounded queue
      // is full, which is the other place (besides drain) where async issue
      // serializes against the drain; report it so a lane can see which.
      ++enqueue_waits_;
      const auto start = std::chrono::steady_clock::now();
      not_full_.wait(lock,
                     [&] { return queue_.size() < max_depth_ || stop_; });
      enqueue_wait_ns_ += static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count());
    }
    if (stop_)
      return -1;
    pending_[task.handle]++;
    if (has_epoch)
      pending_epoch_[taskEpochKey(task)]++;
    if (task.owned) {
      ++owned_enqueued_;
      owned_bytes_ += task.owned_size;
    }
    queue_.push_back(std::move(task));
    ++enqueued_;
    max_queue_depth_ = std::max(max_queue_depth_, queue_.size());
    has_work_.notify_one();
    return 0;
  }

  bool active() {
    std::lock_guard<std::mutex> lock(mtx_);
    return worker_started_ && (!queue_.empty() || busy_);
  }

  // Waits until every queued write for `handle` has drained, then reports
  // only that handle's own first failing rc. A failure queued on an unrelated
  // handle must not be charged here; `drain()` (the close/read/seek join point)
  // still reports the engine-wide sticky rc.
  int flushHandle(int handle) {
    assert(handle >= 0 && "flushHandle expects a caller-validated handle");
    std::unique_lock<std::mutex> lock(mtx_);
    if (!worker_started_)
      return 0; // nothing was ever queued, so nothing can be outstanding
    handle_done_.wait(lock, [&] {
      auto it = pending_.find(handle);
      return it == pending_.end() || it->second == 0;
    });
    auto err = handle_error_.find(handle);
    return err == handle_error_.end() ? 0 : err->second;
  }

  // Waits until every queued write tagged with `epoch` on `handle` has
  // drained, then reports only that epoch's own first failing rc. Writes
  // issued without an epoch hint are not waited on here — they have no epoch
  // to belong to, so flushHandle is their boundary.
  int flushEpoch(int handle, uint64_t epoch) {
    assert(handle >= 0 && "flushEpoch expects a caller-validated handle");
    const EpochKey key{handle, epoch};
    std::unique_lock<std::mutex> lock(mtx_);
    if (!worker_started_)
      return 0; // nothing was ever queued, so nothing can be outstanding
    handle_done_.wait(lock, [&] {
      auto it = pending_epoch_.find(key);
      return it == pending_epoch_.end() || it->second == 0;
    });
    auto err = epoch_error_.find(key);
    return err == epoch_error_.end() ? 0 : err->second;
  }

  // Drops the failure records for a handle and every epoch under it. Called
  // when a handle is closed so a recycled file id does not inherit the
  // previous owner's errors.
  void forgetHandle(int handle) {
    std::lock_guard<std::mutex> lock(mtx_);
    handle_error_.erase(handle);
    eraseHandleRange(pending_epoch_, handle);
    eraseHandleRange(epoch_error_, handle);
  }

  // Range-scoped alternative to drain() for a read or synchronous write of
  // [offset, offset+size) on a file that `handles` are open on: waits until
  // no queued or in-flight task on one of those handles could cover that
  // range, and returns at once when none can. A sequential (no-offset) write
  // has no known range and always counts. This is the wait a dependency
  // actually needs — a read of bytes nobody has queued a write to proceeds
  // while the worker drains unrelated ranges of the same file — and the
  // engine keeps FIFO order among the writes themselves, so two queued writes
  // to one range still land in issue order. Opt-in through
  // LIBOMPFILE_ASYNC_READ_WAIT=range; the caller passes every handle open on
  // the read's path, since a file open twice is still one file.
  void waitForRange(const std::vector<int> &handles, long offset,
                    size_t size) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!worker_started_)
      return;
    const auto covers = [&](int handle, bool has_offset, long task_offset,
                            size_t task_size) {
      if (std::find(handles.begin(), handles.end(), handle) == handles.end())
        return false;
      if (!has_offset)
        return true; // cursor write: range unknown, be conservative
      const long lo = task_offset;
      const long hi = task_offset + static_cast<long>(task_size);
      return lo < offset + static_cast<long>(size) && offset < hi;
    };
    const auto pending = [&] {
      if (current_.active &&
          covers(current_.handle, current_.has_offset, current_.offset,
                 current_.size))
        return true;
      for (const Task &t : queue_)
        if (covers(t.handle, t.has_offset, t.offset, t.payloadSize()))
          return true;
      return false;
    };
    if (!pending()) {
      // Counted only when something was outstanding: these are the reads a
      // drain would have stalled and this wait let through.
      if (current_.active || !queue_.empty())
        ++range_wait_free_;
      return;
    }
    ++range_waits_;
    const auto start = std::chrono::steady_clock::now();
    handle_done_.wait(lock, [&] { return !pending(); });
    range_wait_ns_ += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  }

  // Waits until the whole queue has drained and the worker is idle.
  int drain() {
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mtx_);
    if (!worker_started_)
      return sticky_error_;
    ++drain_calls_;
    idle_.wait(lock, [&] { return queue_.empty() && !busy_; });
    drain_wait_ns_ += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    return sticky_error_;
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stop_ = true;
      has_work_.notify_all();
      not_full_.notify_all();
    }
    if (worker_.joinable())
      worker_.join();
    // Under LIBOMPFILE_OPT_STATS, so a benchmark
    // lane can assert from its log that the owned (zero-copy) submit engaged.
    // Once only: shutdown runs from the client context's destructor and again
    // from this engine's own, and a lane sums the token over the log.
    if (stats_reported_)
      return;
    stats_reported_ = true;
    if (report_stats_)
      io_report("Async IO stats: enqueued=%llu completed=%llu "
             "max_queue_depth=%zu drain_calls=%llu drain_wait_ns=%llu "
             "owned_enqueued=%llu owned_bytes=%llu range_waits=%llu "
             "range_wait_ns=%llu range_wait_free=%llu enqueue_waits=%llu "
             "enqueue_wait_ns=%llu\n",
             static_cast<unsigned long long>(enqueued_),
             static_cast<unsigned long long>(completed_), max_queue_depth_,
             static_cast<unsigned long long>(drain_calls_),
             static_cast<unsigned long long>(drain_wait_ns_),
             static_cast<unsigned long long>(owned_enqueued_),
             static_cast<unsigned long long>(owned_bytes_),
             static_cast<unsigned long long>(range_waits_),
             static_cast<unsigned long long>(range_wait_ns_),
             static_cast<unsigned long long>(range_wait_free_),
             static_cast<unsigned long long>(enqueue_waits_),
             static_cast<unsigned long long>(enqueue_wait_ns_));
  }

private:
  static bool resolveEnabled() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
      return true; // MPI not initialized (unit harness): no concurrent MPI caller to protect
    int level = MPI_THREAD_SINGLE;
    MPI_Query_thread(&level);
    if (level >= MPI_THREAD_SERIALIZED)
      return true;
    io_log("Async IO worker disabled: MPI thread level below SERIALIZED; "
           "async writes run synchronously.\n");
    return false;
  }

  // Erases every entry belonging to `handle` from a (handle, epoch) map.
  template <typename MapTy> static void eraseHandleRange(MapTy &m, int handle) {
    m.erase(m.lower_bound(EpochKey{handle, 0}),
            m.upper_bound(EpochKey{handle, UINT64_MAX}));
  }

  void startWorkerLocked() {
    if (worker_started_)
      return;
    worker_started_ = true;
    worker_ = std::thread([this] {
      ompfile::trace::nameThisThread("ompfile-async-worker");
      workerLoop();
    });
  }

  void workerLoop() {
    std::unique_lock<std::mutex> lock(mtx_);
    for (;;) {
      has_work_.wait(lock, [&] { return !queue_.empty() || stop_; });
      if (queue_.empty()) {
        if (stop_)
          return;
        continue;
      }
      Task task = std::move(queue_.front());
      queue_.pop_front();
      busy_ = true;
      // Keep the in-flight task's range visible to waitForRange while it is
      // no longer in the queue.
      current_ = {true, task.handle, task.has_offset, task.offset,
                  task.payloadSize()};
      not_full_.notify_one();
      lock.unlock();
      int rc = -1;
      {
        // The queued write actually running: its width against the issuing
        // thread's short omp_file_pwrite range is the async overlap.
        ompfile::trace::Scope Trace(ompfile::trace::Domain::LibOmpFile,
                                    "async-write",
                                    ompfile::trace::Color::Io,
                                    task.payloadSize());
        rc = executor_ ? executor_(task) : -1;
      }
      // The write has run, so the caller's buffer goes back now — before the
      // completion is published, so a flush that returns cannot race a
      // release still in flight.
      task.releaseOwned();
      lock.lock();
      busy_ = false;
      current_.active = false;
      auto it = pending_.find(task.handle);
      if (it != pending_.end() && --it->second == 0)
        pending_.erase(it);
      const bool has_epoch = taskHasEpoch(task);
      if (has_epoch) {
        auto ep = pending_epoch_.find(taskEpochKey(task));
        if (ep != pending_epoch_.end() && --ep->second == 0)
          pending_epoch_.erase(ep);
      }
      if (rc != 0) {
        if (sticky_error_ == 0)
          sticky_error_ = rc;
        int &handle_rc = handle_error_[task.handle];
        if (handle_rc == 0)
          handle_rc = rc;
        if (has_epoch) {
          int &epoch_rc = epoch_error_[taskEpochKey(task)];
          if (epoch_rc == 0)
            epoch_rc = rc;
        }
      }
      ++completed_;
      handle_done_.notify_all();
      idle_.notify_all();
    }
  }

  Executor executor_;
  std::mutex mtx_;
  std::condition_variable has_work_;
  std::condition_variable not_full_;
  std::condition_variable handle_done_;
  std::condition_variable idle_;
  std::deque<Task> queue_;
  struct InFlight {
    bool active = false;
    int handle = -1;
    bool has_offset = false;
    long offset = 0;
    size_t size = 0;
  };
  InFlight current_;
  std::unordered_map<int, uint64_t> pending_;
  // First failing rc per handle; entries are dropped on close (see
  // forgetHandle) so a reused file id starts clean.
  std::unordered_map<int, int> handle_error_;
  // The same two, keyed by completion unit. Ordered so forgetHandle can erase
  // a handle's whole epoch range in one pass; both stay small.
  std::map<EpochKey, uint64_t> pending_epoch_;
  std::map<EpochKey, int> epoch_error_;
  std::thread worker_;
  std::once_flag available_once_;
  size_t max_depth_ = 2;
  bool enabled_ = false;
  bool worker_started_ = false;
  bool busy_ = false;
  bool stop_ = false;
  bool stats_reported_ = false;
  bool report_stats_ = false;
  int sticky_error_ = 0;
  uint64_t enqueued_ = 0;
  uint64_t completed_ = 0;
  uint64_t owned_enqueued_ = 0;
  uint64_t owned_bytes_ = 0;
  size_t max_queue_depth_ = 0;
  uint64_t drain_calls_ = 0;
  uint64_t drain_wait_ns_ = 0;
  uint64_t range_waits_ = 0;
  uint64_t range_wait_ns_ = 0;
  uint64_t range_wait_free_ = 0;
  uint64_t enqueue_waits_ = 0;
  uint64_t enqueue_wait_ns_ = 0;
};

#endif // OMPFILE_ASYNC_WRITE_ENGINE_H
