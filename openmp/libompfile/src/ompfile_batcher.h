//===-- ompfile_batcher.h - leader/follower request batcher ----*- C++ -*-===//
//
// One leader/follower batcher for libompfile's inline batching paths: the
// two-phase read window, the write batch and the dirty-owner pread batch.
// Header-only and internal (src/, not installed); depends on nothing in
// libompfile, so tests/batcher_test.cpp unit-tests it without MPI.
//
// Callers queue caller-owned members by pointer. A site object (see
// SiteDefaults) supplies the close policy, the drain key and execute(). A
// site carries no per-member state: the leader's site serves every member it
// drains.
//
// Contract:
//  - mu_ is a leaf lock: nothing else is locked while it is held, and
//    execute() always runs with it released, so no MPP call is made under it.
//    complete() is called with no other lock held.
//  - At most one leader per Batcher (in_progress_, under mu_).
//  - Each member is completed exactly once and never touched afterwards: its
//    owner may return and destroy it as soon as it sees Done.
//  - No exception handling. If execute() throws, in_progress_ stays set and
//    the queued members stay blocked, exactly as before this header.
//
//===----------------------------------------------------------------------===//
#ifndef OMPFILE_BATCHER_H
#define OMPFILE_BATCHER_H

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ompfile {
namespace batch {

using Clock = std::chrono::steady_clock;

// Result slot of every queued member.
struct Completion {
  int Status = 0;
  int Errno = 0;
  bool Done = false;
};

enum class Shape : uint8_t {
  // One drain per leader turn. The window is held under cv_ to an absolute
  // deadline, and enqueue notifications never end it. execute() completes
  // each member through complete(). Followers wake on their own Done or when
  // the turn ends, and may lead the next turn. (Two-phase read, write batch.)
  LeaderPerBatch,
  // The leader drains until the queue is empty. Before its first drain only,
  // it sleeps the window with no lock held. execute() writes results and the
  // batcher marks the drained members Done after it returns. Followers wake
  // on their own Done only; enqueue does not notify. (Dirty-owner pread.)
  LeaderUntilEmpty,
};

enum class CloseRule : uint8_t {
  Fixed,         // hold WindowUs (0: no window)
  Adaptive,      // skip the window iff the leader's member is the whole
                 // queue and no other thread is inside submit(); else Fixed
  ExpectedCount, // close at the deadline or once ExpectedMembers eligible
                 // members (the leader's own included) are queued. The graph
                 // plan's hook. WindowUs > 0 is the mandatory safety net: an
                 // over-estimated count degrades to Fixed and never hangs.
};

struct ClosePolicy {
  CloseRule Rule = CloseRule::Fixed;
  uint64_t WindowUs = 0;
  uint32_t ExpectedMembers = 0;
};

// The site's existing counters; nullptr = not counted.
struct Stats {
  std::atomic<uint64_t> *QueueMaxDepth = nullptr; // max(size after push)
  std::atomic<uint64_t> *LeaderTurns = nullptr;   // per leader turn/tenure
  std::atomic<uint64_t> *FollowerWaits = nullptr; // per follower wait entry
  std::atomic<uint64_t> *FollowerWaitUs = nullptr;
  std::atomic<uint64_t> *WindowWaitUs = nullptr;
  std::atomic<uint64_t> *BatchExecUs = nullptr;   // per execute(), if > 0
  std::atomic<uint64_t> *RequestWaitUs = nullptr; // enqueue -> done, if > 0
};

inline void updateAtomicMax(std::atomic<uint64_t> &counter,
                            uint64_t candidate) {
  uint64_t observed = counter.load(std::memory_order_relaxed);
  while (candidate > observed &&
         !counter.compare_exchange_weak(observed, candidate,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
  }
}

inline int64_t elapsedUs(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
      .count();
}

inline void bump(std::atomic<uint64_t> *counter, uint64_t value) {
  if (counter)
    counter->fetch_add(value, std::memory_order_relaxed);
}

// Base of every site. A site must also define:
//   ClosePolicy closePolicy();              leader, once per turn/tenure
//   void execute(std::vector<M *> &batch);  leader, mu_ released
template <typename M> struct SiteDefaults {
  // 0: a drain takes the whole queue. N > 0: a drain takes the queue front
  // plus later members sameBatch() accepts, in queue order, N at most.
  size_t maxMembers() { return 0; }
  bool sameBatch(const M &, const M &) { return true; } // must be reflexive
  void traceEnqueue(const M &, size_t /*queue_before*/, bool /*in_progress*/) {}
  void traceLeader(const M &, size_t /*queue_size*/) {}
  void traceDrained(const M &, size_t /*batch_size*/, size_t /*queue_after*/) {}
  void traceFollowerWait(const M &, size_t /*queue_size*/) {}
};

// The max-bytes cap, shared by two-phase and write grouping. Sorts `group` in
// place by (offset, size), then folds it into `runs`. A member joins the
// current run when it starts at or before the run's end and the merged
// extent stays within max_bytes (0 = no cap); otherwise it starts a run.
// Rejects go through reject(member, errnum) and join no run: a negative
// offset gets EINVAL, an end past LONG_MAX gets EOVERFLOW.
// Item: default-constructible with `long Start, End; std::vector<M *> Requests;`.
template <typename Item, typename M, typename OffsetOf, typename SizeOf,
          typename Reject>
void coalesceSortedRuns(std::vector<M *> &group, uint64_t max_bytes,
                        std::vector<Item> &runs, OffsetOf offsetOf,
                        SizeOf sizeOf, Reject reject) {
  std::sort(group.begin(), group.end(), [&](const M *lhs, const M *rhs) {
    if (offsetOf(*lhs) != offsetOf(*rhs))
      return offsetOf(*lhs) < offsetOf(*rhs);
    return sizeOf(*lhs) < sizeOf(*rhs);
  });
  runs.reserve(group.size());
  for (M *member : group) {
    const long offset = offsetOf(*member);
    const size_t size = sizeOf(*member);
    if (offset < 0) {
      reject(*member, EINVAL);
      continue;
    }
    if (size > static_cast<size_t>(std::numeric_limits<long>::max()) ||
        offset > std::numeric_limits<long>::max() - static_cast<long>(size)) {
      reject(*member, EOVERFLOW);
      continue;
    }
    const long start = offset;
    const long end = start + static_cast<long>(size);
    if (!runs.empty()) {
      Item &tail = runs.back();
      const bool contiguous = start <= tail.End;
      const long merged_end = std::max(tail.End, end);
      const uint64_t merged_size =
          static_cast<uint64_t>(merged_end - tail.Start);
      const bool exceeds_cap = max_bytes > 0 && merged_size > max_bytes;
      if (contiguous && !exceeds_cap) {
        tail.End = merged_end;
        tail.Requests.push_back(member);
        continue;
      }
    }
    Item item{};
    item.Start = start;
    item.End = end;
    item.Requests.push_back(member);
    runs.push_back(std::move(item));
  }
}

template <typename M> class Batcher {
  static_assert(std::is_base_of_v<Completion, M>,
                "batch members derive from ompfile::batch::Completion");

public:
  Batcher(Shape shape, Stats stats) : shape_(shape), stats_(stats) {}
  Batcher(const Batcher &) = delete;
  Batcher &operator=(const Batcher &) = delete;

  // Queues `member` and returns once it is Done; its Status/Errno, and any
  // result fields execute() wrote, are then final. Returns the member's
  // enqueue-to-done time in microseconds.
  template <typename Site> int64_t submit(M &member, Site &site) {
    struct ActiveScope {
      std::atomic<uint32_t> &Count;
      explicit ActiveScope(std::atomic<uint32_t> &C) : Count(C) {
        Count.fetch_add(1, std::memory_order_acq_rel);
      }
      ~ActiveScope() { Count.fetch_sub(1, std::memory_order_acq_rel); }
    } active(active_);
    const auto enqueue_ts = Clock::now();
    std::unique_lock<std::mutex> lock(mu_);
    const size_t queue_before = queue_.size();
    if (stats_.QueueMaxDepth)
      updateAtomicMax(*stats_.QueueMaxDepth,
                      static_cast<uint64_t>(queue_before + 1));
    site.traceEnqueue(member, queue_before, in_progress_);
    queue_.push_back(&member);
    if (shape_ == Shape::LeaderPerBatch) {
      cv_.notify_all();
      return runLeaderPerBatch(member, site, lock, enqueue_ts);
    }
    return runLeaderUntilEmpty(member, site, lock, enqueue_ts);
  }

  // LeaderPerBatch only: called from execute() on the leader, no other lock
  // held. under_lock runs with mu_ held, after Done is set and before the
  // notify, so it may still read the member.
  void complete(M &member, int status, int errnum) {
    complete(member, status, errnum, [] {});
  }
  template <typename UnderLock>
  void complete(M &member, int status, int errnum, UnderLock &&under_lock) {
    const std::lock_guard<std::mutex> lock(mu_);
    assert(!member.Done && "a batch member is completed exactly once");
    member.Status = status;
    member.Errno = errnum;
    member.Done = true;
    under_lock();
    cv_.notify_all();
  }

#if defined(OMPFILE_ENABLE_TEST_ACCESS)
  // Holds a Fixed window under mu_, exactly as a LeaderPerBatch leader does.
  void holdWindowForTesting(uint64_t window_us) {
    std::unique_lock<std::mutex> lock(mu_);
    SiteDefaults<M> site;
    waitWindowLocked(lock, ClosePolicy{CloseRule::Fixed, window_us, 0}, site);
  }
  void notifyForTesting() { cv_.notify_all(); }
  Shape shapeForTesting() const { return shape_; }
#endif

private:
  template <typename Site> size_t eligibleCountLocked(Site &site) {
    const size_t max_members = site.maxMembers();
    if (max_members == 0)
      return queue_.size();
    if (queue_.empty())
      return 0;
    const M &seed = *queue_.front();
    size_t count = 0;
    for (const M *candidate : queue_)
      if (site.sameBatch(seed, *candidate) && ++count >= max_members)
        break;
    return count;
  }

  template <typename Site>
  bool shouldHoldWindowLocked(const ClosePolicy &policy, Site &site) {
    switch (policy.Rule) {
    case CloseRule::Fixed:
      return policy.WindowUs > 0;
    case CloseRule::Adaptive: {
      const bool nobody_can_join =
          queue_.size() <= 1 && active_.load(std::memory_order_acquire) <= 1;
      return policy.WindowUs > 0 && !nobody_can_join;
    }
    case CloseRule::ExpectedCount:
      return policy.WindowUs > 0 &&
             eligibleCountLocked(site) < policy.ExpectedMembers;
    }
    return false;
  }

  template <typename Site>
  void waitWindowLocked(std::unique_lock<std::mutex> &lock,
                        const ClosePolicy &policy, Site &site) {
    const auto deadline =
        Clock::now() + std::chrono::microseconds(policy.WindowUs);
    if (policy.Rule == CloseRule::ExpectedCount) {
      cv_.wait_until(lock, deadline, [this, &site, &policy] {
        return eligibleCountLocked(site) >= policy.ExpectedMembers;
      });
      return;
    }
    // Enqueue and completion notifications serve followers, but they must
    // not shorten the leader's window. The predicate stays false, so
    // wait_until resumes after notifications and spurious wakeups and
    // returns only at the absolute deadline.
    cv_.wait_until(lock, deadline, [] { return false; });
  }

  template <typename Site>
  void drainLocked(std::vector<M *> &batch, Site &site) {
    const size_t max_members = site.maxMembers();
    if (max_members == 0) {
      while (!queue_.empty()) {
        batch.push_back(queue_.front());
        queue_.pop_front();
      }
      return;
    }
    assert(!queue_.empty() && "a seeded drain needs a queued member");
    M *seed = queue_.front();
    for (auto it = queue_.begin();
         it != queue_.end() && batch.size() < max_members;) {
      if (site.sameBatch(*seed, **it)) {
        batch.push_back(*it);
        it = queue_.erase(it);
      } else {
        ++it;
      }
    }
    assert(!batch.empty() && batch.front() == seed &&
           "sameBatch must accept the seed");
  }

  template <typename Site>
  int64_t runLeaderPerBatch(M &member, Site &site,
                            std::unique_lock<std::mutex> &lock,
                            Clock::time_point enqueue_ts) {
    std::vector<M *> batch;
    while (!member.Done) {
      if (!in_progress_) {
        in_progress_ = true;
        bump(stats_.LeaderTurns, 1);
        site.traceLeader(member, queue_.size());
        const ClosePolicy policy = site.closePolicy();
        if (shouldHoldWindowLocked(policy, site)) {
          const auto window_begin = Clock::now();
          waitWindowLocked(lock, policy, site);
          const int64_t window_us = elapsedUs(window_begin, Clock::now());
          if (window_us > 0)
            bump(stats_.WindowWaitUs, static_cast<uint64_t>(window_us));
        }
        drainLocked(batch, site);
        site.traceDrained(member, batch.size(), queue_.size());
        lock.unlock();
        const auto exec_begin = Clock::now();
        site.execute(batch);
        const int64_t exec_us = elapsedUs(exec_begin, Clock::now());
        if (exec_us > 0)
          bump(stats_.BatchExecUs, static_cast<uint64_t>(exec_us));
        batch.clear();
        lock.lock();
        in_progress_ = false;
        cv_.notify_all();
      } else {
        bump(stats_.FollowerWaits, 1);
        site.traceFollowerWait(member, queue_.size());
        const auto wait_begin = Clock::now();
        cv_.wait(lock, [this, &member] { return member.Done || !in_progress_; });
        const int64_t wait_us = elapsedUs(wait_begin, Clock::now());
        if (wait_us > 0)
          bump(stats_.FollowerWaitUs, static_cast<uint64_t>(wait_us));
      }
    }
    // Still under mu_, as before.
    const int64_t request_us = elapsedUs(enqueue_ts, Clock::now());
    if (request_us > 0)
      bump(stats_.RequestWaitUs, static_cast<uint64_t>(request_us));
    return request_us;
  }

  template <typename Site>
  int64_t runLeaderUntilEmpty(M &member, Site &site,
                              std::unique_lock<std::mutex> &lock,
                              Clock::time_point enqueue_ts) {
    if (in_progress_) {
      bump(stats_.FollowerWaits, 1);
      site.traceFollowerWait(member, queue_.size());
      cv_.wait(lock, [&member] { return member.Done; });
      const int64_t wait_us = elapsedUs(enqueue_ts, Clock::now());
      if (wait_us > 0) {
        bump(stats_.FollowerWaitUs, static_cast<uint64_t>(wait_us));
        bump(stats_.RequestWaitUs, static_cast<uint64_t>(wait_us));
      }
      return wait_us;
    }
    in_progress_ = true;
    bump(stats_.LeaderTurns, 1);
    site.traceLeader(member, queue_.size());
    lock.unlock();
    const ClosePolicy policy = site.closePolicy();
    assert(policy.Rule == CloseRule::Fixed &&
           "LeaderUntilEmpty holds a fixed window only");
    if (policy.WindowUs > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(policy.WindowUs));
      bump(stats_.WindowWaitUs, policy.WindowUs);
    }
    std::vector<M *> batch;
    while (true) {
      batch.clear();
      lock.lock();
      if (queue_.empty()) {
        in_progress_ = false;
        cv_.notify_all();
        lock.unlock();
        break;
      }
      drainLocked(batch, site);
      lock.unlock();
      const auto exec_begin = Clock::now();
      site.execute(batch);
      if (stats_.BatchExecUs) {
        const int64_t exec_us = elapsedUs(exec_begin, Clock::now());
        if (exec_us > 0)
          bump(stats_.BatchExecUs, static_cast<uint64_t>(exec_us));
      }
      lock.lock();
      for (M *drained : batch)
        drained->Done = true;
      lock.unlock();
      cv_.notify_all();
    }
    const int64_t request_us = elapsedUs(enqueue_ts, Clock::now());
    if (request_us > 0)
      bump(stats_.RequestWaitUs, static_cast<uint64_t>(request_us));
    return request_us;
  }

  const Shape shape_;
  const Stats stats_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<M *> queue_;
  bool in_progress_ = false;
  std::atomic<uint32_t> active_{0};
};

} // namespace batch
} // namespace ompfile

#endif // OMPFILE_BATCHER_H
