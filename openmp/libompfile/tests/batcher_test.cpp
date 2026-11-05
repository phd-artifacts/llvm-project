// The leader/follower batcher in src/ompfile_batcher.h, driven by real
// threads: the window holds its deadline across enqueues, Adaptive skips a
// lone leader, ExpectedCount closes early or at its deadline, a follower
// returns before its leader's batch ends, the until-empty shape drains by
// seed key in FIFO order, and the shared coalescer caps and rejects the way
// the two-phase and write-batch grouping always did.
//
// The binary runs after every runtime build, often on a loaded node, so no
// case depends on scheduling luck. A follower is ordered behind its leader
// by a handshake (the leader's traceLeader flag or the queue-depth counter),
// never by a sleep. A Fixed window a follower has to land in lasts 2 s, and a
// window that must close early has a 20 s deadline. Timing checks are lower
// bounds (the window was held) or bounds relative to the window itself, never
// wall-clock maxima. Every spin in this file gives up after 20 s, so a missed
// handshake fails a case instead of hanging the build. Total runtime about
// 4.5 s.

#include "ompfile_batcher.h"
#include "ompfile_test.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace ompfile::batch;

// A Fixed window a follower has to land in. The follower enqueues as soon as
// the window opens, so 2 s absorbs any scheduling delay on a loaded node.
constexpr uint64_t kJoinWindowUs = 2000000;
// The deadline of a window that must close early, and the limit of every
// spin. A working batcher never reaches it; it only bounds how long a broken
// one takes to fail.
constexpr uint64_t kSafetyNetUs = 20000000;

struct Member : Completion {
  int Key = 0;
  int Id = 0;
};

struct Counters {
  std::atomic<uint64_t> QueueMaxDepth{0};
  std::atomic<uint64_t> LeaderTurns{0};
  std::atomic<uint64_t> FollowerWaits{0};
  std::atomic<uint64_t> FollowerWaitUs{0};
  std::atomic<uint64_t> WindowWaitUs{0};
  std::atomic<uint64_t> BatchExecUs{0};
  std::atomic<uint64_t> RequestWaitUs{0};
  Stats stats() {
    Stats S;
    S.QueueMaxDepth = &QueueMaxDepth;
    S.LeaderTurns = &LeaderTurns;
    S.FollowerWaits = &FollowerWaits;
    S.FollowerWaitUs = &FollowerWaitUs;
    S.WindowWaitUs = &WindowWaitUs;
    S.BatchExecUs = &BatchExecUs;
    S.RequestWaitUs = &RequestWaitUs;
    return S;
  }
};

struct Recorder {
  std::mutex Mu;
  std::vector<std::vector<int>> Batches;
  void record(const std::vector<Member *> &Batch) {
    std::vector<int> Ids;
    for (const Member *M : Batch)
      Ids.push_back(M->Id);
    const std::lock_guard<std::mutex> Lock(Mu);
    Batches.push_back(Ids);
  }
  std::vector<std::vector<int>> snapshot() {
    const std::lock_guard<std::mutex> Lock(Mu);
    return Batches;
  }
};

// LeaderPerBatch site: records each batch, then completes every member.
struct PerBatchSite : SiteDefaults<Member> {
  Batcher<Member> &Owner;
  ClosePolicy Policy;
  Recorder &Rec;
  std::atomic<bool> *LeaderSeen = nullptr;
  PerBatchSite(Batcher<Member> &O, ClosePolicy P, Recorder &R)
      : Owner(O), Policy(P), Rec(R) {}
  ClosePolicy closePolicy() { return Policy; }
  void traceLeader(const Member &, size_t) {
    if (LeaderSeen)
      LeaderSeen->store(true);
  }
  void execute(std::vector<Member *> &Batch) {
    Rec.record(Batch);
    for (Member *M : Batch)
      Owner.complete(*M, 0, 0);
  }
};

// Spins until Done() holds; false once kSafetyNetUs has passed.
template <typename Pred> bool spinUntil(Pred Done) {
  const auto Deadline = std::chrono::steady_clock::now() +
                        std::chrono::microseconds(kSafetyNetUs);
  while (!Done()) {
    if (std::chrono::steady_clock::now() >= Deadline)
      return false;
    std::this_thread::yield();
  }
  return true;
}

bool waitUntil(const std::atomic<bool> &Flag) {
  return spinUntil([&Flag] { return Flag.load(); });
}

// A submits on its own thread and leads with `Policy`; once A is leader
// (traceLeader runs under the batcher lock, before the window), F submits
// from this thread. F's enqueue needs that lock, which A releases only inside
// its window wait, so F lands in A's window unless the window ends first.
void runLeaderAndFollower(ClosePolicy Policy, Counters &C, Recorder &Rec) {
  Batcher<Member> B(Shape::LeaderPerBatch, C.stats());
  std::atomic<bool> LeaderSeen{false};
  Member A;
  A.Id = 1;
  Member F;
  F.Id = 2;
  std::thread Leader([&] {
    PerBatchSite Site(B, Policy, Rec);
    Site.LeaderSeen = &LeaderSeen;
    (void)B.submit(A, Site);
  });
  OMPFILE_CHECK(waitUntil(LeaderSeen));
  {
    PerBatchSite Site(B, Policy, Rec);
    (void)B.submit(F, Site);
  }
  Leader.join();
  OMPFILE_CHECK(A.Done && F.Done);
}

} // namespace

OMPFILE_TEST(batcher_window_holds_deadline_across_enqueues) {
  Counters C;
  Recorder Rec;
  runLeaderAndFollower(ClosePolicy{CloseRule::Fixed, kJoinWindowUs, 0}, C,
                       Rec);
  const std::vector<std::vector<int>> Expected = {{1, 2}};
  OMPFILE_CHECK(Rec.snapshot() == Expected);
  // The follower enqueued and notified right after the window opened; the
  // window must still run to its deadline. Under the write batch's
  // predicate-less wait_for it ended at that first notify.
  OMPFILE_CHECK(C.WindowWaitUs.load() >= kJoinWindowUs * 9 / 10);
  OMPFILE_CHECK_EQ(C.LeaderTurns.load(), 1u);
  OMPFILE_CHECK_EQ(C.QueueMaxDepth.load(), 2u);
  OMPFILE_CHECK_EQ(C.FollowerWaits.load(), 1u);
}

OMPFILE_TEST(batcher_adaptive_skips_lone_leader) {
  Counters C;
  Recorder Rec;
  Batcher<Member> B(Shape::LeaderPerBatch, C.stats());
  Member A;
  A.Id = 1;
  PerBatchSite Site(B, ClosePolicy{CloseRule::Adaptive, 100000, 0}, Rec);
  (void)B.submit(A, Site);
  // A held window adds its measured length (at least 100 ms); a skipped one
  // adds nothing, so the counter tells them apart without a clock bound.
  OMPFILE_CHECK_EQ(C.WindowWaitUs.load(), 0u);
  OMPFILE_CHECK_EQ(C.LeaderTurns.load(), 1u);
  const std::vector<std::vector<int>> Expected = {{1}};
  OMPFILE_CHECK(Rec.snapshot() == Expected);
  OMPFILE_CHECK(A.Done && A.Status == 0);
}

OMPFILE_TEST(batcher_fixed_lone_leader_waits) {
  Counters C;
  Recorder Rec;
  Batcher<Member> B(Shape::LeaderPerBatch, C.stats());
  Member A;
  A.Id = 1;
  PerBatchSite Site(B, ClosePolicy{CloseRule::Fixed, 30000, 0}, Rec);
  (void)B.submit(A, Site);
  OMPFILE_CHECK(C.WindowWaitUs.load() >= 24000u);
  OMPFILE_CHECK(A.Done && A.Status == 0);
}

OMPFILE_TEST(batcher_expected_count_closes_on_nth) {
  Counters C;
  Recorder Rec;
  runLeaderAndFollower(ClosePolicy{CloseRule::ExpectedCount, kSafetyNetUs, 2},
                       C, Rec);
  // A window that reaches its deadline measures at least its length, so any
  // value below it is the close on the second member.
  OMPFILE_CHECK(C.WindowWaitUs.load() < kSafetyNetUs);
  const std::vector<std::vector<int>> Expected = {{1, 2}};
  OMPFILE_CHECK(Rec.snapshot() == Expected);
}

OMPFILE_TEST(batcher_expected_count_deadline_fallback) {
  Counters C;
  Recorder Rec;
  // Three expected, two arrive: the window closes at its deadline.
  runLeaderAndFollower(ClosePolicy{CloseRule::ExpectedCount, kJoinWindowUs, 3},
                       C, Rec);
  OMPFILE_CHECK(C.WindowWaitUs.load() >= kJoinWindowUs * 9 / 10);
  const std::vector<std::vector<int>> Expected = {{1, 2}};
  OMPFILE_CHECK(Rec.snapshot() == Expected);
}

namespace {
// Completes Id 2 first, then waits for its owner to return before completing
// Id 1: per-member fan-out. The window closes once both members are queued
// (ExpectedCount 2), so the follower always joins the leader's batch.
struct FanOutSite : SiteDefaults<Member> {
  Batcher<Member> &Owner;
  std::atomic<bool> &FollowerReturned;
  std::atomic<bool> &LeaderSeen;
  std::atomic<bool> &SawReturnedInside;
  FanOutSite(Batcher<Member> &O, std::atomic<bool> &R, std::atomic<bool> &L,
             std::atomic<bool> &S)
      : Owner(O), FollowerReturned(R), LeaderSeen(L), SawReturnedInside(S) {}
  ClosePolicy closePolicy() {
    return ClosePolicy{CloseRule::ExpectedCount, kSafetyNetUs, 2};
  }
  void traceLeader(const Member &, size_t) { LeaderSeen.store(true); }
  void execute(std::vector<Member *> &Batch) {
    Member *First = nullptr;
    Member *Second = nullptr;
    for (Member *M : Batch)
      (M->Id == 1 ? First : Second) = M;
    if (Second)
      Owner.complete(*Second, 0, 0);
    SawReturnedInside.store(waitUntil(FollowerReturned));
    if (First)
      Owner.complete(*First, 0, 0);
  }
};
} // namespace

OMPFILE_TEST(batcher_follower_released_before_batch_end) {
  Counters C;
  Batcher<Member> B(Shape::LeaderPerBatch, C.stats());
  std::atomic<bool> Returned{false};
  std::atomic<bool> LeaderSeen{false};
  std::atomic<bool> SawReturnedInside{false};
  Member A;
  A.Id = 1;
  Member F;
  F.Id = 2;
  std::thread Leader([&] {
    FanOutSite Site(B, Returned, LeaderSeen, SawReturnedInside);
    (void)B.submit(A, Site);
  });
  OMPFILE_CHECK(waitUntil(LeaderSeen));
  std::thread Follower([&] {
    FanOutSite Site(B, Returned, LeaderSeen, SawReturnedInside);
    (void)B.submit(F, Site);
    Returned.store(true);
  });
  Follower.join();
  Leader.join();
  OMPFILE_CHECK(SawReturnedInside.load());
  OMPFILE_CHECK(A.Done && F.Done);
}

namespace {
// True once Depth members were queued: QueueMaxDepth is bumped under the
// batcher lock right before the push, so the member is in the queue by the
// time any other thread gets the lock.
bool waitDepth(Counters &C, uint64_t Depth) {
  return spinUntil([&C, Depth] { return C.QueueMaxDepth.load() >= Depth; });
}

struct UntilEmptySite : SiteDefaults<Member> {
  Recorder &Rec;
  Counters &Gate;
  UntilEmptySite(Recorder &R, Counters &G) : Rec(R), Gate(G) {}
  // LeaderUntilEmpty calls closePolicy() once per tenure, with the batcher
  // lock released and before the first drain. Holding the leader here until
  // all four members are queued makes the drain order a property of the
  // queue alone, not of how fast the other threads start. A gate that times
  // out shows up as a wrong batch order.
  ClosePolicy closePolicy() {
    (void)waitDepth(Gate, 4);
    return ClosePolicy{CloseRule::Fixed, 200000, 0};
  }
  size_t maxMembers() { return 2; }
  bool sameBatch(const Member &Seed, const Member &Candidate) {
    return Seed.Key == Candidate.Key;
  }
  void execute(std::vector<Member *> &Batch) {
    Rec.record(Batch);
    for (Member *M : Batch) {
      M->Status = 0;
      M->Errno = 0;
    }
  }
};
} // namespace

OMPFILE_TEST(batcher_until_empty_drains_seed_key_fifo) {
  Counters C;
  Recorder Rec;
  Batcher<Member> B(Shape::LeaderUntilEmpty, C.stats());
  Member M1, M2, M3, M4;
  M1.Key = 1; M1.Id = 1;
  M2.Key = 2; M2.Id = 2;
  M3.Key = 1; M3.Id = 3;
  M4.Key = 1; M4.Id = 4;
  auto run = [&](Member &M) {
    UntilEmptySite Site(Rec, C);
    (void)B.submit(M, Site);
  };
  // Each thread starts once the previous member is queued: queue order 1-4,
  // and member 1's thread is the leader.
  std::thread T1(run, std::ref(M1));
  OMPFILE_CHECK(waitDepth(C, 1));
  std::thread T2(run, std::ref(M2));
  OMPFILE_CHECK(waitDepth(C, 2));
  std::thread T3(run, std::ref(M3));
  OMPFILE_CHECK(waitDepth(C, 3));
  std::thread T4(run, std::ref(M4));
  T1.join();
  T2.join();
  T3.join();
  T4.join();
  const std::vector<std::vector<int>> Expected = {{1, 3}, {2}, {4}};
  OMPFILE_CHECK(Rec.snapshot() == Expected);
  OMPFILE_CHECK_EQ(C.QueueMaxDepth.load(), 4u);
  OMPFILE_CHECK_EQ(C.WindowWaitUs.load(), 200000u);
  OMPFILE_CHECK_EQ(C.FollowerWaits.load(), 3u);
  OMPFILE_CHECK_EQ(C.LeaderTurns.load(), 1u);
  OMPFILE_CHECK(M1.Done && M2.Done && M3.Done && M4.Done);
}

namespace {
struct Req {
  long Off = 0;
  size_t Sz = 0;
  int Err = 0;
};
struct Run {
  long Start = 0;
  long End = 0;
  std::vector<Req *> Requests;
};

std::vector<Run> coalesce(std::vector<Req> &Reqs, uint64_t Cap) {
  std::vector<Req *> Group;
  for (Req &R : Reqs)
    Group.push_back(&R);
  std::vector<Run> Runs;
  coalesceSortedRuns(
      Group, Cap, Runs, [](const Req &R) { return R.Off; },
      [](const Req &R) { return R.Sz; }, [](Req &R, int E) { R.Err = E; });
  return Runs;
}

std::vector<Req> corpus() {
  return {{100, 50, 0}, {0, 64, 0},  {32, 64, 0},          {200, 10, 0},
          {150, 10, 0}, {-1, 4, 0}, {LONG_MAX - 1, 8, 0}};
}
} // namespace

OMPFILE_TEST(batcher_coalesce_sorted_runs_cap_and_rejects) {
  {
    std::vector<Req> Reqs = corpus();
    const std::vector<Run> Runs = coalesce(Reqs, 0);
    OMPFILE_CHECK_EQ(Runs.size(), size_t{3});
    if (Runs.size() == 3) {
      OMPFILE_CHECK(Runs[0].Start == 0 && Runs[0].End == 96 &&
                    Runs[0].Requests.size() == 2);
      OMPFILE_CHECK(Runs[1].Start == 100 && Runs[1].End == 160 &&
                    Runs[1].Requests.size() == 2);
      OMPFILE_CHECK(Runs[2].Start == 200 && Runs[2].End == 210 &&
                    Runs[2].Requests.size() == 1);
    }
    OMPFILE_CHECK_EQ(Reqs[5].Err, EINVAL);
    OMPFILE_CHECK_EQ(Reqs[6].Err, EOVERFLOW);
  }
  {
    std::vector<Req> Reqs = corpus();
    const std::vector<Run> Runs = coalesce(Reqs, 80);
    OMPFILE_CHECK_EQ(Runs.size(), size_t{4});
    if (Runs.size() == 4) {
      OMPFILE_CHECK(Runs[0].Start == 0 && Runs[0].End == 64);
      OMPFILE_CHECK(Runs[1].Start == 32 && Runs[1].End == 96);
      OMPFILE_CHECK(Runs[2].Start == 100 && Runs[2].End == 160);
      OMPFILE_CHECK(Runs[3].Start == 200 && Runs[3].End == 210);
    }
  }
}
