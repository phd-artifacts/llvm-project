#ifndef MPI_IO_BACKEND_H
#define MPI_IO_BACKEND_H

#include "abstract_backend.h"
#include "ompfile_sched.h"
#include <condition_variable>
#include <atomic>
#include <cstddef> // for size_t
#include <cstdint>
#include <deque>
#include <mpi.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class MPIIOBackend : public IOBackend {
private:
  enum class TwoPhasePolicy {
    Disabled,
    Enabled,
    Auto,
  };

  MPI_Comm file_comm = MPI_COMM_NULL;
  int externally_initialized;
  std::unordered_map<int, MPI_File> file_handle_map;
  std::unordered_map<int, int> remote_file_handle_map;
  std::unordered_set<int> logical_handle_set;
  std::atomic<int> next_file_handle{0};
  std::mutex handle_mutex;
  std::mutex mpp_call_mutex;
  bool mpp_open_enabled = false;
  bool mpp_io_enabled = false;
  TwoPhasePolicy two_phase_policy = TwoPhasePolicy::Disabled;
  bool two_phase_enabled = false;
  uint64_t two_phase_window_us = 0;
  uint64_t two_phase_max_batch_bytes = 0;
  std::atomic<uint64_t> pread_request_count{0};
  std::atomic<uint64_t> remote_pread_event_count{0};
  std::atomic<uint64_t> remote_pread_bytes_total{0};
  std::atomic<uint64_t> short_read_count{0};
  std::atomic<uint64_t> short_read_bytes_total{0};
  std::atomic<uint64_t> two_phase_fallback_count{0};
  std::atomic<uint64_t> two_phase_batch_count{0};
  std::atomic<uint64_t> two_phase_coalesced_read_count{0};
  std::atomic<uint64_t> two_phase_coalesced_bytes_total{0};
  std::atomic<uint64_t> two_phase_planner_batch_count{0};
  std::atomic<uint64_t> two_phase_planner_segment_count{0};
  std::atomic<uint64_t> two_phase_planner_affinity_count{0};
  std::atomic<uint64_t> two_phase_planner_rebalanced_count{0};
  std::atomic<uint64_t> two_phase_planner_scalar_fallback_count{0};
  std::atomic<uint64_t> two_phase_planner_error_count{0};
  std::atomic<uint64_t> two_phase_planner_batch_id{1};
  std::atomic<uint64_t> two_phase_request_id{1};
  bool two_phase_batch_in_progress = false;

  struct TwoPhaseReadRequest {
    uint64_t DebugRequestId = 0;
    int FileHandle = -1;
    int ClientRank = -1;
    long Offset = 0;
    size_t Size = 0;
    void *Buffer = nullptr;
    uint64_t PathKey = 0;
    bool HasPathKey = false;

    int Status = 0;
    int Errno = 0;
    bool Done = false;
  };

  std::mutex two_phase_mutex;
  std::condition_variable two_phase_queue_cv;
  std::deque<TwoPhaseReadRequest *> two_phase_queue;

public:
  MPIIOBackend();
  ~MPIIOBackend();

  int open(const char *filename) override;
  int write(int file_id, const void *data, size_t size) override;
  int read(int file_id, void *data, size_t size) override;
  int close(int file_id) override;
  int seek(int file_id, long offset) override;
  int readAt(int file_id, long offset, void *data, size_t size) override;
  int readAtWithContext(const ompfile::OmpFileReadRequestContext &context,
                        void *data, size_t size) override;
  int writeAt(int file_id, long offset, const void *data, size_t size) override;

private:
  int readAtFallback(int file_id, long offset, void *data, size_t size);
  int readAtFallbackWithBytes(int file_id, long offset, void *data, size_t size,
                              size_t &bytes_read);
  int readAtTwoPhase(const ompfile::OmpFileReadRequestContext &context,
                     void *data, size_t size);
  void processTwoPhaseBatch(std::vector<TwoPhaseReadRequest *> &batch);
  void processTwoPhaseGroup(std::vector<TwoPhaseReadRequest *> &group);
  uint64_t getTwoPhaseGroupKey(const TwoPhaseReadRequest &request) const;
  void completeTwoPhaseRequest(TwoPhaseReadRequest &request, int status,
                               int errnum);
  bool isTwoPhaseActive() const;
  bool hasUsablePlannedRead(const ompfile::OmpFileReadRequestContext &context) const;
  int getNextFileHandle();
  static TwoPhasePolicy parseTwoPhasePolicy(const char *env_value);
  static const char *twoPhasePolicyToString(TwoPhasePolicy policy);
  static bool parseBoolEnv(const char *name, bool default_value);
  static uint64_t parseUint64Env(const char *name, uint64_t default_value);
  static bool shouldReportStats();
  void traceHandleStateLocked(const char *where, int file_id,
                              int remote_handle) const;
  void traceHandleState(const char *where, int file_id, int remote_handle);
  void reportPhase0Stats() const;
};

#endif // MPI_IO_BACKEND_H
