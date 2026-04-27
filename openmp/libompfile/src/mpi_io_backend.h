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
#include <string>
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
  std::unordered_map<int, std::unordered_map<int, int>> remote_read_handle_cache;
  std::unordered_map<int, std::string> file_path_map;
  std::unordered_map<int, uint64_t> file_path_key_map;
  struct WriteEpochEntry {
    long Start = 0;
    long End = 0;
    uint64_t EpochId = 0;
    uint64_t TileId = 0;
    uint64_t Sequence = 0;
    bool HasEpoch = false;
    bool HasTile = false;
  };
  std::unordered_map<int, std::vector<WriteEpochEntry>> file_write_epoch_history;
  uint64_t next_write_sequence = 1;
  std::unordered_set<int> logical_handle_set;
  std::atomic<int> next_file_handle{0};
  std::mutex handle_mutex;
  std::mutex mpp_call_mutex;
  bool mpp_open_enabled = false;
  bool mpp_io_enabled = false;
  bool mpp_requested = false;
  bool mpp_remote_only = false;
  bool mpp_init_succeeded = false;
  bool allow_fallback = false;
  bool strict_mpp_required = false;
  bool strict_mpp_init_failed = false;
  mutable std::atomic<bool> strict_failure_logged{false};
  TwoPhasePolicy two_phase_policy = TwoPhasePolicy::Disabled;
  bool two_phase_enabled = false;
  TwoPhasePolicy write_batch_policy = TwoPhasePolicy::Disabled;
  bool write_batch_enabled = false;
  static constexpr uint64_t kDefaultTwoPhaseWindowUs = 2000;
  static constexpr uint64_t kDefaultTwoPhaseMaxBatchBytes = 8 * 1024 * 1024;
  static constexpr uint64_t kDefaultTwoPhaseSieveBytes = 256 * 1024;
  static constexpr uint64_t kDefaultWriteBatchWindowUs = 2000;
  static constexpr uint64_t kDefaultWriteBatchMaxBatchBytes =
      16 * 1024 * 1024;
  uint64_t two_phase_window_us = 0;
  uint64_t two_phase_max_batch_bytes = 0;
  uint64_t two_phase_sieve_bytes = 0;
  uint64_t write_batch_window_us = 0;
  uint64_t write_batch_max_batch_bytes = 0;
  std::atomic<uint64_t> pread_request_count{0};
  std::atomic<uint64_t> remote_pread_event_count{0};
  std::atomic<uint64_t> remote_pread_bytes_total{0};
  std::atomic<uint64_t> pwrite_request_count{0};
  std::atomic<uint64_t> remote_pwrite_event_count{0};
  std::atomic<uint64_t> remote_pwrite_bytes_total{0};
  std::atomic<uint64_t> short_read_count{0};
  std::atomic<uint64_t> short_read_bytes_total{0};
  std::atomic<uint64_t> short_write_count{0};
  std::atomic<uint64_t> short_write_bytes_total{0};
  std::atomic<uint64_t> two_phase_fallback_count{0};
  std::atomic<uint64_t> two_phase_batch_count{0};
  std::atomic<uint64_t> two_phase_coalesced_read_count{0};
  std::atomic<uint64_t> two_phase_coalesced_bytes_total{0};
  std::atomic<uint64_t> two_phase_planner_batch_count{0};
  std::atomic<uint64_t> two_phase_planner_segment_count{0};
  std::atomic<uint64_t> two_phase_planner_affinity_count{0};
  std::atomic<uint64_t> two_phase_planner_rebalanced_count{0};
  std::atomic<uint64_t> two_phase_planner_rebalanced_applied_count{0};
  std::atomic<uint64_t> two_phase_planner_rebalanced_skipped_count{0};
  std::atomic<uint64_t> two_phase_planner_scalar_fallback_count{0};
  std::atomic<uint64_t> two_phase_planner_error_count{0};
  std::atomic<uint64_t> two_phase_cache_hit_count{0};
  std::atomic<uint64_t> two_phase_planner_batch_id{1};
  std::atomic<uint64_t> two_phase_request_id{1};
  std::atomic<uint64_t> write_batch_count{0};
  std::atomic<uint64_t> write_batch_coalesced_write_count{0};
  std::atomic<uint64_t> write_batch_coalesced_bytes_total{0};
  std::atomic<uint64_t> write_batch_saved_events{0};
  std::atomic<uint64_t> write_batch_failure_count{0};
  std::atomic<uint64_t> write_batch_request_id{1};
  bool two_phase_batch_in_progress = false;
  bool write_batch_in_progress = false;

  struct TwoPhaseReadRequest {
    uint64_t DebugRequestId = 0;
    int FileHandle = -1;
    int ClientRank = -1;
    long Offset = 0;
    size_t Size = 0;
    void *Buffer = nullptr;
    uint64_t PathKey = 0;
    bool HasPathKey = false;
    ompfile::OmpFileIOHint Hint{};

    int Status = 0;
    int Errno = 0;
    bool Done = false;
  };

  std::mutex two_phase_mutex;
  std::condition_variable two_phase_queue_cv;
  std::deque<TwoPhaseReadRequest *> two_phase_queue;

  struct WriteBatchRequest {
    uint64_t DebugRequestId = 0;
    int FileHandle = -1;
    long Offset = 0;
    uint64_t GroupKey = 0;
    std::vector<char> Data;
    int Status = 0;
    int Errno = 0;
    bool Done = false;
  };

  std::mutex write_batch_mutex;
  std::condition_variable write_batch_queue_cv;
  std::deque<WriteBatchRequest *> write_batch_queue;
  struct TwoPhaseReadCacheEntry {
    long Start = 0;
    std::vector<char> Data;
  };
  std::unordered_map<uint64_t, TwoPhaseReadCacheEntry> two_phase_read_cache;

public:
  MPIIOBackend();
  ~MPIIOBackend();

  int open(const char *filename) override;
  int openWithPlan(const char *filename,
                   const ompfile::OmpFileIOPlan *plan) override;
  int write(int file_id, const void *data, size_t size) override;
  int read(int file_id, void *data, size_t size) override;
  int close(int file_id) override;
  int seek(int file_id, long offset) override;
  int readAt(int file_id, long offset, void *data, size_t size) override;
  int readAtWithContext(const ompfile::OmpFileReadRequestContext &context,
                        void *data, size_t size) override;
  int writeAtWithContext(const ompfile::OmpFileWriteRequestContext &context,
                         const void *data, size_t size) override;
  int writeAt(int file_id, long offset, const void *data, size_t size) override;

private:
  int writeAtRemoteHandle(int remote_handle, long offset, const void *data,
                          size_t size, size_t &bytes_written);
  int writeAtBatched(int file_id, long offset, const void *data, size_t size,
                     uint64_t group_key);
  int readAtFallback(int file_id, long offset, void *data, size_t size);
  int readAtFallbackWithBytes(int file_id, long offset, void *data, size_t size,
                              size_t &bytes_read);
  int readAtTwoPhase(const ompfile::OmpFileReadRequestContext &context,
                     void *data, size_t size);
  void processTwoPhaseBatch(std::vector<TwoPhaseReadRequest *> &batch);
  void processTwoPhaseGroup(std::vector<TwoPhaseReadRequest *> &group);
  void processWriteBatch(std::vector<WriteBatchRequest *> &batch);
  void processWriteGroup(std::vector<WriteBatchRequest *> &group);
  uint64_t getTwoPhaseGroupKey(const TwoPhaseReadRequest &request) const;
  uint64_t getWriteBatchGroupKey(const WriteBatchRequest &request) const;
  void completeTwoPhaseRequest(TwoPhaseReadRequest &request, int status,
                               int errnum);
  void completeWriteBatchRequest(WriteBatchRequest &request, int status,
                                 int errnum);
  bool isTwoPhaseActive() const;
  bool isWriteBatchActive() const;
  bool hasUsablePlannedRead(const ompfile::OmpFileReadRequestContext &context) const;
  int getNextFileHandle();
  static TwoPhasePolicy parseTwoPhasePolicy(const char *env_value);
  static const char *twoPhasePolicyToString(TwoPhasePolicy policy);
  static bool parseBoolEnv(const char *name, bool default_value);
  static uint64_t parseUint64Env(const char *name, uint64_t default_value);
  static uint64_t computePathKey(const char *path);
  static uint64_t mixHintIntoKey(uint64_t base_key,
                                 const ompfile::OmpFileIOHint &hint);
  static uint64_t mixWriteHintIntoKey(uint64_t base_key,
                                      const ompfile::OmpFileIOHint &hint);
  static bool shouldReportStats();
  bool getFilePathKey(int file_id, uint64_t &path_key_out);
  void rememberFilePathKey(int file_id, const char *path);
  void rememberFilePath(int file_id, const char *path);
  void forgetFilePath(int file_id);
  void forgetFilePathKey(int file_id);
  uint64_t resolveTwoPhaseKey(
      const ompfile::OmpFileReadRequestContext &context);
  bool rangesOverlap(long a_start, long a_end, long b_start, long b_end) const;
  void recordWriteEpochForContext(int file_id, long offset, size_t size,
                                  const ompfile::OmpFileIOHint &hint);
  bool canApplyRebalancedRead(const ompfile::OmpFileIOHint &hint, int file_id,
                              long start, size_t size,
                              const char *&reason_out,
                              int &reason_errno_out);
  bool getOrCreateRemoteReadHandleForRank(int file_id, int target_rank,
                                          int &remote_handle_out);
  bool readAtRemoteRankWithBytes(int file_id, int target_rank, long offset,
                                 void *data, size_t size,
                                 size_t &bytes_read);
  void collectRemoteReadHandlesForClose(int file_id,
                                        std::vector<int> &handles_out);
  bool tryServeTwoPhaseReadCache(uint64_t key, long offset, void *data,
                                 size_t size);
  void updateTwoPhaseReadCache(uint64_t key, long start,
                               const std::vector<char> &data);
  void invalidateTwoPhaseReadCacheKey(uint64_t key);
  void invalidateTwoPhaseReadCacheForFile(int file_id);
  int failStrictMpp(const char *op_name) const;
  void traceHandleStateLocked(const char *where, int file_id,
                              int remote_handle) const;
  void traceHandleState(const char *where, int file_id, int remote_handle);
  void reportPhase0Stats() const;
};

#endif // MPI_IO_BACKEND_H
