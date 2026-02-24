#ifndef MPI_IO_BACKEND_H
#define MPI_IO_BACKEND_H

#include "abstract_backend.h"
#include <atomic>
#include <cstddef> // for size_t
#include <cstdint>
#include <mpi.h>
#include <unordered_map>
#include <unordered_set>

class MPIIOBackend : public IOBackend {
private:
  MPI_Comm file_comm = MPI_COMM_NULL;
  int externally_initialized;
  std::unordered_map<int, MPI_File> file_handle_map;
  std::unordered_map<int, int> remote_file_handle_map;
  std::unordered_set<int> logical_handle_set;
  std::atomic<int> next_file_handle;
  bool mpp_open_enabled = false;
  bool mpp_io_enabled = false;
  bool two_phase_enabled = false;
  uint64_t two_phase_window_us = 0;
  uint64_t two_phase_max_batch_bytes = 0;
  std::atomic<uint64_t> pread_request_count{0};
  std::atomic<uint64_t> remote_pread_event_count{0};
  std::atomic<uint64_t> remote_pread_bytes_total{0};
  std::atomic<uint64_t> two_phase_fallback_count{0};

public:
  MPIIOBackend();
  ~MPIIOBackend();

  int open(const char *filename) override;
  int write(int file_id, const void *data, size_t size) override;
  int read(int file_id, void *data, size_t size) override;
  int close(int file_id) override;
  int seek(int file_id, long offset) override;
  int readAt(int file_id, long offset, void *data, size_t size) override;
  int writeAt(int file_id, long offset, const void *data, size_t size) override;

private:
  int getNextFileHandle();
  static bool parseBoolEnv(const char *name, bool default_value);
  static uint64_t parseUint64Env(const char *name, uint64_t default_value);
  static bool shouldReportStats();
  void reportPhase0Stats() const;
};

#endif // MPI_IO_BACKEND_H
