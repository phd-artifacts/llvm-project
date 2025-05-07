#ifndef IO_URING_IO_BACKEND_H
#define IO_URING_IO_BACKEND_H

#include "abstract_backend.h"
#include "debug_log.h"

#include <liburing.h>
#include <atomic>
#include <cstddef>
#include <unordered_map>

class IoUringIOBackend : public IOBackend {
private:
  bool is_initialized = false;

  // We maintain a global ring for all requests, but you could also keep
  // a ring per file if you need concurrency or different queue depths.
  io_uring ring;

  // Maps "file_id" -> (fd, current_offset)
  // (We keep a separate offset if you want to emulate a "file position" 
  // for read/write calls that don't have an explicit offset).
  struct FileData {
    int fd;
    off_t offset;
  };
  std::unordered_map<int, FileData> file_handle_map;

  std::atomic<int> next_file_handle;
  // We might want to store the queue depth as well.
  // For example:
  // static constexpr unsigned QUEUE_DEPTH = 64; // or something suitable

public:
  IoUringIOBackend();
  ~IoUringIOBackend();

  int open(const char *filename) override;
  int write(int file_id, const void *data, size_t size) override;
  int read(int file_id, void *data, size_t size) override;
  int close(int file_id) override;
  int seek(int file_id, long offset) override;
  int readAt(int file_id, long offset, void *data, size_t size) override;
  int writeAt(int file_id, long offset, const void *data, size_t size) override;

private:
  int getNextFileHandle();
  // Utility to submit an sqe and wait for the completion (blocking).
  // Returns the res field of the cqe, which is typically #bytes or negative error code.
  int submitAndWait(io_uring_sqe* sqe);
};

#endif // IO_URING_IO_BACKEND_H
