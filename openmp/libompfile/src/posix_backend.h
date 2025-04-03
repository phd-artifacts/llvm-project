#ifndef POSIX_BACKEND_H
#define POSIX_BACKEND_H

#include "abstract_backend.h"
#include "debug_log.h"

#include <atomic>
#include <cstddef> // for size_t
#include <unordered_map>

class POSIXIOBackend : public IOBackend {
private:
  // Maps our internal "file_id" to an actual file descriptor (fd).
  std::unordered_map<int, int> file_handle_map;
  std::atomic<int> next_file_handle;

public:
  POSIXIOBackend();
  ~POSIXIOBackend();

  int open(const char *filename) override;
  int write(int file_id, const void *data, size_t size) override;
  int read(int file_id, void *data, size_t size) override;
  int close(int file_id) override;
  int seek(int file_id, long offset) override;
  int readAt(int file_id, long offset, void *data, size_t size) override;
  int writeAt(int file_id, long offset, const void *data, size_t size) override;

private:
  int getNextFileHandle();
};

#endif // POSIX_IO_BACKEND_H
