#ifndef OMPFILE_IO_SCHEDULER_H
#define OMPFILE_IO_SCHEDULER_H

#include "abstract_backend.h"
#include "ompfile_config.h"
#include "ompfile_sched.h"
#include <cstddef>
#include <memory>

// The scheduler clients OmpFileClientContext routes every call through.
// LocalScheduler calls the backend directly; HeadnodeScheduler asks the
// headnode first. Both are defined in io_scheduler.cpp.
class IOScheduler {
public:
  virtual ~IOScheduler() {}
  virtual int open(const char *filename) = 0;
  virtual int write(int file_handle, const void *data, size_t size) = 0;
  virtual int read(int file_handle, void *data, size_t size) = 0;
  virtual int close(int file_handle) = 0;
  virtual int commit(int file_handle) = 0;
  virtual int seek(int file_handle, long offset) = 0;
  virtual int readAt(int file_handle, long offset, void *data, size_t size) = 0;
  virtual int readAtHint(int file_handle, long offset, void *data, size_t size,
                         const ompfile::OmpFileIOHint *hint) = 0;
  virtual int writeAt(int file_handle, long offset, const void *data,
                      size_t size) = 0;
  virtual int writeAtHint(int file_handle, long offset, const void *data,
                          size_t size,
                          const ompfile::OmpFileIOHint *hint) = 0;
};

std::unique_ptr<IOScheduler> createLocalScheduler(IOBackend &backend);
std::unique_ptr<IOScheduler>
createHeadnodeScheduler(IOBackend &backend,
                        const ompfile::config::ClientConfig &config);

#endif // OMPFILE_IO_SCHEDULER_H
