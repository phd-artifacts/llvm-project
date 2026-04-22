#ifndef OMPFILE_ABSTRACT_BACKEND_H
#define OMPFILE_ABSTRACT_BACKEND_H

#include "ompfile_sched.h"
#include <stddef.h>

enum class IOBackendTy {
  MPI, /* only one implemented, default*/
  POSIX,
  IO_URING,
  HDF5,
};

class IOBackend {
public:
  virtual ~IOBackend() {}

  virtual int open(const char *filename) = 0;
  virtual int openWithPlan(const char *filename,
                           const ompfile::OmpFileIOPlan *plan) {
    return open(filename);
  }
  virtual int write(int file_id, const void *data, size_t size) = 0;
  virtual int read(int file_id, void *data, size_t size) = 0;
  virtual int close(int file_id) = 0;
  virtual int seek(int file_id, long offset) = 0;
  virtual int readAt(int file_id, long offset, void *data, size_t size) = 0;
  virtual int readAtWithContext(const ompfile::OmpFileReadRequestContext &ctx,
                                void *data, size_t size) {
    return readAt(ctx.FileHandle, static_cast<long>(ctx.Offset), data, size);
  }
  virtual int writeAtWithContext(const ompfile::OmpFileWriteRequestContext &ctx,
                                 const void *data, size_t size) {
    return writeAt(ctx.FileHandle, static_cast<long>(ctx.Offset), data, size);
  }
  virtual int writeAt(int file_id, long offset, const void *data,
                      size_t size) = 0;
};

#endif // OMPFILE_ABSTRACT_BACKEND_H
