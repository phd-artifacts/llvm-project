#include "abstract_backend.h"
#include "debug_log.h"
#include "mpi.h"
#include "mpi_io_backend.h"
#include "posix_backend.h"
#include <atomic>
#include <cassert>
#include <memory>
#include <unordered_map>

class OmpFileContext {

private:
  static OmpFileContext *instance;
  std::unique_ptr<IOBackend> io_backend;

public:
  OmpFileContext(IOBackendTy backend_type) {
    switch (backend_type) {
    case IOBackendTy::MPI:
      io_log("MPI backend selected\n");
      io_backend = std::make_unique<MPIIOBackend>();
      break;
    case IOBackendTy::POSIX:
      io_log("POSIX backend selected\n");
      io_backend = std::make_unique<POSIXIOBackend>();
      break;
    case IOBackendTy::IO_URING:
      io_log("IO_URING backend not implemented yet\n");
      break;
    case IOBackendTy::HDF5:
      io_log("HDF5 backend not implemented yet\n");
      break;
    default:
      io_log("Unknown backend type, defaulting to MPI\n");
      io_backend = std::make_unique<MPIIOBackend>();
      break;
    }
    io_log("OmpFileContext constructor called\n");
  }

  ~OmpFileContext() { io_log("Destroying OmpFileContext\n"); }

  static OmpFileContext &getInstance() {
    if (instance == nullptr) {
      io_log("Creating new instance of OmpFileContext\n");

      const char *env = std::getenv("LIBOMPFILE_BACKEND");
      IOBackendTy backend = IOBackendTy::MPI; // Default

      if (env) {
        std::string envStr(env);
        if (envStr == "MPI") {
          backend = IOBackendTy::MPI;
        } else if (envStr == "POSIX") {
          backend = IOBackendTy::POSIX;
        } else if (envStr == "IO_URING") {
          backend = IOBackendTy::IO_URING;
        } else if (envStr == "HDF5") {
          backend = IOBackendTy::HDF5;
        } else {
          io_log("Unknown LIBOMPFILE_BACKEND value, defaulting to MPI\n");
          io_log("Valid options are: MPI, POSIX, IO_URING, HDF5\n");
        }
      } else {
        io_log("LIBOMPFILE_BACKEND not set, defaulting to MPI\n");
      }

      instance = new OmpFileContext(backend);
    }

    return *instance;
  }

  int openFile(const char *filename) { return io_backend->open(filename); }
  int writeFile(int file_handle, const void *data, size_t size) {
    return io_backend->write(file_handle, data, size);
  }
  int readFile(int file_handle, void *data, size_t size) {
    return io_backend->read(file_handle, data, size);
  }
  int closeFile(int file_handle) { return io_backend->close(file_handle); }

  int seekFile(int file_handle, long offset) {
    return io_backend->seek(file_handle, offset);
  }

  int writeFileAt(int file_handle, const void *data, size_t size, long offset) {
    return io_backend->writeAt(file_handle, offset, data, size);
  }
  int readFileAt(int file_handle, void *data, size_t size, long offset) {
    return io_backend->readAt(file_handle, offset, data, size);
  }
  int getFileHandle(int file_handle) { return file_handle; }

  static void finalize() {
    if (instance != nullptr) {
      io_log("Finalizing OmpFileContext\n");
      delete instance;
    }
  }
};

OmpFileContext *OmpFileContext::instance = nullptr;

extern "C" {

int omp_file_open(const char *filename) {

  auto &ctx = OmpFileContext::getInstance();
  auto file_id = ctx.openFile(filename);
  return file_id;
}

int omp_file_write(int file_handle, const void *data, size_t size, int async) {
  if (async) {
    io_log("Error: Asynchronous writes not supported yet\n");
    return -1;
  }

  auto &ctx = OmpFileContext::getInstance();

  return ctx.writeFile(file_handle, data, size);
}

int omp_file_pwrite(int file_handle, long offset, const void *data, size_t size,
                    int async) {
  if (async) {
    io_log("Error: Asynchronous writes not supported yet\n");
    return -1;
  }

  auto &ctx = OmpFileContext::getInstance();

  return ctx.writeFileAt(file_handle, data, size, offset);
}

int omp_file_pread(int file_handle, long offset, void *data, size_t size,
                   int async) {
  if (async) {
    io_log("Error: Asynchronous reads not supported yet\n");
    return -1;
  }

  auto &ctx = OmpFileContext::getInstance();

  return ctx.readFileAt(file_handle, data, size, offset);
}

int omp_file_read(int file_handle, void *data, size_t size, int async) {
  if (async) {
    io_log("Error: Asynchronous reads not supported yet\n");
    return -1;
  }

  auto &ctx = OmpFileContext::getInstance();

  return ctx.readFile(file_handle, data, size);
}

int omp_file_close(int file_handle) {
  auto &ctx = OmpFileContext::getInstance();

  return ctx.closeFile(file_handle);
}

int omp_file_seek(int file_handle, long offset) {
  auto &ctx = OmpFileContext::getInstance();

  return ctx.seekFile(file_handle, offset);
}

} // extern "C"