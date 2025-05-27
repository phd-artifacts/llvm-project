#include "abstract_backend.h"
#include "debug_log.h"
#include "mpi.h"
#include "mpi_io_backend.h"
#include "posix_backend.h"
#include "io_uring_io_backend.h"
#include <atomic>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <cstdlib>

class OmpFileContext {

private:
  static OmpFileContext *instance;
  std::unique_ptr<IOBackend> io_backend;
  std::atomic<int> io_resource_token;

  // RAII guard for IO resource token
  class IOResourceGuard {
    std::atomic<int> &token;
    std::chrono::milliseconds delay{1};
    static constexpr std::chrono::milliseconds max_delay{100};
  public:
    IOResourceGuard(std::atomic<int> &tok) : token(tok) {
      // Acquire a token with exponential backoff when none available
      while (true) {
        int current = token.load(std::memory_order_relaxed);
        if (current > 0) {
          if (token.compare_exchange_strong(current, current - 1,
                                            std::memory_order_acquire)) {
            break;
          }
        } else {
          std::this_thread::sleep_for(delay);
          // Exponential backoff capped at max_delay
          delay = std::min(delay * 2, max_delay);
        }
      }
    }

    ~IOResourceGuard() {
      token.fetch_add(1, std::memory_order_release);
    }

    // disable copying and moving
    IOResourceGuard(const IOResourceGuard &) = delete;
    IOResourceGuard &operator=(const IOResourceGuard &) = delete;
  };

public:
  OmpFileContext(IOBackendTy backend_type)
    : io_resource_token([](){
        const char* env = std::getenv("LIBOMPFILE_IO_TOKENS");
        if (env) {
          try {
            int val = std::stoi(env);
            if (val > 0) return val;
          } catch (...) {}
        }
        // default tokens
        return 4;
      }()) {
    io_log("IO resource slots set to %d\n", io_resource_token.load());

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
      io_log("IO_URING selected\n");
      io_backend = std::make_unique<IoUringIOBackend>();
      break;
    case IOBackendTy::HDF5:
      io_log("HDF5 backend not implemented yet\n");
      break;
    }

    io_log("OmpFileContext constructor called\n");
  }

  ~OmpFileContext() { io_log("Destroying OmpFileContext\n"); }

  static OmpFileContext &getInstance() {
    if (instance == nullptr) {
      io_log("Creating new instance of OmpFileContext\n");

      IOBackendTy backend = IOBackendTy::MPI; // Default
      const char *env = std::getenv("LIBOMPFILE_BACKEND");
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
          io_log("Unknown LIBOMPFILE_BACKEND '%s', defaulting to MPI\n", env);
        }
      } else {
        io_log("LIBOMPFILE_BACKEND not set, defaulting to MPI\n");
      }

      instance = new OmpFileContext(backend);
    }

    return *instance;
  }

  int openFile(const char *filename) {
    IOResourceGuard guard(io_resource_token);
    return io_backend->open(filename);
  }

  int writeFile(int file_handle, const void *data, size_t size) {
    IOResourceGuard guard(io_resource_token);
    return io_backend->write(file_handle, data, size);
  }

  int readFile(int file_handle, void *data, size_t size) {
    IOResourceGuard guard(io_resource_token);
    return io_backend->read(file_handle, data, size);
  }

  int closeFile(int file_handle) {
    IOResourceGuard guard(io_resource_token);
    return io_backend->close(file_handle);
  }

  int seekFile(int file_handle, long offset) {
    IOResourceGuard guard(io_resource_token);
    return io_backend->seek(file_handle, offset);
  }

  int writeFileAt(int file_handle, const void *data, size_t size, long offset) {
    IOResourceGuard guard(io_resource_token);
    return io_backend->writeAt(file_handle, offset, data, size);
  }

  int readFileAt(int file_handle, void *data, size_t size, long offset) {
    IOResourceGuard guard(io_resource_token);
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

inline int acquire_async_not_supported(int async) {
  if (async) {
    io_log("Error: Asynchronous IO not supported yet\n");
    return -1;
  }
  return 0;
}

int omp_file_open(const char *filename) {
  auto &ctx = OmpFileContext::getInstance();
  return ctx.openFile(filename);
}

int omp_file_write(int file_handle, const void *data, size_t size, int async) {
  if (acquire_async_not_supported(async)) return -1;
  auto &ctx = OmpFileContext::getInstance();
  return ctx.writeFile(file_handle, data, size);
}

int omp_file_pwrite(int file_handle, long offset, const void *data, size_t size,
                    int async) {
  if (acquire_async_not_supported(async)) return -1;
  auto &ctx = OmpFileContext::getInstance();
  return ctx.writeFileAt(file_handle, data, size, offset);
}

int omp_file_pread(int file_handle, long offset, void *data, size_t size,
                   int async) {
  if (acquire_async_not_supported(async)) return -1;
  auto &ctx = OmpFileContext::getInstance();
  return ctx.readFileAt(file_handle, data, size, offset);
}

int omp_file_read(int file_handle, void *data, size_t size, int async) {
  if (acquire_async_not_supported(async)) return -1;
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
