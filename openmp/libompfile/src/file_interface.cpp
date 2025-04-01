#include "debug_log.h"
#include "mpi.h"
#include <atomic>
#include <cassert>
#include <unordered_map>

class OmpFileContext {

private:
  static OmpFileContext *instance;
  MPI_Comm file_comm;
  std::unordered_map<int, MPI_File> file_handle_map;
  std::atomic<int> next_file_handle;

public:
  /* Eventually I will change this to be an agnostic interface
  that can call the underlying backend
  that will be important because I want to use libsdtc++ and we cannot
  link omp to the standard lib*/
  OmpFileContext() {
    int provided = 0;
    // Initialize MPI with thread support
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    // Duplicate MPI_COMM_WORLD into our file_comm for file I/O
    MPI_Comm_dup(MPI_COMM_WORLD, &file_comm);

    // Optional: check what level of thread support was granted
    io_log("MPI_Init_thread provided = %d\n", provided);
  }

  ~OmpFileContext() {
    // Finalize MPI
    MPI_Finalize();
  }

  static OmpFileContext &getInstance() {
    if (instance == nullptr) {
      io_log("Creating new instance of OmpFileContext\n");
      instance = new OmpFileContext();
    }

    return *instance;
  }

  static void finalize() {
    if (instance != nullptr) {
      io_log("Finalizing OmpFileContext\n");
      delete instance;
    }
  }

  int openFile(const char *filename) {
    int file_id = getNextFileHandle();
    MPI_File file_handle;

    io_log("Opening file %s with file_id %d\n", filename, file_id);

    int ret = MPI_File_open(file_comm, filename, MPI_MODE_RDWR, MPI_INFO_NULL,
                            &file_handle);
    if (ret != MPI_SUCCESS) {
      io_log("Error: Could not open file %s\n", filename);
      return -1;
    }
    file_handle_map[file_id] = file_handle;
    return file_id;
  }

  int writeFile(int file_id, const void *data, size_t size) {
    if (file_handle_map.find(file_id) == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }

    io_log("Writing %zu bytes to file %d\n", size, file_id);

    MPI_File file = file_handle_map[file_id];

    int ret = MPI_File_write(file, data, size, MPI_BYTE, MPI_STATUS_IGNORE);

    if (ret != MPI_SUCCESS) {
      io_log("Error: Write failed\n");
      return -1;
    }

    io_log("Write completed\n");

    return 0;
  }

  int closeFile(int file_id) {
    if (file_handle_map.find(file_id) == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }

    io_log("Closing file %d\n", file_id);

    MPI_File file = file_handle_map[file_id];

    int ret = MPI_File_close(&file);

    if (ret != MPI_SUCCESS) {
      io_log("Error: Close failed\n");
      return -1;
    }

    io_log("Close completed\n");

    return 0;
  }

  int readFile(int file_id, void *data, size_t size) {
    if (file_handle_map.find(file_id) == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }

    io_log("Reading %zu bytes from file %d\n", size, file_id);

    MPI_File file = file_handle_map[file_id];

    int ret = MPI_File_read(file, data, size, MPI_BYTE, MPI_STATUS_IGNORE);

    if (ret != MPI_SUCCESS) {
      io_log("Error: Read failed\n");
      return -1;
    }

    io_log("Read completed\n");

    return 0;
  }

  int seekFile(int file_id, long offset) {
    io_log("Setting file pointer to offset %ld\n", offset);

    if (file_handle_map.find(file_id) == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }

    MPI_File file = file_handle_map[file_id];

    int ret = MPI_File_seek(file, offset, MPI_SEEK_SET);

    if (ret != MPI_SUCCESS) {
      io_log("Error: Seek failed\n");
      return -1;
    }

    io_log("Seek completed\n");

    return 0;
  }

  int readFileAt(int file_id, void *data, size_t size, MPI_Offset offset) {
    // Check if the file handle is valid
    if (file_handle_map.find(file_id) == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }

    io_log("Reading %zu bytes from file %d at offset %lld\n",
          size, file_id, static_cast<long long>(offset));

    MPI_File file = file_handle_map[file_id];

    // Perform the actual read at the specified offset
    int ret = MPI_File_read_at(file, offset, data, size, MPI_BYTE, MPI_STATUS_IGNORE);
    if (ret != MPI_SUCCESS) {
      io_log("Error: Read at offset failed\n");
      return -1;
    }

    io_log("Read at offset completed\n");
    return 0;
  }

  int writeFileAt(int file_id, const void *data, size_t size, MPI_Offset offset) {
    // Check if the file handle is valid
    if (file_handle_map.find(file_id) == file_handle_map.end()) {
      io_log("Error: Invalid file handle %d\n", file_id);
      return -1;
    }

  io_log("Writing %zu bytes to file %d at offset %lld\n",
         size, file_id, static_cast<long long>(offset));

  MPI_File file = file_handle_map[file_id];

  // Perform the actual write at the specified offset
  int ret = MPI_File_write_at(file, offset, data, size, MPI_BYTE, MPI_STATUS_IGNORE);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Write at offset failed\n");
    return -1;
  }

  io_log("Write at offset completed\n");
  return 0;
}


private:
  int getNextFileHandle() {
    return next_file_handle.fetch_add(1, std::memory_order_relaxed);
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

int omp_file_pwrite(int file_handle, long offset, const void *data, size_t size, int async) {
  if (async) {
    io_log("Error: Asynchronous writes not supported yet\n");
    return -1;
  }

  auto &ctx = OmpFileContext::getInstance();

  return ctx.writeFileAt(file_handle, data, size, offset);
}

int omp_file_pread(int file_handle, long offset, void *data, size_t size, int async) {
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