#include "mpi.h"
#include "debug_log.h"
#include <atomic>
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

  static OmpFileContext& getInstance() {
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
    int ret = MPI_File_open(file_comm, filename, MPI_MODE_RDWR, MPI_INFO_NULL,
                            &file_handle);
    if (ret != MPI_SUCCESS) {
      io_log("Error: Could not open file %s\n", filename);
      return -1;
    }
    file_handle_map[file_id] = file_handle;
    return file_id;
  }

private:
  int getNextFileHandle() {
    return next_file_handle.fetch_add(1, std::memory_order_relaxed);
  }
};

OmpFileContext* OmpFileContext::instance = nullptr;

extern "C" {

int omp_file_open(const char *filename) { 

  auto& ctx = OmpFileContext::getInstance();
  auto file_id = ctx.openFile(filename);
  return file_id;

}

int omp_file_write(const void *data, size_t size) { return 0; }

int omp_file_close() { return 0; }

} // extern "C"