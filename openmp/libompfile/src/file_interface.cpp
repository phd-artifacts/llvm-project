#include "mpi.h"
#include <unordered_map>

class OmpFileContext {
private:
  MPI_Comm file_comm;

public:
  /* Eventually I will change this to be an agnostic interface
  that can call the underlying backend
  that will be important because I want to use libdtc++ and we cannot
  link omp to the standard lib*/
  OmpFileContext() {
    int provided = 0;
    // Initialize MPI with thread support
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    // Duplicate MPI_COMM_WORLD into our file_comm for file I/O
    MPI_Comm_dup(MPI_COMM_WORLD, &file_comm);

    // Optional: check what level of thread support was granted
    printf("MPI_Init_thread provided = %d\n", provided);

    // Initialize the file handle map
    auto file_handles = std::unordered_map<int, MPI_File>();
  }

  ~OmpFileContext() {
    // Finalize MPI
    MPI_Finalize();
  }

  void openFile() { printf("Opening file...\n"); }

  // static void initialize() {
  //   if (instance == nullptr) {
  //     instance = new OmpFileContext();
  //   }
  // }

  static void finalize() {
  }
};

extern "C" {

  int agnostic_file_close(int fh) {
    return 0;
  }

  int agnostic_file_open(const char *filename, int flags, int *fh) {
    return 0;

  }

  int agnostic_file_read(int fh, void *buf, size_t count) {
    return 0;
  }

  int agnostic_file_write(int fh, const void *buf, size_t count) {
    return 0;
  }

}
