#ifndef MPI_IO_BACKEND_H
#define MPI_IO_BACKEND_H

#include "abstract_backend.h"
#include <atomic>
#include <cstddef> // for size_t
#include <mpi.h>
#include <unordered_map>

class MPIIOBackend : public IOBackend {
private:
  MPI_Comm file_comm;
  int externally_initialized;
  std::unordered_map<int, MPI_File> file_handle_map;
  std::atomic<int> next_file_handle;

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
};

#endif // MPI_IO_BACKEND_H