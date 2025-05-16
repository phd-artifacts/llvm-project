#include "mpi_io_backend.h"
#include "debug_log.h"
#include <atomic>
#include <cassert>
#include <mpi.h>
#include <unordered_map>

MPIIOBackend::MPIIOBackend() {
  int provided = 0;
  int initialized = 0;

  // Check if MPI is already initialized
  MPI_Initialized(&initialized);
  if (!initialized) {
    // Initialize MPI with thread support
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    io_log("MPI_Init_thread called, provided = %d\n", provided);
    externally_initialized = 0;
  } else {
    io_log("MPI already initialized by user program.\n");
    externally_initialized = 1;
  }

  // Duplicate MPI_COMM_WORLD into file_comm for file I/O
  MPI_Comm_dup(MPI_COMM_WORLD, &file_comm);
  io_log("MPI_Comm_dup completed for file I/O.\n");

}

MPIIOBackend::~MPIIOBackend() {
  MPI_Comm_free(&file_comm);
  // if(!externally_initialized) {
  //   // Free the duplicated communicator
  //   io_log("MPI_Comm_free completed.\n");
  // }
}

int MPIIOBackend::open(const char *filename) {
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

int MPIIOBackend::write(int file_id, const void *data, size_t size) {
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

int MPIIOBackend::close(int file_id) {
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

  file_handle_map.erase(file_id);

  io_log("Close completed\n");

  return 0;
}

int MPIIOBackend::read(int file_id, void *data, size_t size) {
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

int MPIIOBackend::seek(int file_id, long offset) {
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

int MPIIOBackend::readAt(int file_id, long offset, void *data, size_t size) {
  // Check if the file handle is valid
  if (file_handle_map.find(file_id) == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Reading %zu bytes from file %d at offset %lld\n", size, file_id,
         static_cast<long long>(offset));

  MPI_File file = file_handle_map[file_id];

  // Perform the actual read at the specified offset
  int ret =
      MPI_File_read_at(file, offset, data, size, MPI_BYTE, MPI_STATUS_IGNORE);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Read at offset failed\n");
    return -1;
  }

  io_log("Read at offset completed\n");
  return 0;
}

int MPIIOBackend::writeAt(int file_id, long offset, const void *data,
                          size_t size) {
  // Check if the file handle is valid
  if (file_handle_map.find(file_id) == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Writing %zu bytes to file %d at offset %lld\n", size, file_id,
         static_cast<long long>(offset));

  MPI_File file = file_handle_map[file_id];

  // Perform the actual write at the specified offset
  int ret =
      MPI_File_write_at(file, offset, data, size, MPI_BYTE, MPI_STATUS_IGNORE);
  if (ret != MPI_SUCCESS) {
    io_log("Error: Write at offset failed\n");
    return -1;
  }

  io_log("Write at offset completed\n");
  return 0;
}

int MPIIOBackend::getNextFileHandle() {
  return next_file_handle.fetch_add(1, std::memory_order_relaxed);
}
