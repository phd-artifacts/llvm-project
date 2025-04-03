#include "posix_backend.h"

#include <cassert>
#include <fcntl.h>    // O_RDWR, etc.
#include <sys/stat.h> // S_IRUSR, etc. if needed
#include <unistd.h>   // close, lseek, read, write, pread, pwrite

POSIXIOBackend::POSIXIOBackend() {
  next_file_handle.store(0, std::memory_order_relaxed);
  io_log("POSIXIOBackend initialized\n");
}

POSIXIOBackend::~POSIXIOBackend() { io_log("POSIXIOBackend destroyed\n"); }

int POSIXIOBackend::getNextFileHandle() {
  // Atomically retrieve and increment the file handle counter
  return next_file_handle.fetch_add(1, std::memory_order_relaxed);
}

int POSIXIOBackend::open(const char *filename) {
  // Get a unique file_id for our map
  int file_id = getNextFileHandle();

  io_log("Opening file %s with file_id %d\n", filename, file_id);

  // Open the file for read/write. Add O_CREAT if you wish to create it if
  // nonexistent.
  int fd = ::open(filename, O_RDWR);
  if (fd < 0) {
    io_log("Error: Could not open file %s\n", filename);
    return -1;
  }

  // Store the (file_id -> fd) mapping
  file_handle_map[file_id] = fd;
  return file_id;
}

int POSIXIOBackend::write(int file_id, const void *data, size_t size) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Writing %zu bytes to file %d\n", size, file_id);

  int fd = it->second;
  ssize_t ret = ::write(fd, data, size);
  if (ret < 0) {
    io_log("Error: Write failed\n");
    return -1;
  }

  io_log("Write completed, bytes written: %zd\n", ret);

  return 0;
}

int POSIXIOBackend::read(int file_id, void *data, size_t size) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Reading %zu bytes from file %d\n", size, file_id);

  int fd = it->second;
  ssize_t ret = ::read(fd, data, size);
  if (ret < 0) {
    io_log("Error: Read failed\n");
    return -1;
  }

  io_log("Read completed, bytes read: %zd\n", ret);

  return 0;
}

int POSIXIOBackend::close(int file_id) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Closing file %d\n", file_id);

  int fd = it->second;
  if (::close(fd) < 0) {
    io_log("Error: Close failed\n");
    return -1;
  }

  // Remove from the map after successfully closing
  file_handle_map.erase(file_id);

  io_log("Close completed\n");

  return 0;
}

int POSIXIOBackend::seek(int file_id, long offset) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Setting file pointer to offset %ld\n", offset);

  int fd = it->second;
  off_t ret = ::lseek(fd, offset, SEEK_SET);
  if (ret == (off_t)-1) {
    io_log("Error: Seek failed\n");
    return -1;
  }

  io_log("Seek completed\n");
  return 0;
}

int POSIXIOBackend::readAt(int file_id, long offset, void *data, size_t size) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Reading %zu bytes from file %d at offset %ld\n", size, file_id,
         offset);

  int fd = it->second;
  // Use pread so we do not alter the file descriptor’s file offset
  ssize_t ret = ::pread(fd, data, size, offset);
  if (ret < 0) {
    io_log("Error: Read at offset failed\n");
    return -1;
  }

  io_log("Read at offset completed, bytes read: %zd\n", ret);
  return 0;
}

int POSIXIOBackend::writeAt(int file_id, long offset, const void *data,
                            size_t size) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Writing %zu bytes to file %d at offset %ld\n", size, file_id, offset);

  int fd = it->second;
  // Use pwrite so we do not alter the file descriptor’s file offset
  ssize_t ret = ::pwrite(fd, data, size, offset);
  if (ret < 0) {
    io_log("Error: Write at offset failed\n");
    return -1;
  }

  io_log("Write at offset completed, bytes written: %zd\n", ret);
  return 0;
}
