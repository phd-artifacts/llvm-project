#include "io_uring_io_backend.h"

#include <fcntl.h>     // O_RDWR, etc.
#include <unistd.h>    // read, write, close
#include <cstring>     // strerror
#include <errno.h>
#include <sys/stat.h>  // mode flags if needed
#include <cstdio>      // for printing errors
#include <cassert>

static constexpr unsigned QUEUE_DEPTH = 64;

IoUringIOBackend::IoUringIOBackend() {
  // Initialize our atomic file handle counter
  next_file_handle.store(0, std::memory_order_relaxed);

  // Initialize the io_uring instance
  int ret = io_uring_queue_init(QUEUE_DEPTH, &ring, 0);
  if (ret < 0) {
    io_log("io_uring_queue_init failed: %s\n", strerror(-ret));
    // You may want to handle this error more gracefully
    // or throw an exception if your environment supports it.
  } else {
    io_log("IoUringIOBackend initialized with queue depth %u\n", QUEUE_DEPTH);
  }
}

IoUringIOBackend::~IoUringIOBackend() {
  // Teardown the ring
  io_uring_queue_exit(&ring);
  io_log("IoUringIOBackend destroyed\n");
}

int IoUringIOBackend::getNextFileHandle() {
  return next_file_handle.fetch_add(1, std::memory_order_relaxed);
}

int IoUringIOBackend::open(const char *filename) {
  int file_id = getNextFileHandle();
  io_log("Opening file %s with file_id %d\n", filename, file_id);

  // Plain old POSIX open; you could also do async open with io_uring,
  // but that’s more complicated. 
  // For demonstration, we do standard open + store the fd in the map:
  int fd = ::open(filename, O_RDWR);
  if (fd < 0) {
    io_log("Error: Could not open file %s (%s)\n", filename, strerror(errno));
    return -1;
  }

  // Insert into our map
  FileData fd_data{fd, 0 /* current offset = 0 */};
  file_handle_map[file_id] = fd_data;

  return file_id;
}

int IoUringIOBackend::close(int file_id) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Closing file %d\n", file_id);

  // For demonstration, we do a “normal” close.
  // If you want to do it via io_uring, you’d do:
  //   io_uring_sqe* sqe = io_uring_get_sqe(&ring);
  //   io_uring_prep_close(sqe, it->second.fd);
  //   int ret = submitAndWait(sqe);
  //   ...
  int fd = it->second.fd;
  if (::close(fd) < 0) {
    io_log("Error: close failed on fd=%d (%s)\n", fd, strerror(errno));
    return -1;
  }

  file_handle_map.erase(it);
  io_log("Close completed\n");

  return 0;
}

// Submits a prepared sqe to io_uring and blocks until the cqe is ready.
// Returns the "res" field from the cqe (>= 0 is success, < 0 is -errno).
int IoUringIOBackend::submitAndWait(io_uring_sqe* sqe) {
  // Actually submit the SQE
  int submit_ret = io_uring_submit(&ring);
  if (submit_ret < 0) {
    io_log("Error: io_uring_submit failed: %s\n", strerror(-submit_ret));
    return submit_ret;
  }

  // Wait for completion
  io_uring_cqe* cqe = nullptr;
  int wait_ret = io_uring_wait_cqe(&ring, &cqe);
  if (wait_ret < 0) {
    io_log("Error: io_uring_wait_cqe failed: %s\n", strerror(-wait_ret));
    return wait_ret;
  }

  int result = cqe->res;
  io_uring_cqe_seen(&ring, cqe);

  return result;
}

// “Synchronous” write that calls io_uring_prep_write and blocks for its completion.
int IoUringIOBackend::write(int file_id, const void *data, size_t size) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  int fd = it->second.fd;
  off_t offset = it->second.offset;

  io_log("Writing %zu bytes to file %d at offset %lld\n",
         size, file_id, (long long)offset);

  // Acquire an SQE
  io_uring_sqe* sqe = io_uring_get_sqe(&ring);
  if (!sqe) {
    io_log("Error: could not get SQE (queue full?)\n");
    return -1;
  }

  // Prepare a write request at the file’s current offset
  // Then advance the stored offset by size.
  io_uring_prep_write(sqe, fd, data, size, offset);

  // Submit and wait
  int res = submitAndWait(sqe);
  if (res < 0) {
    io_log("Error: Write failed: %s\n", strerror(-res));
    return -1;
  }

  // On success, res is the number of bytes actually written
  io_log("Write completed, bytes written: %d\n", res);
  it->second.offset += res;

  return 0;
}

// “Synchronous” read using io_uring_prep_read
int IoUringIOBackend::read(int file_id, void *data, size_t size) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  int fd = it->second.fd;
  off_t offset = it->second.offset;

  io_log("Reading %zu bytes from file %d at offset %lld\n",
         size, file_id, (long long)offset);

  io_uring_sqe* sqe = io_uring_get_sqe(&ring);
  if (!sqe) {
    io_log("Error: could not get SQE\n");
    return -1;
  }

  io_uring_prep_read(sqe, fd, data, size, offset);

  int res = submitAndWait(sqe);
  if (res < 0) {
    io_log("Error: Read failed: %s\n", strerror(-res));
    return -1;
  }

  io_log("Read completed, bytes read: %d\n", res);
  it->second.offset += res;

  return 0;
}

int IoUringIOBackend::seek(int file_id, long offset) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }
  io_log("Setting file pointer to offset %ld\n", offset);
  // We just store the offset in our structure. There's no real "seek" in
  // io_uring since we do offset-based I/O calls. But if we want to emulate 
  // "moving the file pointer," we can do:
  it->second.offset = offset;
  io_log("Seek completed\n");
  return 0;
}

// “Synchronous” read at a specific offset
int IoUringIOBackend::readAt(int file_id, long offset, void *data, size_t size) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Reading %zu bytes from file %d at offset %ld\n", size, file_id, offset);

  int fd = it->second.fd;
  io_uring_sqe* sqe = io_uring_get_sqe(&ring);
  if (!sqe) {
    io_log("Error: could not get SQE\n");
    return -1;
  }

  io_uring_prep_read(sqe, fd, data, size, offset);

  int res = submitAndWait(sqe);
  if (res < 0) {
    io_log("Error: ReadAt failed: %s\n", strerror(-res));
    return -1;
  }

  io_log("Read at offset completed, bytes read: %d\n", res);
  return 0;
}

// “Synchronous” write at a specific offset
int IoUringIOBackend::writeAt(int file_id, long offset, const void *data, size_t size) {
  auto it = file_handle_map.find(file_id);
  if (it == file_handle_map.end()) {
    io_log("Error: Invalid file handle %d\n", file_id);
    return -1;
  }

  io_log("Writing %zu bytes to file %d at offset %ld\n", size, file_id, offset);

  int fd = it->second.fd;
  io_uring_sqe* sqe = io_uring_get_sqe(&ring);
  if (!sqe) {
    io_log("Error: could not get SQE\n");
    return -1;
  }

  io_uring_prep_write(sqe, fd, data, size, offset);

  int res = submitAndWait(sqe);
  if (res < 0) {
    io_log("Error: WriteAt failed: %s\n", strerror(-res));
    return -1;
  }

  io_log("Write at offset completed, bytes written: %d\n", res);
  return 0;
}
