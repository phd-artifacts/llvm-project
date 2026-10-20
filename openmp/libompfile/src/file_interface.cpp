#include "client_context.h"
#include "debug_log.h"
#include "file_interface.h"
#include "ompfile_trace.h"
#include <cstdint>
#include <cstdlib>
#include <mutex>

// One range per public call, on the calling thread (thread-local, so
// push/pop). Payload = the request size where there is one.
#define OMPFILE_API_TRACE(Name, Size)                                          \
  ompfile::trace::Scope OmpFileApiTrace(ompfile::trace::Domain::LibOmpFile,   \
                                        Name, ompfile::trace::Color::Io,       \
                                        static_cast<uint64_t>(Size))

extern "C" {

// Reads cannot be fire-and-forget through the fixed C ABI (there is no
// completion handle to return the buffer through later), so an async-requested
// read is served synchronously. The read paths already drain any in-flight
// async writes first, so the data is coherent. Log the downgrade once.
static void note_async_read_synchronous(int async) {
  if (!async)
    return;
  static std::once_flag once;
  std::call_once(once, [] {
    io_log("Async read requested; served synchronously (no async-read "
           "completion handle in the ABI).\n");
  });
}

int omp_file_open(const char *filename) {
  OMPFILE_API_TRACE("omp_file_open", 0);
  io_trace("omp_file_open api enter filename=%s\n",
           filename ? filename : "(null)");
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.openFile(filename);
  io_trace("omp_file_open api exit rc=%d\n", rc);
  return rc;
}

int omp_file_write(int file_handle, const void *data, size_t size, int async) {
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.submitWrite(file_handle, /*offset=*/0, /*has_offset=*/false, data,
                         size, /*hint=*/nullptr, async != 0);
}

int omp_file_pwrite(int file_handle, long offset, const void *data, size_t size,
                    int async) {
  OMPFILE_API_TRACE("omp_file_pwrite", size);
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.submitWrite(file_handle, offset, /*has_offset=*/true, data, size,
                         /*hint=*/nullptr, async != 0);
}

int omp_file_pwrite_hint(int file_handle, long offset, const void *data,
                         size_t size, int async,
                         const omp_file_io_hint_v1 *hint) {
  OMPFILE_API_TRACE("omp_file_pwrite_hint", size);
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.submitWrite(file_handle, offset, /*has_offset=*/true, data, size,
                         hint, async != 0);
}

int omp_file_pwrite_owned(int file_handle, long offset, void *data,
                          size_t size, void (*release)(void *)) {
  OMPFILE_API_TRACE("omp_file_pwrite_owned", size);
  io_trace("omp_file_pwrite_owned api enter file_handle=%d offset=%ld "
           "size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.submitWriteOwned(file_handle, offset, data, size,
                                      release ? release : &::free,
                                      /*hint=*/nullptr);
  io_trace("omp_file_pwrite_owned api exit rc=%d\n", rc);
  return rc;
}

int omp_file_pwrite_owned_hint(int file_handle, long offset, void *data,
                               size_t size, void (*release)(void *),
                               const omp_file_io_hint_v1 *hint) {
  OMPFILE_API_TRACE("omp_file_pwrite_owned_hint", size);
  io_trace("omp_file_pwrite_owned_hint api enter file_handle=%d offset=%ld "
           "size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.submitWriteOwned(file_handle, offset, data, size,
                                      release ? release : &::free, hint);
  io_trace("omp_file_pwrite_owned_hint api exit rc=%d\n", rc);
  return rc;
}

int omp_file_flush(int file_handle) {
  OMPFILE_API_TRACE("omp_file_flush", 0);
  io_trace("omp_file_flush api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.flushFile(file_handle);
  io_trace("omp_file_flush api exit rc=%d\n", rc);
  return rc;
}

int omp_file_flush_epoch(int file_handle, uint64_t epoch) {
  OMPFILE_API_TRACE("omp_file_flush_epoch", epoch);
  io_trace("omp_file_flush_epoch api enter file_handle=%d epoch=%llu\n",
           file_handle, static_cast<unsigned long long>(epoch));
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.flushFileEpoch(file_handle, epoch);
  io_trace("omp_file_flush_epoch api exit rc=%d\n", rc);
  return rc;
}

int omp_file_commit(int file_handle) {
  OMPFILE_API_TRACE("omp_file_commit", 0);
  io_trace("omp_file_commit api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.commitFile(file_handle);
  io_trace("omp_file_commit api exit rc=%d\n", rc);
  return rc;
}

int omp_file_pread(int file_handle, long offset, void *data, size_t size,
                   int async) {
  OMPFILE_API_TRACE("omp_file_pread", size);
  note_async_read_synchronous(async);
  io_trace("omp_file_pread api enter file_handle=%d offset=%ld size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.readFileAt(file_handle, data, size, offset);
  io_trace("omp_file_pread api exit rc=%d\n", rc);
  return rc;
}

int omp_file_pread_hint(int file_handle, long offset, void *data, size_t size,
                        int async, const omp_file_io_hint_v1 *hint) {
  OMPFILE_API_TRACE("omp_file_pread_hint", size);
  note_async_read_synchronous(async);
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.readFileAtHint(file_handle, data, size, offset, hint);
}

int omp_file_read(int file_handle, void *data, size_t size, int async) {
  note_async_read_synchronous(async);
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.readFile(file_handle, data, size);
}

int omp_file_close(int file_handle) {
  OMPFILE_API_TRACE("omp_file_close", 0);
  io_trace("omp_file_close api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = ctx.closeFile(file_handle);
  io_trace("omp_file_close api exit rc=%d\n", rc);
  return rc;
}

int omp_file_seek(int file_handle, long offset) {
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.seekFile(file_handle, offset);
}

} // extern "C"
