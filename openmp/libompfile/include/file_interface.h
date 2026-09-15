#ifndef FILE_INTERFACE_H
#define FILE_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#ifndef OMPFILE_IO_HINT_ABI_VERSION
#define OMPFILE_IO_HINT_ABI_VERSION 1u
#endif

enum omp_file_io_hint_flags {
  OMPFILE_IO_HINT_HAS_EPOCH = 1u << 0,
  OMPFILE_IO_HINT_HAS_STREAM = 1u << 1,
  OMPFILE_IO_HINT_HAS_TILE = 1u << 2,
  OMPFILE_IO_HINT_HAS_ROLE = 1u << 3,
};

enum omp_file_io_hint_role {
  OMPFILE_IO_ROLE_UNKNOWN = 0,
  OMPFILE_IO_ROLE_SOURCE_READ = 1,
  OMPFILE_IO_ROLE_TARGET_READ = 2,
  OMPFILE_IO_ROLE_WRITE = 3,
};

typedef struct omp_file_io_hint_v1 {
  uint32_t abi_version;
  uint32_t hint_flags;
  uint64_t epoch_id;
  uint64_t stream_id;
  uint64_t tile_id;
  uint32_t role;
} omp_file_io_hint_v1;

int omp_file_open(const char *filename);

int omp_file_write(int file_handle, const void *data, size_t size, int async);

int omp_file_pwrite(int file_handle, long offset, const void *data, size_t size,
                    int async);

int omp_file_pwrite_hint(int file_handle, long offset, const void *data,
                         size_t size, int async,
                         const omp_file_io_hint_v1 *hint);

// Feature macro: defined by every runtime whose libompfile exports
// omp_file_pwrite_owned. Guard call sites with
// #ifdef OMPFILE_HAVE_FILE_PWRITE_OWNED when the app must also build against
// older runtime roots.
#define OMPFILE_HAVE_FILE_PWRITE_OWNED 1

// Like omp_file_pwrite with async set, but the runtime takes ownership of
// `data` instead of copying it: the write runs from the caller's buffer in
// place, and once it has completed — succeeded or failed — the runtime calls
// release(data) (free() when release is NULL). The write's own result then
// surfaces the way every queued write's does, through omp_file_flush,
// omp_file_flush_epoch, omp_file_commit or omp_file_close on the handle.
//
// Ownership follows the return value: zero means the runtime has the buffer
// and the caller must not touch or free it again; non-zero means nothing was
// queued, release was not called, and the buffer is still the caller's
// (errno is EBADF when the handle is not open).
//
// The point is the issuing thread: omp_file_pwrite(async=1) spends the
// payload memcpy on the caller's thread before it can return, which on a
// target-region issuer competes with the region's own work. This form spends
// nothing but the enqueue. It fits a buffer that is allocated per write and
// would be freed right after it anyway.
//
// With no async worker available (LIBOMPFILE_ASYNC_DISABLE=1, or an MPI
// thread level below SERIALIZED) the write runs synchronously under the same
// rule: released and zero on success, left with the caller on failure.
int omp_file_pwrite_owned(int file_handle, long offset, void *data,
                          size_t size, void (*release)(void *));

// The owned form of omp_file_pwrite_hint: same ownership contract, carrying
// an I/O hint so the write can be tagged with an epoch and waited on with
// omp_file_flush_epoch.
int omp_file_pwrite_owned_hint(int file_handle, long offset, void *data,
                               size_t size, void (*release)(void *),
                               const omp_file_io_hint_v1 *hint);

// Feature macro: defined by every runtime whose libompfile exports
// omp_file_flush. Applications that must also build against older runtime
// roots should guard their calls with #ifdef OMPFILE_HAVE_FILE_FLUSH.
#define OMPFILE_HAVE_FILE_FLUSH 1

// Wait for the async writes queued for this handle to leave the client-side
// queue, without draining unrelated handles. Returns zero when this handle has
// no outstanding queued write and none of its own writes failed; returns the
// failing write's rc otherwise, or -1 with errno set to EBADF when the handle
// is not open.
//
// This is a queue-completion boundary, not a durability or visibility
// boundary: it issues no fsync, and under proxy write-back staging the bytes
// may still be held in the proxy stage when it returns. Use close (which
// drains every handle and flushes the stage) when another reader must observe
// the data.
int omp_file_flush(int file_handle);

// Feature macro: defined by every runtime whose libompfile exports
// omp_file_flush_epoch.
#define OMPFILE_HAVE_FILE_FLUSH_EPOCH 1

// Same boundary as omp_file_flush, narrowed to the writes tagged with this
// epoch id via omp_file_pwrite_hint's OMPFILE_IO_HINT_HAS_EPOCH. Lets one wave
// of a task graph complete without waiting for the rest of the handle.
//
// Writes issued with no epoch hint belong to the handle alone and are never
// waited on here; omp_file_flush is their boundary. Returns zero when this
// epoch has no outstanding queued write and none of its own writes failed;
// returns the failing write's rc otherwise, or -1 with errno set to EBADF when
// the handle is not open.
//
// Carries the same caveat as omp_file_flush: this is queue completion, not
// durability or visibility.
int omp_file_flush_epoch(int file_handle, uint64_t epoch);

// Feature macro: defined by every runtime whose libompfile exports
// omp_file_commit.
#define OMPFILE_HAVE_FILE_COMMIT 1

// Visibility boundary, as opposed to the queue boundary above. When this
// returns zero, everything written on this handle has reached the source
// filesystem and another rank, handle or process can observe it: queued async
// writes are drained, a proxy write-back stage holding the bytes is pushed to
// the source, and a local backend fdatasyncs the descriptor.
//
// Use omp_file_flush when you only need the caller's buffer back, and this
// when a consumer must read what was written. This one is the expensive call
// of the pair - it belongs at a phase boundary, not inside a write wave.
//
// Returns zero on success, the failing queued write's rc if one of this
// handle's async writes failed, or -1 with errno set to EBADF when the handle
// is not open.
int omp_file_commit(int file_handle);

int omp_file_pread(int file_handle, long offset, void *data, size_t size,
                   int async);

int omp_file_pread_hint(int file_handle, long offset, void *data, size_t size,
                        int async, const omp_file_io_hint_v1 *hint);

int omp_file_read(int file_handle, void *data, size_t size, int async);

int omp_file_close(int file_handle);

int omp_file_seek(int file_handle, long offset);

#ifdef __cplusplus
}
#endif

#endif // FILE_INTERFACE_H
