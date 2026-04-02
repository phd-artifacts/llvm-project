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
};

typedef struct omp_file_io_hint_v1 {
  uint32_t abi_version;
  uint32_t hint_flags;
  uint64_t epoch_id;
  uint64_t stream_id;
  uint64_t tile_id;
} omp_file_io_hint_v1;

int omp_file_open(const char *filename);

int omp_file_write(int file_handle, const void *data, size_t size, int async);

int omp_file_pwrite(int file_handle, long offset, const void *data, size_t size,
                    int async);

int omp_file_pwrite_hint(int file_handle, long offset, const void *data,
                         size_t size, int async,
                         const omp_file_io_hint_v1 *hint);

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
