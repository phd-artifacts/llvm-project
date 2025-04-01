#ifndef FILE_INTERFACE_H
#define FILE_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

int omp_file_open(const char *filename);

int omp_file_write(int file_handle, const void *data, size_t size, int async);

int omp_file_read(int file_handle, void *data, size_t size, int async);

int omp_file_close(int file_handle);

int omp_file_seek(int file_handle, long offset);

#ifdef __cplusplus
}
#endif

#endif // FILE_INTERFACE_H
