#ifdef __cplusplus

#include <stddef.h>

extern "C" {
#endif

int agnostic_file_close(int fh);

int agnostic_file_open(const char *filename, int flags, int *fh);

int agnostic_file_read(int fh, void *buf, size_t count);

int agnostic_file_write(int fh, const void *buf, size_t count);

#ifdef __cplusplus
}
#endif