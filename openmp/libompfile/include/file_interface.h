#ifndef FILE_INTERFACE_H
#define FILE_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

// Example function to open a file
int omp_file_open(const char* filename);

// Example function to write data
int omp_file_write(const void* data, size_t size);

// Example function to close
int omp_file_close();

#ifdef __cplusplus
}
#endif

#endif // FILE_INTERFACE_H
