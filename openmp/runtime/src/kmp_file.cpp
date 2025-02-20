#include "kmp.h" // Include OpenMP runtime headers
#include "file_interface.h"

extern int agnostic_file_close(int fh);

extern "C" { // Ensure C linkage to avoid C++ name mangling
int omp_file_initialize() {
  printf("Initializing OpenMP file support...\n");
  return 0;
}

int omp_file_finalize() {
  printf("Finalizing OpenMP file support...\n");
  return 0;
}

int omp_get_file_support() {
  printf("OpenMP file support enabled.\n");
  return 1; // Dummy function
}

int omp_file_open(const char *filename, int mode) {

  return 0; // Dummy function, always returns 0
}

int omp_file_close(int file_handle_id) {
  agnostic_file_close(file_handle_id);
  return 0; // Dummy function, always returns 0
}

int omp_file_read() {
  return 0; // Dummy function, always returns 0
}

int omp_file_write() {
  return 0; // Dummy function, always returns 0
}
}
