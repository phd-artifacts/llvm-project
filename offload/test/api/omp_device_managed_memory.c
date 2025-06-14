// RUN: %libomptarget-compile-run-and-check-generic

// UNSUPPORTED: nvptx64-nvidia-cuda-mpi

#include <omp.h>
#include <stdio.h>

void *llvm_omp_target_alloc_shared(size_t, int);
void llvm_omp_target_free_shared(void *, int);

int main() {
  const int N = 64;
  const int device = omp_get_default_device();

  int *shared_ptr = llvm_omp_target_alloc_shared(N * sizeof(int), device);

#pragma omp target teams distribute parallel for device(device)                \
    is_device_ptr(shared_ptr)
  for (int i = 0; i < N; ++i) {
    shared_ptr[i] = 1;
  }

  int sum = 0;
  for (int i = 0; i < N; ++i)
    sum += shared_ptr[i];

  llvm_omp_target_free_shared(shared_ptr, device);
  // CHECK: PASS
  if (sum == N)
    printf("PASS\n");
}
