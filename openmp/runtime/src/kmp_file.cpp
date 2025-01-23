#include "kmp.h"  // Include OpenMP runtime headers

extern "C" {  // Ensure C linkage to avoid C++ name mangling
    int omp_get_file_support() {
        return 0;  // Dummy function, always returns 0
    }
}
