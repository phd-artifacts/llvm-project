# Source lists of libompfile, relative to openmp/libompfile/. The library's
# CMakeLists.txt builds from these, and out-of-tree consumers that compile
# the backend directly (application/tests/test-mpp-rebalance-fallback-regression
# builds the MPI backend with OMPFILE_ENABLE_TEST_ACCESS) include this file
# instead of keeping a hand-maintained copy, so a new backend source reaches
# every consumer automatically.

set(OMPFILE_FRONTEND_SOURCES_REL
    src/file_interface.cpp
)

set(OMPFILE_MPI_BACKEND_SOURCES_REL
    src/mpi_io_backend_config.cpp
    src/mpi_io_backend_coherence.cpp
    src/mpi_io_backend_file_ops.cpp
    src/mpi_io_backend_forecast_affinity.cpp
    src/mpi_io_backend_handles.cpp
    src/mpi_io_backend_key_utils.cpp
    src/mpi_io_backend_lifecycle.cpp
    src/mpi_io_backend_metrics.cpp
    src/mpi_io_backend_read_path.cpp
    src/mpi_io_backend_remote_read_handles.cpp
    src/mpi_io_backend_two_phase_cache.cpp
    src/mpi_io_backend_two_phase_scheduler.cpp
    src/mpi_io_backend_write_batch_grouping.cpp
    src/mpi_io_backend_write_path.cpp
    src/mpp_shim.cpp
)

set(OMPFILE_LOCAL_BACKEND_SOURCES_REL
    src/posix_backend.cpp
    src/io_uring_io_backend.cpp
)

# ompfile_prefix_sources(<out-var> <prefix> <relative sources...>)
# Writes <prefix>/<source> for each source into <out-var>.
function(ompfile_prefix_sources out prefix)
  set(result)
  foreach(src IN LISTS ARGN)
    list(APPEND result "${prefix}/${src}")
  endforeach()
  set(${out} "${result}" PARENT_SCOPE)
endfunction()
