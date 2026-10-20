# Source lists of libompfile, relative to openmp/libompfile/. The library's
# CMakeLists.txt builds from these, and out-of-tree consumers that compile
# the backend directly (application/tests/test-mpp-rebalance-fallback-regression
# builds the MPP backend with OMPFILE_ENABLE_TEST_ACCESS) include this file
# instead of keeping a hand-maintained copy, so a new backend source reaches
# every consumer automatically.

# The frontend list is the client side (C ABI, client context, scheduler
# clients); the rebalance regression compiles only the backend list.
set(OMPFILE_FRONTEND_SOURCES_REL
    src/file_interface.cpp
    src/client_context.cpp
    src/io_scheduler.cpp
)

set(OMPFILE_MPP_BACKEND_SOURCES_REL
    src/mpp_backend_config.cpp
    src/mpp_backend_coherence.cpp
    src/mpp_backend_file_ops.cpp
    src/mpp_backend_handles.cpp
    src/mpp_backend_key_utils.cpp
    src/mpp_backend_lifecycle.cpp
    src/mpp_backend_metrics.cpp
    src/mpp_backend_read_path.cpp
    src/mpp_backend_remote_read_handles.cpp
    src/mpp_backend_two_phase_scheduler.cpp
    src/mpp_backend_write_batch_grouping.cpp
    src/mpp_backend_write_path.cpp
    src/mpp_shim.cpp
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
