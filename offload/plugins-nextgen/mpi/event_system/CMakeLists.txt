# Build EventSystem
add_library(EventSystem OBJECT
    EventSystem.cpp
)

# libompfile's headers carry the scheduler wire types and the bridge ABI
# (ompfile_sched.h, ompfile_mpp_abi.h); PUBLIC so the plugin and the proxy
# inherit the path instead of reaching them by a relative include.
set(OMPFILE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../openmp/libompfile/include)
target_include_directories(EventSystem PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OMPFILE_INCLUDE_DIR}
    ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
    ${LIBOMPTARGET_INCLUDE_DIR}
)

target_link_libraries(EventSystem PRIVATE
  MPI::MPI_CXX
  LLVMSupport
)

target_compile_options(EventSystem PUBLIC ${offload_compile_flags})
target_link_options(EventSystem PUBLIC ${offload_link_flags})

set_target_properties(EventSystem PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set C++20 as the target standard for this plugin.
set_target_properties(EventSystem
                      PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED ON)

target_compile_definitions(EventSystem PRIVATE
                           DEBUG_PREFIX="EventSystem")