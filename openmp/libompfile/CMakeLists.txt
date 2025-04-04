cmake_minimum_required(VERSION 3.13)
project(libompfile CXX)

# enable debug logs as default for now
# -DOMP_IO_DEBUG
message(STATUS "OpenMP I/O debug logs enabled.")
add_compile_definitions(OMP_IO_DEBUG)

add_library(ompfile SHARED
    src/file_interface.cpp
    src/mpi_io_backend.cpp
    src/posix_backend.cpp
    src/io_uring_io_backend.cpp
)

find_package(MPI REQUIRED)

find_library(LIBURING_LIBRARY NAMES uring REQUIRED)
find_path(LIBURING_INCLUDE_DIR liburing.h REQUIRED)
target_include_directories(ompfile PRIVATE ${LIBURING_INCLUDE_DIR})
target_link_libraries(ompfile PRIVATE ${LIBURING_LIBRARY})


target_link_libraries(ompfile PRIVATE MPI::MPI_CXX)

target_link_libraries(ompfile PRIVATE ${LIBURING_LIBRARY})

target_link_libraries(ompfile PUBLIC omp)

# Let the library's include/ directory be available as a public include path
target_include_directories(ompfile
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(ompfile PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON
)

# Install the library itself
install(TARGETS ompfile
    EXPORT ompfileTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install the public header
install(FILES
    include/file_interface.h
    DESTINATION include
)

# Optionally export the CMake targets if you want a config file 
# for third-party projects to `find_package(ompfile)`
install(EXPORT ompfileTargets
    FILE ompfileConfig.cmake
    NAMESPACE ompfile::
    DESTINATION lib/cmake/ompfile
)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")

# so that I can run from the build folder without having to install
add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/include/file_interface.h"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/include/file_interface.h"
            "${CMAKE_BINARY_DIR}/include/file_interface.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/file_interface.h"
    COMMENT "Copying file_interface.h to build folder"
)

add_custom_target(copy_ompfile_headers ALL
    DEPENDS "${CMAKE_BINARY_DIR}/include/file_interface.h"
)

