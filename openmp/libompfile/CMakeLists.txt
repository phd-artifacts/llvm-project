add_library(
    ompfile STATIC
    src/file_interface.cpp
)
target_include_directories(ompfile PUBLIC include)

target_include_directories(ompfile
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(MPI REQUIRED)
target_link_libraries(ompfile PRIVATE MPI::MPI_CXX)

# Add these lines to export ompfile
install(TARGETS ompfile
  EXPORT LLVMExports  # Match the export name used by LLVM/OpenMP
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)