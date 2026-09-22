# Unit tests of libompfile's pure, header-only units. A plain executable in
# the default target so every runtime build compiles and links it (that alone
# catches a wire-struct or helper signature drift); the build jobs run the
# binary after the build, and `ctest --test-dir <build>/openmp/libompfile`
# runs it by hand. No MPI, no liburing, no libompfile link.

add_executable(ompfile_unit_tests
    ompfile_unit_tests_main.cpp
    sched_wire_test.cpp
    extents_test.cpp
    env_test.cpp
)

target_include_directories(ompfile_unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

set_target_properties(ompfile_unit_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)

add_test(NAME ompfile_unit COMMAND ompfile_unit_tests)
