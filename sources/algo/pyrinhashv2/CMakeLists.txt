if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

# ============================================================================
# Standalone CPU-reference / OpenCL-KAT build for PyrinHashV2 (PYI).
#
# This is the correctness oracle for the algorithm. It is intentionally
# dependency-free (no Boost/CUDA) so it can be built and tested on a minimal
# toolchain. The PoW hashers are plain BLAKE3 (vendored crypto/reference/blake3),
# the matrix is the kHeavyHash-family 64x64 nibble matrix, and the reduction is
# the V2 nibble-XOR-fold (algo_updated == true).
#
# Build + test in isolation:
#   cmake -S sources/algo/pyrinhashv2 -B build-pyrinhashv2 [-DPYRINHASHV2_BUILD_OPENCL=ON]
#   cmake --build build-pyrinhashv2
#   ctest --test-dir build-pyrinhashv2 --output-on-failure
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(pyrinhashv2_reference CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendored official BLAKE3 C reference (shared crypto primitive).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../crypto/reference/blake3 ${CMAKE_BINARY_DIR}/blake3_ref)

add_library(pyrinhashv2_reference STATIC
    matrix.cpp
    hashers.cpp
    pyrinhashv2.cpp)

# sources include themselves as <algo/pyrinhashv2/...> and <algo/crypto/...>, so the
# include root is sources/.
target_include_directories(pyrinhashv2_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(pyrinhashv2_reference PUBLIC blake3_ref)
target_compile_options(pyrinhashv2_reference PRIVATE -Wall -Wextra)

option(PYRINHASHV2_BUILD_TESTS "Build the PyrinHashV2 known-answer test suite" ON)
if(PYRINHASHV2_BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
    add_executable(pyrinhashv2_unit_test
        tests/pyrinhashv2_test.cpp)
    target_include_directories(pyrinhashv2_unit_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(pyrinhashv2_unit_test PRIVATE pyrinhashv2_reference GTest::gtest GTest::gtest_main)
    add_test(NAME pyrinhashv2_unit_test COMMAND pyrinhashv2_unit_test)
endif()

# OpenCL kernel KAT harness. Runs the .cl (with the shared blake3.cl concatenated ahead of
# it) on any OpenCL device the ICD exposes (POCL/CPU in the dev image, a real GPU otherwise)
# and asserts bit-identity with the CPU reference. Off by default so the pure-CPU build needs
# no ICD.
option(PYRINHASHV2_BUILD_OPENCL "Build the PyrinHashV2 OpenCL known-answer test suite" OFF)
if(PYRINHASHV2_BUILD_OPENCL)
    find_package(GTest REQUIRED)
    find_package(OpenCL REQUIRED)
    enable_testing()
    add_executable(pyrinhashv2_opencl_test opencl/tests/opencl_kat_test.cpp)
    target_include_directories(pyrinhashv2_opencl_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_compile_definitions(pyrinhashv2_opencl_test
        PRIVATE P2_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/opencl/pyrinhashv2.cl"
                P2_BLAKE3_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../crypto/opencl/blake3.cl")
    target_compile_options(pyrinhashv2_opencl_test PRIVATE -Wall -Wextra)
    target_link_libraries(pyrinhashv2_opencl_test
        PRIVATE pyrinhashv2_reference OpenCL::OpenCL GTest::gtest GTest::gtest_main)
    add_test(NAME pyrinhashv2_opencl_test COMMAND pyrinhashv2_opencl_test)
endif()

else()

# ============================================================================
# Integrated into the main miner build. The CPU reference compiles into the
# miner so the AMD resolver can generate the per-job matrix host-side, and the
# CPU known-answer tests fold into the main unit_test target.
# ============================================================================
file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

if (BUILD_EXE_MINER)
    target_sources(${MINER_EXE} PUBLIC ${HEADERS} ${SOURCES})
endif()

if (BUILD_EXE_UNIT_TEST)
    target_sources(${UNIT_TEST_EXE} PUBLIC ${HEADERS} ${SOURCES})
endif()

if (BUILD_EXE_BENCHMARK)
    target_sources(${BENCH_EXE} PUBLIC ${HEADERS} ${SOURCES})
endif()

add_subdirectory(opencl)
add_subdirectory(tests)

set(SOURCES_ALGO_PYRINHASHV2 ${HEADERS} ${SOURCES} PARENT_SCOPE)

endif()
