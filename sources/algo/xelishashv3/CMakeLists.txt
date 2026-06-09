if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

# ============================================================================
# Standalone CPU-reference / OpenCL-KAT build for XelisHash V3 (XEL).
#
# Correctness oracle for the algorithm, intentionally dependency-free (no
# Boost/CUDA) so it builds on a minimal toolchain. Stage 1 derives 4 ChaCha8
# keys with BLAKE3 (vendored crypto/reference/blake3) and fills a ~531 KiB
# scratchpad; stage 3 is the branchy 16-way memory-hard mixer (needs 128-bit
# integer math — GCC/Clang __uint128_t); stage 4 is BLAKE3 over the scratchpad.
#
# Build + test in isolation:
#   cmake -S sources/algo/xelishashv3 -B build-xelis [-DXELISHASHV3_BUILD_OPENCL=ON]
#   cmake --build build-xelis
#   ctest --test-dir build-xelis --output-on-failure
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(xelishashv3_reference CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendored official BLAKE3 C reference (shared crypto primitive).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../crypto/reference/blake3 ${CMAKE_BINARY_DIR}/blake3_ref)

add_library(xelishashv3_reference STATIC
    chacha8.cpp
    aes.cpp
    xelishashv3.cpp)

# Sources include themselves as <algo/xelishashv3/...>, so the include root is sources/.
target_include_directories(xelishashv3_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(xelishashv3_reference PUBLIC blake3_ref)
target_compile_options(xelishashv3_reference PRIVATE -Wall -Wextra)

option(XELISHASHV3_BUILD_TESTS "Build the XelisHash V3 known-answer test suite" ON)
if(XELISHASHV3_BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
    add_executable(xelishashv3_unit_test tests/xelishashv3_test.cpp)
    target_include_directories(xelishashv3_unit_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(xelishashv3_unit_test PRIVATE xelishashv3_reference GTest::gtest GTest::gtest_main)
    add_test(NAME xelishashv3_unit_test COMMAND xelishashv3_unit_test)
endif()

# OpenCL kernel KAT harness. Runs the .cl (with the shared blake3.cl concatenated ahead of it)
# on any OpenCL device the ICD exposes (POCL/CPU in the dev image, a real GPU otherwise) and
# asserts bit-identity with the CPU reference + gold. Off by default so the pure-CPU build
# needs no ICD.
option(XELISHASHV3_BUILD_OPENCL "Build the XelisHash V3 OpenCL known-answer test suite" OFF)
if(XELISHASHV3_BUILD_OPENCL)
    find_package(OpenCL REQUIRED)
    enable_testing()

    # The KAT harness needs GoogleTest; the benchmark below does not (plain main + printf), so it
    # can cross-compile on a toolchain without GTest. Build the KAT only when GTest is present.
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_executable(xelishashv3_opencl_test opencl/tests/opencl_kat_test.cpp)
        target_include_directories(xelishashv3_opencl_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_compile_definitions(xelishashv3_opencl_test
            PRIVATE XV3_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/opencl/xelishashv3.cl"
                    XV3_BLAKE3_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../crypto/opencl/blake3.cl")
        target_compile_options(xelishashv3_opencl_test PRIVATE -Wall -Wextra)
        target_link_libraries(xelishashv3_opencl_test
            PRIVATE xelishashv3_reference OpenCL::OpenCL GTest::gtest GTest::gtest_main)
        add_test(NAME xelishashv3_opencl_test COMMAND xelishashv3_opencl_test)
    else()
        message(STATUS "GTest not found: skipping xelishashv3_opencl_test (KAT); building bench only")
    endif()

    # Head-to-head benchmark of the stage-3 128/64 divmod variants (XV3_DIVMOD_IMPL=0..3).
    # Not a ctest: it measures throughput on a real device and prints a table; run by hand.
    # Header-only deps (gold vectors + types); compares GPU digests against the upstream gold
    # vectors, so it needs no CPU reference (and avoids __int128 compiler-rt builtins on clang-cl).
    add_executable(xelishashv3_opencl_bench opencl/tests/opencl_divmod_bench.cpp)
    target_include_directories(xelishashv3_opencl_bench
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/../..)
    target_compile_definitions(xelishashv3_opencl_bench
        PRIVATE XV3_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/opencl/xelishashv3.cl"
                XV3_BLAKE3_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../crypto/opencl/blake3.cl")
    target_compile_options(xelishashv3_opencl_bench PRIVATE -Wall -Wextra)
    target_link_libraries(xelishashv3_opencl_bench PRIVATE OpenCL::OpenCL)
endif()

else()

# ============================================================================
# Integrated into the main miner build. The AMD resolver hashes on the GPU and
# only needs the headers (types/result, via the include path) plus the OpenCL
# kernel deployed next to the binary at runtime — so the integrated build just
# deploys the kernel. The CPU reference and the CPU/OpenCL KATs live in the
# standalone oracle build above (run in CI), keeping the miner build minimal.
# ============================================================================
add_subdirectory(opencl)

set(SOURCES_ALGO_XELISHASHV3 "" PARENT_SCOPE)

endif()
