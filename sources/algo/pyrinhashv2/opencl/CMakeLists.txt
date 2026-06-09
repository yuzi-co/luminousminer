# Deploy the PyrinHashV2 OpenCL kernel (and the shared BLAKE3 primitive it pulls in) next to
# the binary so the AMD resolver can load them at runtime via kernel/pyrinhashv2/ (same scheme
# as the ethash/progpow/kheavyhash kernels). The bit-identity KAT harness that also consumes
# these .cl lives in the standalone build (PYRINHASHV2_BUILD_OPENCL) and is folded into the
# main unit_test on the AMD path below.
set(OUT_PYRINHASHV2 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/pyrinhashv2)

file(MAKE_DIRECTORY ${OUT_PYRINHASHV2})

add_custom_command(
  OUTPUT ${OUT_PYRINHASHV2}/pyrinhashv2.cl
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/pyrinhashv2.cl ${OUT_PYRINHASHV2}/pyrinhashv2.cl
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/pyrinhashv2.cl
  COMMENT "Copy pyrinhashv2.cl")
add_custom_command(
  OUTPUT ${OUT_PYRINHASHV2}/blake3.cl
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/../../crypto/opencl/blake3.cl ${OUT_PYRINHASHV2}/blake3.cl
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../crypto/opencl/blake3.cl
  COMMENT "Copy blake3.cl (shared) for pyrinhashv2")

add_custom_target(copy_pyrinhashv2_opencl_files ALL
  DEPENDS
    ${OUT_PYRINHASHV2}/pyrinhashv2.cl
    ${OUT_PYRINHASHV2}/blake3.cl)

# Fold the OpenCL PyrinHashV2 KAT into the main unit_test on the AMD path so the bit-identity
# vectors for the shipped `search` kernel run on the actual GPU. The .cl files are loaded at
# runtime; the P2_*_CL_PATH compile-time paths are the source files, with an env override
# resolved in the test itself.
if (BUILD_EXE_UNIT_TEST AND BUILD_AMD)
    target_sources(${UNIT_TEST_EXE} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests/opencl_kat_test.cpp)
    target_include_directories(${UNIT_TEST_EXE} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests
        ${OpenCL_INCLUDE_DIRS})
    target_compile_definitions(${UNIT_TEST_EXE}
        PRIVATE P2_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/pyrinhashv2.cl"
                P2_BLAKE3_CL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../crypto/opencl/blake3.cl")
endif()
