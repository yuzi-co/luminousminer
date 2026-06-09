# Deploy the XelisHash V3 OpenCL kernel (and the shared BLAKE3 primitive it pulls in) next to
# the binary so the AMD resolver can load them at runtime via kernel/xelishashv3/ (same scheme
# as the other algo kernels). The bit-identity KAT harness that consumes these .cl lives in the
# standalone build (XELISHASHV3_BUILD_OPENCL) and is folded into the main unit_test on the AMD
# path below.
set(OUT_XELISHASHV3 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/xelishashv3)

file(MAKE_DIRECTORY ${OUT_XELISHASHV3})

add_custom_command(
  OUTPUT ${OUT_XELISHASHV3}/xelishashv3.cl
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/xelishashv3.cl ${OUT_XELISHASHV3}/xelishashv3.cl
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/xelishashv3.cl
  COMMENT "Copy xelishashv3.cl")
add_custom_command(
  OUTPUT ${OUT_XELISHASHV3}/blake3.cl
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/../../crypto/opencl/blake3.cl ${OUT_XELISHASHV3}/blake3.cl
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../crypto/opencl/blake3.cl
  COMMENT "Copy blake3.cl (shared) for xelishashv3")

add_custom_target(copy_xelishashv3_opencl_files ALL
  DEPENDS
    ${OUT_XELISHASHV3}/xelishashv3.cl
    ${OUT_XELISHASHV3}/blake3.cl)
