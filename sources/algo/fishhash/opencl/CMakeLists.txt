set(OUT_FISHHASH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/fishhash)

file(MAKE_DIRECTORY ${OUT_FISHHASH})

set(FISHHASH_FILES
  fishhash_dag.cl
  fishhash_result.cl
  fishhash_search.cl
)

foreach(FILE ${FISHHASH_FILES})
  add_custom_command(
    OUTPUT ${OUT_FISHHASH}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_FISHHASH}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_fishhash_opencl_files ALL
  DEPENDS
    ${OUT_FISHHASH}/fishhash_dag.cl
    ${OUT_FISHHASH}/fishhash_result.cl
    ${OUT_FISHHASH}/fishhash_search.cl
)
