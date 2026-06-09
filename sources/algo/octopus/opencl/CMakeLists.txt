set(OUT_OCTOPUS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/octopus)

file(MAKE_DIRECTORY ${OUT_OCTOPUS})

set(OCTOPUS_FILES
  octopus_dag.cl
)

foreach(FILE ${OCTOPUS_FILES})
  add_custom_command(
    OUTPUT ${OUT_OCTOPUS}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_OCTOPUS}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_octopus_opencl_files ALL
  DEPENDS
    ${OUT_OCTOPUS}/octopus_dag.cl
)
