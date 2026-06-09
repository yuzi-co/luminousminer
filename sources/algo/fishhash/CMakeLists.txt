file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

# Vendored upstream reference (iron-fish/fish-hash). FishHash.cpp is C++; keccak is
# compiled as a C translation unit (its header carries extern "C" guards). BLAKE3 now
# comes from the shared blake3_ref lib (sources/algo/crypto/reference/blake3) instead
# of a per-algo copy. See reference/FishHash.cpp vendor-patch note.
set(FISHHASH_REFERENCE
    ${CMAKE_CURRENT_SOURCE_DIR}/reference/FishHash.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reference/3rdParty/keccak.c
)

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/reference/3rdParty/keccak.c
    PROPERTIES LANGUAGE C
)

if (BUILD_EXE_MINER)
    target_sources(${MINER_EXE} PUBLIC ${HEADERS} ${SOURCES} ${FISHHASH_REFERENCE})
    target_link_libraries(${MINER_EXE} PUBLIC blake3_ref)
endif()

if (BUILD_EXE_UNIT_TEST)
    target_sources(${UNIT_TEST_EXE} PUBLIC ${HEADERS} ${SOURCES} ${FISHHASH_REFERENCE})
    target_link_libraries(${UNIT_TEST_EXE} PUBLIC blake3_ref)
endif()

if (BUILD_EXE_BENCHMARK)
    target_sources(${BENCH_EXE} PUBLIC ${HEADERS} ${SOURCES} ${FISHHASH_REFERENCE})
    target_link_libraries(${BENCH_EXE} PUBLIC blake3_ref)
endif()

add_subdirectory(opencl)
add_subdirectory(tests)

set(SOURCES_ALGO_FISHHASH ${HEADERS} ${SOURCES} ${FISHHASH_REFERENCE} PARENT_SCOPE)
set(SOURCES_ALGO_FISHHASH_TESTS ${SOURCES_ALGO_FISHHASH_TESTS} PARENT_SCOPE)
