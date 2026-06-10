file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

# Vendored tevador/RandomX static library (target: randomx).
add_subdirectory(randomx)

# Host PoW wrapper (randomx_pow.*) + result.hpp compiled into each executable, with the
# vendored randomx library linked to whichever exes consume it. Kept in lockstep so a
# target never compiles the wrapper without the library it calls.
if (BUILD_EXE_MINER)
    target_sources(${MINER_EXE} PUBLIC ${HEADERS} ${SOURCES})
    target_link_libraries(${MINER_EXE} PRIVATE randomx)
endif()

if (BUILD_EXE_BENCHMARK)
    target_sources(${BENCH_EXE} PUBLIC ${HEADERS} ${SOURCES})
    target_link_libraries(${BENCH_EXE} PRIVATE randomx)
endif()

if (BUILD_EXE_UNIT_TEST)
    target_sources(${UNIT_TEST_EXE} PUBLIC ${HEADERS} ${SOURCES})
    target_link_libraries(${UNIT_TEST_EXE} PRIVATE randomx)
endif()

add_subdirectory(tests)

set(SOURCES_ALGO_RANDOMX ${HEADERS} ${SOURCES} PARENT_SCOPE)
set(SOURCES_ALGO_RANDOMX_TESTS ${SOURCES_ALGO_RANDOMX_TESTS} PARENT_SCOPE)
