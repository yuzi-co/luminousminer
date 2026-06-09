cmake_minimum_required(VERSION 3.9)
project(LibFishHash CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_OSX_DEPLOYMENT_TARGET 12.0)

add_library(
    FishHash
    STATIC
    FishHash.cpp
    3rdParty/keccak.c
)

# BLAKE3 now comes from the shared vendored reference (sources/algo/crypto/reference/blake3)
# instead of a per-algo copy. blake3_ref's PUBLIC include dir provides blake3.h.
target_link_libraries(FishHash PUBLIC blake3_ref)

install(TARGETS FishHash DESTINATION .)
