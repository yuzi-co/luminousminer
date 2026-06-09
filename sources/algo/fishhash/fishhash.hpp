#pragma once

#include <cstdint>

#include <algo/hash.hpp>


namespace algo
{
    namespace fishhash
    {
        // FishHash is a fixed-seed, no-epoch, ethash-like memory-hard PoW (Iron Fish).
        // Reference: iron-fish/fish-hash (cpp/FishHash.cpp), vendored under reference/.
        constexpr uint32_t FNV_PRIME{ 0x01000193u };
        constexpr int32_t  FULL_DATASET_ITEM_PARENTS{ 512 };
        constexpr int32_t  NUM_DATASET_ACCESSES{ 32 };
        constexpr int32_t  LIGHT_CACHE_ROUNDS{ 3 };
        constexpr int32_t  LIGHT_CACHE_NUM_ITEMS{ 1179641 };
        constexpr int32_t  FULL_DATASET_NUM_ITEMS{ 37748717 };

        // Iron Fish block header layout (see ironfish-rust mining::mine).
        constexpr uint64_t HEADER_SIZE{ 180ull };
        constexpr uint64_t RANDOMNESS_OFFSET{ 172ull }; // 8-byte big-endian nonce at [172..180]

        // Fixed seed (no epochs). Matches iron-fish/fish-hash cpp/FishHash.cpp.
        constexpr algo::hash256 SEED{ .ubytes = { 0xeb, 0x01, 0x63, 0xae, 0xf2, 0xab, 0x1c, 0x5a,
                                                  0x66, 0x31, 0x0c, 0x1c, 0x14, 0xd6, 0x0f, 0x42,
                                                  0x55, 0xa9, 0xb3, 0x9b, 0x0e, 0xdf, 0x26, 0x53,
                                                  0x98, 0x44, 0xf1, 0x17, 0xad, 0x67, 0x21, 0x19 } };

        // Opaque handle over the vendored FishHash::fishhash_context (light cache +
        // optional 4.6 GB full dataset). Managed as a process-wide singleton by the
        // reference; freeContext() is a no-op kept for call-site symmetry.
        struct Context;

        // full=false -> ~75 MB light cache only (lazy dataset calc, used by CPU/tests).
        // full=true  -> also allocates the ~4.6 GB full dataset (call prebuildDataset).
        Context* getContext(bool full);
        void     freeContext(Context* ctx);
        void     prebuildDataset(Context* ctx, uint32_t numThreads);

        // Iron Fish hash: blake3(header)->seed512; 32-access mix over dataset; blake3 final.
        // output: 32 bytes. header/headerSize: the 180-byte Iron Fish header.
        void hash(uint8_t* output, Context const* ctx, uint8_t const* header, uint64_t headerSize);

        // Accessors for GPU upload of the light cache (layout-compatible with algo::hash512).
        algo::hash512 const* lightCache(Context const* ctx);
        int32_t              lightCacheNumItems(Context const* ctx);

        // Reference dataset item (128 bytes) for a given index — used to validate the
        // GPU-built DAG. Requires a light (or full) context.
        algo::hash1024 datasetItem(Context const* ctx, uint32_t index);
    }
}
