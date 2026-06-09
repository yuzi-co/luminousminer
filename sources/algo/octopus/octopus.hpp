#pragma once

#include <cstdint>


namespace algo
{
    namespace octopus
    {
        // Conflux (CFX) Octopus PoW — ethash-family memory-hard hash with an extra
        // prime-field polynomial step (NTT, GF(1032193)) in dataset-item generation.
        // Epoch is derived directly from the block number (NO ethash seedhash chain).
        // Constants verified against Conflux-Chain/open-cfxmine (octopus_params.h, light.cc).

        constexpr uint64_t EPOCH_LENGTH{ 1ull << 19 };  // 524288 blocks
        constexpr uint32_t NUM_DAG_ACCESSES{ 32u };
        constexpr uint32_t MIX_BYTES{ 256u };
        constexpr uint32_t HASH_BYTES{ 64u };
        constexpr uint32_t DAG_ITEM_PARENTS{ 256u };
        constexpr uint32_t CACHE_ROUNDS{ 3u };

        // NTT / polynomial step (dataset-item generation)
        constexpr uint32_t NTT_NK{ 10u };
        constexpr uint32_t NTT_N{ 1u << NTT_NK };  // 1024
        constexpr uint32_t NTT_MOD{ 1032193u };
        constexpr uint32_t NTT_B{ 11u };

        constexpr uint64_t LIGHT_CACHE_INIT_SIZE{ 1ull << 24 };  // 16 MiB
        constexpr uint64_t LIGHT_CACHE_GROWTH{ 1ull << 16 };     // 64 KiB / epoch

        constexpr uint64_t DAG_INIT_SIZE{ 1ull << 32 };  // 4 GiB (>= AMD single-alloc limit)
        constexpr uint64_t DAG_GROWTH{ 1ull << 24 };     // 16 MiB / epoch

        // Size helpers — faithful port of open-cfxmine/src/light.cc.
        uint64_t getEpoch(uint64_t const blockNumber);
        uint64_t getCacheSize(uint64_t const blockNumber);
        uint64_t getDataSize(uint64_t const blockNumber);
    }
}
