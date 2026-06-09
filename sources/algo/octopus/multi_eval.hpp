#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <algo/hash.hpp>
#include <algo/octopus/octopus.hpp>


namespace algo
{
    namespace octopus
    {
        // The distinctive Octopus step: a siphash-seeded polynomial (NTT_N coeffs over
        // GF(NTT_MOD)) evaluated at points derived from the header params (a,b,c,w).
        // Its output `result` vector feeds the DAG access indices in the hashimoto loop.
        // Faithful port of open-cfxmine/src/light.cc (compute_d / OctopusABCW / multi_eval).

        struct Abcw
        {
            uint32_t a{ 0u };
            uint32_t b{ 0u };
            uint32_t c{ 0u };
            uint32_t w{ 0u };
        };

        Abcw makeAbcw(algo::hash256 const& header);

        // Fills d with NTT_N coefficients (warp-interleaved layout d[i*WARP_SIZE + lid]).
        void computeD(algo::hash256 const& header, uint64_t nonce, uint32_t* const d);

        // Returns (threadResult, result) where result has DATA_PER_THREAD entries.
        std::pair<uint64_t, std::vector<uint32_t>> multiEval(algo::hash256 const& header, uint64_t const nonce);
    }
}
