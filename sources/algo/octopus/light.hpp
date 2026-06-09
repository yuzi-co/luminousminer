#pragma once

#include <cstdint>
#include <vector>

#include <algo/hash.hpp>


namespace algo
{
    namespace octopus
    {
        // CPU oracle for the Octopus light-cache path — faithful port of
        // open-cfxmine/src/light.cc. Node = 64 bytes = algo::hash512.

        struct LightCache
        {
            std::vector<algo::hash512> nodes{};
            uint64_t                   numNodes{ 0ull };
            uint64_t                   blockNumber{ 0ull };
        };

        LightCache    buildLightCache(uint64_t const blockNumber);
        algo::hash512 calcDatasetItem(LightCache const& light, uint32_t const nodeIndex);

        algo::hash256 octopusHash(algo::hash256 const&  header,
                                  uint64_t const        nonce,
                                  algo::octopus::LightCache const& light,
                                  uint64_t const        fullSize);

        // Convenience: full_size derived from light.blockNumber.
        algo::hash256 octopusLightCompute(algo::hash256 const&  header,
                                          uint64_t const        nonce,
                                          algo::octopus::LightCache const& light);
    }
}
