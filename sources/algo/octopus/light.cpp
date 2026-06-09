#include <cstring>

#include <algo/crypto/fnv1.hpp>
#include <algo/keccak.hpp>
#include <algo/octopus/light.hpp>
#include <algo/octopus/multi_eval.hpp>
#include <algo/octopus/octopus.hpp>


namespace
{
    constexpr uint32_t NODE_WORDS{ algo::octopus::HASH_BYTES / 4u };  // 16
    constexpr uint32_t MIX_WORDS{ algo::octopus::MIX_BYTES / 4u };    // 64
    constexpr uint32_t MIX_NODES{ MIX_WORDS / NODE_WORDS };           // 4
}


algo::octopus::LightCache algo::octopus::buildLightCache(uint64_t const blockNumber)
{
    // seed = SHA3-256 chain applied `epoch` times starting from zero (no ethash seed search).
    algo::hash256  seed{};
    uint64_t const epoch{ algo::octopus::getEpoch(blockNumber) };
    for (uint64_t i{ 0ull }; i < epoch; ++i)
    {
        seed = algo::keccak(seed);
    }

    uint64_t const cacheSize{ algo::octopus::getCacheSize(blockNumber) };
    uint64_t const numNodes{ cacheSize / algo::octopus::HASH_BYTES };

    algo::octopus::LightCache light{};
    light.numNodes = numNodes;
    light.blockNumber = blockNumber;
    light.nodes.resize(numNodes);

    light.nodes[0] = algo::keccak<algo::hash512, algo::hash256>(seed);
    for (uint64_t i{ 1ull }; i < numNodes; ++i)
    {
        light.nodes[i] = algo::keccak(light.nodes[i - 1ull]);
    }

    uint32_t const numNodes32{ static_cast<uint32_t>(numNodes) };
    for (uint32_t round{ 0u }; round < algo::octopus::CACHE_ROUNDS; ++round)
    {
        for (uint32_t i{ 0u }; i < numNodes32; ++i)
        {
            uint32_t const       idx{ light.nodes[i].word32[0] % numNodes32 };
            algo::hash512        data{ light.nodes[(numNodes32 - 1u + i) % numNodes32] };
            algo::hash512 const& other{ light.nodes[idx] };
            for (uint32_t w{ 0u }; w < NODE_WORDS; ++w)
            {
                data.word32[w] ^= other.word32[w];
            }
            light.nodes[i] = algo::keccak(data);
        }
    }

    return light;
}


algo::hash512 algo::octopus::calcDatasetItem(algo::octopus::LightCache const& light, uint32_t const nodeIndex)
{
    uint32_t const numParent{ static_cast<uint32_t>(light.numNodes) };
    algo::hash512  ret{ light.nodes[nodeIndex % numParent] };
    ret.word32[0] ^= nodeIndex;
    ret = algo::keccak(ret);
    for (uint32_t i{ 0u }; i < algo::octopus::DAG_ITEM_PARENTS; ++i)
    {
        uint32_t const       parentIndex{ algo::fnv1(nodeIndex ^ i, ret.word32[i % NODE_WORDS]) % numParent };
        algo::hash512 const& parent{ light.nodes[parentIndex] };
        for (uint32_t w{ 0u }; w < NODE_WORDS; ++w)
        {
            ret.word32[w] = algo::fnv1(ret.word32[w], parent.word32[w]);
        }
    }
    return algo::keccak(ret);
}


algo::hash256 algo::octopus::octopusHash(algo::hash256 const&             header,
                                         uint64_t const                  nonce,
                                         algo::octopus::LightCache const& light,
                                         uint64_t const                  fullSize)
{
    std::pair<uint64_t, std::vector<uint32_t>> const ev{ algo::octopus::multiEval(header, nonce) };
    uint64_t const                                   threadResult{ ev.first };
    std::vector<uint32_t> const&                     result{ ev.second };

    // s_mix[0] = seed node; mix = s_mix[1..MIX_NODES] is a contiguous MIX_WORDS array.
    algo::hash512 sMix[MIX_NODES + 1u]{};
    std::memcpy(sMix[0].ubytes, header.ubytes, 32);
    sMix[0].word64[4] = threadResult;
    {
        algo::hash512 seeded{};
        algo::keccak(seeded.word64, 512u, sMix[0].ubytes, 40u);
        sMix[0] = seeded;
    }

    uint32_t* const mix{ reinterpret_cast<uint32_t*>(&sMix[1]) };
    for (uint32_t w{ 0u }; w < MIX_WORDS; ++w)
    {
        mix[w] = sMix[0].word32[w % NODE_WORDS];
    }

    uint32_t const pageSize{ 4u * MIX_WORDS };  // 256
    uint32_t const numFullPages{ static_cast<uint32_t>(fullSize / pageSize) };

    for (uint32_t i{ 0u }; i < algo::octopus::NUM_DAG_ACCESSES; ++i)
    {
        uint32_t const index{ algo::fnv1(sMix[0].word32[0] ^ i ^ result[i], mix[i % MIX_WORDS]) % numFullPages };
        for (uint32_t n{ 0u }; n < MIX_NODES; ++n)
        {
            algo::hash512 const dagNode{ algo::octopus::calcDatasetItem(light, index * MIX_NODES + n) };
            uint32_t* const     mixNode{ mix + n * NODE_WORDS };
            for (uint32_t w{ 0u }; w < NODE_WORDS; ++w)
            {
                mixNode[w] = algo::fnv1(mixNode[w], dagNode.word32[w]);
            }
        }
    }

    for (uint32_t w{ 0u }; w < MIX_WORDS; w += 4u)
    {
        uint32_t reduction{ mix[w] };
        reduction = algo::fnv1(reduction, mix[w + 1u]);
        reduction = algo::fnv1(reduction, mix[w + 2u]);
        reduction = algo::fnv1(reduction, mix[w + 3u]);
        mix[w / 4u] = reduction;
    }
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        mix[i] = algo::fnv1(mix[i], mix[8u + i]);
    }

    algo::hash256 out{};
    algo::keccak(out.word64, 256u, sMix[0].ubytes, 96u);
    return out;
}


algo::hash256 algo::octopus::octopusLightCompute(algo::hash256 const&             header,
                                                 uint64_t const                  nonce,
                                                 algo::octopus::LightCache const& light)
{
    uint64_t const fullSize{ algo::octopus::getDataSize(light.blockNumber) };
    return algo::octopus::octopusHash(header, nonce, light, fullSize);
}
