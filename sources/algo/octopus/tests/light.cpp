#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/octopus/light.hpp>
#include <algo/octopus/octopus.hpp>


// Expected values captured bit-exactly from Conflux-Chain/open-cfxmine
// (src/light.cc), see reference_vectors.md. Header = bytes 0x00..0x1f.

namespace
{
    std::string toHex(uint8_t const* const p, uint32_t const n)
    {
        static char const* const digits{ "0123456789abcdef" };
        std::string              s{};
        s.reserve(n * 2u);
        for (uint32_t i{ 0u }; i < n; ++i)
        {
            s += digits[p[i] >> 4];
            s += digits[p[i] & 0x0fu];
        }
        return s;
    }

    // Same (self-consistent) FNV-1a-64 constants as the capture harness.
    uint64_t fnv1a64(uint8_t const* const p, uint64_t const n)
    {
        uint64_t h{ 1469598103934665603ull };
        for (uint64_t i{ 0ull }; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }
}


struct OctopusLightTest : public testing::Test
{
    algo::hash256             header{};
    algo::octopus::LightCache light{};

    OctopusLightTest()
    {
        for (uint8_t i{ 0u }; i < 32u; ++i)
        {
            header.ubytes[i] = i;
        }
        light = algo::octopus::buildLightCache(0ull);
    }
    ~OctopusLightTest() = default;
};


TEST_F(OctopusLightTest, lightCacheEpoch0)
{
    uint64_t const cacheSize{ light.numNodes * algo::octopus::HASH_BYTES };
    ASSERT_EQ(cacheSize, 16776896ull);

    uint8_t const* const cache{ light.nodes[0].ubytes };
    EXPECT_EQ(fnv1a64(cache, cacheSize), 0xd36a5e79fad00334ull);
    EXPECT_EQ(toHex(cache, 64),
              "5e493e76a1318e50815c6ce77950425532964ebbb8dcf94718991fa9a82eaf37"
              "658de68ca6fe078884e803da3a26a4aa56420a6867ebcd9ab0f29b08d1c48fed");
    EXPECT_EQ(toHex(cache + cacheSize - 64, 64),
              "724f2f86c24c487809dc3897acbbd32d5d791e4536aa1520e65e93891a40dde5"
              "887899ffc556cbd174f426e32ae2ab711be859601c024d1514b29a27370b662e");
}


TEST_F(OctopusLightTest, datasetItemsEpoch0)
{
    EXPECT_EQ(toHex(algo::octopus::calcDatasetItem(light, 0u).ubytes, 64),
              "22db2229cc516c46d2210086f1ab417e0bd1c3827c5ecc6af7d3a33f8dae332b"
              "ab5aa31fc58e71cff27666e81bf418775e74839743ca9d410fdf514d009bcec2");
    EXPECT_EQ(toHex(algo::octopus::calcDatasetItem(light, 1u).ubytes, 64),
              "e5263184c4985ca0570d1ebdf507049e427dc86c7e96485739c0960a2ce4e6eb"
              "386d5aa39471876225c23c5b69443f6d5db8120fe3204cedcfefd0347f69ec1d");
    EXPECT_EQ(toHex(algo::octopus::calcDatasetItem(light, 1000u).ubytes, 64),
              "8e6094037ad186a0fde024e3ef505627e2aae8a6ffdcb8f2fd3b6a85b654db8f"
              "c3144afa2b98e5d8a45f94b0a3521cf04accaec4298b9274d4f0de7d802bed71");
    EXPECT_EQ(toHex(algo::octopus::calcDatasetItem(light, 1234567u).ubytes, 64),
              "c0c58b2a628da8005001fced0e7556e20c0f06cdd1174f541c21a717ddf05dd3"
              "bfeb04455c50ac49a7bb4fb2f1c51f5c4fc1e85ff21e48a9717032cd617b6e69");
}


TEST_F(OctopusLightTest, fullHashEpoch0)
{
    algo::hash256 const h0{ algo::octopus::octopusLightCompute(header, 0ull, light) };
    EXPECT_EQ(toHex(h0.ubytes, 32), "aeb06e4738269d0d60ced2206d21ec5c331690342a54f5c37f1d8eef4abdaf94");

    algo::hash256 const h1{ algo::octopus::octopusLightCompute(header, 12345ull, light) };
    EXPECT_EQ(toHex(h1.ubytes, 32), "d8e2e19a7ab253fba2a9c085157dae170288d3e9853f7967e5f71363eeeb85e9");
}
