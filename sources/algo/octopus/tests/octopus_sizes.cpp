#include <gtest/gtest.h>

#include <algo/octopus/octopus.hpp>


struct OctopusSizesTest : public testing::Test
{
    OctopusSizesTest() = default;
    ~OctopusSizesTest() = default;
};


// All expected values captured bit-exactly from Conflux-Chain/open-cfxmine
// (src/light.cc octopus_get_cachesize / octopus_get_datasize), see
// sources/algo/octopus/tests/reference_vectors.md.


TEST_F(OctopusSizesTest, epoch)
{
    EXPECT_EQ(algo::octopus::getEpoch(0ull), 0ull);
    EXPECT_EQ(algo::octopus::getEpoch(524287ull), 0ull);
    EXPECT_EQ(algo::octopus::getEpoch(524288ull), 1ull);
    EXPECT_EQ(algo::octopus::getEpoch(34078720ull), 65ull);
    EXPECT_EQ(algo::octopus::getEpoch(149499627ull), 285ull);
}


TEST_F(OctopusSizesTest, cacheSize)
{
    EXPECT_EQ(algo::octopus::getCacheSize(0ull), 16776896ull);
    EXPECT_EQ(algo::octopus::getCacheSize(34078720ull), 21035968ull);
    EXPECT_EQ(algo::octopus::getCacheSize(149499627ull), 35454784ull);
}


TEST_F(OctopusSizesTest, dataSize)
{
    EXPECT_EQ(algo::octopus::getDataSize(0ull), 4294966528ull);
    EXPECT_EQ(algo::octopus::getDataSize(34078720ull), 5385477376ull);
    EXPECT_EQ(algo::octopus::getDataSize(149499627ull), 9076465408ull);
}


TEST_F(OctopusSizesTest, dataSizeIsPrimeFlooredInMixUnits)
{
    uint64_t const size{ algo::octopus::getDataSize(0ull) };
    EXPECT_EQ(size % algo::octopus::MIX_BYTES, 0ull);
}
