#include <vector>

#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/octopus/multi_eval.hpp>
#include <algo/octopus/octopus.hpp>


// Expected values captured bit-exactly from Conflux-Chain/open-cfxmine
// (src/light.cc compute_d / multi_eval), see reference_vectors.md.
// Fixed test header = bytes 0x00 0x01 ... 0x1f.


struct OctopusMultiEvalTest : public testing::Test
{
    algo::hash256 header{};

    OctopusMultiEvalTest()
    {
        for (uint8_t i{ 0u }; i < 32u; ++i)
        {
            header.ubytes[i] = i;
        }
    }
    ~OctopusMultiEvalTest() = default;
};


TEST_F(OctopusMultiEvalTest, computeDNonce0)
{
    std::vector<uint32_t> d(algo::octopus::NTT_N);
    algo::octopus::computeD(header, 0ull, d.data());

    uint32_t const expected[8]{ 349892u, 834550u, 588346u, 449621u, 85820u, 185537u, 368936u, 447534u };
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        EXPECT_EQ(d[i], expected[i]);
    }
    EXPECT_EQ(d[1023], 585399u);
}


TEST_F(OctopusMultiEvalTest, computeDNonce12345)
{
    std::vector<uint32_t> d(algo::octopus::NTT_N);
    algo::octopus::computeD(header, 12345ull, d.data());

    uint32_t const expected[8]{ 628584u, 456832u, 993474u, 590150u, 215332u, 330854u, 627355u, 138534u };
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        EXPECT_EQ(d[i], expected[i]);
    }
    EXPECT_EQ(d[1023], 103108u);
}


TEST_F(OctopusMultiEvalTest, multiEvalNonce0)
{
    std::pair<uint64_t, std::vector<uint32_t>> const pr{ algo::octopus::multiEval(header, 0ull) };
    EXPECT_EQ(pr.first, 16285858431497212756ull);
    ASSERT_EQ(pr.second.size(), algo::octopus::DATA_PER_THREAD);

    uint32_t const expected[8]{ 188101u, 213632u, 115490u, 721627u, 326911u, 925788u, 246724u, 906503u };
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        EXPECT_EQ(pr.second[i], expected[i]);
    }
}


TEST_F(OctopusMultiEvalTest, multiEvalNonce12345)
{
    std::pair<uint64_t, std::vector<uint32_t>> const pr{ algo::octopus::multiEval(header, 12345ull) };
    EXPECT_EQ(pr.first, 17518503130972107457ull);
    ASSERT_EQ(pr.second.size(), algo::octopus::DATA_PER_THREAD);

    uint32_t const expected[8]{ 823148u, 687249u, 759189u, 521781u, 980657u, 460517u, 728407u, 731710u };
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        EXPECT_EQ(pr.second[i], expected[i]);
    }
}
