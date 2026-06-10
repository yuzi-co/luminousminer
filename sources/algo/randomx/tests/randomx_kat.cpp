#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/randomx/randomx_pow.hpp>


namespace
{
    std::string toHex(algo::hash256 const& h)
    {
        static char const* const digits{ "0123456789abcdef" };
        std::string              out(2u * sizeof(h.ubytes), '0');
        for (size_t i{ 0u }; i < sizeof(h.ubytes); ++i)
        {
            out[2u * i] = digits[h.ubytes[i] >> 4];
            out[2u * i + 1u] = digits[h.ubytes[i] & 0x0f];
        }
        return out;
    }


    std::string rxHash(std::string const& key, std::string const& input)
    {
        algo::hash256 out{};
        algo::randomx::calculateHash(key.data(), key.size(), input.data(), input.size(), out);
        return toHex(out);
    }
}


struct RandomXKatTest : public testing::Test
{
    RandomXKatTest() = default;
    ~RandomXKatTest() = default;
};


// Canonical RandomX (non-V2 / light) test vectors from tevador/RandomX src/tests/tests.cpp.
TEST_F(RandomXKatTest, officialVectors)
{
    EXPECT_EQ(rxHash("test key 000", "This is a test"),
              "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f");

    EXPECT_EQ(rxHash("test key 000", "Lorem ipsum dolor sit amet"),
              "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969");

    EXPECT_EQ(rxHash("test key 000",
                     "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"),
              "c36d4ed4191e617309867ed66a443be4075014e2b061bcdaf9ce7b721d2b77a8");
}


// A re-keyed cache must produce that key's vector, proving init() rebuilds correctly.
TEST_F(RandomXKatTest, reKeyRebuildsCache)
{
    algo::randomx::RandomXHasher hasher{};

    ASSERT_TRUE(hasher.init("test key 000", 12u, true));
    algo::hash256 a{};
    hasher.hash("This is a test", 14u, a);
    EXPECT_EQ(toHex(a), "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f");

    ASSERT_TRUE(hasher.init("test key 001", 12u, true));
    algo::hash256 b{};
    hasher.hash("This is a test", 14u, b);
    EXPECT_EQ(toHex(b), "351562e03304417135a5f8946164e8900f24fd45b444c092b029b439ddbc1450");
}
