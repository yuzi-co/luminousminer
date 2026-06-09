#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <algo/fishhash/fishhash.hpp>


namespace
{
    std::string toHex(uint8_t const* bytes, size_t len)
    {
        static char const* const hexDigits{ "0123456789abcdef" };
        std::string             out;
        out.reserve(len * 2u);
        for (size_t i{ 0u }; i < len; ++i)
        {
            out.push_back(hexDigits[bytes[i] >> 4]);
            out.push_back(hexDigits[bytes[i] & 0x0f]);
        }
        return out;
    }

    // Writes randomness as big-endian into header bytes [172..180].
    void setRandomness(std::vector<uint8_t>& header, uint64_t randomness)
    {
        for (size_t i{ 0u }; i < 8u; ++i)
        {
            header[algo::fishhash::RANDOMNESS_OFFSET + i] =
                static_cast<uint8_t>((randomness >> (8u * (7u - i))) & 0xffu);
        }
    }
}


struct FishHashTest : public testing::Test
{
    algo::fishhash::Context* ctx{ nullptr };

    FishHashTest() { ctx = algo::fishhash::getContext(false); }
    ~FishHashTest() { algo::fishhash::freeContext(ctx); }
};


// Bootstrap KAT from ironfish-rust mining::mine test: header = [0..9] x 18 (180 bytes),
// randomness 45 is the first value (from start 43) whose hash <= target.
TEST_F(FishHashTest, bootstrapKatMineBatch)
{
    std::vector<uint8_t> header;
    for (int i{ 0 }; i < 18; ++i)
    {
        for (uint8_t b{ 0u }; b < 10u; ++b)
        {
            header.push_back(b);
        }
    }
    ASSERT_EQ(header.size(), algo::fishhash::HEADER_SIZE);

    static uint8_t const target[32]{ 59,  125, 43,  4,   254, 19,  32,  88,  203, 188, 220,
                                     43,  193, 139, 194, 164, 61,  99,  44,  90,  151, 122,
                                     236, 65,  253, 171, 148, 82,  130, 54,  122, 195 };

    uint64_t winner{ 0u };
    bool     found{ false };
    for (uint64_t r{ 43u }; r <= 46u && false == found; ++r)
    {
        setRandomness(header, r);
        uint8_t out[32]{};
        algo::fishhash::hash(out, ctx, header.data(), header.size());
        if (std::memcmp(out, target, 32) <= 0)
        {
            found = true;
            winner = r;
        }
    }

    EXPECT_TRUE(found);
    EXPECT_EQ(winner, 45u);
}


// Golden vectors generated from the vendored upstream reference: zero 180-byte header,
// randomness (8-byte big-endian) at [172..180].
TEST_F(FishHashTest, goldenVectorsZeroHeader)
{
    struct Vector
    {
        uint64_t    randomness;
        char const* expected;
    };

    static Vector const vectors[]{
        { 0u, "f96b6d14b00a8437cb30448faf04b7cf2928ab8488f95ff8a1ed8dc02f0fb71d" },
        { 1u, "37363f3616f477becf56c3aa7af26f6c39903233a32d580382f80f8e7a3ecb83" },
        { 65535u, "e61c88a9d22ef8248055482ba5b3925c676cd446d68546e801e3896d5878fd5c" },
    };

    for (Vector const& v : vectors)
    {
        std::vector<uint8_t> header(algo::fishhash::HEADER_SIZE, 0u);
        setRandomness(header, v.randomness);
        uint8_t out[32]{};
        algo::fishhash::hash(out, ctx, header.data(), header.size());
        EXPECT_EQ(toHex(out, 32), v.expected) << "randomness=" << v.randomness;
    }
}
