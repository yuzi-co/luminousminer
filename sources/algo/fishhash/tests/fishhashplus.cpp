#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <algo/fishhash/fishhash.hpp>


namespace
{
    std::string toHex(uint8_t const* bytes, size_t len)
    {
        static char const* const hexDigits{ "0123456789abcdef" };
        std::string              out;
        out.reserve(len * 2u);
        for (size_t i{ 0u }; i < len; ++i)
        {
            out.push_back(hexDigits[bytes[i] >> 4]);
            out.push_back(hexDigits[bytes[i] & 0x0f]);
        }
        return out;
    }

    // Builds an 80-byte KarlsenHashV2 header: prePowHash[32] || timestamp_le[8] ||
    // zero[32] || nonce_le[8]. With fill=false the header is all zero except the
    // timestamp/nonce fields. The nonce is little-endian at RANDOMNESS_OFFSET_PLUS.
    std::vector<uint8_t> buildHeader(bool fill, uint64_t timestamp, uint64_t nonce)
    {
        std::vector<uint8_t> header(algo::fishhash::HEADER_SIZE_PLUS, 0u);
        if (true == fill)
        {
            for (uint8_t i{ 0u }; i < 32u; ++i)
            {
                header[i] = i;
            }
        }
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            header[32u + i] = static_cast<uint8_t>(timestamp >> (8u * i));
        }
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            header[algo::fishhash::RANDOMNESS_OFFSET_PLUS + i] = static_cast<uint8_t>(nonce >> (8u * i));
        }
        return header;
    }
}


struct FishHashPlusTest : public testing::Test
{
    algo::fishhash::Context* ctx{ nullptr };

    FishHashPlusTest() { ctx = algo::fishhash::getContext(false); }
    ~FishHashPlusTest() { algo::fishhash::freeContext(ctx); }
};


// Golden vectors for KarlsenHashV2 (FishHashPlus), light context. Cross-verified
// bit-for-bit against the patched upstream karlsen-network/fish-hash-plus C++ reference
// (independent source file, seed zero-extended per the CUDA/Rust miner). See
// memory/fishhashplus-karlsen-spec.md.
TEST_F(FishHashPlusTest, goldenVectors)
{
    struct Vector
    {
        bool        fill;
        uint64_t    timestamp;
        uint64_t    nonce;
        char const* expected;
    };

    static Vector const vectors[]{
        { false, 0ull, 0ull, "2a788445d05b55676de63d74d8ec827c25881ca7dd2398058ee832f353284d07" },
        { true, 0x1122334455667788ull, 1ull, "69a918bd1555d834459e9d7f08f9c307164dc36954cdb3ab91771d9506de465d" },
        { true, 0x1122334455667788ull, 1234ull, "a530dec8d5e157434dfad212f00000079954500cdc8c77a4a46ecc098762a14b" },
    };

    for (Vector const& v : vectors)
    {
        std::vector<uint8_t> const header{ buildHeader(v.fill, v.timestamp, v.nonce) };
        uint8_t                    out[32]{};
        algo::fishhash::hashPlus(out, ctx, header.data(), header.size());
        EXPECT_EQ(toHex(out, 32), v.expected) << "nonce=" << v.nonce;
    }
}


// Same input must hash to the same output (lazy dataset writes must not perturb results).
TEST_F(FishHashPlusTest, deterministic)
{
    std::vector<uint8_t> const header{ buildHeader(true, 0x99aabbccddeeff00ull, 7ull) };
    uint8_t                    first[32]{};
    uint8_t                    second[32]{};
    algo::fishhash::hashPlus(first, ctx, header.data(), header.size());
    algo::fishhash::hashPlus(second, ctx, header.data(), header.size());
    EXPECT_EQ(toHex(first, 32), toHex(second, 32));
}
