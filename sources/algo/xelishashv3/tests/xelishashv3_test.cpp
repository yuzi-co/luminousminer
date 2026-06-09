#include <cstring>

#include "xelishashv3_test_vectors.hpp"
#include <gtest/gtest.h>

#include <algo/xelishashv3/xelishashv3.hpp>


// Full pipeline on the all-zero 112-byte input. This single vector transitively pins every
// stage to the upstream reference: stage1 (BLAKE3 key-derive + ChaCha8 fill), stage3 (AES
// round + 16-way mixing + 128-bit ALU), stage4 (BLAKE3 over the scratchpad). Any deviation
// in any primitive changes this digest. (C reference gold / Go TestZeroHash.)
TEST(XelisHashV3, ZeroInputMatchesReference)
{
    xelishashv3::ScratchPad    pad;
    xelishashv3::Hash256 const out{ xelishashv3::hash(xelishashv3::kat::ZERO_INPUT.data(), pad) };
    EXPECT_EQ(0, std::memcmp(out.data(), xelishashv3::kat::ZERO_EXPECTED.data(), 32));
}


// A second, non-trivial 112-byte input (Go TestVerifyOutput) to guard against an input that
// happens to be special-cased by the zero vector.
TEST(XelisHashV3, VerifyOutputMatchesReference)
{
    xelishashv3::ScratchPad    pad;
    xelishashv3::Hash256 const out{ xelishashv3::hash(xelishashv3::kat::VERIFY_INPUT.data(), pad) };
    EXPECT_EQ(0, std::memcmp(out.data(), xelishashv3::kat::VERIFY_EXPECTED.data(), 32));
}


// The scratchpad is fully overwritten each call, so reusing one allocation across hashes
// must be deterministic (Go TestReusedScratchpad). Mirrors how the resolver recycles a single
// per-thread scratchpad in the hot loop.
TEST(XelisHashV3, ReusedScratchpadIsDeterministic)
{
    xelishashv3::ScratchPad    pad;
    xelishashv3::Hash256 const first{ xelishashv3::hash(xelishashv3::kat::VERIFY_INPUT.data(), pad) };
    xelishashv3::Hash256 const second{ xelishashv3::hash(xelishashv3::kat::VERIFY_INPUT.data(), pad) };
    EXPECT_EQ(first, second);
}
