#include <gtest/gtest.h>

#include <algo/pyrinhashv2/hashers.hpp>
#include <algo/pyrinhashv2/matrix.hpp>
#include <algo/pyrinhashv2/pyrinhashv2.hpp>
#include "pyrinhashv2_test_vectors.hpp"


// Layer 0: matrix generation. xoshiro256++ seeded from pre_pow_hash, regenerated until rank 64.
// Byte-identical to the reference node matrix.rs::test_generate_matrix (shared with kHeavyHash).
TEST(PyrinHashV2Pipeline, GenerateMatrixMatchesReference)
{
    pyrinhashv2::Matrix const out{ pyrinhashv2::generateMatrix(pyrinhashv2::kat::GEN_SEED) };
    EXPECT_EQ(out, pyrinhashv2::kat::GEN_EXPECTED_MATRIX);
}


// Layer 1a: PowHash = plain BLAKE3. Expected is the reference node's own concrete
// pow_hashers.rs::test_heavy_hash() vector (pre=[0x2a;32]).
TEST(PyrinHashV2Pipeline, PowHashMatchesReference)
{
    pyrinhashv2::Hash256 pre{};
    for (size_t i{ 0 }; i < 32; ++i)
    {
        pre[i] = pyrinhashv2::kat::POW_KAT_PRE[i];
    }
    pyrinhashv2::Hash256 const out{
        pyrinhashv2::powHash(pre, pyrinhashv2::kat::POW_KAT_TIMESTAMP, pyrinhashv2::kat::POW_KAT_NONCE)
    };
    EXPECT_EQ(out, pyrinhashv2::kat::POW_KAT_EXPECTED);
}


// Layer 1b: KHeavyHash = plain BLAKE3 over 32 bytes.
TEST(PyrinHashV2Pipeline, KHeavyHashMatchesReference)
{
    pyrinhashv2::Hash256 const out{ pyrinhashv2::kHeavyHash(pyrinhashv2::kat::HEAVY_INPUT) };
    EXPECT_EQ(out, pyrinhashv2::kat::KHEAVY_EXPECTED);
}


// Layer 2: heavy step (matrix multiply + V2 nibble-XOR-fold + XOR + KHeavyHash).
// Same matrix+input as the reference matrix.rs::test_heavy_hash(false); only the V2 reduction
// differs, so this pins the one-line V2 change against the reference's own explicit matrix.
TEST(PyrinHashV2Pipeline, HeavyHashV2MatchesReference)
{
    pyrinhashv2::Hash256 const out{
        pyrinhashv2::heavyHash(pyrinhashv2::kat::HEAVY_TEST_MATRIX, pyrinhashv2::kat::HEAVY_INPUT)
    };
    EXPECT_EQ(out, pyrinhashv2::kat::HEAVY_EXPECTED);
}


// Layer 3: full pipeline incl. matrix generation. Oracle-minted, reference-cross-validated.
TEST(PyrinHashV2Pipeline, CalculatePowMatchesReference)
{
    pyrinhashv2::Hash256 const out{
        pyrinhashv2::calculatePow(pyrinhashv2::kat::FP_PRE, pyrinhashv2::kat::FP_TIMESTAMP, pyrinhashv2::kat::FP_NONCE)
    };
    EXPECT_EQ(out, pyrinhashv2::kat::FP_FINAL);
}


// Accept test: pow <= target (little-endian 256-bit).
TEST(PyrinHashV2Pipeline, MeetsTargetCompare)
{
    EXPECT_TRUE(pyrinhashv2::meetsTarget(pyrinhashv2::kat::FP_FINAL, pyrinhashv2::kat::FP_TARGET_PASS));
    EXPECT_FALSE(pyrinhashv2::meetsTarget(pyrinhashv2::kat::FP_FINAL, pyrinhashv2::kat::FP_TARGET_FAIL));
}
