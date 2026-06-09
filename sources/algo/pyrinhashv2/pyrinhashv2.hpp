#pragma once

#include <algo/pyrinhashv2/types.hpp>


namespace pyrinhashv2
{
    // Heavy step: expand hash1 to 64 nibbles, matrix-multiply, fold each 12-bit row sum to a
    // nibble by XORing its three low nibbles (the V2 / algo_updated reduction), XOR with hash1,
    // then KHeavyHash. reference node consensus/pow/src/matrix.rs::heavy_hash(_, true).
    Hash256 heavyHash(Matrix const& matrix, Hash256 const& hash1);

    // Full per-nonce PoW: generate matrix from pre_pow_hash, PowHash, heavy step.
    // Returns the 32-byte little-endian PoW value. reference node lib.rs::calculate_pow.
    Hash256 calculatePow(Hash256 const& prePowHash, uint64_t timestamp, uint64_t nonce);

    // Accept test: powValue <= target, both interpreted as little-endian 256-bit ints.
    bool meetsTarget(Hash256 const& powValueLe, Hash256 const& targetLe);
}
