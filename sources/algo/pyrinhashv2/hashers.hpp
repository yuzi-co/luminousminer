#pragma once

#include <algo/pyrinhashv2/types.hpp>


namespace pyrinhashv2
{
    // hash1 = BLAKE3 (plain, unkeyed) over
    //   pre_pow_hash[32] || timestamp_u64_LE || zero[32] || nonce_u64_LE
    // (reference node Pyrinpyi/pyrin crypto/hashes/src/pow_hashers.rs::PowHash). 32 LE bytes.
    Hash256 powHash(Hash256 const& prePowHash, uint64_t timestamp, uint64_t nonce);

    // hash2 step = BLAKE3 (plain, unkeyed) over 32 bytes (pow_hashers.rs::KHeavyHash).
    Hash256 kHeavyHash(Hash256 const& input);
}
