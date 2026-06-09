#include <algo/crypto/reference/blake3/blake3.h>
#include <algo/pyrinhashv2/hashers.hpp>


namespace pyrinhashv2
{
    namespace
    {
        // Append a u64 as 8 little-endian bytes to the running BLAKE3 state.
        inline void updateLe64(blake3_hasher& hasher, uint64_t const value)
        {
            uint8_t bytes[8];
            for (int b{ 0 }; b < 8; ++b)
            {
                bytes[b] = static_cast<uint8_t>((value >> (8 * b)) & 0xFF);
            }
            blake3_hasher_update(&hasher, bytes, sizeof(bytes));
        }
    }


    Hash256 powHash(Hash256 const& prePowHash, uint64_t const timestamp, uint64_t const nonce)
    {
        // BLAKE3( pre_pow_hash[32] || timestamp_LE || zero[32] || nonce_LE ).
        blake3_hasher hasher;
        blake3_hasher_init(&hasher);
        blake3_hasher_update(&hasher, prePowHash.data(), prePowHash.size());
        updateLe64(hasher, timestamp);
        uint8_t const zero[32]{};
        blake3_hasher_update(&hasher, zero, sizeof(zero));
        updateLe64(hasher, nonce);

        Hash256 out{};
        blake3_hasher_finalize(&hasher, out.data(), out.size());
        return out;
    }


    Hash256 kHeavyHash(Hash256 const& input)
    {
        blake3_hasher hasher;
        blake3_hasher_init(&hasher);
        blake3_hasher_update(&hasher, input.data(), input.size());

        Hash256 out{};
        blake3_hasher_finalize(&hasher, out.data(), out.size());
        return out;
    }
}
