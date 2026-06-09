#include <algo/pyrinhashv2/hashers.hpp>
#include <algo/pyrinhashv2/matrix.hpp>
#include <algo/pyrinhashv2/pyrinhashv2.hpp>


namespace pyrinhashv2
{
    Hash256 heavyHash(Matrix const& matrix, Hash256 const& hash1)
    {
        // Expand hash1 to 64 nibbles, high nibble first.
        uint16_t vec[64];
        for (int i{ 0 }; i < 32; ++i)
        {
            vec[2 * i] = static_cast<uint16_t>(hash1[i] >> 4);
            vec[2 * i + 1] = static_cast<uint16_t>(hash1[i] & 0x0F);
        }

        // Matrix-vector multiply; each pair of rows collapses to one output byte. V2 reduction
        // (algo_updated == true): fold the three low nibbles of each 12-bit sum with XOR.
        Hash256 product{};
        for (int i{ 0 }; i < 32; ++i)
        {
            uint16_t sum1{ 0 };
            uint16_t sum2{ 0 };
            for (int j{ 0 }; j < 64; ++j)
            {
                sum1 = static_cast<uint16_t>(sum1 + matrix[2 * i][j] * vec[j]);
                sum2 = static_cast<uint16_t>(sum2 + matrix[2 * i + 1][j] * vec[j]);
            }
            uint16_t const hi{ static_cast<uint16_t>((sum1 & 0xF) ^ ((sum1 >> 4) & 0xF) ^ ((sum1 >> 8) & 0xF)) };
            uint16_t const lo{ static_cast<uint16_t>((sum2 & 0xF) ^ ((sum2 >> 4) & 0xF) ^ ((sum2 >> 8) & 0xF)) };
            product[i] = static_cast<uint8_t>((hi << 4) | lo);
        }

        // XOR with the original hash1 bytes, then KHeavyHash.
        for (int i{ 0 }; i < 32; ++i)
        {
            product[i] ^= hash1[i];
        }
        return kHeavyHash(product);
    }


    Hash256 calculatePow(Hash256 const& prePowHash, uint64_t const timestamp, uint64_t const nonce)
    {
        Matrix const  matrix{ generateMatrix(prePowHash) };
        Hash256 const hash1{ powHash(prePowHash, timestamp, nonce) };
        return heavyHash(matrix, hash1);
    }


    bool meetsTarget(Hash256 const& powValueLe, Hash256 const& targetLe)
    {
        // Compare as little-endian 256-bit integers: scan from most-significant byte.
        for (int i{ 31 }; i >= 0; --i)
        {
            if (powValueLe[i] != targetLe[i])
            {
                return powValueLe[i] < targetLe[i];
            }
        }
        return true;  // equal => pow <= target
    }
}
