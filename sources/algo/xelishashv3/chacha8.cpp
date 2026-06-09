#include <cstring>

#include <algo/xelishashv3/chacha8.hpp>


namespace
{
    inline uint32_t rotl32(uint32_t const x, int const n)
    {
        return (x << n) | (x >> (32 - n));
    }


    inline uint32_t readLe32(uint8_t const* p)
    {
        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
             | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    }


    inline void writeLe32(uint8_t* p, uint32_t const v)
    {
        p[0] = static_cast<uint8_t>(v);
        p[1] = static_cast<uint8_t>(v >> 8);
        p[2] = static_cast<uint8_t>(v >> 16);
        p[3] = static_cast<uint8_t>(v >> 24);
    }


    inline void quarterRound(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d)
    {
        a += b;
        d ^= a;
        d = rotl32(d, 16);
        c += d;
        b ^= c;
        b = rotl32(b, 12);
        a += b;
        d ^= a;
        d = rotl32(d, 8);
        c += d;
        b ^= c;
        b = rotl32(b, 7);
    }
}


namespace xelishashv3
{
    void chacha8Keystream(uint8_t const key[32], uint8_t const nonce[12], uint8_t* out, size_t len)
    {
        uint32_t state[16];
        // "expand 32-byte k"
        state[0] = 0x61707865u;
        state[1] = 0x3320646eu;
        state[2] = 0x79622d32u;
        state[3] = 0x6b206574u;
        for (int i{ 0 }; i < 8; ++i)
        {
            state[4 + i] = readLe32(key + 4 * i);
        }
        state[12] = 0u;  // 32-bit block counter
        state[13] = readLe32(nonce + 0);
        state[14] = readLe32(nonce + 4);
        state[15] = readLe32(nonce + 8);

        size_t offset{ 0 };
        while (offset < len)
        {
            uint32_t work[16];
            std::memcpy(work, state, sizeof(work));

            for (int i{ 0 }; i < 4; ++i)  // 8 rounds = 4 double-rounds
            {
                quarterRound(work[0], work[4], work[8], work[12]);
                quarterRound(work[1], work[5], work[9], work[13]);
                quarterRound(work[2], work[6], work[10], work[14]);
                quarterRound(work[3], work[7], work[11], work[15]);
                quarterRound(work[0], work[5], work[10], work[15]);
                quarterRound(work[1], work[6], work[11], work[12]);
                quarterRound(work[2], work[7], work[8], work[13]);
                quarterRound(work[3], work[4], work[9], work[14]);
            }

            uint8_t  keystream[64];
            for (int i{ 0 }; i < 16; ++i)
            {
                writeLe32(keystream + 4 * i, work[i] + state[i]);
            }

            size_t const take{ (len - offset < 64) ? (len - offset) : 64 };
            std::memcpy(out + offset, keystream, take);
            offset += 64;
            ++state[12];  // next block
        }
    }
}
