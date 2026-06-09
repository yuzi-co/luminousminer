#include <cmath>
#include <cstring>

#include <blake3.h>  // vendored official reference, via blake3_ref

#include <algo/xelishashv3/aes.hpp>
#include <algo/xelishashv3/chacha8.hpp>
#include <algo/xelishashv3/xelishashv3.hpp>


namespace
{
    using xelishashv3::BUFSIZE;
    using xelishashv3::MEMSIZE;


    void blake3(uint8_t const* in, size_t const len, uint8_t* out32)
    {
        blake3_hasher hasher;
        blake3_hasher_init(&hasher);
        blake3_hasher_update(&hasher, in, len);
        blake3_hasher_finalize(&hasher, out32, BLAKE3_OUT_LEN);
    }


    // x86 rolq/rorq mask the count to 6 bits (mod 64); replicate that exactly.
    inline uint64_t rotl64(uint64_t const x, uint32_t r)
    {
        r &= 63u;
        return (0u == r) ? x : ((x << r) | (x >> (64u - r)));
    }


    inline uint64_t rotr64(uint64_t const x, uint32_t r)
    {
        r &= 63u;
        return (0u == r) ? x : ((x >> r) | (x << (64u - r)));
    }


    inline uint64_t murmurhash3(uint64_t seed)
    {
        seed ^= seed >> 55;
        seed *= 0xff51afd7ed558ccdULL;
        seed ^= seed >> 32;
        seed *= 0xc4ceb9fe1a85ec53ULL;
        seed ^= seed >> 15;
        return seed;
    }


    inline uint64_t mapIndex(uint64_t x)
    {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        return static_cast<uint64_t>((static_cast<__uint128_t>(x) * BUFSIZE) >> 64);
    }


    inline bool pickHalf(uint64_t const seed)
    {
        return (murmurhash3(seed) & (1ULL << 58)) != 0u;
    }


    // floor(sqrt(n)) via FP64 approximation + integer adjust — replicated bit-for-bit,
    // including the wrapping (approx+1)*(approx+1) the reference relies on.
    inline uint64_t isqrt(uint64_t const n)
    {
        if (n < 2u)
        {
            return n;
        }
        uint64_t approx{ static_cast<uint64_t>(std::sqrt(static_cast<double>(n))) };
        if (approx * approx > n)
        {
            return approx - 1u;
        }
        if ((approx + 1u) * (approx + 1u) <= n)
        {
            return approx + 1u;
        }
        return approx;
    }


    inline uint64_t modularPower(uint64_t base, uint64_t exp, uint64_t const mod)
    {
        uint64_t result{ 1u };
        base %= mod;
        while (exp > 0u)
        {
            if (exp & 1u)
            {
                result = static_cast<uint64_t>((static_cast<__uint128_t>(result) * base) % mod);
            }
            base = static_cast<uint64_t>((static_cast<__uint128_t>(base) * base) % mod);
            exp /= 2u;
        }
        return result;
    }


    inline __uint128_t combine(uint64_t const high, uint64_t const low)
    {
        return (static_cast<__uint128_t>(high) << 64) | low;
    }


    inline uint64_t leBytesToU64(uint8_t const* b)
    {
        uint64_t v{ 0u };
        for (int i{ 7 }; i >= 0; --i)
        {
            v = (v << 8) | b[i];
        }
        return v;
    }


    inline void u64ToLeBytes(uint64_t value, uint8_t* bytes)
    {
        for (int i{ 0 }; i < 8; ++i)
        {
            bytes[i] = static_cast<uint8_t>(value & 0xFFu);
            value >>= 8;
        }
    }
}


namespace xelishashv3
{
    void stage1(uint8_t const input[INPUT_LEN], uint8_t* scratch)
    {
        uint8_t key[CHUNK_SIZE * CHUNKS]{};  // 128 bytes, zero-padded
        uint8_t inputHash[HASH_SIZE];
        uint8_t buffer[CHUNK_SIZE * 2];      // 64 bytes
        std::memcpy(key, input, INPUT_LEN);
        blake3(input, INPUT_LEN, buffer);    // buffer[0..32] = BLAKE3(input)

        size_t const chunkLen{ MEMSIZE_BYTES / CHUNKS };
        uint8_t*     t{ scratch };

        // Chunk 0: nonce is the first 12 bytes of BLAKE3(input) (still in buffer[0..32]).
        std::memcpy(buffer + CHUNK_SIZE, key + 0 * CHUNK_SIZE, CHUNK_SIZE);
        blake3(buffer, CHUNK_SIZE * 2, inputHash);
        chacha8Keystream(inputHash, buffer, t, chunkLen);

        // Chunks 1..3: key chained from the previous ChaCha key; nonce is the last 12 bytes
        // of the previously written chunk.
        for (size_t k{ 1 }; k < CHUNKS; ++k)
        {
            t += chunkLen;
            std::memcpy(buffer, inputHash, CHUNK_SIZE);
            std::memcpy(buffer + CHUNK_SIZE, key + k * CHUNK_SIZE, CHUNK_SIZE);
            blake3(buffer, CHUNK_SIZE * 2, inputHash);
            chacha8Keystream(inputHash, t - NONCE_SIZE, t, chunkLen);
        }
    }


    void stage3(uint64_t* scratch)
    {
        uint64_t* memBufferA{ scratch };
        uint64_t* memBufferB{ &scratch[BUFSIZE] };

        uint64_t addrA{ memBufferB[BUFSIZE - 1] };
        uint64_t addrB{ memBufferA[BUFSIZE - 1] >> 32 };
        uint32_t r{ 0u };

        for (uint32_t i{ 0u }; i < ITERS; ++i)
        {
            uint64_t const memA0{ memBufferA[mapIndex(addrA)] };
            uint64_t const memB0{ memBufferB[mapIndex(memA0 ^ addrB)] };

            uint8_t block[16];
            u64ToLeBytes(memB0, block);
            u64ToLeBytes(memA0, block + 8);
            aesSingleRound(block, reinterpret_cast<uint8_t const*>(KEY));

            uint64_t const hash1{ leBytesToU64(block) };
            uint64_t const hash2{ leBytesToU64(block + 8) };
            uint64_t       result{ ~(hash1 ^ hash2) };

            for (uint32_t j{ 0u }; j < BUFSIZE; ++j)
            {
                uint64_t const a{ memBufferA[mapIndex(result)] };
                uint64_t const b{ memBufferB[mapIndex(a ^ ~rotr64(result, r))] };
                uint64_t const c{ (r < BUFSIZE) ? memBufferA[r] : memBufferB[r - BUFSIZE] };
                r = (r < MEMSIZE - 1) ? r + 1 : 0u;

                uint64_t    v{ 0u };
                __uint128_t t1{};
                __uint128_t t2{};
                switch (rotl64(result, static_cast<uint32_t>(c)) & 0xfu)
                {
                    case 0:
                    {
                        t1 = combine(a + i, isqrt(b + j));
                        uint64_t const denom{ murmurhash3(c ^ result ^ i ^ j) | 1u };
                        v = static_cast<uint64_t>(t1 % denom);
                        break;
                    }
                    case 1:
                        v = rotl64((c + i) % isqrt(b | 2u), i + j) * isqrt(a + j);
                        break;
                    case 2:
                        v = (isqrt(a + i) * isqrt(c + j)) ^ (b + i + j);
                        break;
                    case 3:
                        v = ((a + b) * c);
                        break;
                    case 4:
                        v = ((b - c) * a);
                        break;
                    case 5:
                        v = (c - a + b);
                        break;
                    case 6:
                        v = (a - b + c);
                        break;
                    case 7:
                        v = (b * c + a);
                        break;
                    case 8:
                        v = (c * a + b);
                        break;
                    case 9:
                        v = (a * b * c);
                        break;
                    case 10:
                        t1 = combine(a, b);
                        v = static_cast<uint64_t>(t1 % (c | 1u));
                        break;
                    case 11:
                        t1 = combine(b, c);
                        t2 = combine(rotl64(result, r), a | 2u);
                        v = (t2 > t1) ? c : static_cast<uint64_t>(t1 % t2);
                        break;
                    case 12:
                        v = static_cast<uint64_t>(combine(c, a) / (b | 4u));
                        break;
                    case 13:
                        t1 = combine(rotl64(result, r), b);
                        t2 = combine(a, c | 8u);
                        v = (t1 > t2) ? static_cast<uint64_t>(t1 / t2) : (a ^ b);
                        break;
                    case 14:
                        t1 = combine(b, a);
                        v = static_cast<uint64_t>((t1 * c) >> 64);
                        break;
                    case 15:
                        t1 = combine(a, c);
                        t2 = combine(rotr64(result, r), b);
                        v = static_cast<uint64_t>((t1 * t2) >> 64);
                        break;
                }

                uint64_t const idxSeed{ v ^ result };
                result = rotl64(idxSeed, r);

                bool const     useBufferB{ pickHalf(v) };
                uint64_t const idxT{ mapIndex(idxSeed) };
                uint64_t const t{ (useBufferB ? memBufferB[idxT] : memBufferA[idxT]) ^ result };

                uint64_t const idxA{ mapIndex(t ^ result ^ 0x9e3779b97f4a7c15ULL) };
                uint64_t const idxB{ mapIndex(idxA ^ ~result ^ 0xd2b74407b1ce6e93ULL) };

                uint64_t const memA{ memBufferA[idxA] };
                memBufferA[idxA] = t;
                memBufferB[idxB] ^= memA ^ rotr64(t, i + j);
            }

            addrA = modularPower(addrA, addrB, result);
            addrB = isqrt(result) * (r + 1u) * isqrt(addrA);
        }
    }


    Hash256 hash(uint8_t const input[INPUT_LEN], ScratchPad& pad)
    {
        stage1(input, pad.bytes());
        stage3(pad.u64());
        Hash256 out{};
        blake3(pad.bytes(), MEMSIZE_BYTES, out.data());
        return out;
    }
}
