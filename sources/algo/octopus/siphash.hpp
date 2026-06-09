#pragma once

#include <cstdint>


namespace algo
{
    namespace octopus
    {
        // Faithful port of open-cfxmine/src/siphash.h (used by compute_d to seed the
        // polynomial coefficient vector). Default rotE=21 matches `siphash_state<>`.
        template<int rotE = 21>
        class SipHashState
        {
          public:
            uint64_t v0{ 0ull };
            uint64_t v1{ 0ull };
            uint64_t v2{ 0ull };
            uint64_t v3{ 0ull };

            explicit SipHashState(uint64_t const* const sk)
                : v0{ sk[0] }
                , v1{ sk[1] }
                , v2{ sk[2] }
                , v3{ sk[3] }
            {
            }

            uint64_t xorLanes() const
            {
                return (v0 ^ v1) ^ (v2 ^ v3);
            }

            static uint64_t rotl(uint64_t const x, uint64_t const b)
            {
                return (x << b) | (x >> (64ull - b));
            }

            void sipRound()
            {
                v0 += v1;
                v2 += v3;
                v1 = rotl(v1, 13);
                v3 = rotl(v3, 16);
                v1 ^= v0;
                v3 ^= v2;
                v0 = rotl(v0, 32);
                v2 += v1;
                v0 += v3;
                v1 = rotl(v1, 17);
                v3 = rotl(v3, rotE);
                v1 ^= v2;
                v3 ^= v0;
                v2 = rotl(v2, 32);
            }

            void hash24(uint64_t const nonce)
            {
                v3 ^= nonce;
                sipRound();
                sipRound();
                v0 ^= nonce;
                v2 ^= 0xffull;
                sipRound();
                sipRound();
                sipRound();
                sipRound();
            }
        };
    }
}
