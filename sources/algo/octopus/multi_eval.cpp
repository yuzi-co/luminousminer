#include <algo/octopus/multi_eval.hpp>
#include <algo/octopus/siphash.hpp>


namespace
{
    constexpr uint32_t MOD{ algo::octopus::NTT_MOD };

    uint32_t castU32(uint64_t const v)
    {
        return static_cast<uint32_t>(v);
    }

    uint32_t gcd(uint32_t const a, uint32_t const b)
    {
        return 0u != b ? gcd(b, a % b) : a;
    }

    uint32_t powerMod(uint32_t a, uint32_t n)
    {
        uint32_t res{ 1u };
        while (0u != n)
        {
            if (0u != (n & 1u))
            {
                res = castU32((uint64_t) res * a % MOD);
            }
            a = castU32((uint64_t) a * a % MOD);
            n >>= 1u;
        }
        return res;
    }

    uint32_t remapParam(uint64_t const h)
    {
        uint32_t e{ castU32(h % (MOD - 2u) + 1u) };
        while (true)
        {
            uint32_t const g{ gcd(e, MOD - 1u) };
            if (1u == g)
            {
                break;
            }
            e /= g;
        }
        return powerMod(algo::octopus::NTT_B, e);
    }

    // 64-bit FNV fold (reference fnv.h macro: x*0x01000193 ^ y) for thread_result.
    uint64_t fnv64(uint64_t const x, uint64_t const y)
    {
        return x * 0x01000193ull ^ y;
    }
}


algo::octopus::Abcw algo::octopus::makeAbcw(algo::hash256 const& header)
{
    algo::octopus::Abcw p{};
    p.a = remapParam(header.word64[0]);
    p.b = remapParam(header.word64[1]);
    {
        uint64_t h{ header.word64[2] };
        while (true)
        {
            p.c = remapParam(h);
            if (((uint64_t) p.b * p.b % MOD) != ((uint64_t) 4u * p.a * p.c % MOD))
            {
                break;
            }
            ++h;
        }
    }
    p.w = remapParam(header.word64[3]);
    return p;
}


void algo::octopus::computeD(algo::hash256 const& header, uint64_t nonce, uint32_t* const d)
{
    nonce /= algo::octopus::WARP_SIZE;
    for (uint32_t lid{ 0u }; lid < algo::octopus::WARP_SIZE; ++lid)
    {
        algo::octopus::SipHashState<> state{ header.word64 };
        state.hash24(nonce * algo::octopus::WARP_SIZE + lid);
        for (uint32_t i{ 0u }; i < algo::octopus::DATA_PER_THREAD; ++i)
        {
            state.sipRound();
            d[i * algo::octopus::WARP_SIZE + lid] = castU32((state.xorLanes() & UINT32_MAX) % MOD);
        }
    }
}


std::pair<uint64_t, std::vector<uint32_t>> algo::octopus::multiEval(algo::hash256 const& header, uint64_t const nonce)
{
    algo::octopus::Abcw const p{ algo::octopus::makeAbcw(header) };
    uint32_t const           a{ p.a };
    uint32_t const           b{ p.b };
    uint32_t const           c{ p.c };
    uint32_t const           w{ p.w };
    uint32_t const           w2{ castU32((uint64_t) w * w % MOD) };

    std::vector<uint32_t> d(algo::octopus::NTT_N);
    algo::octopus::computeD(header, nonce, d.data());

    uint32_t wpow{ 1u };
    uint32_t w2pow{ 1u };
    for (uint32_t lid{ 0u }; lid < nonce % algo::octopus::WARP_SIZE; ++lid)
    {
        wpow = castU32((uint64_t) wpow * w % MOD);
        w2pow = castU32((uint64_t) w2pow * w2 % MOD);
    }
    uint32_t fullWpow{ wpow };
    uint32_t fullW2pow{ w2pow };
    for (uint32_t lid{ castU32(nonce % algo::octopus::WARP_SIZE) }; lid < algo::octopus::WARP_SIZE; ++lid)
    {
        fullWpow = castU32((uint64_t) fullWpow * w % MOD);
        fullW2pow = castU32((uint64_t) fullW2pow * w2 % MOD);
    }

    uint64_t              threadResult{ 0ull };
    std::vector<uint32_t> result(algo::octopus::DATA_PER_THREAD);
    for (uint32_t i{ 0u }; i < algo::octopus::DATA_PER_THREAD; ++i)
    {
        uint32_t const x{ castU32(((uint64_t) a * w2pow + (uint64_t) b * wpow + c) % MOD) };
        uint32_t       pv{ 0u };
        for (uint32_t j{ algo::octopus::NTT_N }; j--;)
        {
            pv = castU32(((uint64_t) pv * x + d[j]) % MOD);
        }
        result[i] = pv;
        threadResult = fnv64(threadResult, (uint64_t) pv);
        if (i + 1u < algo::octopus::DATA_PER_THREAD)
        {
            wpow = castU32((uint64_t) wpow * fullWpow % MOD);
            w2pow = castU32((uint64_t) w2pow * fullW2pow % MOD);
        }
    }
    return std::make_pair(threadResult, result);
}
