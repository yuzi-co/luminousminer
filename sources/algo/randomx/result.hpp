#pragma once

#include <cstdint>
#include <string>

#include <algo/hash.hpp>


// NOTE: these types are the result contract for the forthcoming CPU resolver and are not
// referenced yet in this branch. The layout is provisional and may change when the
// resolver lands.
namespace algo
{
    namespace randomx
    {
        // RandomX is a single-share-per-nonce CPU PoW (Monero rx/0): a worker tests one
        // nonce at a time, so unlike the GPU batch resolvers there is no MAX_RESULT fan-out.
        struct Result
        {
            bool          found{ false };
            uint64_t      nonce{ 0ull };
            algo::hash256 hash{};
        };

        // Share payload handed to the stratum submit (Monero: job_id + nonce + result hash).
        struct ResultShare
        {
            bool        found{ false };
            uint64_t    nonce{ 0ull };
            std::string jobId{};
            std::string nonceHex{};
            std::string resultHex{};
        };
    }
}
