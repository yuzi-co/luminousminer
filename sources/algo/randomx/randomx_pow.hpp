#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <algo/hash.hpp>


// Forward declarations of the opaque vendored RandomX handles, so this header does not
// pull in <randomx.h> (kept private to randomx_pow.cpp). These match the struct tags in
// the vendored randomx.h.
struct randomx_cache;
struct randomx_dataset;
struct randomx_vm;


// Host-side RandomX (Monero rx/0) hashing wrapper over the vendored tevador/RandomX
// library (sources/algo/randomx/randomx). This is the device-neutral primitive: the
// oracle for the KAT tests today, and the per-nonce hash a future CPU resolver will
// call. No GPU path -- RandomX is a CPU PoW.
namespace algo
{
    namespace randomx
    {
        class RandomXHasher
        {
          public:
            RandomXHasher() = default;
            ~RandomXHasher();

            RandomXHasher(RandomXHasher const&) = delete;
            RandomXHasher(RandomXHasher&&) = delete;
            RandomXHasher& operator=(RandomXHasher const&) = delete;
            RandomXHasher& operator=(RandomXHasher&&) = delete;

            // Initialize (or re-key) for a RandomX seed (the Monero `seed_hash`). The cache
            // and VM are allocated on first use; later calls only rebuild the cache when the
            // seed actually changes. `lightMode == true` uses the ~256 MiB cache-backed VM;
            // `false` allocates the full ~2 GiB dataset for maximum throughput. Returns false
            // on allocation failure.
            bool init(void const* seed, size_t const seedSize, bool const lightMode = true);

            // Hash `input` (the Monero hashing blob with the nonce already patched in at
            // offset 39); `out` receives the 32-byte result. Requires a prior init().
            void hash(void const* input, size_t const inputSize, algo::hash256& out);

            bool ready() const;

          private:
            void release();

            ::randomx_cache*   cache{ nullptr };
            ::randomx_dataset* dataset{ nullptr };
            ::randomx_vm*      vm{ nullptr };
            std::string        seedKey{};
            bool               seedSet{ false };
            bool               full{ false };
        };

        // Stateless one-shot hash used by the KAT / verification paths: allocates, hashes,
        // and frees. Not for the mining hot loop -- use RandomXHasher there. Returns false
        // (leaving `out` zeroed) on allocation failure rather than reporting a bogus hash.
        bool calculateHash(
            void const*    seed,
            size_t const   seedSize,
            void const*    input,
            size_t const   inputSize,
            algo::hash256& out);
    }
}
