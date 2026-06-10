#include <cstring>

#include <randomx.h>

#include <algo/randomx/randomx_pow.hpp>


algo::randomx::RandomXHasher::~RandomXHasher()
{
    release();
}


void algo::randomx::RandomXHasher::release()
{
    if (nullptr != vm)
    {
        randomx_destroy_vm(vm);
        vm = nullptr;
    }
    if (nullptr != dataset)
    {
        randomx_release_dataset(dataset);
        dataset = nullptr;
    }
    if (nullptr != cache)
    {
        randomx_release_cache(cache);
        cache = nullptr;
    }
    seedSet = false;
    seedKey.clear();
    full = false;
}


bool algo::randomx::RandomXHasher::ready() const
{
    return nullptr != vm;
}


bool algo::randomx::RandomXHasher::init(void const* seed, size_t const seedSize, bool const lightMode)
{
    std::string const newSeed{ static_cast<char const*>(seed), seedSize };

    // Same key AND same mode, already built: nothing to do (RandomX re-keys only per
    // Monero epoch). A mode change with an unchanged seed must still rebuild.
    if (true == seedSet && nullptr != vm && newSeed == seedKey && full == (false == lightMode))
    {
        return true;
    }

    // Seed (or mode) changed -- rebuild from scratch. Epoch changes are rare, so the
    // simplicity of a full rebuild outweighs the cost of an in-place cache/dataset swap.
    release();

    randomx_flags flags{ randomx_get_flags() };

    cache = randomx_alloc_cache(flags);
    if (nullptr == cache)
    {
        return false;
    }
    randomx_init_cache(cache, seed, seedSize);

    if (false == lightMode)
    {
        flags = flags | RANDOMX_FLAG_FULL_MEM;

        dataset = randomx_alloc_dataset(flags);
        if (nullptr == dataset)
        {
            release();
            return false;
        }

        // Single-threaded dataset init. A CPU resolver can parallelize this across the
        // item range with several threads; correctness is identical.
        randomx_init_dataset(dataset, cache, 0u, randomx_dataset_item_count());

        // The full-memory VM reads from the dataset; the cache is no longer needed.
        randomx_release_cache(cache);
        cache = nullptr;

        vm = randomx_create_vm(flags, nullptr, dataset);
    }
    else
    {
        vm = randomx_create_vm(flags, cache, nullptr);
    }

    if (nullptr == vm)
    {
        release();
        return false;
    }

    seedKey = newSeed;
    seedSet = true;
    full = (false == lightMode);
    return true;
}


void algo::randomx::RandomXHasher::hash(void const* input, size_t const inputSize, algo::hash256& out)
{
    randomx_calculate_hash(vm, input, inputSize, out.ubytes);
}


bool algo::randomx::calculateHash(
    void const*    seed,
    size_t const   seedSize,
    void const*    input,
    size_t const   inputSize,
    algo::hash256& out)
{
    algo::randomx::RandomXHasher hasher{};
    if (false == hasher.init(seed, seedSize, true))
    {
        std::memset(out.ubytes, 0, sizeof(out.ubytes));
        return false;
    }
    hasher.hash(input, inputSize, out);
    return true;
}
