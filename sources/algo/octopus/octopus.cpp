#include <algo/octopus/octopus.hpp>


namespace
{
    bool isPrimeUnits(uint64_t const number)
    {
        if (number < 2ull)
        {
            return false;
        }
        if (0ull == number % 2ull)
        {
            return 2ull == number;
        }
        for (uint64_t i{ 3ull }; i * i <= number; i += 2ull)
        {
            if (0ull == number % i)
            {
                return false;
            }
        }
        return true;
    }
}


uint64_t algo::octopus::getEpoch(uint64_t const blockNumber)
{
    return blockNumber / algo::octopus::EPOCH_LENGTH;
}


uint64_t algo::octopus::getCacheSize(uint64_t const blockNumber)
{
    uint64_t size{ algo::octopus::LIGHT_CACHE_INIT_SIZE
                   + algo::octopus::LIGHT_CACHE_GROWTH * algo::octopus::getEpoch(blockNumber) };
    size -= algo::octopus::HASH_BYTES;
    while (false == isPrimeUnits(size / algo::octopus::HASH_BYTES))
    {
        size -= 2ull * algo::octopus::HASH_BYTES;
    }
    return size;
}


uint64_t algo::octopus::getDataSize(uint64_t const blockNumber)
{
    uint64_t size{ algo::octopus::DAG_INIT_SIZE
                   + algo::octopus::DAG_GROWTH * algo::octopus::getEpoch(blockNumber) };
    size -= algo::octopus::MIX_BYTES;
    while (false == isPrimeUnits(size / algo::octopus::MIX_BYTES))
    {
        size -= 2ull * algo::octopus::MIX_BYTES;
    }
    return size;
}
