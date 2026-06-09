#include <cstring>

#include <algo/fishhash/fishhash.hpp>
#include <algo/fishhash/reference/FishHash.h>


namespace FishHash
{
    // Internal reference function (not exposed in FishHash.h).
    hash1024 calculate_dataset_item_1024(fishhash_context const& ctx, uint32_t index) noexcept;
}


namespace
{
    inline FishHash::fishhash_context const* native(algo::fishhash::Context const* ctx)
    {
        return reinterpret_cast<FishHash::fishhash_context const*>(ctx);
    }

    inline FishHash::fishhash_context* native(algo::fishhash::Context* ctx)
    {
        return reinterpret_cast<FishHash::fishhash_context*>(ctx);
    }
}


algo::fishhash::Context* algo::fishhash::getContext(bool full)
{
    return reinterpret_cast<algo::fishhash::Context*>(FishHash::get_context(full));
}


void algo::fishhash::freeContext([[maybe_unused]] algo::fishhash::Context* ctx)
{
    // The reference manages a process-wide shared context; nothing to free per-call.
}


void algo::fishhash::prebuildDataset(algo::fishhash::Context* ctx, uint32_t numThreads)
{
    FishHash::prebuild_dataset(native(ctx), numThreads);
}


void algo::fishhash::hash(
    uint8_t*                       output,
    algo::fishhash::Context const* ctx,
    uint8_t const*                 header,
    uint64_t                       headerSize)
{
    FishHash::hash(output, native(ctx), header, headerSize);
}


void algo::fishhash::hashPlus(
    uint8_t*                       output,
    algo::fishhash::Context const* ctx,
    uint8_t const*                 header,
    uint64_t                       headerSize)
{
    FishHash::hash_karlsen(output, native(ctx), header, headerSize);
}


algo::hash512 const* algo::fishhash::lightCache(algo::fishhash::Context const* ctx)
{
    return reinterpret_cast<algo::hash512 const*>(native(ctx)->light_cache);
}


int32_t algo::fishhash::lightCacheNumItems(algo::fishhash::Context const* ctx)
{
    return native(ctx)->light_cache_num_items;
}


algo::hash1024 algo::fishhash::datasetItem(algo::fishhash::Context const* ctx, uint32_t index)
{
    FishHash::hash1024 const item{ FishHash::calculate_dataset_item_1024(*native(ctx), index) };
    algo::hash1024     out{};
    std::memcpy(out.ubytes, item.bytes, sizeof(out.ubytes));
    return out;
}
