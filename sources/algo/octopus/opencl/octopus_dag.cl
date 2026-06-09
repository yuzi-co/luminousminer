#include "kernel/common/rotate_byte.cl"
#include "kernel/common/to_u4.cl"
#include "kernel/common/xor.cl"

#include "kernel/crypto/fnv1.cl"
#include "kernel/crypto/keccak_f1600.cl"


// Octopus DAG-item generation. Identical structure to ethash (Keccak-512 +
// DAG_ITEM_PARENTS FNV passes + Keccak-512) but emits ONE 64-byte node per
// work-item (the search reads nodes index*MIX_NODES+n), not ethash's 2-node
// hash1024 packing. Verified bit-exact vs algo::octopus::calcDatasetItem.
inline
void build_item_mix(
    __global uint4 const* const restrict cache,
    uint4* const restrict item,
    uint const cache_number_item,
    uint const cache_index)
{
    uint const cache_index_access = (cache_index % cache_number_item) * 4u;
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 4u; ++i)
    {
        uint4 cache_tmp = cache[cache_index_access + i];
        fnv1_v4_from_v4(&item[i], &cache_tmp);
    }
}


__kernel
void octopus_build_dag(
    __global uint4* const restrict dag,
    __global uint4 const* const restrict cache,
    uint const dag_item_parent,
    uint const dag_number_item,
    uint const cache_number_item)
{
    ////////////////////////////////////////////////////////////////////////////
    uint const node_index = get_global_id(0) + get_global_id(1) * GROUP_SIZE;
    if (node_index >= dag_number_item)
    {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint const cache_index = (node_index % cache_number_item) * 4u;
    uint4      item[4];
    __attribute__((opencl_unroll_hint))
    for (uchar i = 0; i < 4; ++i)
    {
        item[i] = cache[cache_index + i];
    }
    item[0].x ^= node_index;

    ////////////////////////////////////////////////////////////////////////////
    keccak_f1600(item);

    ////////////////////////////////////////////////////////////////////////////
    uint k = 0;
    __attribute__((opencl_unroll_hint))
    for (uchar i = 0; i < DAG_LOOP; ++i)
    {
        __attribute__((opencl_unroll_hint))
        for (uchar j = 0; j < 4u; ++j)
        {
            build_item_mix(cache, item, cache_number_item, fnv1_u32(node_index ^ k,        item[j & 3u].x));
            build_item_mix(cache, item, cache_number_item, fnv1_u32(node_index ^ (k + 1u), item[j & 3u].y));
            build_item_mix(cache, item, cache_number_item, fnv1_u32(node_index ^ (k + 2u), item[j & 3u].z));
            build_item_mix(cache, item, cache_number_item, fnv1_u32(node_index ^ (k + 3u), item[j & 3u].w));
            k += 4u;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    keccak_f1600(item);

    ////////////////////////////////////////////////////////////////////////////
    uint const gap_index = node_index * 4u;
    __attribute__((opencl_unroll_hint))
    for (uint x = 0u; x < 4u; ++x)
    {
        dag[gap_index + x] = item[x];
    }
}
