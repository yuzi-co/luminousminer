// FishHash full-dataset item generation (calculate_dataset_item_1024).
// Mirrors iron-fish/fish-hash reference: 2 sub-items per dataset item, each is
// keccak512(cache[idx] with word0^=idx) then FULL_DATASET_ITEM_PARENTS (512) fnv1
// mixes against parent cache lines, then keccak512. Two 64-byte halves -> 128-byte item.
//
// The full dataset is ~4.83 GB, which exceeds the 4 GB single-buffer / 32-bit byte
// addressing limit of AMD OpenCL. It is therefore split into chunks, each built into
// its own < 4 GB buffer: this kernel writes one chunk, items [base, base+count), to
// the chunk-local offset.

#include "kernel/common/rotate_byte.cl"
#include "kernel/common/to_u4.cl"
#include "kernel/common/xor.cl"

#include "kernel/crypto/fnv1.cl"
#include "kernel/crypto/keccak_f1600.cl"


// item = fnv1(item, cache[parent]) over all 16 word32 (4x uint4).
inline void fishhash_parent_mix(
    __global uint4 const* const restrict cache,
    uint4* const restrict item,
    uint const cache_number_item,
    uint const parent_index)
{
    uint const base = (parent_index % cache_number_item) * 4u;
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 4u; ++i)
    {
        uint4 c = cache[base + i];
        fnv1_v4_from_v4(&item[i], &c);
    }
}


// Returns word32[j] (j in 0..15) of the 4x uint4 item.
inline uint fishhash_item_word(uint4 const* const item, uint const j)
{
    uint4 const v = item[j >> 2];
    uint const  l = j & 3u;
    return (l == 0u) ? v.x : (l == 1u) ? v.y : (l == 2u) ? v.z : v.w;
}


__kernel
void fishhash_build_dag(
    __global uint4* const restrict dag,             // chunk buffer (chunk-local indexing)
    __global uint4 const* const restrict cache,
    uint const dag_base,                            // global index of this chunk's first item
    uint const chunk_count,                         // number of items in this chunk
    uint const cache_number_item)
{
    uint const local_index = get_global_id(0) + get_global_id(1) * GROUP_SIZE;
    if (local_index >= chunk_count)
    {
        return;
    }
    uint const dag_index = dag_base + local_index; // global item index (used by the algorithm)

    __attribute__((opencl_unroll_hint))
    for (uchar loop = 0; loop < 2; ++loop)
    {
        uint const index = (dag_index * 2u) + loop;
        uint const cache_index = (index % cache_number_item) * 4u;

        uint4 item[4];
        __attribute__((opencl_unroll_hint))
        for (uchar i = 0; i < 4; ++i)
        {
            item[i] = cache[cache_index + i];
        }
        item[0].x ^= index;

        keccak_f1600(item);

        // 512 parents: t = fnv1(index ^ round, item.word32[round & 15]); item = fnv1(item, cache[t]).
        for (uint round = 0u; round < FISHHASH_DAG_ITEM_PARENTS; ++round)
        {
            uint const w = fishhash_item_word(item, round & 15u);
            uint const t = fnv1_u32(index ^ round, w);
            fishhash_parent_mix(cache, item, cache_number_item, t);
        }

        keccak_f1600(item);

        // chunk-local write offset (stays < 4 GB because each chunk is < 4 GB).
        size_t const gap_index = ((size_t)local_index * 8ul) + ((size_t)loop * 4ul);
        __attribute__((opencl_unroll_hint))
        for (uint x = 0u; x < 4u; ++x)
        {
            dag[gap_index + x] = item[x];
        }
    }
}
