// FishHash search kernel (Iron Fish). Per work-item:
//   - write big-endian nonce into header[172..180]
//   - seed = blake3(header, 180)            (64-byte XOF)
//   - mix(1024) = {seed, seed}; 32 dataset-access rounds of fnv1/xor + 64-bit mul-add
//   - collapse mix(1024) -> mixHash(256) via fnv1 chains
//   - out = blake3(seed[64] || mixHash[32]) (32 bytes)
//   - record nonce when out <= boundary (32-byte big-endian compare)
//
// KarlsenHashV2 (FISHHASH_PLUS) reuses the same DAG but differs in: an 80-byte header with a
// little-endian nonce at [72..80], a 32-byte blake3 seed zero-extended to 64, the mix-group
// index derivation, and a final blake3 over the 32-byte mixHash only. Base (Iron Fish) default.

#include "kernel/crypto/blake3.cl"
#include "kernel/crypto/fnv1.cl"
#include "kernel/fishhash/fishhash_result.cl"


// The dataset is split into chunks (each < 4 GB) across up to 8 buffers. Select the
// chunk buffer for a global item index. chunk = index / chunk_items.
inline __global uint4 const* fishhash_chunk(
    __global uint4 const* const restrict b0,
    __global uint4 const* const restrict b1,
    __global uint4 const* const restrict b2,
    __global uint4 const* const restrict b3,
    __global uint4 const* const restrict b4,
    __global uint4 const* const restrict b5,
    __global uint4 const* const restrict b6,
    __global uint4 const* const restrict b7,
    uint const chunk)
{
    switch (chunk)
    {
        case 0u: return b0;
        case 1u: return b1;
        case 2u: return b2;
        case 3u: return b3;
        case 4u: return b4;
        case 5u: return b5;
        case 6u: return b6;
        default: return b7;
    }
}


inline void fishhash_load_item(
    __global uint4 const* const restrict b0,
    __global uint4 const* const restrict b1,
    __global uint4 const* const restrict b2,
    __global uint4 const* const restrict b3,
    __global uint4 const* const restrict b4,
    __global uint4 const* const restrict b5,
    __global uint4 const* const restrict b6,
    __global uint4 const* const restrict b7,
    uint const chunk_items,
    uint const index,
    uint       out[32])
{
    uint const                   chunk = index / chunk_items;
    uint const                   off = index - chunk * chunk_items; // chunk-local item index
    __global uint4 const* const  dag = fishhash_chunk(b0, b1, b2, b3, b4, b5, b6, b7, chunk);
    size_t const                 base = (size_t)off * 8ul;
    __attribute__((opencl_unroll_hint))
    for (uint w = 0u; w < 8u; ++w)
    {
        uint4 const v = dag[base + w];
        out[w * 4u + 0u] = v.x;
        out[w * 4u + 1u] = v.y;
        out[w * 4u + 2u] = v.z;
        out[w * 4u + 3u] = v.w;
    }
}


inline uint fishhash_le32(uchar const* const b)
{
    return (uint)b[0] | ((uint)b[1] << 8) | ((uint)b[2] << 16) | ((uint)b[3] << 24);
}


// true if a <= b as 32-byte big-endian integers (matches reference bytes_lte).
inline bool fishhash_bytes_lte(uchar const* const a, __global uchar const* const b)
{
    for (uint i = 0u; i < 32u; ++i)
    {
        if (a[i] < b[i])
        {
            return true;
        }
        if (a[i] > b[i])
        {
            return false;
        }
    }
    return true;
}


// true if a <= b as 32-byte LITTLE-endian integers (KarlsenHashV2 / Kaspa target compare:
// byte 31 is most significant). The stratum supplies the boundary little-endian.
inline bool fishhash_bytes_lte_le(uchar const* const a, __global uchar const* const b)
{
    for (int i = 31; i >= 0; --i)
    {
        if (a[i] < b[i])
        {
            return true;
        }
        if (a[i] > b[i])
        {
            return false;
        }
    }
    return true;
}


__kernel
void fishhash_search(
    __global uint4 const* const restrict dag0, // dataset chunk buffers (each < 4 GB)
    __global uint4 const* const restrict dag1,
    __global uint4 const* const restrict dag2,
    __global uint4 const* const restrict dag3,
    __global uint4 const* const restrict dag4,
    __global uint4 const* const restrict dag5,
    __global uint4 const* const restrict dag6,
    __global uint4 const* const restrict dag7,
    __global t_result* const restrict result,
    __global uchar const* const restrict header, // 180 bytes (xn already set by pool)
    __global uchar const* const restrict boundary, // 32-byte big-endian target
    ulong const start_nonce,
    uint const dag_number_item,
    uint const chunk_items
#ifdef FISHHASH_DEBUG_HASH
    ,
    __global uchar* const restrict dbg_hash // 32 bytes, written for global id 0
#endif
)
{
    // Same 2D thread->nonce mapping as ethash_search: id(1)*GROUP_SIZE + id(0).
    size_t const thread_id = get_global_id(1) * GROUP_SIZE + get_global_id(0);
    ulong const  nonce = start_nonce + (ulong)thread_id;

#ifdef FISHHASH_PLUS
    // KarlsenHashV2 header: 80 bytes with the little-endian nonce at [72..80].
    uchar h[80];
    for (uint i = 0u; i < 80u; ++i)
    {
        h[i] = header[i];
    }
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        h[72u + i] = (uchar)(nonce >> (8u * i));
    }

    // seed = blake3(header) -> 32 bytes, zero-extended into 64 (upper half MUST be zero).
    uchar seed[64];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 64u; ++i)
    {
        seed[i] = 0u;
    }
    blake3_hash_chunk(h, 80u, 32u, seed);
#else
    // Local header copy with the big-endian nonce at [172..180].
    uchar h[180];
    for (uint i = 0u; i < 180u; ++i)
    {
        h[i] = header[i];
    }
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        h[172u + i] = (uchar)(nonce >> (8u * (7u - i)));
    }

    // seed = blake3(header) -> 64 bytes.
    uchar seed[64];
    blake3_hash_chunk(h, 180u, 64u, seed);
#endif

    // mix(1024) = {seed512, seed512} as 32x uint32 (little-endian word view).
    uint mix[32];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 16u; ++i)
    {
        uint const w = fishhash_le32(seed + i * 4u);
        mix[i] = w;
        mix[i + 16u] = w;
    }

    uint const M = dag_number_item;

    for (uint it = 0u; it < 32u; ++it)
    {
#ifdef FISHHASH_PLUS
        uint mixGroup[8];
        __attribute__((opencl_unroll_hint))
        for (uint c = 0u; c < 8u; ++c)
        {
            mixGroup[c] = mix[4u * c] ^ mix[4u * c + 1u] ^ mix[4u * c + 2u] ^ mix[4u * c + 3u];
        }
        uint const p0 = (mixGroup[0] ^ mixGroup[3] ^ mixGroup[6]) % M;
        uint const p1 = (mixGroup[1] ^ mixGroup[4] ^ mixGroup[7]) % M;
        uint const p2 = (mixGroup[2] ^ mixGroup[5] ^ it) % M;
#else
        uint const p0 = mix[0] % M;
        uint const p1 = mix[4] % M;
        uint const p2 = mix[8] % M;
#endif

        uint f0[32];
        uint f1[32];
        uint f2[32];
        fishhash_load_item(dag0, dag1, dag2, dag3, dag4, dag5, dag6, dag7, chunk_items, p0, f0);
        fishhash_load_item(dag0, dag1, dag2, dag3, dag4, dag5, dag6, dag7, chunk_items, p1, f1);
        fishhash_load_item(dag0, dag1, dag2, dag3, dag4, dag5, dag6, dag7, chunk_items, p2, f2);

        __attribute__((opencl_unroll_hint))
        for (uint j = 0u; j < 32u; ++j)
        {
            f1[j] = fnv1_u32(mix[j], f1[j]);
            f2[j] = mix[j] ^ f2[j];
        }

        __attribute__((opencl_unroll_hint))
        for (uint k = 0u; k < 16u; ++k)
        {
            ulong const a = ((ulong)f0[2u * k + 1u] << 32) | f0[2u * k];
            ulong const b = ((ulong)f1[2u * k + 1u] << 32) | f1[2u * k];
            ulong const c = ((ulong)f2[2u * k + 1u] << 32) | f2[2u * k];
            ulong const r = a * b + c;
            mix[2u * k] = (uint)r;
            mix[2u * k + 1u] = (uint)(r >> 32);
        }
    }

#ifdef FISHHASH_PLUS
    // Collapse mix(1024) -> mixHash(256). KarlsenHashV2 hashes ONLY the 32-byte mixHash
    // (not seed || mixHash): fnv1 chain over each group of 4 words.
    uchar finalData[32];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        uint const base = i * 4u;
        uint const h1 = fnv1_u32(mix[base], mix[base + 1u]);
        uint const h2 = fnv1_u32(h1, mix[base + 2u]);
        uint const h3 = fnv1_u32(h2, mix[base + 3u]);
        finalData[i * 4u + 0u] = (uchar)(h3);
        finalData[i * 4u + 1u] = (uchar)(h3 >> 8);
        finalData[i * 4u + 2u] = (uchar)(h3 >> 16);
        finalData[i * 4u + 3u] = (uchar)(h3 >> 24);
    }

    // out = blake3(mixHash[32]).
    uchar out[32];
    blake3_hash_chunk(finalData, 32u, 32u, out);
#else
    // Collapse mix(1024) -> mixHash(256): fnv1 chain over each group of 4 words.
    uchar finalData[96];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 64u; ++i)
    {
        finalData[i] = seed[i];
    }
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        uint const base = i * 4u;
        uint const h1 = fnv1_u32(mix[base], mix[base + 1u]);
        uint const h2 = fnv1_u32(h1, mix[base + 2u]);
        uint const h3 = fnv1_u32(h2, mix[base + 3u]);
        finalData[64u + i * 4u + 0u] = (uchar)(h3);
        finalData[64u + i * 4u + 1u] = (uchar)(h3 >> 8);
        finalData[64u + i * 4u + 2u] = (uchar)(h3 >> 16);
        finalData[64u + i * 4u + 3u] = (uchar)(h3 >> 24);
    }

    // out = blake3(seed[64] || mixHash[32]).
    uchar out[32];
    blake3_hash_chunk(finalData, 96u, 32u, out);
#endif

#ifdef FISHHASH_DEBUG_HASH
    if (thread_id == 0u)
    {
        for (uint i = 0u; i < 32u; ++i)
        {
            dbg_hash[i] = out[i];
        }
    }
#endif

#ifdef FISHHASH_PLUS
    if (fishhash_bytes_lte_le(out, boundary)) // Kaspa target is little-endian
#else
    if (fishhash_bytes_lte(out, boundary))
#endif
    {
        uint const slot = atomic_inc(&result->count);
        result->found = true;
        if (slot < 4u)
        {
            result->nonces[slot] = nonce;
        }
    }
}
