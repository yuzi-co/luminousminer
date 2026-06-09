#include "kernel/common/rotate_byte.cl"
#include "kernel/common/to_u4.cl"
#include "kernel/common/xor.cl"

#include "kernel/crypto/fnv1.cl"
#include "kernel/crypto/keccak_f1600.cl"

#include "kernel/ethash/ethash_result.cl"


// Octopus search kernel — one work-item per nonce. Reproduces the CPU oracle
// algo::octopus::octopusHash bit-for-bit:
//   multi_eval (siphash-seeded polynomial over GF(OCT_MOD))
//   -> seed = Keccak-512(header[32] || thread_result[8])
//   -> hashimoto (OCT_ACCESSES x MIX_NODES dataset reads, reads prebuilt DAG)
//   -> Keccak-256(seed[64] || mix[32]) -> 256-bit hash compared big-endian to boundary.
// DAG node j occupies dag[j*4 .. j*4+3] (4 uint4 = 64 bytes), matching octopus_dag.cl.
// The >= 4 GiB dataset is split across up to 8 chunk buffers (each < 4 GiB); a node's
// global index selects its chunk = nodeIndex / chunk_items. A single-buffer test binds
// the same buffer to all 8 slots with chunk_items > total nodes (everything in chunk 0).
// Correctness-first: every nonce recomputes the full d[OCT_N] polynomial (the
// reference shares it across a 32-lane warp; that optimization is deferred).

#define OCT_MOD             1032193u
#define OCT_N               1024u
#define OCT_B               11u
#define OCT_WARP            32u
#define OCT_DATA_PER_THREAD 32u
#define OCT_MIX_WORDS       64u
#define OCT_MIX_NODES       4u
#define OCT_NODE_WORDS      16u
#define OCT_ACCESSES        32u


__constant ulong OCTOPUS_KECCAK_RC[24] =
{
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL,
    0x8000000080008000UL, 0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008AUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};


inline
void octopus_keccak_permute(ulong* const restrict state)
{
    __attribute__((opencl_unroll_hint(1)))
    for (uint round = 0u; round < 24u; ++round)
    {
        keccak_f1600_round(state, OCTOPUS_KECCAK_RC[round]);
    }
}


inline
ulong octopus_swap_u64(ulong const x)
{
    return as_ulong(as_uchar8(x).s76543210);
}


////////////////////////////////////////////////////////////////////////////////
// Prime-field helpers (GF(OCT_MOD)).
inline
uint octopus_mulmod(uint const a, uint const b)
{
    return (uint)(((ulong)a * (ulong)b) % OCT_MOD);
}


inline
uint octopus_powermod(uint a, uint n)
{
    uint res = 1u;
    while (0u != n)
    {
        if (0u != (n & 1u))
        {
            res = octopus_mulmod(res, a);
        }
        a = octopus_mulmod(a, a);
        n >>= 1u;
    }
    return res;
}


inline
uint octopus_gcd(uint a, uint b)
{
    while (0u != b)
    {
        uint const t = a % b;
        a = b;
        b = t;
    }
    return a;
}


inline
uint octopus_remap_param(ulong const h)
{
    uint e = (uint)(h % (ulong)(OCT_MOD - 2u) + 1u);
    while (true)
    {
        uint const g = octopus_gcd(e, OCT_MOD - 1u);
        if (1u == g)
        {
            break;
        }
        e /= g;
    }
    return octopus_powermod(OCT_B, e);
}


////////////////////////////////////////////////////////////////////////////////
// SipHash (open-cfxmine siphash.h, rotE=21) used to seed the polynomial.
inline
void octopus_sip_round(ulong* const restrict v)
{
    v[0] += v[1];
    v[2] += v[3];
    v[1] = rol_u64(v[1], 13u);
    v[3] = rol_u64(v[3], 16u);
    v[1] ^= v[0];
    v[3] ^= v[2];
    v[0] = rol_u64(v[0], 32u);
    v[2] += v[1];
    v[0] += v[3];
    v[1] = rol_u64(v[1], 17u);
    v[3] = rol_u64(v[3], 21u);
    v[1] ^= v[2];
    v[3] ^= v[0];
    v[2] = rol_u64(v[2], 32u);
}


inline
void octopus_sip_hash24(ulong* const restrict v, ulong const nonce)
{
    v[3] ^= nonce;
    octopus_sip_round(v);
    octopus_sip_round(v);
    v[0] ^= nonce;
    v[2] ^= 0xffUL;
    octopus_sip_round(v);
    octopus_sip_round(v);
    octopus_sip_round(v);
    octopus_sip_round(v);
}


inline
void octopus_compute_d(
    ulong const* const restrict header,
    ulong nonce,
    uint* const restrict d)
{
    nonce /= OCT_WARP;
    for (uint lid = 0u; lid < OCT_WARP; ++lid)
    {
        ulong v[4];
        v[0] = header[0];
        v[1] = header[1];
        v[2] = header[2];
        v[3] = header[3];
        octopus_sip_hash24(v, nonce * OCT_WARP + lid);
        for (uint i = 0u; i < OCT_DATA_PER_THREAD; ++i)
        {
            octopus_sip_round(v);
            ulong const lanes = (v[0] ^ v[1]) ^ (v[2] ^ v[3]);
            d[i * OCT_WARP + lid] = (uint)((lanes & 0xffffffffUL) % OCT_MOD);
        }
    }
}


// Returns thread_result; fills result[OCT_DATA_PER_THREAD].
inline
ulong octopus_multi_eval(
    ulong const* const restrict header,
    ulong const nonce,
    uint* const restrict d,
    uint* const restrict result)
{
    uint const a = octopus_remap_param(header[0]);
    uint const b = octopus_remap_param(header[1]);
    uint       c;
    {
        ulong h = header[2];
        while (true)
        {
            c = octopus_remap_param(h);
            uint const lhs = octopus_mulmod(b, b);
            uint const rhs = (uint)(((ulong)4u * (ulong)a * (ulong)c) % OCT_MOD);
            if (lhs != rhs)
            {
                break;
            }
            ++h;
        }
    }
    uint const w  = octopus_remap_param(header[3]);
    uint const w2 = octopus_mulmod(w, w);

    octopus_compute_d(header, nonce, d);

    uint wpow  = 1u;
    uint w2pow = 1u;
    uint const lane = (uint)(nonce % OCT_WARP);
    for (uint lid = 0u; lid < lane; ++lid)
    {
        wpow  = octopus_mulmod(wpow, w);
        w2pow = octopus_mulmod(w2pow, w2);
    }
    uint fullWpow  = wpow;
    uint fullW2pow = w2pow;
    for (uint lid = lane; lid < OCT_WARP; ++lid)
    {
        fullWpow  = octopus_mulmod(fullWpow, w);
        fullW2pow = octopus_mulmod(fullW2pow, w2);
    }

    ulong threadResult = 0ul;
    for (uint i = 0u; i < OCT_DATA_PER_THREAD; ++i)
    {
        uint const x = (uint)(((ulong)a * (ulong)w2pow + (ulong)b * (ulong)wpow + (ulong)c) % OCT_MOD);
        uint       pv = 0u;
        __attribute__((opencl_unroll_hint(1)))
        for (uint j = OCT_N; j--;)
        {
            pv = (uint)(((ulong)pv * (ulong)x + (ulong)d[j]) % OCT_MOD);
        }
        result[i] = pv;
        threadResult = threadResult * 0x01000193ul ^ (ulong)pv;
        if (i + 1u < OCT_DATA_PER_THREAD)
        {
            wpow  = octopus_mulmod(wpow, fullWpow);
            w2pow = octopus_mulmod(w2pow, fullW2pow);
        }
    }
    return threadResult;
}


// Select the chunk buffer holding a node's global index. chunk = nodeIndex / chunk_items.
inline
__global uint4 const* octopus_chunk(
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


__kernel
void octopus_search(
    __global uint4 const* const restrict dag0,  // dataset chunk buffers (each < 4 GiB)
    __global uint4 const* const restrict dag1,
    __global uint4 const* const restrict dag2,
    __global uint4 const* const restrict dag3,
    __global uint4 const* const restrict dag4,
    __global uint4 const* const restrict dag5,
    __global uint4 const* const restrict dag6,
    __global uint4 const* const restrict dag7,
    __global t_result* const restrict result,
    __constant ulong const* const restrict header,
    ulong const start_nonce,
    uint const num_full_pages,
    uint const chunk_items,                     // DAG nodes per chunk buffer
    ulong4 const boundary)
{
    ulong const nonce = start_nonce + (ulong)(get_global_id(1) * GROUP_SIZE + get_global_id(0));

    ulong header4[4];
    header4[0] = header[0];
    header4[1] = header[1];
    header4[2] = header[2];
    header4[3] = header[3];

    uint  d[OCT_N];
    uint  result_v[OCT_DATA_PER_THREAD];
    ulong const threadResult = octopus_multi_eval(header4, nonce, d, result_v);

    ////////////////////////////////////////////////////////////////////////////
    // seed = Keccak-512(header[32] || thread_result[8])  (40-byte input, rate 72)
    ulong state[25];
    state[0] = header4[0];
    state[1] = header4[1];
    state[2] = header4[2];
    state[3] = header4[3];
    state[4] = threadResult;
    state[5] = 0x0000000000000001UL;
    state[6] = 0ul;
    state[7] = 0ul;
    state[8] = 0x8000000000000000UL;
    __attribute__((opencl_unroll_hint))
    for (uint i = 9u; i < 25u; ++i)
    {
        state[i] = 0ul;
    }
    octopus_keccak_permute(state);

    ulong seed64[8];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        seed64[i] = state[i];
    }

    uint seedW[16];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        seedW[2u * i]      = (uint)seed64[i];
        seedW[2u * i + 1u] = (uint)(seed64[i] >> 32);
    }

    ////////////////////////////////////////////////////////////////////////////
    // hashimoto
    uint mix[OCT_MIX_WORDS];
    __attribute__((opencl_unroll_hint))
    for (uint w = 0u; w < OCT_MIX_WORDS; ++w)
    {
        mix[w] = seedW[w % OCT_NODE_WORDS];
    }
    uint const seedWord0 = seedW[0];

    for (uint i = 0u; i < OCT_ACCESSES; ++i)
    {
        uint const index = fnv1_u32(seedWord0 ^ i ^ result_v[i], mix[i]) % num_full_pages;
        for (uint n = 0u; n < OCT_MIX_NODES; ++n)
        {
            uint const nodeIndex = index * OCT_MIX_NODES + n;
            uint const chunk     = nodeIndex / chunk_items;
            uint const off       = nodeIndex - chunk * chunk_items;
            uint const base      = off * 4u;
            __global uint4 const* const restrict dag =
                octopus_chunk(dag0, dag1, dag2, dag3, dag4, dag5, dag6, dag7, chunk);
            uint* const mixNode = mix + n * OCT_NODE_WORDS;
            __attribute__((opencl_unroll_hint))
            for (uint k = 0u; k < 4u; ++k)
            {
                uint4 const dn = dag[base + k];
                mixNode[k * 4u]      = fnv1_u32(mixNode[k * 4u],      dn.x);
                mixNode[k * 4u + 1u] = fnv1_u32(mixNode[k * 4u + 1u], dn.y);
                mixNode[k * 4u + 2u] = fnv1_u32(mixNode[k * 4u + 2u], dn.z);
                mixNode[k * 4u + 3u] = fnv1_u32(mixNode[k * 4u + 3u], dn.w);
            }
        }
    }

    __attribute__((opencl_unroll_hint))
    for (uint w = 0u; w < OCT_MIX_WORDS; w += 4u)
    {
        uint reduction = mix[w];
        reduction = fnv1_u32(reduction, mix[w + 1u]);
        reduction = fnv1_u32(reduction, mix[w + 2u]);
        reduction = fnv1_u32(reduction, mix[w + 3u]);
        mix[w / 4u] = reduction;
    }
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        mix[i] = fnv1_u32(mix[i], mix[8u + i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // out = Keccak-256(seed[64] || mix[32])  (96-byte input, rate 136)
    state[0] = seed64[0];
    state[1] = seed64[1];
    state[2] = seed64[2];
    state[3] = seed64[3];
    state[4] = seed64[4];
    state[5] = seed64[5];
    state[6] = seed64[6];
    state[7] = seed64[7];
    state[8]  = ((ulong)mix[1] << 32) | (ulong)mix[0];
    state[9]  = ((ulong)mix[3] << 32) | (ulong)mix[2];
    state[10] = ((ulong)mix[5] << 32) | (ulong)mix[4];
    state[11] = ((ulong)mix[7] << 32) | (ulong)mix[6];
    state[12] = 0x0000000000000001UL;
    state[13] = 0ul;
    state[14] = 0ul;
    state[15] = 0ul;
    state[16] = 0x8000000000000000UL;
    __attribute__((opencl_unroll_hint))
    for (uint i = 17u; i < 25u; ++i)
    {
        state[i] = 0ul;
    }
    octopus_keccak_permute(state);

    ////////////////////////////////////////////////////////////////////////////
    // Big-endian 256-bit compare: hash <= boundary (octopus_check_difficulty).
    ulong const h0 = octopus_swap_u64(state[0]);
    ulong const h1 = octopus_swap_u64(state[1]);
    ulong const h2 = octopus_swap_u64(state[2]);
    ulong const h3 = octopus_swap_u64(state[3]);

    bool win;
    if (h0 != boundary.s0)
    {
        win = h0 < boundary.s0;
    }
    else if (h1 != boundary.s1)
    {
        win = h1 < boundary.s1;
    }
    else if (h2 != boundary.s2)
    {
        win = h2 < boundary.s2;
    }
    else if (h3 != boundary.s3)
    {
        win = h3 < boundary.s3;
    }
    else
    {
        win = true;
    }

    if (win)
    {
        result->found = true;
        uint const index = atomic_inc(&result->count);
        if (4u > index)
        {
            result->nonces[index] = nonce;
        }
    }
}
