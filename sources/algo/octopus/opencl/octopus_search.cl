#include "kernel/common/rotate_byte.cl"
#include "kernel/common/to_u4.cl"
#include "kernel/common/xor.cl"

#include "kernel/crypto/fnv1.cl"
#include "kernel/crypto/keccak_f1600.cl"

#include "kernel/ethash/ethash_result.cl"


// Octopus search kernel. Reproduces the CPU oracle algo::octopus::octopusHash:
//   multi_eval (siphash-seeded polynomial over GF(OCT_MOD))
//   -> seed = Keccak-512(header[32] || thread_result[8])
//   -> hashimoto (OCT_ACCESSES x MIX_NODES dataset reads, reads prebuilt DAG)
//   -> Keccak-256(seed[64] || mix[32]) -> 256-bit hash compared big-endian to boundary.
//
// Three compile-time optimization switches (all ON by default; the benchmark sweep flips
// them to produce baselines):
//   OCT_COOP_D    1 = a 32-lane subgroup shares one d[OCT_N] in LDS (computed once per 32
//                     nonces); 0 = each work-item builds its own d[OCT_N] in private memory
//                     (spills to scratch -> the original slow baseline).
//   OCT_USE_BARRETT 1 = Barrett reduction for GF(MOD); 0 = hardware 64-bit `% MOD` divide.
//   OCT_INTERLEAVE  number of independent polynomial evaluations run together in the Horner
//                   inner loop (ILP + one d[j] read per group); 1 = no interleave.
//
// start_nonce must be 32-aligned so the 32 lanes of a subgroup share a warp base
// (nonce / 32); lane = nonce % 32. DAG node j occupies dag[j*4 .. j*4+3]. The >= 4 GiB
// dataset is split across up to 16 chunk buffers (each 1 GiB, under the AMD ~2 GiB signed
// buffer-addressing limit); chunk = nodeIndex / chunk_items.

#ifndef OCT_COOP_D
#define OCT_COOP_D 1
#endif
#ifndef OCT_USE_BARRETT
#define OCT_USE_BARRETT 1
#endif
#ifndef OCT_INTERLEAVE
#define OCT_INTERLEAVE 16u
#endif
// OCT_LAZY_HORNER 1 = the Horner accumulator stays partially reduced in [0, 2*MOD); the
//   per-step canonical conditional subtraction is dropped (it is the hottest instruction in
//   the kernel — 32*1024 modmuls/nonce) and applied only once on store. 0 = canonical every
//   step (the pre-lazy baseline).
#ifndef OCT_LAZY_HORNER
#define OCT_LAZY_HORNER 1
#endif
// OCT_USE_MONT 1 = the Horner modmul uses Montgomery reduction (R = 2^32) instead of
//   Barrett (the default / production path: ~+27% on gfx1201). Montgomery costs ~5
//   multiplies/step vs Barrett's ~7 (the reduction shift is a free word-select, not a
//   cross-word >>41), and the accumulator is kept lazily in [0, 2*MOD). Coefficients d[] and
//   the evaluation point xs are pre-converted to Montgomery form; result_v is converted back.
//   0 = Barrett (OCT_LAZY_HORNER path above). Bit-exact vs the CPU oracle either way.
//   (A 24-bit-mul split of the product was measured at -22% on gfx1201 and dropped — RDNA4
//   runs 32-bit mul_lo/mul_hi at full rate, so mul24 only adds shift/mask overhead.)
#ifndef OCT_USE_MONT
#define OCT_USE_MONT 1
#endif
// Lets the benchmark build several variants (different switches) into distinctly named
// kernels from this one source; production uses the default name.
#ifndef OCT_KERNEL_NAME
#define OCT_KERNEL_NAME octopus_search
#endif

#define OCT_MOD             1032193u
#define OCT_N               1024u
#define OCT_B               11u
#define OCT_WARP            32u
#define OCT_DATA_PER_THREAD 32u
#define OCT_MIX_WORDS       64u
#define OCT_MIX_NODES       4u
#define OCT_NODE_WORDS      16u
#define OCT_ACCESSES        32u

#define OCT_NUM_WARPS       (GROUP_SIZE / OCT_WARP)

// Barrett: OCT_MU = floor(2^41 / OCT_MOD). For t < 2^41 (products of two residues are
// < 2^40; the largest sum a*w2pow+b*wpow+c < 2^41) q = (t*MU) >> 41 is exact to within -1,
// so one conditional subtraction lands in [0, MOD).
#define OCT_BARRETT_SHIFT   41
#define OCT_MU              2130438UL


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
uint octopus_redumod(ulong const t)  // t < 2^41
{
#if OCT_USE_BARRETT
    ulong const q = (t * OCT_MU) >> OCT_BARRETT_SHIFT;
    ulong       r = t - q * (ulong)OCT_MOD;
    if (r >= (ulong)OCT_MOD)
    {
        r -= (ulong)OCT_MOD;
    }
    return (uint)r;
#else
    return (uint)(t % OCT_MOD);
#endif
}


// Lazy Barrett: r ≡ t (mod MOD) in [0, 2*MOD), without the canonical conditional subtraction.
// Valid for t < 2^41 — the Horner accumulator is kept < 2*MOD, so pv*xs + dj < 2*MOD*MOD + MOD
// ≈ 2.131e12 < 2^41 (2.199e12), still inside the shift-41 Barrett bound (verified bit-exact
// over the KAT). Canonicalise once with octopus_final_reduce when leaving the loop.
inline
uint octopus_redumod_lazy(ulong const t)  // t < 2^41 -> [0, 2*MOD)
{
#if OCT_USE_BARRETT
    ulong const q = (t * OCT_MU) >> OCT_BARRETT_SHIFT;
    return (uint)(t - q * (ulong)OCT_MOD);
#else
    return (uint)(t % OCT_MOD);
#endif
}


inline
uint octopus_final_reduce(uint const r)  // [0, 2*MOD) -> [0, MOD)
{
    return (r >= OCT_MOD) ? (r - OCT_MOD) : r;
}


inline
uint octopus_mulmod(uint const a, uint const b)
{
    return octopus_redumod((ulong)a * (ulong)b);
}


////////////////////////////////////////////////////////////////////////////////
// Montgomery form (R = 2^32). OCT_NINV = -MOD^{-1} mod 2^32, OCT_R2 = 2^64 mod MOD.
// Used only when OCT_USE_MONT is set; see the switch comment. All three sequences are
// validated bit-exact vs the reference modmul over the KAT range.
#define OCT_NINV 0xf00fbfffu
#define OCT_R2   765937u


inline
uint octopus_mont_mul(uint const a, uint const b)  // canonical: a, b < MOD -> [0, MOD)
{
    ulong const t = (ulong)a * (ulong)b;
    uint const  m = (uint)t * OCT_NINV;
    ulong const r = (t + (ulong)m * (ulong)OCT_MOD) >> 32;
    return (r >= (ulong)OCT_MOD) ? (uint)(r - (ulong)OCT_MOD) : (uint)r;
}


inline
uint octopus_mont_mul_lazy(uint const a, uint const b)  // a < 2*MOD, b < MOD -> [0, 2*MOD)
{
    ulong const t = (ulong)a * (ulong)b;
    uint const  m = (uint)t * OCT_NINV;
    return (uint)((t + (ulong)m * (ulong)OCT_MOD) >> 32);
}


inline
uint octopus_to_mont(uint const a)  // a < MOD -> Montgomery form
{
    return octopus_mont_mul(a, OCT_R2);
}


inline
uint octopus_from_mont(uint const a)  // a < 2*MOD (Montgomery) -> [0, MOD)
{
    ulong const t = (ulong)a;
    uint const  m = (uint)t * OCT_NINV;
    ulong const r = (t + (ulong)m * (ulong)OCT_MOD) >> 32;
    return (r >= (ulong)OCT_MOD) ? (uint)(r - (ulong)OCT_MOD) : (uint)r;
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


// Select the chunk buffer holding a node's global index. chunk = nodeIndex / chunk_items.
inline
__global uint4 const* octopus_chunk(
    __global uint4 const* const restrict b00,
    __global uint4 const* const restrict b01,
    __global uint4 const* const restrict b02,
    __global uint4 const* const restrict b03,
    __global uint4 const* const restrict b04,
    __global uint4 const* const restrict b05,
    __global uint4 const* const restrict b06,
    __global uint4 const* const restrict b07,
    __global uint4 const* const restrict b08,
    __global uint4 const* const restrict b09,
    __global uint4 const* const restrict b10,
    __global uint4 const* const restrict b11,
    __global uint4 const* const restrict b12,
    __global uint4 const* const restrict b13,
    __global uint4 const* const restrict b14,
    __global uint4 const* const restrict b15,
    uint const chunk)
{
    switch (chunk)
    {
        case 0u:  return b00;
        case 1u:  return b01;
        case 2u:  return b02;
        case 3u:  return b03;
        case 4u:  return b04;
        case 5u:  return b05;
        case 6u:  return b06;
        case 7u:  return b07;
        case 8u:  return b08;
        case 9u:  return b09;
        case 10u: return b10;
        case 11u: return b11;
        case 12u: return b12;
        case 13u: return b13;
        case 14u: return b14;
        default:  return b15;
    }
}


#if OCT_LAZY_HORNER
#define OCT_HORNER_REDUCE(T) octopus_redumod_lazy(T)
#define OCT_HORNER_STORE(P)  octopus_final_reduce(P)
#else
#define OCT_HORNER_REDUCE(T) octopus_redumod(T)
#define OCT_HORNER_STORE(P)  (P)
#endif

// Coefficient / evaluation-point preparation, one Horner step, and the result finalize —
// switched between Barrett (default) and Montgomery (OCT_USE_MONT). With Montgomery the
// d[] coefficients and xs are carried in Montgomery form and converted back on store.
#if OCT_USE_MONT
#define OCT_XS_PREP(X)           octopus_to_mont(X)
#define OCT_D_PREP(X)            octopus_to_mont(X)
#define OCT_HORNER_STEP(P, X, D) (octopus_mont_mul_lazy((P), (X)) + (D))
#define OCT_HORNER_FIN(P)        octopus_from_mont(P)
#else
#define OCT_XS_PREP(X)           (X)
#define OCT_D_PREP(X)            (X)
#define OCT_HORNER_STEP(P, X, D) OCT_HORNER_REDUCE((ulong)(P) * (ulong)(X) + (ulong)(D))
#define OCT_HORNER_FIN(P)        OCT_HORNER_STORE(P)
#endif


// Interleaved Horner: OCT_INTERLEAVE independent evaluations share each d[j] read and
// expose ILP. DPTR is the coefficient vector (LDS or private). Fills result_v[0..31];
// advances wpow/w2pow exactly as the serial reference does.
#define OCTOPUS_MULTI_EVAL(DPTR)                                                                   \
    for (uint i0 = 0u; i0 < OCT_DATA_PER_THREAD; i0 += OCT_INTERLEAVE)                             \
    {                                                                                             \
        uint xs[OCT_INTERLEAVE];                                                                  \
        uint pv[OCT_INTERLEAVE];                                                                  \
        __attribute__((opencl_unroll_hint)) for (uint k = 0u; k < OCT_INTERLEAVE; ++k)            \
        {                                                                                         \
            xs[k] = OCT_XS_PREP(octopus_redumod((ulong)a * (ulong)w2pow + (ulong)b * (ulong)wpow + (ulong)c)); \
            pv[k] = 0u;                                                                           \
            if (i0 + k + 1u < OCT_DATA_PER_THREAD)                                                \
            {                                                                                     \
                wpow  = octopus_mulmod(wpow, fullWpow);                                            \
                w2pow = octopus_mulmod(w2pow, fullW2pow);                                          \
            }                                                                                     \
        }                                                                                         \
        __attribute__((opencl_unroll_hint(1))) for (uint j = OCT_N; j--;)                         \
        {                                                                                         \
            uint const dj = (DPTR)[j];                                                            \
            __attribute__((opencl_unroll_hint)) for (uint k = 0u; k < OCT_INTERLEAVE; ++k)        \
            {                                                                                     \
                pv[k] = OCT_HORNER_STEP(pv[k], xs[k], dj);                                        \
            }                                                                                     \
        }                                                                                         \
        __attribute__((opencl_unroll_hint)) for (uint k = 0u; k < OCT_INTERLEAVE; ++k)            \
        {                                                                                         \
            result_v[i0 + k] = OCT_HORNER_FIN(pv[k]);                                             \
        }                                                                                         \
    }


__kernel
void OCT_KERNEL_NAME(
    __global uint4 const* const restrict dag00,  // dataset chunk buffers (each 1 GiB)
    __global uint4 const* const restrict dag01,
    __global uint4 const* const restrict dag02,
    __global uint4 const* const restrict dag03,
    __global uint4 const* const restrict dag04,
    __global uint4 const* const restrict dag05,
    __global uint4 const* const restrict dag06,
    __global uint4 const* const restrict dag07,
    __global uint4 const* const restrict dag08,
    __global uint4 const* const restrict dag09,
    __global uint4 const* const restrict dag10,
    __global uint4 const* const restrict dag11,
    __global uint4 const* const restrict dag12,
    __global uint4 const* const restrict dag13,
    __global uint4 const* const restrict dag14,
    __global uint4 const* const restrict dag15,
    __global t_result* const restrict result,
    __constant ulong const* const restrict header,
    ulong const start_nonce,
    uint const num_full_pages,
    uint const chunk_items,                     // DAG nodes per chunk buffer
    ulong4 const boundary)
{
    ulong const nonce = start_nonce + (ulong)(get_global_id(1) * GROUP_SIZE + get_global_id(0));
    uint const  lane  = (uint)(nonce & (OCT_WARP - 1u));  // nonce % 32

    ulong header4[4];
    header4[0] = header[0];
    header4[1] = header[1];
    header4[2] = header[2];
    header4[3] = header[3];

    ////////////////////////////////////////////////////////////////////////////
    // multi_eval setup (header-derived, lane-dependent) — independent of d.
    uint const a = octopus_remap_param(header4[0]);
    uint const b = octopus_remap_param(header4[1]);
    uint       c;
    {
        ulong h = header4[2];
        while (true)
        {
            c = octopus_remap_param(h);
            uint const lhs = octopus_mulmod(b, b);
            uint const rhs = octopus_mulmod(octopus_mulmod(a, c), 4u);
            if (lhs != rhs)
            {
                break;
            }
            ++h;
        }
    }
    uint const w  = octopus_remap_param(header4[3]);
    uint const w2 = octopus_mulmod(w, w);

    uint wpow  = 1u;
    uint w2pow = 1u;
    for (uint k = 0u; k < lane; ++k)
    {
        wpow  = octopus_mulmod(wpow, w);
        w2pow = octopus_mulmod(w2pow, w2);
    }
    uint fullWpow  = wpow;
    uint fullW2pow = w2pow;
    for (uint k = lane; k < OCT_WARP; ++k)
    {
        fullWpow  = octopus_mulmod(fullWpow, w);
        fullW2pow = octopus_mulmod(fullW2pow, w2);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Coefficient vector d[OCT_N] + the (interleaved) polynomial evaluation.
    uint result_v[OCT_DATA_PER_THREAD];
#if OCT_COOP_D
    // 32-lane subgroup shares one d in LDS: lane L fills d[i*32 + L] from siphash(nonce).
    __local uint        dShared[OCT_NUM_WARPS * OCT_N];
    __local uint* const d = dShared + (get_local_id(0) / OCT_WARP) * OCT_N;
    {
        ulong v[4];
        v[0] = header4[0];
        v[1] = header4[1];
        v[2] = header4[2];
        v[3] = header4[3];
        octopus_sip_hash24(v, nonce);
        __attribute__((opencl_unroll_hint(1)))
        for (uint i = 0u; i < OCT_DATA_PER_THREAD; ++i)
        {
            octopus_sip_round(v);
            ulong const lanes = (v[0] ^ v[1]) ^ (v[2] ^ v[3]);
            d[i * OCT_WARP + lane] = OCT_D_PREP(octopus_redumod(lanes & 0xffffffffUL));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    OCTOPUS_MULTI_EVAL(d)
#else
    // Baseline: each work-item builds the full d[OCT_N] in private memory (spills to scratch).
    uint        d[OCT_N];
    ulong const nonceBase = (nonce / OCT_WARP) * OCT_WARP;
    for (uint lid = 0u; lid < OCT_WARP; ++lid)
    {
        ulong v[4];
        v[0] = header4[0];
        v[1] = header4[1];
        v[2] = header4[2];
        v[3] = header4[3];
        octopus_sip_hash24(v, nonceBase + lid);
        __attribute__((opencl_unroll_hint(1)))
        for (uint i = 0u; i < OCT_DATA_PER_THREAD; ++i)
        {
            octopus_sip_round(v);
            ulong const lanes = (v[0] ^ v[1]) ^ (v[2] ^ v[3]);
            d[i * OCT_WARP + lid] = OCT_D_PREP(octopus_redumod(lanes & 0xffffffffUL));
        }
    }
    OCTOPUS_MULTI_EVAL(d)
#endif

    ulong threadResult = 0ul;
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < OCT_DATA_PER_THREAD; ++i)
    {
        threadResult = threadResult * 0x01000193ul ^ (ulong)result_v[i];
    }

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
    for (uint wrd = 0u; wrd < OCT_MIX_WORDS; ++wrd)
    {
        mix[wrd] = seedW[wrd % OCT_NODE_WORDS];
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
                octopus_chunk(dag00, dag01, dag02, dag03, dag04, dag05, dag06, dag07,
                              dag08, dag09, dag10, dag11, dag12, dag13, dag14, dag15, chunk);
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
    for (uint wrd = 0u; wrd < OCT_MIX_WORDS; wrd += 4u)
    {
        uint reduction = mix[wrd];
        reduction = fnv1_u32(reduction, mix[wrd + 1u]);
        reduction = fnv1_u32(reduction, mix[wrd + 2u]);
        reduction = fnv1_u32(reduction, mix[wrd + 3u]);
        mix[wrd / 4u] = reduction;
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
