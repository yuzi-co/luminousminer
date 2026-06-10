// XelisHash V3 OpenCL kernel. Bit-identical to the CPU reference
// (sources/algo/xelishashv3) and to upstream xelis-project/xelis-hash (C/xelis_hash_v3.c).
//
// Pulls in the shared BLAKE3 primitive (crypto/opencl/blake3.cl) ahead of this file (the
// kernel generator / KAT harness concatenates them), reusing blake3_compress for stage 4's
// 531-chunk tree and blake3_hash_chunk for stage 1's single-chunk hashes. ChaCha8, the AES
// single round, and the 128-bit integer math (no native u128 on GPU) are implemented here.

#ifndef LM_XELISHASHV3_CL
#define LM_XELISHASHV3_CL

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define XV3_MEMSIZE 67968u        // u64 words (531 * 128)
#define XV3_MEMSIZE_BYTES 543744u // 531 * 1024
#define XV3_BUFSIZE 33984u        // MEMSIZE / 2
#define XV3_ITERS 2u
#define XV3_NONCE_SIZE 12u
#define XV3_CHUNK_BYTES 1024u
#define XV3_NUM_CHUNKS 531u

#define B3_FLAG_PARENT 4u

__constant uchar XV3_AES_KEY[16] = {
    'x', 'e', 'l', 'i', 's', 'h', 'a', 's', 'h', '-', 'p', 'o', 'w', '-', 'v', '3'
};

__constant uchar XV3_SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};


// ───────────────────────── 64-bit helpers ─────────────────────────

static inline ulong xv3_rotl64(ulong x, uint r)
{
    r &= 63u;
    return (0u == r) ? x : ((x << r) | (x >> (64u - r)));
}


static inline ulong xv3_rotr64(ulong x, uint r)
{
    r &= 63u;
    return (0u == r) ? x : ((x >> r) | (x << (64u - r)));
}


static inline ulong xv3_murmur(ulong seed)
{
    seed ^= seed >> 55;
    seed *= 0xff51afd7ed558ccdUL;
    seed ^= seed >> 32;
    seed *= 0xc4ceb9fe1a85ec53UL;
    seed ^= seed >> 15;
    return seed;
}


// (u128)x * BUFSIZE >> 64, with the murmur-ish premix — matches CPU map_index.
static inline ulong xv3_map_index(ulong x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdUL;
    return mul_hi(x, (ulong)XV3_BUFSIZE);
}


static inline int xv3_pick_half(ulong seed)
{
    return (xv3_murmur(seed) & (1UL << 58)) != 0UL;
}


// floor(sqrt(n)) — bit-identical to the CPU reference (xelishashv3.cpp isqrt): an FP
// approximation truncated to u64, then a single +/-1 integer adjust, INCLUDING the wrapping
// (approx+1)*(approx+1) the reference relies on for n near 2^64. XV3_ISQRT_IMPL selects the seed
// at build time; 0 (FP64) is consensus-correct and the production default. The whole correction
// body is shared via the macro so every variant differs only in the one sqrt seed.
#ifndef XV3_ISQRT_IMPL
#define XV3_ISQRT_IMPL 0
#endif

#define XV3_ISQRT_BODY(SEED)                          \
    if (n < 2UL)                                      \
    {                                                 \
        return n;                                     \
    }                                                 \
    ulong approx = (SEED);                            \
    if (approx * approx > n)                          \
    {                                                 \
        return approx - 1UL;                          \
    }                                                 \
    if ((approx + 1UL) * (approx + 1UL) <= n)         \
    {                                                 \
        return approx + 1UL;                          \
    }                                                 \
    return approx;

#if XV3_ISQRT_IMPL == 1
// PROBE ONLY — NON-CONSENSUS. f32 seed: the 24-bit mantissa is too coarse for a single +/-1 adjust
// to reach floor(sqrt(n)) at large n, so digests diverge from the reference. Exists solely to
// measure the throughput ceiling of removing the slow consumer-RDNA4 FP64 sqrt before committing
// to a bit-exact integer replacement.
static inline ulong xv3_isqrt(ulong n)
{
    XV3_ISQRT_BODY((ulong)sqrt((float)n))
}
#else
static inline ulong xv3_isqrt(ulong n)
{
    XV3_ISQRT_BODY((ulong)sqrt((double)n))
}
#endif


// ───────────────────── 128-bit integer emulation ─────────────────────
// CPU reference uses __uint128_t; replicate exactly (incl. mod-2^128 truncation).

static inline int xv3_ge128(ulong ahi, ulong alo, ulong bhi, ulong blo)
{
    return (ahi > bhi) || (ahi == bhi && alo >= blo);
}


// Full 128/128 binary long division — num / den and num % den (the baseline). Always used for
// a true 128-bit divisor (dhi != 0); the optimized variants below only specialize dhi == 0.
static inline void xv3_divmod_full(ulong nhi, ulong nlo, ulong dhi, ulong dlo,
                                   ulong* qhi, ulong* qlo, ulong* rhi, ulong* rlo)
{
    ulong q_hi = 0UL, q_lo = 0UL, r_hi = 0UL, r_lo = 0UL;
    for (int i = 127; i >= 0; --i)
    {
        // r <<= 1
        r_hi = (r_hi << 1) | (r_lo >> 63);
        r_lo <<= 1;
        // bring down bit i of num
        ulong bit = (i >= 64) ? ((nhi >> (i - 64)) & 1UL) : ((nlo >> i) & 1UL);
        r_lo |= bit;
        if (xv3_ge128(r_hi, r_lo, dhi, dlo))
        {
            // r -= d
            ulong borrow = (r_lo < dlo) ? 1UL : 0UL;
            r_lo -= dlo;
            r_hi -= dhi + borrow;
            if (i >= 64)
            {
                q_hi |= (1UL << (i - 64));
            }
            else
            {
                q_lo |= (1UL << i);
            }
        }
    }
    *qhi = q_hi;
    *qlo = q_lo;
    *rhi = r_hi;
    *rlo = r_lo;
}


// ───────────── 128 / 64 divmod variants (dhi == 0, d != 0), benchmark-selectable ─────────────
// Most stage-3 divisors (cases 0/10/12 and modpow) fit in 64 bits. With a 64-bit divisor the
// remainder is always < d < 2^64, so the running remainder needs only one 64-bit register; the
// one bit shifted out of it is tracked as an explicit carry (the true pre-subtract value is then
// >= 2^64 > d, so a subtract is forced — and `r -= d` lands on the correct value mod 2^64).
// XV3_DIVMOD_IMPL selects the implementation at build time; 0 keeps the original full path so
// the deployed kernel is byte-for-byte unchanged unless the host overrides the define.
#ifndef XV3_DIVMOD_IMPL
#define XV3_DIVMOD_IMPL 0
#endif

#if XV3_DIVMOD_IMPL == 1
// V1: bit-serial, single 64-bit remainder register (half the per-bit ALU of the full path).
static inline void xv3_divmod_by64(ulong nhi, ulong nlo, ulong d, ulong* qhi, ulong* qlo, ulong* rlo)
{
    ulong q_hi = 0UL, q_lo = 0UL, r = 0UL;
    for (int i = 127; i >= 0; --i)
    {
        ulong carry = r >> 63;
        r = (r << 1) | ((i >= 64) ? ((nhi >> (i - 64)) & 1UL) : ((nlo >> i) & 1UL));
        if (0UL != carry || r >= d)
        {
            r -= d;
            if (i >= 64)
            {
                q_hi |= (1UL << (i - 64));
            }
            else
            {
                q_lo |= (1UL << i);
            }
        }
    }
    *qhi = q_hi;
    *qlo = q_lo;
    *rlo = r;
}
#endif

#if XV3_DIVMOD_IMPL == 2
// V2: peel the high word with one native u64 divide (q_hi = nhi / d, r0 = nhi % d < d), then a
// 64-iteration serial tail over the low word — half the loop trips of V1.
static inline void xv3_divmod_by64(ulong nhi, ulong nlo, ulong d, ulong* qhi, ulong* qlo, ulong* rlo)
{
    ulong q_lo = 0UL;
    ulong r = nhi % d;
    for (int i = 63; i >= 0; --i)
    {
        ulong carry = r >> 63;
        r = (r << 1) | ((nlo >> i) & 1UL);
        if (0UL != carry || r >= d)
        {
            r -= d;
            q_lo |= (1UL << i);
        }
    }
    *qhi = nhi / d;
    *qlo = q_lo;
    *rlo = r;
}
#endif

#if XV3_DIVMOD_IMPL == 3 || XV3_DIVMOD_IMPL == 4
// (u1:u0) / v with u1 < v (quotient fits in 64 bits), v != 0 — Hacker's Delight base-2^32
// long division (Knuth Algorithm D, 2 limbs), no per-bit loop. *rem gets the remainder.
static inline ulong xv3_udiv_128_by_64(ulong u1, ulong u0, ulong v, ulong* rem)
{
    ulong const b = 0x100000000UL; // 2^32
    int   s = (int)clz(v);
    v <<= s;
    ulong vn1 = v >> 32;
    ulong vn0 = v & 0xffffffffUL;
    ulong un32 = (0 == s) ? u1 : ((u1 << s) | (u0 >> (64 - s)));
    ulong un10 = u0 << s;
    ulong un1 = un10 >> 32;
    ulong un0 = un10 & 0xffffffffUL;

    ulong q1 = un32 / vn1;
    ulong rhat = un32 - q1 * vn1;
    while (q1 >= b || q1 * vn0 > b * rhat + un1)
    {
        --q1;
        rhat += vn1;
        if (rhat >= b)
        {
            break;
        }
    }

    ulong un21 = un32 * b + un1 - q1 * v;
    ulong q0 = un21 / vn1;
    rhat = un21 - q0 * vn1;
    while (q0 >= b || q0 * vn0 > b * rhat + un0)
    {
        --q0;
        rhat += vn1;
        if (rhat >= b)
        {
            break;
        }
    }

    *rem = (un21 * b + un0 - q0 * v) >> s;
    return q1 * b + q0;
}
#endif

#if XV3_DIVMOD_IMPL == 3
// V3: native fold + divlu tail — no per-bit loop at all.
static inline void xv3_divmod_by64(ulong nhi, ulong nlo, ulong d, ulong* qhi, ulong* qlo, ulong* rlo)
{
    *qhi = nhi / d;
    *qlo = xv3_udiv_128_by_64(nhi % d, nlo, d, rlo);
}
#endif

#if XV3_DIVMOD_IMPL == 4
// V4: identical to V3 (native fold + divlu) but drops the high quotient word. No stage-3 caller
// ever reads qhi or rhi (cases 0/10/11 and modpow use rl; cases 12/13 use ql), so *qhi = nhi / d
// is provably dead — this variant tests whether the compiler already eliminates that emulated
// u64 divide across the (non-inlined, RGA-opaque) impl path or leaves it in. The low dividend
// still needs nhi % d, so the modulo remains.
static inline void xv3_divmod_by64(ulong nhi, ulong nlo, ulong d, ulong* qhi, ulong* qlo, ulong* rlo)
{
    *qhi = 0UL;
    *qlo = xv3_udiv_128_by_64(nhi % d, nlo, d, rlo);
}
#endif


// num / den and num % den. Dispatches to the 64-bit-divisor fast path when the divisor fits in
// 64 bits (and the chosen variant provides one); otherwise the full 128-bit long division.
static inline void xv3_divmod128(ulong nhi, ulong nlo, ulong dhi, ulong dlo,
                                  ulong* qhi, ulong* qlo, ulong* rhi, ulong* rlo)
{
#if XV3_DIVMOD_IMPL == 0
    xv3_divmod_full(nhi, nlo, dhi, dlo, qhi, qlo, rhi, rlo);
#else
    if (0UL == dhi && 0UL != dlo)
    {
        xv3_divmod_by64(nhi, nlo, dlo, qhi, qlo, rlo);
        *rhi = 0UL;
    }
    else
    {
        xv3_divmod_full(nhi, nlo, dhi, dlo, qhi, qlo, rhi, rlo);
    }
#endif
}


// low 64 bits of ((hi:lo) * c) — i.e. ((u128)t * (u64)c) >> 64, truncated to 128 first.
static inline ulong xv3_mul128x64_hi(ulong hi, ulong lo, ulong c)
{
    return mul_hi(lo, c) + hi * c;  // wrapping; high word of (t*c) mod 2^128
}


// ((ah:al) * (bh:bl)) mod 2^128, then >> 64 — matches __uint128_t * truncation.
static inline ulong xv3_mul128x128_hi(ulong ah, ulong al, ulong bh, ulong bl)
{
    return mul_hi(al, bl) + al * bh + ah * bl;  // wrapping
}


// base, result and mod are all < 2^64; products are taken to 128 bits before the modulo
// (the CPU reference uses __uint128_t), so a plain 64-bit multiply here would be wrong.
static inline ulong xv3_modpow(ulong base, ulong exp, ulong mod)
{
    ulong result = 1UL;
    ulong qh, ql, rh, rl;
    base %= mod;
    while (exp > 0UL)
    {
        if (exp & 1UL)
        {
            xv3_divmod128(mul_hi(result, base), result * base, 0UL, mod, &qh, &ql, &rh, &rl);
            result = rl;
        }
        xv3_divmod128(mul_hi(base, base), base * base, 0UL, mod, &qh, &ql, &rh, &rl);
        base = rl;
        exp /= 2UL;
    }
    return result;
}


// ───────────────────────── ChaCha8 ─────────────────────────

static inline uint xv3_rotl32(uint x, int n)
{
    return (x << n) | (x >> (32 - n));
}


#define XV3_QR(a, b, c, d)                  \
    a += b; d ^= a; d = xv3_rotl32(d, 16);  \
    c += d; b ^= c; b = xv3_rotl32(b, 12);  \
    a += b; d ^= a; d = xv3_rotl32(d, 8);   \
    c += d; b ^= c; b = xv3_rotl32(b, 7);


// Writes `len` bytes of ChaCha8 keystream (RFC-8439, 8 rounds, counter 0) to __global out.
static void xv3_chacha8(uchar const key[32], uchar const nonce[12], __global uchar* out, uint len)
{
    uint base[16];
    base[0] = 0x61707865u;
    base[1] = 0x3320646eu;
    base[2] = 0x79622d32u;
    base[3] = 0x6b206574u;
    for (int i = 0; i < 8; ++i)
    {
        base[4 + i] = ((uint)key[4 * i]) | ((uint)key[4 * i + 1] << 8)
                    | ((uint)key[4 * i + 2] << 16) | ((uint)key[4 * i + 3] << 24);
    }
    base[12] = 0u;
    base[13] = ((uint)nonce[0]) | ((uint)nonce[1] << 8) | ((uint)nonce[2] << 16) | ((uint)nonce[3] << 24);
    base[14] = ((uint)nonce[4]) | ((uint)nonce[5] << 8) | ((uint)nonce[6] << 16) | ((uint)nonce[7] << 24);
    base[15] = ((uint)nonce[8]) | ((uint)nonce[9] << 8) | ((uint)nonce[10] << 16) | ((uint)nonce[11] << 24);

    uint offset = 0u;
    while (offset < len)
    {
        uint w[16];
        for (int i = 0; i < 16; ++i)
        {
            w[i] = base[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            XV3_QR(w[0], w[4], w[8], w[12]);
            XV3_QR(w[1], w[5], w[9], w[13]);
            XV3_QR(w[2], w[6], w[10], w[14]);
            XV3_QR(w[3], w[7], w[11], w[15]);
            XV3_QR(w[0], w[5], w[10], w[15]);
            XV3_QR(w[1], w[6], w[11], w[12]);
            XV3_QR(w[2], w[7], w[8], w[13]);
            XV3_QR(w[3], w[4], w[9], w[14]);
        }
        for (int i = 0; i < 16; ++i)
        {
            uint v = w[i] + base[i];
            uint pos = offset + (uint)i * 4u;
            if (pos + 0u < len) out[pos + 0u] = (uchar)(v);
            if (pos + 1u < len) out[pos + 1u] = (uchar)(v >> 8);
            if (pos + 2u < len) out[pos + 2u] = (uchar)(v >> 16);
            if (pos + 3u < len) out[pos + 3u] = (uchar)(v >> 24);
        }
        offset += 64u;
        ++base[12];
    }
}


// ───────────────────────── AES single round ─────────────────────────

static inline uchar xv3_xtime(uchar x)
{
    return (uchar)((x << 1) ^ ((x >> 7) * 0x1bu));
}


// block <- MixColumns(ShiftRows(SubBytes(block))) XOR XV3_AES_KEY  (== _mm_aesenc_si128)
static void xv3_aes_round(uchar block[16])
{
    uchar s[16];
    for (int c = 0; c < 4; ++c)
    {
        for (int r = 0; r < 4; ++r)
        {
            s[c * 4 + r] = XV3_SBOX[block[((c + r) & 3) * 4 + r]];
        }
    }
    for (int c = 0; c < 4; ++c)
    {
        uchar a0 = s[c * 4 + 0], a1 = s[c * 4 + 1], a2 = s[c * 4 + 2], a3 = s[c * 4 + 3];
        block[c * 4 + 0] = (uchar)(xv3_xtime(a0) ^ (xv3_xtime(a1) ^ a1) ^ a2 ^ a3 ^ XV3_AES_KEY[c * 4 + 0]);
        block[c * 4 + 1] = (uchar)(a0 ^ xv3_xtime(a1) ^ (xv3_xtime(a2) ^ a2) ^ a3 ^ XV3_AES_KEY[c * 4 + 1]);
        block[c * 4 + 2] = (uchar)(a0 ^ a1 ^ xv3_xtime(a2) ^ (xv3_xtime(a3) ^ a3) ^ XV3_AES_KEY[c * 4 + 2]);
        block[c * 4 + 3] = (uchar)((xv3_xtime(a0) ^ a0) ^ a1 ^ a2 ^ xv3_xtime(a3) ^ XV3_AES_KEY[c * 4 + 3]);
    }
}


// ───────────────────────── stage 1 ─────────────────────────

static void xv3_stage1(uchar const input[112], __global uchar* scratch)
{
    uchar key[128];
    for (int i = 0; i < 112; ++i)
    {
        key[i] = input[i];
    }
    for (int i = 112; i < 128; ++i)
    {
        key[i] = 0u;
    }

    uchar buffer[64];
    blake3_hash_chunk(input, 112u, 32u, buffer);  // buffer[0..32] = BLAKE3(input)

    uchar inputHash[32];
    uint const chunkLen = XV3_MEMSIZE_BYTES / 4u;
    uint        toff = 0u;

    // chunk 0: nonce = first 12 bytes of BLAKE3(input)
    for (int i = 0; i < 32; ++i)
    {
        buffer[32 + i] = key[i];
    }
    blake3_hash_chunk(buffer, 64u, 32u, inputHash);
    xv3_chacha8(inputHash, buffer, scratch + toff, chunkLen);

    for (uint k = 1u; k < 4u; ++k)
    {
        toff += chunkLen;
        for (int i = 0; i < 32; ++i)
        {
            buffer[i] = inputHash[i];
            buffer[32 + i] = key[k * 32u + i];
        }
        blake3_hash_chunk(buffer, 64u, 32u, inputHash);
        // nonce = last 12 bytes of the previously written chunk
        uchar nonce[12];
        for (uint i = 0u; i < 12u; ++i)
        {
            nonce[i] = scratch[toff - XV3_NONCE_SIZE + i];
        }
        xv3_chacha8(inputHash, nonce, scratch + toff, chunkLen);
    }
}


// ───────────────────────── stage 3 ─────────────────────────

static void xv3_stage3(__global ulong* scratch)
{
    // A = scratch[0..BUFSIZE), B = scratch[BUFSIZE..MEMSIZE). c == scratch[r] for any r.
    __global ulong* A = scratch;
    __global ulong* B = scratch + XV3_BUFSIZE;

    ulong addrA = B[XV3_BUFSIZE - 1u];
    ulong addrB = A[XV3_BUFSIZE - 1u] >> 32;
    uint  r = 0u;

    for (uint i = 0u; i < XV3_ITERS; ++i)
    {
        ulong memA0 = A[xv3_map_index(addrA)];
        ulong memB0 = B[xv3_map_index(memA0 ^ addrB)];

        uchar block[16];
        for (int k = 0; k < 8; ++k)
        {
            block[k]     = (uchar)(memB0 >> (8 * k));
            block[8 + k] = (uchar)(memA0 >> (8 * k));
        }
        xv3_aes_round(block);

        ulong hash1 = 0UL, hash2 = 0UL;
        for (int k = 7; k >= 0; --k)
        {
            hash1 = (hash1 << 8) | block[k];
            hash2 = (hash2 << 8) | block[8 + k];
        }
        ulong result = ~(hash1 ^ hash2);

        for (uint j = 0u; j < XV3_BUFSIZE; ++j)
        {
            ulong a = A[xv3_map_index(result)];
            ulong b = B[xv3_map_index(a ^ ~xv3_rotr64(result, r))];
            ulong c = scratch[r];
            r = (r < XV3_MEMSIZE - 1u) ? r + 1u : 0u;

            ulong v = 0UL;
            ulong qh, ql, rh, rl;
            switch (xv3_rotl64(result, (uint)c) & 0xfUL)
            {
                case 0:
                {
                    ulong denom = xv3_murmur(c ^ result ^ i ^ j) | 1UL;
                    xv3_divmod128(a + i, xv3_isqrt(b + j), 0UL, denom, &qh, &ql, &rh, &rl);
                    v = rl;
                    break;
                }
                case 1:
                    v = xv3_rotl64((c + i) % xv3_isqrt(b | 2UL), i + j) * xv3_isqrt(a + j);
                    break;
                case 2:
                    v = (xv3_isqrt(a + i) * xv3_isqrt(c + j)) ^ (b + i + j);
                    break;
                case 3:
                    v = ((a + b) * c);
                    break;
                case 4:
                    v = ((b - c) * a);
                    break;
                case 5:
                    v = (c - a + b);
                    break;
                case 6:
                    v = (a - b + c);
                    break;
                case 7:
                    v = (b * c + a);
                    break;
                case 8:
                    v = (c * a + b);
                    break;
                case 9:
                    v = (a * b * c);
                    break;
                case 10:
                    xv3_divmod128(a, b, 0UL, (c | 1UL), &qh, &ql, &rh, &rl);
                    v = rl;
                    break;
                case 11:
                {
                    // t1 = (b:c), t2 = (rotl(result,r):(a|2)); v = (t2>t1)? c : t1 % t2
                    ulong t2h = xv3_rotl64(result, r), t2l = (a | 2UL);
                    if (xv3_ge128(t2h, t2l, b, c) && !(t2h == b && t2l == c))
                    {
                        v = c;
                    }
                    else
                    {
                        xv3_divmod128(b, c, t2h, t2l, &qh, &ql, &rh, &rl);
                        v = rl;
                    }
                    break;
                }
                case 12:
                    // (c:a) / (b|4)
                    xv3_divmod128(c, a, 0UL, (b | 4UL), &qh, &ql, &rh, &rl);
                    v = ql;
                    break;
                case 13:
                {
                    // t1 = (rotl(result,r):b), t2 = (a:(c|8)); v = (t1>t2)? t1/t2 : a^b
                    ulong t1h = xv3_rotl64(result, r), t1l = b;
                    ulong t2h = a, t2l = (c | 8UL);
                    if (xv3_ge128(t1h, t1l, t2h, t2l) && !(t1h == t2h && t1l == t2l))
                    {
                        xv3_divmod128(t1h, t1l, t2h, t2l, &qh, &ql, &rh, &rl);
                        v = ql;
                    }
                    else
                    {
                        v = a ^ b;
                    }
                    break;
                }
                case 14:
                    // (b:a) * c >> 64
                    v = xv3_mul128x64_hi(b, a, c);
                    break;
                case 15:
                    // (a:c) * (rotr(result,r):b) >> 64
                    v = xv3_mul128x128_hi(a, c, xv3_rotr64(result, r), b);
                    break;
            }

            ulong idxSeed = v ^ result;
            result = xv3_rotl64(idxSeed, r);

            int   useB = xv3_pick_half(v);
            ulong idxT = xv3_map_index(idxSeed);
            ulong t = (useB ? B[idxT] : A[idxT]) ^ result;

            ulong idxA = xv3_map_index(t ^ result ^ 0x9e3779b97f4a7c15UL);
            ulong idxB = xv3_map_index(idxA ^ ~result ^ 0xd2b74407b1ce6e93UL);

            ulong memA = A[idxA];
            A[idxA] = t;
            B[idxB] ^= memA ^ xv3_rotr64(t, i + j);
        }

        addrA = xv3_modpow(addrA, addrB, result);
        addrB = xv3_isqrt(result) * (ulong)(r + 1u) * xv3_isqrt(addrA);
    }
}


// ───────────────────────── stage 4 (multi-chunk BLAKE3) ─────────────────────────

// Non-root chaining value of one 1024-byte chunk read from __global, counter = chunkIdx.
static void xv3_b3_chunk_cv(__global uchar const* data, uint chunkIdx, uint cvOut[8])
{
    uint cv[8] = { B3_IV0, B3_IV1, B3_IV2, B3_IV3, B3_IV4, B3_IV5, B3_IV6, B3_IV7 };
    __global uchar const* p = data + chunkIdx * XV3_CHUNK_BYTES;

    for (uint blk = 0u; blk < 16u; ++blk)
    {
        uint m[16];
        uint b = blk * 64u;
        for (uint w = 0u; w < 16u; ++w)
        {
            uint o = b + w * 4u;
            m[w] = ((uint)p[o]) | ((uint)p[o + 1u] << 8) | ((uint)p[o + 2u] << 16) | ((uint)p[o + 3u] << 24);
        }
        uint flags = 0u;
        if (0u == blk) flags |= B3_FLAG_CHUNK_START;
        if (15u == blk) flags |= B3_FLAG_CHUNK_END;

        uint out16[16];
        blake3_compress(cv, m, chunkIdx, 0u, 64u, flags, out16);
        for (int k = 0; k < 8; ++k)
        {
            cv[k] = out16[k];
        }
    }
    for (int k = 0; k < 8; ++k)
    {
        cvOut[k] = cv[k];
    }
}


// BLAKE3 of the whole scratchpad (531 full chunks) -> 32-byte root, written to private out.
static void xv3_stage4(__global uchar const* data, uchar* out)
{
    uint const iv[8] = { B3_IV0, B3_IV1, B3_IV2, B3_IV3, B3_IV4, B3_IV5, B3_IV6, B3_IV7 };
    uint stack[16][8];
    int  sp = 0;

    // chunks 0..529 pushed with the cv-stack merge rule
    for (uint ci = 0u; ci < XV3_NUM_CHUNKS - 1u; ++ci)
    {
        uint cv[8];
        xv3_b3_chunk_cv(data, ci, cv);

        uint t = ci + 1u;
        while (0u == (t & 1u))
        {
            --sp;
            uint m[16];
            for (int k = 0; k < 8; ++k)
            {
                m[k]     = stack[sp][k];
                m[8 + k] = cv[k];
            }
            uint out16[16];
            blake3_compress(iv, m, 0u, 0u, 64u, B3_FLAG_PARENT, out16);
            for (int k = 0; k < 8; ++k)
            {
                cv[k] = out16[k];
            }
            t >>= 1;
        }
        for (int k = 0; k < 8; ++k)
        {
            stack[sp][k] = cv[k];
        }
        ++sp;
    }

    // last chunk is the root candidate; merge with the stack, ROOT on the outermost parent
    uint cur[8];
    xv3_b3_chunk_cv(data, XV3_NUM_CHUNKS - 1u, cur);

    for (int s = sp - 1; s >= 0; --s)
    {
        uint m[16];
        for (int k = 0; k < 8; ++k)
        {
            m[k]     = stack[s][k];
            m[8 + k] = cur[k];
        }
        uint flags = B3_FLAG_PARENT;
        if (0 == s) flags |= B3_FLAG_ROOT;
        uint out16[16];
        blake3_compress(iv, m, 0u, 0u, 64u, flags, out16);
        for (int k = 0; k < 8; ++k)
        {
            cur[k] = out16[k];
        }
    }

    for (uint i = 0u; i < 32u; ++i)
    {
        out[i] = (uchar)(cur[i >> 2] >> (8u * (i & 3u)));
    }
}


// Full XelisHash V3 of a 112-byte work blob into the given scratchpad slice -> hash[32].
static void xv3_hash(uchar const input[112], __global ulong* scratch, uchar hash[32])
{
    __global uchar* scratchBytes = (__global uchar*)scratch;
    xv3_stage1(input, scratchBytes);
    xv3_stage3(scratch);
    xv3_stage4(scratchBytes, hash);
}


// ───────────────────────── KAT entry point ─────────────────────────
// One work-item computes one full XelisHash V3 into its own scratchpad slice.
__kernel void xelis_kat(__global uchar const* in,      // 112 bytes
                        __global ulong*       scratchAll, // XV3_MEMSIZE u64 per work-item
                        __global uchar*       out)       // 32 bytes
{
    uint gid = (uint)get_global_id(0);
    uchar input[112];
    for (int i = 0; i < 112; ++i)
    {
        input[i] = in[i];
    }
    uchar h[32];
    xv3_hash(input, scratchAll + (size_t)gid * XV3_MEMSIZE, h);
    __global uchar* o = out + (size_t)gid * 32u;
    for (int i = 0; i < 32; ++i)
    {
        o[i] = h[i];
    }
}


// ───────────────────────── mining search kernel ─────────────────────────
// Result buffer shared with the host (mirrors algo::kheavyhash/blake3 Result). MAX_RESULT
// is overridable by the host kernel generator (addDefine).
#ifndef MAX_RESULT
#define MAX_RESULT 4
#endif

typedef struct __attribute__((aligned(8)))
{
    uchar found;
    uint  count;
    ulong nonces[MAX_RESULT];
} Result;


static inline void xv3_publish(__global Result* result, ulong const nonce)
{
    uint const idx = atomic_inc(&result->count);
    result->found = 1;
    if (idx < MAX_RESULT)
    {
        result->nonces[idx] = nonce;
    }
}


// Xelis difficulty check: the 32-byte hash is read BIG-ENDIAN as a U256 and must be <= target
// (target = U256::MAX / difficulty, also big-endian). Compare MSB-first.
static inline int xv3_meets_target(uchar const hash[32], __global uchar const* target)
{
    for (int i = 0; i < 32; ++i)
    {
        uchar h = hash[i], t = target[i];
        if (h != t)
        {
            return h < t;
        }
    }
    return 1;  // equal => accepted
}


// Real mining kernel: each work-item tries nonce = startNonce + global_id(0), written
// big-endian at MinerWork offset 40, and publishes its nonce on pow <= target.
__kernel void search(__global uchar const* baseInput,  // 112-byte MinerWork template
                     __global uchar const* target,     // 32-byte big-endian target
                     ulong const           startNonce,
                     __global ulong*       scratchAll,  // XV3_MEMSIZE u64 per work-item
                     __global Result*      result)
{
    uint const gid = (uint)get_global_id(0);
    ulong const nonce = startNonce + (ulong)gid;

    uchar input[112];
    for (int i = 0; i < 112; ++i)
    {
        input[i] = baseInput[i];
    }
    // nonce -> bytes [40..48], big-endian
    for (int i = 0; i < 8; ++i)
    {
        input[40 + i] = (uchar)(nonce >> (8 * (7 - i)));
    }

    uchar h[32];
    xv3_hash(input, scratchAll + (size_t)gid * XV3_MEMSIZE, h);

    if (xv3_meets_target(h, target))
    {
        xv3_publish(result, nonce);
    }
}

#endif // LM_XELISHASHV3_CL
