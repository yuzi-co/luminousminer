// PyrinHashV2 (PYI) OpenCL kernels.
//
// Correctness Layer 3: this kernel must be BIT-IDENTICAL to the CPU reference in
// sources/algo/pyrinhashv2/*.cpp. The matrix is generated host-side once per job
// (CPU reference generateMatrix) and uploaded; per nonce the kernel computes
// powHash -> heavyHash -> little-endian target compare.
//
// The PoW hashers are plain unkeyed BLAKE3: this file requires the shared BLAKE3
// primitive (sources/algo/crypto/opencl/blake3.cl, exposing blake3_hash_chunk) to be
// in scope. The resolver's kernel generator pulls it in via addInclude; the KAT harness
// concatenates it ahead of this file. This is a straightforward correctness kernel — the
// BLAKE3 midstate / udot4 matmul perf levers are a later pass.
//
// The test_* entry points exist purely so the host KAT harness can check each stage
// against the same known-answer vectors the CPU reference is gated on.

inline void storeLe64(ulong const value, uchar* p)
{
    for (int b = 0; b < 8; ++b)
    {
        p[b] = (uchar)((value >> (8 * b)) & 0xFF);
    }
}


// hash1 = BLAKE3( pre_pow_hash[32] || timestamp_LE || zero[32] || nonce_LE ).
void powHash(uchar const* prePowHash, ulong const timestamp, ulong const nonce, uchar* out)
{
    uchar buf[80];
    for (int i = 0; i < 32; ++i)
    {
        buf[i] = prePowHash[i];
    }
    storeLe64(timestamp, buf + 32);
    for (int i = 0; i < 32; ++i)
    {
        buf[40 + i] = 0;
    }
    storeLe64(nonce, buf + 72);
    blake3_hash_chunk(buf, 80u, 32u, out);
}


// hash2 step = BLAKE3 over 32 bytes.
void kHeavyHash(uchar const* input, uchar* out)
{
    blake3_hash_chunk(input, 32u, 32u, out);
}


// Heavy step: matrix * nibble-vector; fold each 12-bit row sum to a nibble by XORing its
// three low nibbles (V2 / algo_updated reduction), XOR with hash1, then KHeavyHash.
// matrix is row-major ushort[64*64], values 0..15.
void heavyHash(__global ushort const* matrix, uchar const* hash1, uchar* out)
{
    ushort vec[64];
    for (int i = 0; i < 32; ++i)
    {
        vec[2 * i] = (ushort)(hash1[i] >> 4);
        vec[2 * i + 1] = (ushort)(hash1[i] & 0x0F);
    }

    uchar product[32];
    for (int i = 0; i < 32; ++i)
    {
        ushort sum1 = 0;
        ushort sum2 = 0;
        for (int j = 0; j < 64; ++j)
        {
            sum1 = (ushort)(sum1 + matrix[(2 * i) * 64 + j] * vec[j]);
            sum2 = (ushort)(sum2 + matrix[(2 * i + 1) * 64 + j] * vec[j]);
        }
        ushort const hi = (ushort)((sum1 & 0xF) ^ ((sum1 >> 4) & 0xF) ^ ((sum1 >> 8) & 0xF));
        ushort const lo = (ushort)((sum2 & 0xF) ^ ((sum2 >> 4) & 0xF) ^ ((sum2 >> 8) & 0xF));
        product[i] = (uchar)((hi << 4) | lo);
    }
    for (int i = 0; i < 32; ++i)
    {
        product[i] ^= hash1[i];
    }
    kHeavyHash(product, out);
}


// pow <= target as little-endian 256-bit integers (scan from most-significant byte).
inline bool meetsTarget(uchar const* powLe, uchar const* targetLe)
{
    for (int i = 31; i >= 0; --i)
    {
        if (powLe[i] != targetLe[i])
        {
            return powLe[i] < targetLe[i];
        }
    }
    return true;
}


__kernel void test_pow_hash(__global uchar const* prePowHash,
                            ulong const           timestamp,
                            ulong const           nonce,
                            __global uchar*       out)
{
    uchar pre[32];
    for (int i = 0; i < 32; ++i)
    {
        pre[i] = prePowHash[i];
    }
    uchar h[32];
    powHash(pre, timestamp, nonce, h);
    for (int i = 0; i < 32; ++i)
    {
        out[i] = h[i];
    }
}


__kernel void test_kheavy(__global uchar const* input, __global uchar* out)
{
    uchar in[32];
    for (int i = 0; i < 32; ++i)
    {
        in[i] = input[i];
    }
    uchar h[32];
    kHeavyHash(in, h);
    for (int i = 0; i < 32; ++i)
    {
        out[i] = h[i];
    }
}


__kernel void test_heavy_hash(__global ushort const* matrix,
                              __global uchar const*  hash1,
                              __global uchar*        out)
{
    uchar h1[32];
    for (int i = 0; i < 32; ++i)
    {
        h1[i] = hash1[i];
    }
    uchar h[32];
    heavyHash(matrix, h1, h);
    for (int i = 0; i < 32; ++i)
    {
        out[i] = h[i];
    }
}


// Result buffer shared with the host. MAX_RESULT is overridable by the host kernel
// generator (addDefine).
#ifndef MAX_RESULT
#define MAX_RESULT 4
#endif

typedef struct __attribute__((aligned(8)))
{
    uchar found;
    uint  count;
    ulong nonces[MAX_RESULT];
} Result;


inline void publishHit(__global Result* result, ulong const nonce)
{
    uint const idx = atomic_inc(&result->count);
    result->found = 1;
    if (idx < MAX_RESULT)
    {
        result->nonces[idx] = nonce;
    }
}


// Real mining kernel: each work-item tries nonce = startNonce + global_id(0).
// On a hit (pow <= target, little-endian) it publishes its nonce into result.
__kernel void search(__global ushort const* matrix,
                     __global uchar const*  header,
                     __global uchar const*  target,
                     ulong const            timestamp,
                     ulong const            startNonce,
                     __global Result*       result)
{
    uchar pre[32];
    for (int i = 0; i < 32; ++i)
    {
        pre[i] = header[i];
    }
    uchar tgt[32];
    for (int i = 0; i < 32; ++i)
    {
        tgt[i] = target[i];
    }

    ulong const nonce = startNonce + (ulong)get_global_id(0);

    uchar h1[32];
    powHash(pre, timestamp, nonce, h1);
    uchar product_pow[32];
    heavyHash(matrix, h1, product_pow);

    if (meetsTarget(product_pow, tgt))
    {
        publishHit(result, nonce);
    }
}
