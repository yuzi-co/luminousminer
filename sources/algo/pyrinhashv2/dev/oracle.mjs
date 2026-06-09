// PyrinHashV2 (PYI) known-answer-test oracle / generator.
//
// Regenerates sources/algo/pyrinhashv2/tests/pyrinhashv2_test_vectors.hpp.
//
//   npm i @noble/hashes && node sources/algo/pyrinhashv2/dev/oracle.mjs > \
//     sources/algo/pyrinhashv2/tests/pyrinhashv2_test_vectors.hpp
//
// The oracle is an INDEPENDENT reimplementation of the reference node
// (github.com/Pyrinpyi/pyrin: consensus/pow/{matrix,xoshiro}.rs + crypto/hashes
// pow_hashers.rs). Before emitting, it cross-validates every component against the
// reference node's own test vectors, so the generated V2 vectors are trustworthy:
//   * generateMatrix(seed=[42;32])      == matrix.rs::test_generate_matrix
//   * heavyHash(test_matrix, input, V1) == matrix.rs::test_heavy_hash (algo_updated=false)
//   * powHash(pre=[42;32], ...)          == pow_hashers.rs::test_heavy_hash BLAKE3 vector
// Only after those pass does it print the V2 (algo_updated=true) expecteds.
import { blake3 } from "@noble/hashes/blake3.js";

const RAW = "https://raw.githubusercontent.com/Pyrinpyi/pyrin/master/";
const matSrc = await (await fetch(RAW + "consensus/pow/src/matrix.rs")).text();

const grabMatrix = (src, anchor) => {
  const b = src.indexOf("Matrix([", src.indexOf(anchor));
  const re = /\[([0-9,\s]+?)\]/g;
  re.lastIndex = b + 7;
  const rows = [];
  let m;
  while (rows.length < 64 && (m = re.exec(src)))
    rows.push(m[1].split(",").map((s) => parseInt(s.trim())).filter((x) => !isNaN(x)));
  return rows;
};
const grabHash = (src, anchor) => {
  const b = src.indexOf("from_bytes([", src.indexOf(anchor));
  return src.slice(b + 12, src.indexOf("]", b)).split(",").map((x) => parseInt(x.trim())).filter((x) => !isNaN(x));
};

const TEST_MATRIX = grabMatrix(matSrc, "let test_matrix");
const HEAVY_INPUT = grabHash(matSrc, "let hash = Hash::from_bytes");
const REF_V1 = grabHash(matSrc, "fn test_heavy_hash");
const GEN_EXPECTED = grabMatrix(matSrc, "fn test_generate_matrix");

const b3 = (buf) => Buffer.from(blake3(buf, { dkLen: 32 }));
const le64 = (n) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };

const heavyHash = (matrix, hash, v2) => {
  const vec = new Array(64);
  for (let i = 0; i < 32; i++) { vec[2 * i] = hash[i] >> 4; vec[2 * i + 1] = hash[i] & 0xf; }
  const p = Buffer.alloc(32);
  for (let i = 0; i < 32; i++) {
    let s1 = 0, s2 = 0;
    for (let j = 0; j < 64; j++) { s1 = (s1 + matrix[2 * i][j] * vec[j]) & 0xffff; s2 = (s2 + matrix[2 * i + 1][j] * vec[j]) & 0xffff; }
    const hi = v2 ? (s1 & 0xf) ^ ((s1 >> 4) & 0xf) ^ ((s1 >> 8) & 0xf) : s1 >> 10;
    const lo = v2 ? (s2 & 0xf) ^ ((s2 >> 4) & 0xf) ^ ((s2 >> 8) & 0xf) : s2 >> 10;
    p[i] = ((hi << 4) | lo) & 0xff;
  }
  for (let i = 0; i < 32; i++) p[i] ^= hash[i];
  return b3(p);
};

const M = (1n << 64n) - 1n;
const rotl = (x, k) => { x &= M; return ((x << BigInt(k)) | (x >> BigInt(64 - k))) & M; };
class Xo {
  constructor(h) { this.s = []; for (let i = 0; i < 4; i++) this.s.push(h.readBigUInt64LE(i * 8)); }
  next() { const s = this.s; const r = (s[0] + rotl((s[0] + s[3]) & M, 23)) & M; const t = (s[1] << 17n) & M; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45); return r; }
}
const rank = (mat) => {
  const E = 1e-9, f = mat.map((r) => r.map(Number)); let rk = 0; const sel = Array(64).fill(false);
  for (let i = 0; i < 64; i++) {
    let j = 0; for (; j < 64; j++) if (!sel[j] && Math.abs(f[j][i]) > E) break;
    if (j !== 64) { rk++; sel[j] = true; for (let p = i + 1; p < 64; p++) f[j][p] /= f[j][i]; for (let k = 0; k < 64; k++) if (k !== j && Math.abs(f[k][i]) > E) for (let p = i + 1; p < 64; p++) f[k][p] -= f[j][p] * f[k][i]; }
  }
  return rk;
};
const generate = (h) => { const g = new Xo(h); for (;;) { const m = []; for (let i = 0; i < 64; i++) { const row = Array(64); let v = 0n; for (let j = 0; j < 64; j++) { const sh = j % 16; if (sh === 0) v = g.next(); row[j] = Number((v >> BigInt(4 * sh)) & 0xfn); } m.push(row); } if (rank(m) === 64) return m; } };
const powHash = (pre, ts, nonce) => b3(Buffer.concat([Buffer.from(pre), le64(ts), Buffer.alloc(32), le64(nonce)]));

// ---- cross-validate against the reference node's own vectors ----
const eqMat = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const assert = (cond, msg) => { if (!cond) { console.error("ORACLE VALIDATION FAILED: " + msg); process.exit(1); } };
assert(heavyHash(TEST_MATRIX, HEAVY_INPUT, false).equals(Buffer.from(REF_V1)), "V1 heavy_hash");
assert(eqMat(generate(Buffer.alloc(32, 42)), GEN_EXPECTED), "generateMatrix");
assert(powHash(Buffer.alloc(32, 42), 1715521488610n, 11171827086635415026n).toString("hex") ===
  "e369d9a9ab793622e8fc83b6bcb540f553510c8adf5b9b9964982e8348117626", "powHash BLAKE3");

// ---- emit ----
const ts = 1715521488610n, nonce = 11171827086635415026n;
const FP_PRE = Buffer.from([99, 231, 29, 85, 153, 225, 235, 207, 36, 237, 3, 55, 106, 21, 221, 122, 28, 51, 249, 76, 190, 128, 153, 244, 189, 104, 26, 178, 170, 4, 177, 103]);
const HEAVY_V2 = heavyHash(TEST_MATRIX, HEAVY_INPUT, true);
const KHEAVY = b3(Buffer.from(HEAVY_INPUT));
const POW_EXP = powHash(Buffer.alloc(32, 42), ts, nonce);
const FP_HASH1 = powHash(FP_PRE, ts, nonce);
const FP_FINAL = heavyHash(generate(FP_PRE), FP_HASH1, true);
const FP_FAIL = Buffer.from(FP_FINAL); FP_FAIL[0] = (FP_FAIL[0] - 1) & 0xff;

const hexArr = (b) => "{ " + [...b].map((x) => "0x" + x.toString(16).padStart(2, "0")).join(", ") + " }";
const decRow = (r) => "    {" + r.join(", ") + "}";
const matrixLines = (m) => m.map(decRow).join(",\n");
const h64 = (n) => "0x" + n.toString(16).padStart(16, "0") + "ULL";

process.stdout.write(`// GENERATED by sources/algo/pyrinhashv2/dev/oracle.mjs — DO NOT EDIT BY HAND.
// Known-answer vectors for the PyrinHashV2 (PYI) CPU reference.
// Provenance: reference node Pyrinpyi/pyrin master (consensus/pow matrix.rs + crypto/hashes
// pow_hashers.rs) + official BLAKE3. The matrix-generation and explicit-matrix vectors are
// byte-identical to the reference test suite; the PoW hashers are plain unkeyed BLAKE3 and the
// matrix reduction is the V2 nibble-XOR-fold (algo_updated == true). Oracle is cross-validated
// against the reference's own test_generate_matrix / test_heavy_hash(false) / test_pow_hash.
#pragma once
#include <array>
#include <cstdint>

namespace pyrinhashv2::kat
{
inline constexpr std::array<uint8_t, 32> GEN_SEED{{ ${[...Buffer.alloc(32, 42)].map((x) => "0x" + x.toString(16).padStart(2, "0")).join(", ")} }};

inline constexpr std::array<std::array<uint16_t, 64>, 64> GEN_EXPECTED_MATRIX{{
${matrixLines(GEN_EXPECTED)}
}};

inline constexpr std::array<std::array<uint16_t, 64>, 64> HEAVY_TEST_MATRIX{{
${matrixLines(TEST_MATRIX)}
}};

inline constexpr std::array<uint8_t, 32> HEAVY_INPUT{{ ${hexArr(Buffer.from(HEAVY_INPUT)).slice(2, -2)} }};

// heavyHash(HEAVY_TEST_MATRIX, HEAVY_INPUT, /*algo_updated=*/true) — V2 nibble-XOR-fold.
// Same matrix+input as the reference's test_heavy_hash(false); only the reduction differs.
inline constexpr std::array<uint8_t, 32> HEAVY_EXPECTED${hexArr(HEAVY_V2).replace("{ ", "{{ ").replace(" }", " }}")};

// kHeavyHash(HEAVY_INPUT) = plain BLAKE3 of the 32-byte input.
inline constexpr std::array<uint8_t, 32> KHEAVY_EXPECTED${hexArr(KHEAVY).replace("{ ", "{{ ").replace(" }", " }}")};

// powHash KAT: pre=[0x2a;32], reference pow_hashers.rs::test_pow_hash inputs. Expected is the
// reference node's own concrete test_heavy_hash() BLAKE3 vector (cross-check).
inline constexpr uint8_t POW_KAT_PRE[32]{ ${[...Buffer.alloc(32, 42)].map((x) => "0x" + x.toString(16).padStart(2, "0")).join(", ")} };
inline constexpr uint64_t POW_KAT_TIMESTAMP{ ${h64(ts)} };
inline constexpr uint64_t POW_KAT_NONCE{ ${h64(nonce)} };
inline constexpr std::array<uint8_t, 32> POW_KAT_EXPECTED${hexArr(POW_EXP).replace("{ ", "{{ ").replace(" }", " }}")};

// Full pipeline (generateMatrix + powHash + V2 heavy step). pre/ts/nonce from test_pow_hash.
inline constexpr std::array<uint8_t, 32> FP_PRE${hexArr(FP_PRE).replace("{ ", "{{ ").replace(" }", " }}")};

inline constexpr uint64_t FP_TIMESTAMP{ ${h64(ts)} };
inline constexpr uint64_t FP_NONCE{ ${h64(nonce)} };
inline constexpr std::array<uint8_t, 32> FP_HASH1${hexArr(FP_HASH1).replace("{ ", "{{ ").replace(" }", " }}")};

inline constexpr std::array<uint8_t, 32> FP_FINAL${hexArr(FP_FINAL).replace("{ ", "{{ ").replace(" }", " }}")};

inline constexpr std::array<uint8_t, 32> FP_TARGET_PASS${hexArr(FP_FINAL).replace("{ ", "{{ ").replace(" }", " }}")};

inline constexpr std::array<uint8_t, 32> FP_TARGET_FAIL${hexArr(FP_FAIL).replace("{ ", "{{ ").replace(" }", " }}")};

}
`);
