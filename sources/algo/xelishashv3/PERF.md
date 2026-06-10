# XelisHash V3 — AMD/OpenCL performance notes (gfx1201 / RX 9070 XT)

Engineering record for the AMD resolver (`sources/resolver/amd/xelishashv3.cpp`) and OpenCL
kernel (`sources/algo/xelishashv3/opencl/xelishashv3.cl`).

## Verdict

XelisHash V3 on RDNA4 is **memory-roofline-bound**. Production ships the `divmod_divlu` stage-3
variant at an occupancy of 6144 in-flight nonces and is **at the optimum** for the kernel itself.
No compute, geometry, or register lever remains; the only further gain is a **+2.5% occupancy bump**
that is already reachable from the CLI (`--blocks 120`), and the only untested lever with material
headroom is **memory-clock overclock** (operational, outside the kernel).

## Why stage 3 is the hot path

Per nonce: stage 1 (BLAKE3 + ChaCha8) fills a ~531 KiB scratchpad, stage 3 runs
`ITERS * BUFSIZE = 2 * 33984 = 67968` **serial, data-dependent** iterations over it, stage 4
hashes the whole scratchpad. Stage 3 dominates: each iteration issues several *random* 8-byte
global gathers whose address depends on the previous iteration's `result` (`mapIndex(result)`,
`B[mapIndex(a ^ ...)]`, the write-back to `A[idxA]`/`B[idxB]`). That data dependency is the
latency wall — it cannot be prefetched or pipelined within a nonce. The only parallelism is
*across* nonces, i.e. occupancy.

## The optimization journey

| Iter | Lever | Result |
|---|---|---|
| 1 | Stage-3 128/64 divmod: full long-div → native fold + base-2^32 divlu (`XV3_DIVMOD_IMPL=3`) | **+38%** (3491→4813 H/s live). Shipped. |
| 2 | f32 isqrt seed (`XV3_ISQRT_IMPL=1`, non-consensus probe) | +1.6% only — FP64 cost is hidden behind the scratchpad gather/scatter. **Dropped.** |
| 2 | divmod remainder-only (`XV3_DIVMOD_IMPL=4`, drops the provably-dead high quotient word) | **+0.0%** — compiler already DCEs it / it's latency-hidden. **Dropped.** |
| 3 | Occupancy (in-flight nonce count) | +2.5% max, then collapses (see below). |
| 3 | Workgroup geometry (threads/block at fixed occupancy) | **Neutral** — total in-flight waves are fixed by globalSize; no LDS/cross-lane to exploit. |

### Stage-3 divmod variants (gfx1201, 6144 nonces, bit-exact vs gold)

| `XV3_DIVMOD_IMPL` | Variant | H/s | vs baseline |
|---|---|---|---|
| 0 | full 128-bit binary long division | 3503 | — |
| 1 | single 64-bit remainder, bit-serial | 3640 | 1.04× |
| 2 | native high-word fold + 64-iter tail | 4623 | 1.32× |
| **3** | **native fold + Hacker's Delight divlu (no per-bit loop)** | **4887** | **1.40× (shipped)** |

### Occupancy curve (globalSize = threads × blocks, divmod_divlu)

| globalSize | scratchpad | H/s | vs 6144 |
|---|---|---|---|
| 4096 | 2.2 GiB | 3986 | −18% |
| **6144 (default)** | 3.1 GiB | 4887 | — |
| 7168 | 3.6 GiB | 4987 | +2.0% |
| **7680** | **3.89 GiB** | **5006** | **+2.5% — safe ceiling** |
| 8192 | 4.15 GiB | 4199 / 2773 | **collapses, unstable** |

The marginal return decays from +98% (1k→2k nonces) to +2% (6k→7k): the saturation knee is at
~6144. **8192 overruns AMD's ~4 GiB single-allocation limit**, so the scratch buffer can no longer
be placed healthily and throughput collapses rather than failing cleanly. `7680` is the safe
maximum; pushing more nonces past the single-buffer limit (multi-buffer split) chases an asymptote
worth ≤~1% beyond 7680 — not worth the complexity.

## Tuning guidance

- Default occupancy (6144, ~3.1 GiB) is conservative for broad VRAM safety.
- On a card with VRAM headroom: `--threads 64 --blocks 120` (7680 nonces, 3.89 GiB) for **+2.5%**.
  Do **not** exceed ~7680 — `--blocks 128` (8192) collapses, it does not fail cleanly.
- The biggest remaining lever is **memory/VRAM overclock**: the kernel is bandwidth-bound, so MCLK
  raises throughput directly and can exceed the +2.5% occupancy gain. This is operational (driver /
  Adrenalin / external tool), not a kernel change.

## RGA resource analysis (Radeon GPU Analyzer 2.14, gfx1201)

For the `search` kernel built at `XV3_DIVMOD_IMPL=0` (the only fully-inlined, trustworthy build):
**64/256 VGPR, 71/106 SGPR, 0 LDS, 0 register spills, wave32.** 64 VGPR on RDNA4's 1536-VGPR/SIMD
file means register occupancy is already hardware-maxed (16 waves/SIMD); registers and LDS are *not*
the limiter — the per-nonce 531 KiB scratchpad is.

> Note: at `XV3_DIVMOD_IMPL=3` the offline Lightning compiler does **not** inline `xv3_hash` into
> `search`, so RGA reports a bogus module-level 116 VGPR / 3192-byte ISA. Trust the inlined
> `IMPL=0` figures and the on-GPU benchmark, not RGA's per-kernel occupancy for `IMPL=3`.

## Reproducing the benchmarks

Two harnesses, both built via the Windows cross-compile docker and run natively on the host (docker
cannot see the GPU). Free the GPU first (stop any running miner).

**Framework benchmark** (`sources/benchmark/amd/xelishashv3.cpp`, kawpow-style, config-driven —
occupancy/variants are read from `config.json` at runtime, so sweeps need no rebuild):

```sh
docker build -f docker/Dockerfile.windows-cross --build-arg GPU=amd --build-arg BUILD_BENCH=ON \
  --target artifact -o dist/windows-cross-amd-bench .
# run from the artifact dir (kernels load relative); edit config.json amd.algorithms.xelishashv3
# threads/blocks/kernels and re-run to sweep occupancy/geometry.
cd dist/windows-cross-amd-bench && ./benchmark.exe
```

The framework times each ~1.26 s launch individually, so its early samples are contaminated by GPU
clock-ramp — run one variant at a time with `loop >= 8` and read the steady tail.

**Standalone divmod bench** (`sources/algo/xelishashv3/opencl/tests/opencl_divmod_bench.cpp`) — builds
all four divmod variants in one process, KAT-gates each against the gold vectors, and times the real
`search` kernel as a continuous average (cleaner for occupancy sweeps):

```sh
# build (image lm-xcross-unittest-verify, no gtest/CPU-ref needed — compares vs gold header):
cmake -S sources/algo/xelishashv3 -DXELISHASHV3_BUILD_OPENCL=ON -DXELISHASHV3_BUILD_TESTS=OFF ...
# run native; env points the loader at host .cl paths:
XV3_CL=.../opencl/xelishashv3.cl XV3_BLAKE3_CL=.../crypto/opencl/blake3.cl \
  ./xelishashv3_opencl_bench <globalSize> <minSeconds>
```
