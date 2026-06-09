# FishHash (Iron Fish) — implementation notes

Status: **working & live-verified** on AMD (OpenCL). Live accepted shares, 0 rejected,
**~24–25 MH/s** on RX 9070 XT vs `fishhash.unmineable.com:3333` (2026-06-09). FishHash is
**memory-bandwidth bound** (random lookups into the ~4.83 GB DAG), so ~25 MH/s is close to
this card's ceiling — kernel tuning won't move it much.

## Algorithm

Fixed-seed, no-epoch, ethash-like memory-hard PoW with blake3 endpoints. Reference:
`iron-fish/fish-hash` (vendored under `sources/algo/fishhash/reference/`).

- Light cache: 1,179,641 × 64 B (~75 MB), built on host, uploaded.
- Full dataset: 37,748,717 × 128 B (**~4.83 GB**), built once on-GPU (fixed seed).
- Hash: `seed=blake3(header,180)` → 32× dataset-access mix → `blake3(seed‖mixHash)`.
- Header 180 B; randomness (nonce) is the **last 8 bytes** [172:180], big-endian.
- Win: `hash <= target` as 32-byte big-endian.

## Key gotchas (hard-won)

1. **4 GB DAG limit (the big one).** AMD OpenCL caps a single buffer at 4 GB (32-bit
   byte addressing). A 4.83 GB DAG silently leaves items at offset ≥ 2³² unwritten
   (zero) → search reads zeros for ~13% of accesses → wrong hash → all shares
   rejected. Fix: split the DAG into 1 GiB chunk buffers (`MAX_DAG_CHUNKS`,
   `DAG_CHUNK_ITEMS`); the search kernel selects a chunk per access. A startup
   `DAG check[...] OK` self-check guards against regressions.
2. **Vendored `keccak.c` is `inline`.** The windows-cross toolchain compiles the
   `.c` as C++, where an unused inline def emits no symbol → `undefined symbol:
   keccak`. Dropped `inline`. Also guarded the ARM-only NEON include in `blake3.c`.
3. **unmineable `xn="00"`** → `setExtraNonce` yields `startNonce=0`, which
   `isValidJob()` rejects. Bump a zero start-nonce to 1 (xn only fixes the leading
   byte, so the submitted randomness still starts with xn).

## unmineable / Iron Fish stratum

unmineable's FishHash endpoint speaks the **native Iron Fish protocol** (v3):
envelope `{id, method, body}`; `mining.subscribe → subscribed{clientId,xn}`,
`mining.set_target{target}`, `mining.notify{miningRequestId,header}`,
`mining.submit{miningRequestId, randomness}` (full 8-byte BE hex, must start with xn).

Run: `miner --algo fishhash --host fishhash.unmineable.com --port 3333
--wallet <alias.worker> --workername <name> --devices_disable=<igpu>`

**unmineable alias login (gotcha):** in alias mode the login is `<alias>.<worker>` and the
miner sends `wallet` verbatim as the subscribe `publicAddress`. Pass the combined string as
`--wallet`, e.g. `--wallet=4forward.lm-test` — passing `--wallet=4forward --workername=lm-test`
sends `publicAddress="4forward"` alone and the pool rejects it ("Invalid address").

## Test coverage

- CPU reference KAT: `sources/algo/fishhash/tests/fishhash.cpp` (golden vectors + bootstrap).
- On-GPU AMD resolver: `sources/resolver/amd/tests/fishhash.cpp`
  (`ResolverFishhashAmdTest.findNonce`) — builds the DAG on the device, runs the integrity
  check, compiles the kernels with the device compiler, finds the bootstrap winner (nonce 45).
- Offline stratum protocol: `sources/stratum/tests/fishhash_protocol.cpp` (notify/set_target
  left-pad / zero-xnonce guard / submitted callback).

## Follow-ups (not done here)

- CUDA/NVIDIA resolver (mirror of the OpenCL path).
- FishHashPlus / Karlsen (kernel index-derivation already `#ifdef`-guarded; final-hash
  wiring + stratum TBD).
