# Octopus CPU-oracle reference vectors

Ground-truth captured **bit-exactly** from the reference miner `Conflux-Chain/open-cfxmine`
(`src/light.cc` + `src/sha3.cc`, pure C++ — no CUDA), via the standalone harness at
`D:\projects\octopus-oracle\` (compiled in `gcc:13` docker). Re-capture with:

```
docker run --rm -v "D:\projects\octopus-oracle:/work" -w /work gcc:13 \
  bash -c "g++ -O2 -std=c++17 light.cc sha3.cc capture_main.cc -o capture && ./capture"
```

The LuminousMiner octopus CPU oracle (M1) MUST reproduce every value below.

## Fixed test inputs
- `header` = 32 bytes `0x00 0x01 0x02 … 0x1f`
- `nonce` ∈ { 0, 12345 }
- `OCTOPUS_N` = 1024, `OCTOPUS_MOD` = 1032193, `EPOCH_LENGTH` = 524288

## 1. DAG / cache sizes (matches analytical formula exactly)
| blockNumber | epoch | cache_size (bytes) | data_size (bytes) | data (GiB) |
|---|---|---|---|---|
| 0          | 0   | 16776896  | 4294966528 | 4.000 |
| 34078720   | 65  | 21035968  | 5385477376 | 5.016 |
| 149499627  | 285 | 35454784  | 9076465408 | 8.453 |

## 2. compute_d — NTT/polynomial coefficients (length 1024)
| nonce | d[0..7] | d[1023] | fnv1a-64(d, 4096B) |
|---|---|---|---|
| 0     | 349892,834550,588346,449621,85820,185537,368936,447534 | 585399 | d0349c4bb6c48497 |
| 12345 | 628584,456832,993474,590150,215332,330854,627355,138534 | 103108 | 790ed3a92c0b99ca |

## 3. multi_eval — (thread_result, vector<u32> len 32)
| nonce | thread_result | v[0..7] | fnv1a-64(v, 128B) |
|---|---|---|---|
| 0     | 16285858431497212756 | 188101,213632,115490,721627,326911,925788,246724,906503 | 278e0a7d5a90255c |
| 12345 | 17518503130972107457 | 823148,687249,759189,521781,980657,460517,728407,731710 | 92423205d5801c4e |

## 4. Light cache @ epoch 0
- `cache_size` = 16776896
- `fnv1a-64(cache)` = d36a5e79fad00334
- first 64 bytes = `5e493e76a1318e50815c6ce77950425532964ebbb8dcf94718991fa9a82eaf37658de68ca6fe078884e803da3a26a4aa56420a6867ebcd9ab0f29b08d1c48fed`
- last 64 bytes = `724f2f86c24c487809dc3897acbbd32d5d791e4536aa1520e65e93891a40dde5887899ffc556cbd174f426e32ae2ab711be859601c024d1514b29a27370b662e`

## 5. End-to-end octopus_light_compute @ epoch 0 (32-byte pow hash)
| nonce | success | hash |
|---|---|---|
| 0     | 1 | aeb06e4738269d0d60ced2206d21ec5c331690342a54f5c37f1d8eef4abdaf94 |
| 12345 | 1 | d8e2e19a7ab253fba2a9c085157dae170288d3e9853f7967e5f71363eeeb85e9 |

## Still to capture (when M1.4 needs them)
- Per-index dataset-item vectors: `octopus_calculate_dag_item` is `static` in `light.cc`;
  expose it (drop `static inline` + add a prototype) and dump items at indices {0,1,1000,last} for epoch 0.
- Additional epochs/headers as needed for broader coverage.
