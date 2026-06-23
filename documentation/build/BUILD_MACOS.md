# Build — macOS (Apple Silicon, OpenCL GPU + CPU)

> macOS has no CUDA. It does ship an OpenCL **1.2** runtime that exposes the
> integrated GPU as a real OpenCL device, so GPU mining via Apple's
> `OpenCL.framework` is supported (Blake3 verified on Apple M5). NVIDIA is off.
> This is a **native** build: Docker and podman on macOS run a *Linux* VM and
> would build the Linux miner, not a Mach-O one, and Apple forbids running macOS
> in a container — so the build uses the vcpkg + CMake-preset flow directly.

## Requirements (via [Homebrew](https://brew.sh))

```sh
brew install cmake ninja pkg-config   # pkg-config is needed by vcpkg's openssl port
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_ROOT=~/vcpkg
```

## Configure and build (GPU + CPU)

```sh
cmake --preset macos-opencl
cmake --build --preset macos-opencl -j$(sysctl -n hw.ncpu)
```

Binaries are written to `bin/miner` and `bin/benchmark`. Boost, OpenSSL, and the
Khronos OpenCL headers come from vcpkg (`vcpkg.json`); the OpenCL **runtime** is
Apple's `OpenCL.framework`. The first configure is slow because vcpkg compiles
Boost from source.

### How the OpenCL path works on macOS

- Apple's framework ships only `<OpenCL/*.h>`; the Khronos `<CL/opencl.hpp>` C++
  header the code includes comes from the vcpkg `opencl` port.
- The framework is the OpenCL platform. The Khronos ICD **loader** finds no
  vendor on macOS, so the build links the framework directly (`-framework
  OpenCL`), not the loader.
- The framework is frozen at OpenCL **1.2** (no `clCreateCommandQueueWithProperties`),
  so the C++ wrapper is compiled with `CL_HPP_TARGET_OPENCL_VERSION=120`.
- Kernel `-cl-std` is chosen at runtime from `CL_DEVICE_OPENCL_C_VERSION`
  (Apple → `CL1.2`; capable AMD on other platforms → `CL3.0`). `-O3` (a
  non-standard build option Apple rejects) is only passed on non-Apple vendors.

For a CPU-only build instead, use the `macos-cpu` preset (both GPU backends off).

## Algorithm support on macOS (Apple OpenCL 1.2, Apple M5)

All OpenCL kernels compile and execute under Apple's 1.2 compiler. The table
reports the benchmark's measured **throughput**; the benchmark does not validate
share correctness, so only Blake3 (the primary target) is treated as verified
end-to-end.

| Algorithm            | Kernel builds | Benchmark throughput | Notes |
|----------------------|:-------------:|----------------------|-------|
| blake3               | ✅ | ~310 MH/s            | Verified (primary target) |
| kheavyhash           | ✅ | ~91 MH/s             | Builds + runs |
| autolykos_v2         | ✅ | ~165 MH/s (search)   | Builds + runs; DAG build is slow |
| ethash               | ✅ | ~10 MH/s             | Builds + runs |
| ethash_light_cache   | ✅ | ~9.5 KH/s            | Cache generation step |
| progpow              | ✅ | ~7.4 MH/s            | Builds + runs |
| kawpow               | ✅ | not measured         | Light cache + DAG kernel build; search throughput not captured in the probe window (slow DAG) |

> Throughput on the integrated GPU is far below discrete AMD hardware and varies
> by Mac. Treat non-Blake3 rows as "compiles and executes", pending share-level
> validation against a pool.
