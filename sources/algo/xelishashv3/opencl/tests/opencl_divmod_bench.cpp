// Head-to-head benchmark of the stage-3 128/64 divmod variants (XV3_DIVMOD_IMPL=0..3).
//
// stage 3 is the hot path of XelisHash V3 (67,968 serial, data-dependent iterations per hash),
// and its single heaviest op is the 128-bit binary long division. This harness builds the SAME
// kernel once per variant (injecting -D XV3_DIVMOD_IMPL=N into the OpenCL build options), proves
// each variant is bit-exact against the CPU reference + gold vectors, then times the real
// `search` kernel over a batch of nonces on whatever device the ICD exposes — so all variants
// are compared in one process, on one device, free of pool/job-stream noise.
//
//   variant 0  full 128-bit long division              (baseline; deployed default)
//   variant 1  single 64-bit remainder, bit-serial     (half the per-bit ALU)
//   variant 2  native high-word fold + 64-iter tail     (half the loop trips of v1)
//   variant 3  native fold + Hacker's Delight divlu      (no per-bit loop)
//
// Usage: xelishashv3_opencl_bench [globalSize] [minSeconds]   (defaults: 4096, 2.0)

#define CL_TARGET_OPENCL_VERSION 300

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <CL/cl.h>

#include <algo/xelishashv3/types.hpp>
#include "xelishashv3_test_vectors.hpp"


namespace
{
    using xelishashv3::Hash256;

    constexpr int VARIANT_COUNT{ 4 };
    char const* const VARIANT_LABEL[VARIANT_COUNT]{
        "v0 full 128-bit long-div (baseline)",
        "v1 64-bit remainder, bit-serial",
        "v2 native fold + 64-iter tail",
        "v3 native fold + divlu (no loop)"
    };

    struct GpuResult
    {
        uint8_t  found{ 0 };
        uint32_t count{ 0 };
        uint64_t nonces[4]{ 0, 0, 0, 0 };
    } __attribute__((aligned(8)));


    void clDie(cl_int const err, char const* const what)
    {
        if (CL_SUCCESS != err)
        {
            std::fprintf(stderr, "FATAL: %s failed: %d\n", what, err);
            std::exit(2);
        }
    }


    std::string readFile(char const* const path)
    {
        std::ifstream in{ path };
        if (false == in.good())
        {
            std::fprintf(stderr, "FATAL: cannot open %s\n", path);
            std::exit(2);
        }
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
}


int main(int argc, char** argv)
{
    size_t const globalSize{ (argc > 1) ? static_cast<size_t>(std::strtoull(argv[1], nullptr, 10)) : 4096u };
    double const minSeconds{ (argc > 2) ? std::strtod(argv[2], nullptr) : 2.0 };

    // ─── device ───
    cl_platform_id platform{};
    cl_uint        numPlatforms{ 0 };
    clDie(clGetPlatformIDs(1, &platform, &numPlatforms), "clGetPlatformIDs");

    cl_device_id device{};
    cl_uint      numDevices{ 0 };
    if (CL_SUCCESS != clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices) || 0u == numDevices)
    {
        clDie(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &numDevices), "clGetDeviceIDs");
    }
    char devName[256]{};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devName), devName, nullptr);

    cl_int err{ CL_SUCCESS };
    cl_context context{ clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err) };
    clDie(err, "clCreateContext");
    cl_command_queue queue{ clCreateCommandQueueWithProperties(context, device, nullptr, &err) };
    clDie(err, "clCreateCommandQueue");

    // Kernel paths are baked at build time, but a cross-built exe runs on a different host than
    // it was compiled on — let env vars override so the .cl can be located at runtime.
    char const* const b3Path{ std::getenv("XV3_BLAKE3_CL") ? std::getenv("XV3_BLAKE3_CL") : XV3_BLAKE3_CL_PATH };
    char const* const xvPath{ std::getenv("XV3_CL") ? std::getenv("XV3_CL") : XV3_CL_PATH };
    std::string const src{ readFile(b3Path) + "\n" + readFile(xvPath) };
    char const*  srcPtr{ src.c_str() };
    size_t const srcLen{ src.size() };

    std::printf("device      : %s\n", devName);
    std::printf("batch       : %zu nonces/launch, >= %.1fs per variant\n\n", globalSize, minSeconds);

    // ─── shared device buffers (reused across variants) ───
    uint8_t const zeroTarget[32]{};
    cl_mem inBuf{ clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xelishashv3::INPUT_LEN,
                                 const_cast<uint8_t*>(xelishashv3::kat::VERIFY_INPUT.data()), &err) };
    clDie(err, "buf in");
    cl_mem tgtBuf{ clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32u,
                                  const_cast<uint8_t*>(zeroTarget), &err) };
    clDie(err, "buf target");
    cl_mem scratchBuf{ clCreateBuffer(context, CL_MEM_READ_WRITE, globalSize * xelishashv3::MEMSIZE_BYTES,
                                      nullptr, &err) };
    clDie(err, "buf scratch (try a smaller globalSize if this fails)");
    cl_mem oneScratch{ clCreateBuffer(context, CL_MEM_READ_WRITE, xelishashv3::MEMSIZE_BYTES, nullptr, &err) };
    clDie(err, "buf oneScratch");

    double hashrate[VARIANT_COUNT]{};
    bool   correct[VARIANT_COUNT]{};

    for (int impl = 0; impl < VARIANT_COUNT; ++impl)
    {
        cl_program program{ clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err) };
        clDie(err, "clCreateProgramWithSource");

        std::string const opts{ "-D XV3_DIVMOD_IMPL=" + std::to_string(impl) };
        if (CL_SUCCESS != clBuildProgram(program, 1, &device, opts.c_str(), nullptr, nullptr))
        {
            size_t logSize{ 0 };
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::string log(logSize, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::fprintf(stderr, "variant %d build failed:\n%s\n", impl, log.c_str());
            clReleaseProgram(program);
            continue;
        }

        // ─── correctness: one full hash via xelis_kat for ZERO and VERIFY ───
        auto runKat = [&](uint8_t const* input) -> Hash256
        {
            cl_mem ib{ clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xelishashv3::INPUT_LEN,
                                      const_cast<uint8_t*>(input), &err) };
            clDie(err, "kat in");
            cl_mem ob{ clCreateBuffer(context, CL_MEM_WRITE_ONLY, 32u, nullptr, &err) };
            clDie(err, "kat out");
            cl_kernel k{ clCreateKernel(program, "xelis_kat", &err) };
            clDie(err, "kernel xelis_kat");
            clDie(clSetKernelArg(k, 0, sizeof(cl_mem), &ib), "kat a0");
            clDie(clSetKernelArg(k, 1, sizeof(cl_mem), &oneScratch), "kat a1");
            clDie(clSetKernelArg(k, 2, sizeof(cl_mem), &ob), "kat a2");
            size_t one{ 1 };
            clDie(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr), "kat ndr");
            clDie(clFinish(queue), "kat finish");
            Hash256 got{};
            clDie(clEnqueueReadBuffer(queue, ob, CL_TRUE, 0, 32u, got.data(), 0, nullptr, nullptr), "kat read");
            clReleaseKernel(k);
            clReleaseMemObject(ib);
            clReleaseMemObject(ob);
            return got;
        };

        Hash256 const gpuZero{ runKat(xelishashv3::kat::ZERO_INPUT.data()) };
        Hash256 const gpuVerify{ runKat(xelishashv3::kat::VERIFY_INPUT.data()) };
        correct[impl] = (0 == std::memcmp(gpuZero.data(), xelishashv3::kat::ZERO_EXPECTED.data(), 32))
                     && (0 == std::memcmp(gpuVerify.data(), xelishashv3::kat::VERIFY_EXPECTED.data(), 32));
        if (false == correct[impl])
        {
            std::fprintf(stderr, "variant %d: DIGEST MISMATCH vs gold vectors — not bit-exact\n", impl);
            clReleaseProgram(program);
            continue;
        }

        // ─── timing: real `search` kernel over `globalSize` nonces, repeated to >= minSeconds ───
        cl_kernel search{ clCreateKernel(program, "search", &err) };
        clDie(err, "kernel search");
        clDie(clSetKernelArg(search, 0, sizeof(cl_mem), &inBuf), "s0");
        clDie(clSetKernelArg(search, 1, sizeof(cl_mem), &tgtBuf), "s1");
        clDie(clSetKernelArg(search, 3, sizeof(cl_mem), &scratchBuf), "s3");

        GpuResult zero{};
        cl_mem    resBuf{ clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(GpuResult),
                                         &zero, &err) };
        clDie(err, "buf result");
        clDie(clSetKernelArg(search, 4, sizeof(cl_mem), &resBuf), "s4");

        // warm-up launch (driver JIT / first-touch allocation) — not timed
        cl_ulong sn{ 0 };
        clDie(clSetKernelArg(search, 2, sizeof(cl_ulong), &sn), "s2");
        clDie(clEnqueueNDRangeKernel(queue, search, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr), "warm");
        clDie(clFinish(queue), "warm finish");

        size_t nonces{ 0 };
        auto   t0{ std::chrono::steady_clock::now() };
        double elapsed{ 0.0 };
        do
        {
            sn += globalSize;
            clDie(clSetKernelArg(search, 2, sizeof(cl_ulong), &sn), "s2");
            clDie(clEnqueueNDRangeKernel(queue, search, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr),
                  "ndr");
            clDie(clFinish(queue), "finish");
            nonces += globalSize;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } while (elapsed < minSeconds);

        hashrate[impl] = static_cast<double>(nonces) / elapsed;
        clReleaseMemObject(resBuf);
        clReleaseKernel(search);
        clReleaseProgram(program);

        std::printf("  built+measured variant %d (%s): %.1f H/s\n", impl, VARIANT_LABEL[impl], hashrate[impl]);
    }

    clReleaseMemObject(oneScratch);
    clReleaseMemObject(scratchBuf);
    clReleaseMemObject(tgtBuf);
    clReleaseMemObject(inBuf);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // ─── results table ───
    std::printf("\n  %-38s %10s %8s\n", "variant", "H/s", "vs v0");
    std::printf("  %-38s %10s %8s\n", "--------------------------------------", "----------", "--------");
    for (int impl = 0; impl < VARIANT_COUNT; ++impl)
    {
        if (false == correct[impl])
        {
            std::printf("  %-38s %10s %8s\n", VARIANT_LABEL[impl], "FAILED", "-");
            continue;
        }
        char speed[16]{ "-" };
        if (impl > 0 && hashrate[0] > 0.0)
        {
            std::snprintf(speed, sizeof(speed), "%.3fx", hashrate[impl] / hashrate[0]);
        }
        std::printf("  %-38s %10.1f %8s\n", VARIANT_LABEL[impl], hashrate[impl], speed);
    }
    return 0;
}
