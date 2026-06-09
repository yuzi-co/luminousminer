// Host harness that runs the PyrinHashV2 OpenCL kernels on whatever OpenCL device
// the ICD exposes (POCL/CPU in CI, a real GPU otherwise) and asserts each stage is
// BIT-IDENTICAL to the CPU reference's known-answer vectors.
//
// The kernel depends on the shared BLAKE3 primitive (crypto/opencl/blake3.cl), so the
// harness concatenates blake3.cl ahead of pyrinhashv2.cl before building — the same
// scheme the resolver's kernel generator uses (addInclude blake3 then the algo kernel).
// Both .cl files are loaded at runtime (paths baked in), so there is no second copy.

#define CL_TARGET_OPENCL_VERSION 300

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <CL/cl.h>
#include <gtest/gtest.h>

#include <algo/pyrinhashv2/matrix.hpp>
#include "pyrinhashv2_test_vectors.hpp"

namespace
{
    using pyrinhashv2::Hash256;
    using pyrinhashv2::Matrix;

    void clCheck(cl_int const err, char const* what)
    {
        ASSERT_EQ(CL_SUCCESS, err) << what << " failed: " << err;
    }


    std::vector<uint16_t> flatten(Matrix const& matrix)
    {
        std::vector<uint16_t> flat;
        flat.reserve(64 * 64);
        for (int r{ 0 }; r < 64; ++r)
        {
            for (int c{ 0 }; c < 64; ++c)
            {
                flat.push_back(matrix[r][c]);
            }
        }
        return flat;
    }


    std::string readFile(char const* const path)
    {
        std::ifstream in{ path };
        if (false == in.good())
        {
            return std::string{};
        }
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }


    // One OpenCL context/program, built once and shared by every test.
    class OpenClKat : public ::testing::Test
    {
    protected:
        static cl_context       context;
        static cl_command_queue queue;
        static cl_program       program;
        static cl_device_id     device;

        static void SetUpTestSuite()
        {
            cl_platform_id platform{};
            cl_uint        numPlatforms{ 0 };
            clCheck(clGetPlatformIDs(1, &platform, &numPlatforms), "clGetPlatformIDs");
            ASSERT_GT(numPlatforms, 0u) << "no OpenCL platform";

            cl_uint numDevices{ 0 };
            clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &numDevices), "clGetDeviceIDs");
            ASSERT_GT(numDevices, 0u) << "no OpenCL device";

            cl_int err{ CL_SUCCESS };
            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            clCheck(err, "clCreateContext");
            queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
            clCheck(err, "clCreateCommandQueue");

            // Shared BLAKE3 primitive first (env override for the rig, then compile-time path),
            // then the algo kernel. The concatenation is what puts blake3_hash_chunk in scope.
            std::string blake3Src{ readFile(std::getenv("P2_BLAKE3_CL_PATH") ? std::getenv("P2_BLAKE3_CL_PATH")
                                                                             : P2_BLAKE3_CL_PATH) };
            ASSERT_FALSE(blake3Src.empty()) << "cannot open blake3.cl (P2_BLAKE3_CL_PATH=" << P2_BLAKE3_CL_PATH << ")";
            std::string kernelSrc{ readFile(std::getenv("P2_CL_PATH") ? std::getenv("P2_CL_PATH") : P2_CL_PATH) };
            ASSERT_FALSE(kernelSrc.empty()) << "cannot open pyrinhashv2.cl (P2_CL_PATH=" << P2_CL_PATH << ")";

            std::string const src{ blake3Src + "\n" + kernelSrc };
            char const*       srcPtr{ src.c_str() };
            size_t const      srcLen{ src.size() };

            program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
            clCheck(err, "clCreateProgramWithSource");
            err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
            if (CL_SUCCESS != err)
            {
                size_t logSize{ 0 };
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                std::string log(logSize, '\0');
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
                FAIL() << "clBuildProgram failed:\n" << log;
            }
        }

        static void TearDownTestSuite()
        {
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }

        cl_mem makeBuf(cl_mem_flags const flags, size_t const bytes, void const* host)
        {
            cl_int err{ CL_SUCCESS };
            cl_mem buf{ clCreateBuffer(context, flags, bytes, const_cast<void*>(host), &err) };
            clCheck(err, "clCreateBuffer");
            return buf;
        }

        cl_kernel kernel(char const* name)
        {
            cl_int    err{ CL_SUCCESS };
            cl_kernel k{ clCreateKernel(program, name, &err) };
            clCheck(err, "clCreateKernel");
            return k;
        }

        void run(cl_kernel k, size_t const global)
        {
            clCheck(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel");
            clCheck(clFinish(queue), "clFinish");
        }
    };

    cl_context       OpenClKat::context{ nullptr };
    cl_command_queue OpenClKat::queue{ nullptr };
    cl_program       OpenClKat::program{ nullptr };
    cl_device_id     OpenClKat::device{ nullptr };


    Hash256 toHash(std::array<uint8_t, 32> const& a)
    {
        Hash256 h{};
        for (int i{ 0 }; i < 32; ++i)
        {
            h[i] = a[i];
        }
        return h;
    }
}


TEST_F(OpenClKat, PowHashMatchesReference)
{
    std::array<uint8_t, 32> pre{};
    for (int i{ 0 }; i < 32; ++i)
    {
        pre[i] = pyrinhashv2::kat::POW_KAT_PRE[i];
    }
    std::array<uint8_t, 32> out{};

    cl_mem    preBuf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, pre.data()) };
    cl_mem    outBuf{ makeBuf(CL_MEM_WRITE_ONLY, 32, nullptr) };
    cl_kernel k{ kernel("test_pow_hash") };
    cl_ulong  ts{ pyrinhashv2::kat::POW_KAT_TIMESTAMP };
    cl_ulong  nonce{ pyrinhashv2::kat::POW_KAT_NONCE };
    clCheck(clSetKernelArg(k, 0, sizeof(cl_mem), &preBuf), "arg0");
    clCheck(clSetKernelArg(k, 1, sizeof(cl_ulong), &ts), "arg1");
    clCheck(clSetKernelArg(k, 2, sizeof(cl_ulong), &nonce), "arg2");
    clCheck(clSetKernelArg(k, 3, sizeof(cl_mem), &outBuf), "arg3");
    run(k, 1);
    clCheck(clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0, 32, out.data(), 0, nullptr, nullptr), "read");

    EXPECT_EQ(out, pyrinhashv2::kat::POW_KAT_EXPECTED);

    clReleaseKernel(k);
    clReleaseMemObject(preBuf);
    clReleaseMemObject(outBuf);
}


TEST_F(OpenClKat, KHeavyHashMatchesReference)
{
    std::array<uint8_t, 32> out{};
    cl_mem    inBuf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32,
                             const_cast<uint8_t*>(pyrinhashv2::kat::HEAVY_INPUT.data())) };
    cl_mem    outBuf{ makeBuf(CL_MEM_WRITE_ONLY, 32, nullptr) };
    cl_kernel k{ kernel("test_kheavy") };
    clCheck(clSetKernelArg(k, 0, sizeof(cl_mem), &inBuf), "arg0");
    clCheck(clSetKernelArg(k, 1, sizeof(cl_mem), &outBuf), "arg1");
    run(k, 1);
    clCheck(clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0, 32, out.data(), 0, nullptr, nullptr), "read");

    EXPECT_EQ(out, pyrinhashv2::kat::KHEAVY_EXPECTED);

    clReleaseKernel(k);
    clReleaseMemObject(inBuf);
    clReleaseMemObject(outBuf);
}


TEST_F(OpenClKat, HeavyHashV2MatchesReference)
{
    std::vector<uint16_t> const flat{ flatten(pyrinhashv2::kat::HEAVY_TEST_MATRIX) };
    std::array<uint8_t, 32>     out{};

    cl_mem    matBuf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flat.size() * sizeof(uint16_t),
                              const_cast<uint16_t*>(flat.data())) };
    cl_mem    h1Buf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32,
                             const_cast<uint8_t*>(pyrinhashv2::kat::HEAVY_INPUT.data())) };
    cl_mem    outBuf{ makeBuf(CL_MEM_WRITE_ONLY, 32, nullptr) };
    cl_kernel k{ kernel("test_heavy_hash") };
    clCheck(clSetKernelArg(k, 0, sizeof(cl_mem), &matBuf), "arg0");
    clCheck(clSetKernelArg(k, 1, sizeof(cl_mem), &h1Buf), "arg1");
    clCheck(clSetKernelArg(k, 2, sizeof(cl_mem), &outBuf), "arg2");
    run(k, 1);
    clCheck(clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0, 32, out.data(), 0, nullptr, nullptr), "read");

    EXPECT_EQ(out, pyrinhashv2::kat::HEAVY_EXPECTED);

    clReleaseKernel(k);
    clReleaseMemObject(matBuf);
    clReleaseMemObject(h1Buf);
    clReleaseMemObject(outBuf);
}


// Full per-nonce mining kernel: the work item whose nonce reproduces FP_FINAL must
// report a hit at exactly FP_NONCE, and must NOT report one against a target one unit
// below FP_FINAL.
class SearchKernel : public OpenClKat, public ::testing::WithParamInterface<char const*>
{
protected:
    struct alignas(8) Result
    {
        uint8_t  found{ 0 };
        uint32_t count{ 0 };
        uint64_t nonces[4]{ 0, 0, 0, 0 };
    };

    cl_ulong runSearch(std::array<uint8_t, 32> const& target, cl_ulong nonceStart, size_t globalSize,
                       cl_uint& foundCount)
    {
        Matrix const                matrix{ pyrinhashv2::generateMatrix(toHash(pyrinhashv2::kat::FP_PRE)) };
        std::vector<uint16_t> const flat{ flatten(matrix) };

        Result result{};
        cl_mem matBuf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flat.size() * sizeof(uint16_t),
                               const_cast<uint16_t*>(flat.data())) };
        cl_mem preBuf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32,
                               const_cast<uint8_t*>(pyrinhashv2::kat::FP_PRE.data())) };
        cl_mem tgtBuf{ makeBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32,
                               const_cast<uint8_t*>(target.data())) };
        cl_mem resBuf{ makeBuf(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(Result), &result) };

        cl_kernel k{ kernel(GetParam()) };
        cl_ulong  ts{ pyrinhashv2::kat::FP_TIMESTAMP };
        clCheck(clSetKernelArg(k, 0, sizeof(cl_mem), &matBuf), "arg0");
        clCheck(clSetKernelArg(k, 1, sizeof(cl_mem), &preBuf), "arg1");
        clCheck(clSetKernelArg(k, 2, sizeof(cl_mem), &tgtBuf), "arg2");
        clCheck(clSetKernelArg(k, 3, sizeof(cl_ulong), &ts), "arg3");
        clCheck(clSetKernelArg(k, 4, sizeof(cl_ulong), &nonceStart), "arg4");
        clCheck(clSetKernelArg(k, 5, sizeof(cl_mem), &resBuf), "arg5");
        run(k, globalSize);
        clCheck(clEnqueueReadBuffer(queue, resBuf, CL_TRUE, 0, sizeof(Result), &result, 0, nullptr, nullptr),
                "read result");

        clReleaseKernel(k);
        clReleaseMemObject(matBuf);
        clReleaseMemObject(preBuf);
        clReleaseMemObject(tgtBuf);
        clReleaseMemObject(resBuf);
        foundCount = result.count;
        return result.nonces[0];
    }
};


// FP_TARGET_PASS == FP_FINAL and FP_TARGET_FAIL == FP_FINAL - 1 (LSB decremented).
// A single work-item at exactly FP_NONCE that passes the first and fails the second pins
// the kernel's end-to-end pow output to FP_FINAL bit-for-bit, and proves the hit-reporting
// (atomic_inc + foundNonce = nonceStart + gid) path.
TEST_P(SearchKernel, ReportsHitAtWinningNonce)
{
    cl_uint        count{ 0 };
    cl_ulong const found{ runSearch(pyrinhashv2::kat::FP_TARGET_PASS, pyrinhashv2::kat::FP_NONCE, 1, count) };

    EXPECT_EQ(1u, count);
    EXPECT_EQ(pyrinhashv2::kat::FP_NONCE, found);
}


TEST_P(SearchKernel, NoHitWhenPowExceedsTarget)
{
    cl_uint count{ 0 };
    runSearch(pyrinhashv2::kat::FP_TARGET_FAIL, pyrinhashv2::kat::FP_NONCE, 1, count);

    EXPECT_EQ(0u, count);
}


INSTANTIATE_TEST_SUITE_P(Search,
                         SearchKernel,
                         ::testing::Values("search"),
                         [](::testing::TestParamInfo<char const*> const& info)
                         { return std::string{ info.param }; });
