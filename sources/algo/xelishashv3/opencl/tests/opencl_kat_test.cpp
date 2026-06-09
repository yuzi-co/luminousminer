// Host harness that runs the XelisHash V3 OpenCL kernel on whatever OpenCL device the ICD
// exposes (POCL/CPU in CI, a real GPU otherwise) and asserts the GPU digest is BIT-IDENTICAL
// to both the CPU reference and the upstream gold vectors.
//
// The kernel depends on the shared BLAKE3 primitive (crypto/opencl/blake3.cl), so the harness
// concatenates blake3.cl ahead of xelishashv3.cl before building — the same scheme the
// resolver's kernel generator uses. Both .cl are loaded at runtime (paths baked in), so there
// is no second copy.

#define CL_TARGET_OPENCL_VERSION 300

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <CL/cl.h>
#include <gtest/gtest.h>

#include <algo/xelishashv3/types.hpp>
#include <algo/xelishashv3/xelishashv3.hpp>
#include "xelishashv3_test_vectors.hpp"


namespace
{
    using xelishashv3::Hash256;

    void clCheck(cl_int const err, char const* what)
    {
        ASSERT_EQ(CL_SUCCESS, err) << what << " failed: " << err;
    }


    std::string readFile(char const* const path)
    {
        std::ifstream in{ path };
        EXPECT_TRUE(in.good()) << "cannot open " << path;
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }


    class XelisOpenClKat : public ::testing::Test
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

            std::string const src{ readFile(XV3_BLAKE3_CL_PATH) + "\n" + readFile(XV3_CL_PATH) };
            char const*  srcPtr{ src.c_str() };
            size_t const srcLen{ src.size() };

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

        struct GpuResult
        {
            uint8_t  found{ 0 };
            uint32_t count{ 0 };
            uint64_t nonces[4]{ 0, 0, 0, 0 };
        } __attribute__((aligned(8)));

        // Runs the mining `search` kernel over `globalSize` nonces from startNonce and returns
        // the device-side Result (found flag + winning nonces).
        GpuResult runSearch(uint8_t const* baseInput, uint8_t const* target, uint64_t startNonce, size_t globalSize)
        {
            cl_int err{ CL_SUCCESS };
            cl_mem inBuf{ clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         xelishashv3::INPUT_LEN, const_cast<uint8_t*>(baseInput), &err) };
            clCheck(err, "buf base");
            cl_mem tgtBuf{ clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          32, const_cast<uint8_t*>(target), &err) };
            clCheck(err, "buf target");
            cl_mem scratchBuf{ clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              globalSize * xelishashv3::MEMSIZE_BYTES, nullptr, &err) };
            clCheck(err, "buf scratch");
            GpuResult zero{};
            cl_mem resBuf{ clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(GpuResult), &zero, &err) };
            clCheck(err, "buf result");

            cl_kernel k{ clCreateKernel(program, "search", &err) };
            clCheck(err, "kernel search");
            cl_ulong sn{ startNonce };
            clCheck(clSetKernelArg(k, 0, sizeof(cl_mem), &inBuf), "s0");
            clCheck(clSetKernelArg(k, 1, sizeof(cl_mem), &tgtBuf), "s1");
            clCheck(clSetKernelArg(k, 2, sizeof(cl_ulong), &sn), "s2");
            clCheck(clSetKernelArg(k, 3, sizeof(cl_mem), &scratchBuf), "s3");
            clCheck(clSetKernelArg(k, 4, sizeof(cl_mem), &resBuf), "s4");

            clCheck(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr), "ndr");
            clCheck(clFinish(queue), "finish");

            GpuResult got{};
            clCheck(clEnqueueReadBuffer(queue, resBuf, CL_TRUE, 0, sizeof(GpuResult), &got, 0, nullptr, nullptr),
                    "read");

            clReleaseKernel(k);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(tgtBuf);
            clReleaseMemObject(scratchBuf);
            clReleaseMemObject(resBuf);
            return got;
        }

        // Runs one full XelisHash V3 on the device for `input` and returns the 32-byte digest.
        Hash256 runDevice(uint8_t const* input)
        {
            cl_int err{ CL_SUCCESS };
            cl_mem inBuf{ clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         xelishashv3::INPUT_LEN, const_cast<uint8_t*>(input), &err) };
            clCheck(err, "clCreateBuffer in");
            cl_mem scratchBuf{ clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              xelishashv3::MEMSIZE_BYTES, nullptr, &err) };
            clCheck(err, "clCreateBuffer scratch");
            cl_mem outBuf{ clCreateBuffer(context, CL_MEM_WRITE_ONLY, 32, nullptr, &err) };
            clCheck(err, "clCreateBuffer out");

            cl_kernel k{ clCreateKernel(program, "xelis_kat", &err) };
            clCheck(err, "clCreateKernel");
            clCheck(clSetKernelArg(k, 0, sizeof(cl_mem), &inBuf), "arg0");
            clCheck(clSetKernelArg(k, 1, sizeof(cl_mem), &scratchBuf), "arg1");
            clCheck(clSetKernelArg(k, 2, sizeof(cl_mem), &outBuf), "arg2");

            size_t global{ 1 };
            clCheck(clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "ndr");
            clCheck(clFinish(queue), "finish");

            Hash256 got{};
            clCheck(clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0, 32, got.data(), 0, nullptr, nullptr), "read");

            clReleaseKernel(k);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(scratchBuf);
            clReleaseMemObject(outBuf);
            return got;
        }
    };

    cl_context       XelisOpenClKat::context{ nullptr };
    cl_command_queue XelisOpenClKat::queue{ nullptr };
    cl_program       XelisOpenClKat::program{ nullptr };
    cl_device_id     XelisOpenClKat::device{ nullptr };
}


TEST_F(XelisOpenClKat, ZeroInputMatchesCpuAndGold)
{
    xelishashv3::ScratchPad pad;
    Hash256 const cpu{ xelishashv3::hash(xelishashv3::kat::ZERO_INPUT.data(), pad) };
    Hash256 const gpu{ runDevice(xelishashv3::kat::ZERO_INPUT.data()) };

    EXPECT_EQ(0, std::memcmp(cpu.data(), xelishashv3::kat::ZERO_EXPECTED.data(), 32)) << "CPU != gold";
    EXPECT_EQ(gpu, cpu) << "GPU != CPU";
    EXPECT_EQ(0, std::memcmp(gpu.data(), xelishashv3::kat::ZERO_EXPECTED.data(), 32)) << "GPU != gold";
}


TEST_F(XelisOpenClKat, VerifyInputMatchesCpuAndGold)
{
    xelishashv3::ScratchPad pad;
    Hash256 const cpu{ xelishashv3::hash(xelishashv3::kat::VERIFY_INPUT.data(), pad) };
    Hash256 const gpu{ runDevice(xelishashv3::kat::VERIFY_INPUT.data()) };

    EXPECT_EQ(gpu, cpu) << "GPU != CPU";
    EXPECT_EQ(0, std::memcmp(gpu.data(), xelishashv3::kat::VERIFY_EXPECTED.data(), 32)) << "GPU != gold";
}


// The search kernel writes nonce = startNonce + gid at offset 40 (big-endian) and accepts on
// big-endian hash <= target. Feeding VERIFY_INPUT's own nonce as startNonce makes gid 0
// reproduce VERIFY_EXPECTED; with target == that hash the share is exactly at the boundary
// (accepted), and the published nonce must equal the embedded nonce.
TEST_F(XelisOpenClKat, SearchFindsNonceAtTargetBoundary)
{
    uint64_t embeddedNonce{ 0 };
    for (int i = 0; i < 8; ++i)
    {
        embeddedNonce = (embeddedNonce << 8) | xelishashv3::kat::VERIFY_INPUT[40 + i];  // big-endian
    }

    GpuResult const r{ runSearch(xelishashv3::kat::VERIFY_INPUT.data(),
                                 xelishashv3::kat::VERIFY_EXPECTED.data(), embeddedNonce, 1) };
    EXPECT_EQ(1u, r.found) << "search did not accept a hash equal to target";
    ASSERT_GE(r.count, 1u);
    EXPECT_EQ(embeddedNonce, r.nonces[0]) << "published nonce mismatch";
}


// An all-zero target is unreachable (no non-zero hash is <= 0), so nothing is published.
TEST_F(XelisOpenClKat, SearchRejectsUnreachableTarget)
{
    uint8_t const zeroTarget[32]{};
    GpuResult const r{ runSearch(xelishashv3::kat::VERIFY_INPUT.data(), zeroTarget, 0, 4) };
    EXPECT_EQ(0u, r.found) << "search accepted against a zero target";
}
