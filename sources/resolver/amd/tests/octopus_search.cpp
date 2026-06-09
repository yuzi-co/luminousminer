#include <cstring>
#include <vector>

#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/octopus/light.hpp>
#include <algo/octopus/octopus.hpp>
#include <algo/octopus/result.hpp>
#include <common/cast.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/tests/amd.hpp>


namespace
{
    // Big-endian 64-bit word of bytes b[k..k+7] with b[k] as most-significant byte.
    uint64_t swapU64(uint64_t const x)
    {
        uint64_t r{ 0ull };
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            r = (r << 8) | ((x >> (8u * i)) & 0xffull);
        }
        return r;
    }

    // a < b as 32-byte big-endian integers (octopus_check_difficulty ordering).
    bool hashLt(algo::hash256 const& a, algo::hash256 const& b)
    {
        for (uint32_t i{ 0u }; i < 32u; ++i)
        {
            if (a.ubytes[i] != b.ubytes[i])
            {
                return a.ubytes[i] < b.ubytes[i];
            }
        }
        return false;
    }

    void decrement256(uint64_t* const b)
    {
        for (int32_t i{ 3 }; i >= 0; --i)
        {
            if (0ull != b[i])
            {
                b[i] -= 1ull;
                return;
            }
            b[i] = ~0ull;
        }
    }
}


struct OctopusSearchAmdTest : public testing::Test
{
    resolver::tests::Properties properties{};

    OctopusSearchAmdTest()
    {
        resolver::tests::initializeOpenCL(properties);
    }
    ~OctopusSearchAmdTest() = default;
};


// The warp-cooperative search kernel must reproduce the CPU oracle octopusHash bit-for-bit.
// A small synthetic full_size keeps the touched dataset items inside a buildable DAG. We
// launch one 32-lane warp (the unit that shares the polynomial coefficients in LDS) and
// target the unique minimum-hash nonce, so exactly one lane wins -- a clean two-sided proof
// (winner at boundary == minHash, miss at minHash - 1) that GPU == CPU for every lane.
TEST_F(OctopusSearchAmdTest, searchMatchesCpu)
{
    algo::octopus::LightCache const light{ algo::octopus::buildLightCache(0ull) };

    uint32_t const numFullPages{ 4096u };
    uint64_t const pageSize{ 4ull * (algo::octopus::MIX_BYTES / 4u) };  // 256
    uint64_t const fullSizeBytes{ static_cast<uint64_t>(numFullPages) * pageSize };

    uint32_t const       dagNodes{ numFullPages * 4u };
    std::vector<uint8_t> dagHost(static_cast<size_t>(dagNodes) * algo::octopus::HASH_BYTES);
    for (uint32_t j{ 0u }; j < dagNodes; ++j)
    {
        algo::hash512 const node{ algo::octopus::calcDatasetItem(light, j) };
        std::memcpy(dagHost.data() + static_cast<size_t>(j) * 64u, node.ubytes, 64u);
    }

    cl::Buffer dagBuf{ properties.clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dagHost.size(), dagHost.data() };

    algo::hash256 header{};
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        header.ubytes[i] = static_cast<uint8_t>(i);
    }
    cl::Buffer headerBuf{ properties.clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32u, header.word64 };
    cl::Buffer resultBuf{ properties.clContext, CL_MEM_READ_WRITE, sizeof(algo::octopus::Result) };

    constexpr uint32_t WARP{ 32u };
    common::KernelGeneratorOpenCL generator{};
    generator.setKernelName("octopus_search");
    generator.addDefine("GROUP_SIZE", WARP);  // one subgroup = one LDS coefficient vector
    ASSERT_TRUE(generator.appendFile("kernel/octopus/octopus_search.cl"));
    ASSERT_TRUE(generator.build(&properties.clDevice, &properties.clContext));

    // start_nonce must be 32-aligned so the 32 lanes share one warp base.
    uint64_t const startNonce{ 16384ull };

    // CPU oracle over the whole warp; the unique minimum hash is the only winner.
    uint64_t      targetNonce{ startNonce };
    algo::hash256 minHash{ algo::octopus::octopusHash(header, startNonce, light, fullSizeBytes) };
    for (uint32_t lane{ 1u }; lane < WARP; ++lane)
    {
        algo::hash256 const h{ algo::octopus::octopusHash(header, startNonce + lane, light, fullSizeBytes) };
        if (true == hashLt(h, minHash))
        {
            minHash = h;
            targetNonce = startNonce + lane;
        }
    }

    auto const runKernel{ [&](cl_ulong4 const boundary) -> algo::octopus::Result
    {
        algo::octopus::Result zero{};
        properties.clQueue.enqueueWriteBuffer(resultBuf, CL_TRUE, 0u, sizeof(zero), &zero);

        for (uint32_t i{ 0u }; i < 16u; ++i)
        {
            generator.clKernel.setArg(i, dagBuf);
        }
        generator.clKernel.setArg(16u, resultBuf);
        generator.clKernel.setArg(17u, headerBuf);
        generator.clKernel.setArg(18u, startNonce);
        generator.clKernel.setArg(19u, numFullPages);
        generator.clKernel.setArg(20u, dagNodes);
        generator.clKernel.setArg(21u, boundary);

        properties.clQueue.enqueueNDRangeKernel(generator.clKernel, cl::NullRange, cl::NDRange(WARP), cl::NDRange(WARP));
        properties.clQueue.finish();

        algo::octopus::Result out{};
        properties.clQueue.enqueueReadBuffer(resultBuf, CL_TRUE, 0u, sizeof(out), &out);
        return out;
    } };

    cl_ulong4 boundary{};
    for (uint32_t w{ 0u }; w < 4u; ++w)
    {
        boundary.s[w] = swapU64(minHash.word64[w]);
    }

    // boundary == minHash => exactly the minimum-hash nonce wins.
    algo::octopus::Result const eq{ runKernel(boundary) };
    ASSERT_TRUE(eq.found) << "GPU did not find the minimum-hash nonce";
    EXPECT_EQ(eq.count, 1u);
    EXPECT_EQ(eq.nonces[0], targetNonce);

    // boundary == minHash - 1 => no lane qualifies (proves GPU min == CPU min exactly).
    decrement256(boundary.s);
    algo::octopus::Result const lt{ runKernel(boundary) };
    EXPECT_FALSE(lt.found) << "no nonce should beat (minHash - 1)";
}
