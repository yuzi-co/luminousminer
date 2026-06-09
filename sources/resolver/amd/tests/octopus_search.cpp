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
    // hash256.word64[w] holds bytes 8w..8w+7 little-endian (b[8w] = LSB), so the
    // big-endian numeric word is its byteswap — matching the kernel's swap_u64.
    uint64_t swapU64(uint64_t const x)
    {
        uint64_t r{ 0ull };
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            r = (r << 8) | ((x >> (8u * i)) & 0xffull);
        }
        return r;
    }

    // (B0,B1,B2,B3) is the hash as a big-endian 256-bit number (B0 most significant,
    // B3 least significant). Decrement by one with borrow.
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


// GPU search kernel must reproduce the CPU oracle octopusHash bit-for-bit. We use a
// small synthetic full_size so the touched dataset items fit in a buildable DAG.
TEST_F(OctopusSearchAmdTest, searchMatchesCpu)
{
    algo::octopus::LightCache const light{ algo::octopus::buildLightCache(0ull) };

    uint32_t const numFullPages{ 4096u };
    // page_size = 4 * MIX_WORDS = 256; full_size = numFullPages * 256.
    uint64_t const pageSize{ 4ull * (algo::octopus::MIX_BYTES / 4u) };
    EXPECT_EQ(pageSize, 256ull);
    uint64_t const fullSizeBytes{ static_cast<uint64_t>(numFullPages) * pageSize };

    // Build the small DAG on CPU (verified path) — one 64-byte node per item.
    uint32_t const       dagNodes{ numFullPages * 4u };
    std::vector<uint8_t> dagHost(static_cast<size_t>(dagNodes) * algo::octopus::HASH_BYTES);
    for (uint32_t j{ 0u }; j < dagNodes; ++j)
    {
        algo::hash512 const node{ algo::octopus::calcDatasetItem(light, j) };
        std::memcpy(dagHost.data() + static_cast<size_t>(j) * 64u, node.ubytes, 64u);
    }

    cl::Buffer dagBuf{ properties.clContext,
                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       dagHost.size(),
                       dagHost.data() };

    algo::hash256 header{};
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        header.ubytes[i] = static_cast<uint8_t>(i);
    }
    cl::Buffer headerBuf{ properties.clContext,
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          32u,
                          header.word64 };

    cl::Buffer resultBuf{ properties.clContext, CL_MEM_READ_WRITE, sizeof(algo::octopus::Result) };

    common::KernelGeneratorOpenCL generator{};
    generator.setKernelName("octopus_search");
    generator.addDefine("GROUP_SIZE", castU32(256));
    ASSERT_TRUE(generator.appendFile("kernel/octopus/octopus_search.cl"));
    ASSERT_TRUE(generator.build(&properties.clDevice, &properties.clContext));

    auto const runKernel{ [&](uint64_t const startNonce, cl_ulong4 const boundary) -> algo::octopus::Result
    {
        algo::octopus::Result zero{};
        properties.clQueue.enqueueWriteBuffer(resultBuf, CL_TRUE, 0u, sizeof(zero), &zero);

        // Single-buffer test: bind the same DAG to all 8 chunk slots and make
        // chunk_items exceed the total node count so every node lands in chunk 0.
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            generator.clKernel.setArg(i, dagBuf);
        }
        generator.clKernel.setArg(8u, resultBuf);
        generator.clKernel.setArg(9u, headerBuf);
        generator.clKernel.setArg(10u, startNonce);
        generator.clKernel.setArg(11u, numFullPages);
        generator.clKernel.setArg(12u, dagNodes);
        generator.clKernel.setArg(13u, boundary);

        properties.clQueue.enqueueNDRangeKernel(generator.clKernel, cl::NullRange, cl::NDRange(1u), cl::NullRange);
        properties.clQueue.finish();

        algo::octopus::Result out{};
        properties.clQueue.enqueueReadBuffer(resultBuf, CL_TRUE, 0u, sizeof(out), &out);
        return out;
    } };

    for (uint64_t const nonce : { 0ull, 7ull, 12345ull })
    {
        algo::hash256 const cpu{ algo::octopus::octopusHash(header, nonce, light, fullSizeBytes) };

        cl_ulong4 boundary{};
        for (uint32_t w{ 0u }; w < 4u; ++w)
        {
            boundary.s[w] = swapU64(cpu.word64[w]);
        }

        // boundary == cpuHash  => hash <= boundary => the nonce is a winner.
        algo::octopus::Result const eq{ runKernel(nonce, boundary) };
        ASSERT_TRUE(eq.found) << "GPU did not find winner at nonce " << nonce;
        EXPECT_EQ(eq.nonces[0], nonce);

        // boundary == cpuHash - 1  => GPU hash (== cpuHash) is now ABOVE target.
        // Passing both directions proves the GPU hash equals the CPU hash exactly.
        cl_ulong4 below{ boundary };
        decrement256(below.s);
        algo::octopus::Result const lt{ runKernel(nonce, below) };
        EXPECT_FALSE(lt.found) << "GPU hash should exceed (cpuHash - 1) at nonce " << nonce;
    }
}
