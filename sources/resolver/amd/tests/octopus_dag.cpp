#include <string>
#include <vector>

#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/octopus/light.hpp>
#include <algo/octopus/octopus.hpp>
#include <common/cast.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/tests/amd.hpp>


namespace
{
    std::string toHex(uint8_t const* const p, uint32_t const n)
    {
        static char const* const digits{ "0123456789abcdef" };
        std::string              s{};
        s.reserve(n * 2u);
        for (uint32_t i{ 0u }; i < n; ++i)
        {
            s += digits[p[i] >> 4];
            s += digits[p[i] & 0x0fu];
        }
        return s;
    }
}


struct OctopusDagAmdTest : public testing::Test
{
    resolver::tests::Properties properties{};

    OctopusDagAmdTest()
    {
        resolver::tests::initializeOpenCL(properties);
    }
    ~OctopusDagAmdTest() = default;
};


// GPU DAG-item kernel must be bit-identical to the CPU oracle calcDatasetItem.
TEST_F(OctopusDagAmdTest, dagItemsMatchCpu)
{
    algo::octopus::LightCache const light{ algo::octopus::buildLightCache(0ull) };
    uint32_t const                  numNodes{ castU32(light.numNodes) };
    uint64_t const                  cacheBytes{ light.numNodes * algo::octopus::HASH_BYTES };

    uint32_t const numGenerate{ 2048u };

    cl::Buffer cacheBuf{ properties.clContext,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         static_cast<size_t>(cacheBytes),
                         const_cast<void*>(static_cast<void const*>(light.nodes.data())) };
    cl::Buffer dagBuf{ properties.clContext, CL_MEM_READ_WRITE,
                       static_cast<size_t>(numGenerate) * algo::octopus::HASH_BYTES };

    common::KernelGeneratorOpenCL generator{};
    generator.setKernelName("octopus_build_dag");
    generator.addDefine("GROUP_SIZE", castU32(256));
    generator.addDefine("DAG_LOOP", algo::octopus::DAG_ITEM_PARENTS / 4u / 4u);
    ASSERT_TRUE(generator.appendFile("kernel/octopus/octopus_dag.cl"));
    ASSERT_TRUE(generator.build(&properties.clDevice, &properties.clContext));

    uint32_t const parents{ algo::octopus::DAG_ITEM_PARENTS };
    uint32_t const dagBase{ 0u };
    generator.clKernel.setArg(0u, dagBuf);
    generator.clKernel.setArg(1u, cacheBuf);
    generator.clKernel.setArg(2u, parents);
    generator.clKernel.setArg(3u, dagBase);
    generator.clKernel.setArg(4u, numGenerate);
    generator.clKernel.setArg(5u, numNodes);

    properties.clQueue.enqueueNDRangeKernel(generator.clKernel, cl::NullRange, cl::NDRange(numGenerate), cl::NullRange);
    properties.clQueue.finish();

    std::vector<uint8_t> gpu(static_cast<size_t>(numGenerate) * algo::octopus::HASH_BYTES);
    properties.clQueue.enqueueReadBuffer(dagBuf, CL_TRUE, 0u, gpu.size(), gpu.data());

    for (uint32_t const idx : { 0u, 1u, 1000u, 2047u })
    {
        algo::hash512 const cpu{ algo::octopus::calcDatasetItem(light, idx) };
        EXPECT_EQ(toHex(gpu.data() + static_cast<size_t>(idx) * 64u, 64u), toHex(cpu.ubytes, 64u))
            << "dataset item mismatch at index " << idx;
    }
}
