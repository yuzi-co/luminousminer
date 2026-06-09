#if defined(AMD_ENABLE)

#include <cstring>
#include <vector>

#include <CL/opencl.hpp>

#include <algo/hash.hpp>
#include <algo/octopus/light.hpp>
#include <algo/octopus/octopus.hpp>
#include <algo/octopus/result.hpp>
#include <benchmark/workflow.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/kernel_generator/opencl.hpp>


// Benchmark the octopus_search optimization variants (octopus_lm1..lm6) on AMD. A moderate
// synthetic dataset is built once with the production octopus_dag kernel; each variant is
// then JIT-built (the lmN.cl wrappers flip OCT_COOP_D / OCT_USE_BARRETT / OCT_INTERLEAVE)
// and timed over the configured grid. The variants are bit-exact (see the unit-test sweep);
// this measures throughput. Pure compute comparison — the DAG is small enough to stay cache
// resident, so absolute MH/s is higher than the live memory-bound hashrate.
bool benchmark::BenchmarkWorkflow::runAmdOctopus()
{
    logInfo() << "Running benchmark AMD Octopus";

    if (false == config.amd.isAlgoEnabled("octopus"))
    {
        return true;
    }

    common::Dashboard            dashboard{ createNewDashboard("[AMD] OCTOPUS") };
    benchmark::AlgoConfig const& algo{ config.amd.getAlgo("octopus") };

    ////////////////////////////////////////////////////////////////////////////
    // Light cache + a single-buffer synthetic DAG (~256 MiB) built on the GPU.
    algo::octopus::LightCache const light{ algo::octopus::buildLightCache(0ull) };
    if (true == light.nodes.empty())
    {
        logErr() << "Octopus: cannot build light cache.";
        return false;
    }

    uint32_t const cacheNumberItem{ castU32(light.numNodes) };
    uint32_t const numFullPages{ 1u << 20 };                 // 1,048,576 pages
    uint32_t const dagNumberItem{ numFullPages * 4u };       // 4,194,304 nodes -> 256 MiB
    uint64_t const cacheBytes{ light.numNodes * algo::octopus::HASH_BYTES };
    uint64_t const dagBytes{ static_cast<uint64_t>(dagNumberItem) * algo::octopus::HASH_BYTES };

    cl::Buffer cacheBuf{ propertiesAmd.clContext,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         static_cast<size_t>(cacheBytes),
                         const_cast<void*>(static_cast<void const*>(light.nodes.data())) };
    cl::Buffer dagBuf{ propertiesAmd.clContext, CL_MEM_READ_WRITE, static_cast<size_t>(dagBytes) };

    algo::hash256 header{};
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        header.ubytes[i] = static_cast<uint8_t>(i);
    }
    cl::Buffer headerBuf{ propertiesAmd.clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32u, header.word64 };
    cl::Buffer resultBuf{ propertiesAmd.clContext, CL_MEM_READ_WRITE, sizeof(algo::octopus::Result) };

    ////////////////////////////////////////////////////////////////////////////
    // Build the DAG once with the production generator.
    {
        common::KernelGeneratorOpenCL generator{};
        generator.setKernelName("octopus_build_dag");
        generator.addDefine("GROUP_SIZE", 256u);
        generator.addDefine("DAG_LOOP", algo::octopus::DAG_ITEM_PARENTS / 4u / 4u);
        if (false == generator.appendFile("kernel/octopus/octopus_dag.cl")
            || false == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            logErr() << "Octopus: cannot build octopus_dag.cl.";
            return false;
        }
        auto& clKernel{ generator.clKernel };
        OPENCL_ER(clKernel.setArg(0u, dagBuf));
        OPENCL_ER(clKernel.setArg(1u, cacheBuf));
        OPENCL_ER(clKernel.setArg(2u, algo::octopus::DAG_ITEM_PARENTS));
        OPENCL_ER(clKernel.setArg(3u, 0u));
        OPENCL_ER(clKernel.setArg(4u, dagNumberItem));
        OPENCL_ER(clKernel.setArg(5u, cacheNumberItem));

        uint32_t const threadKernel{ (dagNumberItem + 255u) / 256u };
        OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
            clKernel, cl::NullRange, cl::NDRange(256u, threadKernel, 1), cl::NDRange(256u, 1, 1)));
        OPENCL_ER(propertiesAmd.clQueue.finish());
    }

    ////////////////////////////////////////////////////////////////////////////
    auto const benchOctopus = [&](std::string const& kernelName, uint32_t const loop, uint32_t const groupSize,
                                  uint32_t const blockCount) -> bool
    {
        common::KernelGeneratorOpenCL generator{};
        generator.setKernelName(kernelName);
        generator.addDefine("GROUP_SIZE", groupSize);
        if (false == generator.appendFile("kernel/octopus/" + kernelName + ".cl")
            || false == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            logErr() << "Octopus: cannot build " << kernelName;
            return false;
        }

        auto&     clKernel{ generator.clKernel };
        cl_ulong4 boundary{};  // nothing wins -> pure timing, no atomics
        for (uint32_t i{ 0u }; i < 16u; ++i)
        {
            OPENCL_ER(clKernel.setArg(i, dagBuf));  // single buffer bound to all chunk slots
        }
        OPENCL_ER(clKernel.setArg(16u, resultBuf));
        OPENCL_ER(clKernel.setArg(17u, headerBuf));
        OPENCL_ER(clKernel.setArg(18u, 0ull));            // start_nonce (32-aligned)
        OPENCL_ER(clKernel.setArg(19u, numFullPages));
        OPENCL_ER(clKernel.setArg(20u, dagNumberItem));   // chunk_items >= total -> all chunk 0
        OPENCL_ER(clKernel.setArg(21u, boundary));

        setGrid(groupSize, blockCount);

        for (uint32_t i{ 0u }; i < loop; ++i)
        {
            startChrono(kernelName);
            OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
                clKernel, cl::NullRange, cl::NDRange(groupSize, blockCount, 1), cl::NDRange(groupSize, 1, 1)));
            OPENCL_ER(propertiesAmd.clQueue.finish());
            stopChrono(dashboard);
        }
        return true;
    };

    ////////////////////////////////////////////////////////////////////////////
    auto const runKernel = [&](std::string const& name)
    {
        if (false == algo.isKernelEnabled(name))
        {
            return;
        }
        KernelParams const p{ algo.resolveKernel(name) };
        benchOctopus(name, p.loop, p.threads, p.blocks);
    };

    runKernel("octopus_lm1");
    runKernel("octopus_lm2");
    runKernel("octopus_lm3");
    runKernel("octopus_lm4");
    runKernel("octopus_lm5");
    runKernel("octopus_lm6");

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    return true;
}

#endif
