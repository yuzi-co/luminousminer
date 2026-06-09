#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/xelishashv3/result.hpp>
#include <algo/xelishashv3/types.hpp>
#include <benchmark/workflow.hpp>
#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <common/log/log.hpp>
#include <common/opencl/buffer_wrapper.hpp>


bool benchmark::BenchmarkWorkflow::runAmdXelisHashV3()
{
    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark AMD XelisHashV3";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.amd.isAlgoEnabled("xelishashv3"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[AMD] XELISHASHV3") };
    benchmark::AlgoConfig const& algo{ config.amd.getAlgo("xelishashv3") };

    ////////////////////////////////////////////////////////////////////////////
    // Variant axis: the stage-3 128/64 divmod implementation, injected as a build define and
    // dispatched by the deployed kernel's XV3_DIVMOD_IMPL switch. Unlike the DAG algorithms
    // (one .cl per variant), XelisHash V3's variants differ only in this one hot inner op, so a
    // single kernel file is built N times — the production resolver ships divmod_divlu (=3).
    struct Variant
    {
        char const* name;
        uint32_t    impl;
    };
    static constexpr Variant VARIANTS[]{
        { "divmod_full", 0u },    // baseline: full 128-bit binary long division
        { "divmod_bit64", 1u },   // single 64-bit remainder, bit-serial
        { "divmod_fold64", 2u },  // native high-word fold + 64-iter serial tail
        { "divmod_divlu", 3u },   // native fold + base-2^32 division (no per-bit loop)
    };

    ////////////////////////////////////////////////////////////////////////////
    // XelisHash V3 is memory-hard: one ~531 KiB scratchpad per in-flight nonce, so the grid is
    // intentionally small (config defaults threads=64, blocks=96 -> 6144 nonces ~3.1 GiB) to
    // stay under AMD's 4 GiB per-allocation limit — NOT the 256x1024 of the DAG algorithms.
    common::opencl::Buffer<uint8_t>                   headerCache{ CL_MEM_READ_ONLY };
    common::opencl::Buffer<uint8_t>                   targetCache{ CL_MEM_READ_ONLY };
    common::opencl::Buffer<uint8_t>                   scratchCache{ CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
    common::opencl::Buffer<algo::xelishashv3::Result> resultCache{ CL_MEM_READ_WRITE };

    headerCache.setSize(xelishashv3::INPUT_LEN);
    targetCache.setSize(xelishashv3::HASH_SIZE);
    resultCache.setSize(sizeof(algo::xelishashv3::Result));

    if (   false == headerCache.alloc(propertiesAmd.clContext)
        || false == targetCache.alloc(propertiesAmd.clContext)
        || false == resultCache.alloc(propertiesAmd.clContext))
    {
        logErr() << "XelisHashV3 benchmark: failed to allocate IO buffers";
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // A fixed 112-byte work template plus a zero target: the difficulty check is never met
    // (so no spurious publishes, the result buffer stays clean) and every work-item runs the
    // full three-stage hash — i.e. the mining hot path, with no pool/job-stream noise.
    uint8_t header[xelishashv3::INPUT_LEN];
    for (uint32_t i{ 0u }; i < xelishashv3::INPUT_LEN; ++i)
    {
        header[i] = static_cast<uint8_t>(0x11u * (i + 1u));
    }
    uint8_t const                     target[xelishashv3::HASH_SIZE]{};
    algo::xelishashv3::Result const   cleanResult{};
    headerCache.write(header, sizeof(header), &propertiesAmd.clQueue);
    targetCache.write(const_cast<uint8_t*>(target), sizeof(target), &propertiesAmd.clQueue);
    resultCache.write(const_cast<algo::xelishashv3::Result*>(&cleanResult), sizeof(cleanResult),
                      &propertiesAmd.clQueue);

    ////////////////////////////////////////////////////////////////////////////
    auto benchVariant = [&](std::string const& name, uint32_t const impl,
                            benchmark::KernelParams const& params) -> bool
    {
        ////////////////////////////////////////////////////////////////////////
        uint32_t const groupSize{ params.threads };
        uint32_t const blocksCount{ params.blocks };
        size_t const   globalSize{ static_cast<size_t>(groupSize) * static_cast<size_t>(blocksCount) };

        ////////////////////////////////////////////////////////////////////////
        // The scratchpad scales with the grid; (re)size for this variant's occupancy.
        scratchCache.setSize(globalSize * xelishashv3::MEMSIZE_BYTES);
        if (false == scratchCache.alloc(propertiesAmd.clContext))
        {
            logErr() << "XelisHashV3 benchmark: scratchpad alloc failed (lower threads/blocks)";
            return false;
        }

        ////////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};
        generator.setKernelName("search");
        generator.addDefine("MAX_RESULT", algo::xelishashv3::MAX_RESULT);
        generator.addDefine("XV3_DIVMOD_IMPL", impl);
        // The shared BLAKE3 primitive must precede the algorithm kernel (stage 1 & 4).
        if (   false == generator.appendFile("kernel/crypto/blake3.cl")
            || false == generator.appendFile("kernel/xelishashv3/xelishashv3.cl"))
        {
            return false;
        }
        if (false == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            return false;
        }

        ////////////////////////////////////////////////////////////////////////
        auto& clKernel{ generator.clKernel };
        OPENCL_ER(clKernel.setArg(0u, *headerCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(1u, *targetCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(2u, 0ull));
        OPENCL_ER(clKernel.setArg(3u, *scratchCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(4u, *resultCache.getBuffer()));

        ////////////////////////////////////////////////////////////////////////
        setGrid(groupSize, blocksCount);

        ////////////////////////////////////////////////////////////////////////
        // Warm-up launch (driver JIT first-touch / lazy scratch commit) — not timed.
        OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
            clKernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(groupSize)));
        OPENCL_ER(propertiesAmd.clQueue.finish());

        ////////////////////////////////////////////////////////////////////////
        for (uint32_t i{ 0u }; i < params.loop; ++i)
        {
            startChrono(name);
            OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
                clKernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(groupSize)));
            OPENCL_ER(propertiesAmd.clQueue.finish());
            stopChrono(dashboard);
        }

        return true;
    };

    ////////////////////////////////////////////////////////////////////////////
    for (auto const& variant : VARIANTS)
    {
        if (false == algo.isKernelEnabled(variant.name))
        {
            continue;
        }
        benchmark::KernelParams const params{ algo.resolveKernel(variant.name) };
        if (false == benchVariant(variant.name, variant.impl, params))
        {
            logErr() << "XelisHashV3 variant " << variant.name << " failed";
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    scratchCache.free();
    headerCache.free();
    targetCache.free();
    resultCache.free();

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
