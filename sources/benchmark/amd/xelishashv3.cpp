#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/xelishashv3/result.hpp>
#include <algo/xelishashv3/tests/xelishashv3_test_vectors.hpp>
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
    // Two orthogonal build-define axes drive the variants, both dispatched inside the single
    // deployed kernel (unlike the DAG algorithms' one-.cl-per-variant scheme — XelisHash V3's
    // variants differ only in a couple of hot inner ops, so one kernel file is built N times):
    //   * XV3_DIVMOD_IMPL — the stage-3 128/64 divmod (production ships divlu = 3, +38%);
    //   * XV3_ISQRT_IMPL  — the stage-3 floor-sqrt seed (production ships FP64 = 0; f32 = 1 is a
    //                       non-consensus probe to measure the throughput ceiling of dropping the
    //                       slow consumer-RDNA4 FP64 sqrt before building a bit-exact integer one).
    struct Variant
    {
        char const* name;
        uint32_t    divmod;
        uint32_t    isqrt;
    };
    static constexpr Variant VARIANTS[]{
        { "divmod_full", 0u, 0u },   // absolute baseline: full 128-bit long division, FP64 sqrt
        { "divmod_divlu", 3u, 0u },  // production: native fold + base-2^32 divlu, FP64 sqrt
        { "divmod_remhi", 4u, 0u },  // divlu minus the provably-dead high quotient word (nhi/d)
        { "isqrt_f32", 3u, 1u },     // divlu + f32 sqrt probe (NON-CONSENSUS: digest != gold)
    };

    ////////////////////////////////////////////////////////////////////////////
    // XelisHash V3 is memory-hard: one ~531 KiB scratchpad per in-flight nonce, so the grid is
    // intentionally small (config defaults threads=64, blocks=96 -> 6144 nonces ~3.1 GiB) to
    // stay under AMD's 4 GiB per-allocation limit — NOT the 256x1024 of the DAG algorithms.
    common::opencl::Buffer<uint8_t>                   headerCache{ CL_MEM_READ_ONLY };
    common::opencl::Buffer<uint8_t>                   targetCache{ CL_MEM_READ_ONLY };
    common::opencl::Buffer<uint8_t>                   scratchCache{ CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
    common::opencl::Buffer<algo::xelishashv3::Result> resultCache{ CL_MEM_READ_WRITE };
    common::opencl::Buffer<uint8_t>                   katInCache{ CL_MEM_READ_ONLY };
    common::opencl::Buffer<uint8_t>                   katOutCache{ CL_MEM_READ_WRITE };

    headerCache.setSize(xelishashv3::INPUT_LEN);
    targetCache.setSize(xelishashv3::HASH_SIZE);
    resultCache.setSize(sizeof(algo::xelishashv3::Result));
    katInCache.setSize(xelishashv3::INPUT_LEN);
    katOutCache.setSize(xelishashv3::HASH_SIZE);

    if (   false == headerCache.alloc(propertiesAmd.clContext)
        || false == targetCache.alloc(propertiesAmd.clContext)
        || false == resultCache.alloc(propertiesAmd.clContext)
        || false == katInCache.alloc(propertiesAmd.clContext)
        || false == katOutCache.alloc(propertiesAmd.clContext))
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
    uint8_t const                   target[xelishashv3::HASH_SIZE]{};
    algo::xelishashv3::Result const cleanResult{};
    headerCache.write(header, sizeof(header), &propertiesAmd.clQueue);
    targetCache.write(const_cast<uint8_t*>(target), sizeof(target), &propertiesAmd.clQueue);
    resultCache.write(const_cast<algo::xelishashv3::Result*>(&cleanResult), sizeof(cleanResult),
                      &propertiesAmd.clQueue);

    ////////////////////////////////////////////////////////////////////////////
    // Gold known-answer vector (upstream xelis-project/xelis-hash). A correctness gate runs the
    // xelis_kat kernel of every variant against this before timing, so a variant that silently
    // breaks consensus (the f32 probe is expected to) is flagged rather than reported as a "win".
    katInCache.write(const_cast<uint8_t*>(xelishashv3::kat::VERIFY_INPUT.data()), xelishashv3::INPUT_LEN,
                     &propertiesAmd.clQueue);

    ////////////////////////////////////////////////////////////////////////////
    // Build the deployed kernel (shared BLAKE3 primitive first, for stages 1 & 4) for one variant
    // and one entry point, injecting both define axes. Build time is not part of the measurement,
    // so the KAT gate and the timed run each build their own program (clProgram is not exposed for
    // a second getKernel-by-name on this branch).
    auto buildKernel = [&](char const* const entry, uint32_t const divmod, uint32_t const isqrt,
                           common::KernelGeneratorOpenCL& generator) -> bool
    {
        generator.setKernelName(entry);
        generator.addDefine("MAX_RESULT", algo::xelishashv3::MAX_RESULT);
        generator.addDefine("XV3_DIVMOD_IMPL", divmod);
        generator.addDefine("XV3_ISQRT_IMPL", isqrt);
        return    true == generator.appendFile("kernel/crypto/blake3.cl")
               && true == generator.appendFile("kernel/xelishashv3/xelishashv3.cl")
               && true == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext);
    };

    ////////////////////////////////////////////////////////////////////////////
    // Run one work-item of the xelis_kat entry on the gold input and compare the 32-byte digest
    // to the gold output. Returns true on bit-exact match. Reuses scratch slice 0.
    auto verifyVariant = [&](uint32_t const divmod, uint32_t const isqrt) -> bool
    {
        common::KernelGeneratorOpenCL generator{};
        if (false == buildKernel("xelis_kat", divmod, isqrt, generator))
        {
            return false;
        }

        auto& clKernel{ generator.clKernel };
        OPENCL_ER(clKernel.setArg(0u, *katInCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(1u, *scratchCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(2u, *katOutCache.getBuffer()));

        OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
            clKernel, cl::NullRange, cl::NDRange(1u), cl::NDRange(1u)));
        OPENCL_ER(propertiesAmd.clQueue.finish());

        uint8_t digest[xelishashv3::HASH_SIZE]{};
        OPENCL_ER(propertiesAmd.clQueue.enqueueReadBuffer(
            *katOutCache.getBuffer(), CL_TRUE, 0, xelishashv3::HASH_SIZE, digest));

        for (uint32_t i{ 0u }; i < xelishashv3::HASH_SIZE; ++i)
        {
            if (digest[i] != xelishashv3::kat::VERIFY_EXPECTED[i])
            {
                return false;
            }
        }
        return true;
    };

    ////////////////////////////////////////////////////////////////////////////
    auto benchVariant = [&](std::string const& name, uint32_t const divmod, uint32_t const isqrt,
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
        // Correctness gate first (reuses the freshly (re)sized scratchpad). The f32 probe is
        // expected to fail here — that confirms it is non-consensus, not a regression.
        bool const consensusOk{ verifyVariant(divmod, isqrt) };
        if (true == consensusOk)
        {
            logInfo() << "XelisHashV3 [" << name << "] KAT: OK (digest == gold)";
        }
        else
        {
            logErr() << "XelisHashV3 [" << name << "] KAT: MISMATCH (NON-CONSENSUS) — timing only";
        }

        ////////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};
        if (false == buildKernel("search", divmod, isqrt, generator))
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
        if (false == benchVariant(variant.name, variant.divmod, variant.isqrt, params))
        {
            logErr() << "XelisHashV3 variant " << variant.name << " failed";
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    scratchCache.free();
    headerCache.free();
    targetCache.free();
    resultCache.free();
    katInCache.free();
    katOutCache.free();

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
