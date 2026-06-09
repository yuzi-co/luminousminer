#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
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


// Bulk throughput sweep of the octopus_search compile-time switches: OCT_COOP_D (LDS-shared
// vs private/scratch d[]), OCT_USE_BARRETT (Barrett vs hardware % MOD), OCT_INTERLEAVE and
// GROUP_SIZE. Each variant is JIT-built, checked for bit-exact correctness against the CPU
// oracle (the unique minimum-hash nonce in one warp must win), then timed. Prints a MH/s
// table with speedup vs the fully-unoptimized baseline.
//
// NOTE: the synthetic DAG is cache-hot, so the hashimoto memory latency of the real 8.45 GiB
// dataset is not modelled; these numbers compare the *compute* path (multi_eval), not the
// live memory-bound hashrate. Run: unit_test.exe --gtest_filter=OctopusBenchAmdTest.sweep
namespace
{
    uint64_t swapU64(uint64_t const x)
    {
        uint64_t r{ 0ull };
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            r = (r << 8) | ((x >> (8u * i)) & 0xffull);
        }
        return r;
    }

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

    struct Variant
    {
        char const* label;
        uint32_t    coop;
        uint32_t    barrett;
        uint32_t    interleave;
        uint32_t    groupSize;
        uint64_t    nonces;
        uint32_t    repeats;
    };
}


struct OctopusBenchAmdTest : public testing::Test
{
    resolver::tests::Properties properties{};

    OctopusBenchAmdTest()
    {
        resolver::tests::initializeOpenCL(properties);
    }
    ~OctopusBenchAmdTest() = default;
};


TEST_F(OctopusBenchAmdTest, sweep)
{
    constexpr uint32_t WARP{ 32u };
    constexpr uint64_t startNonce{ 16384ull };  // 32-aligned

    // Private-d baselines are ~100x slower, so they use a far smaller grid (the MH/s is
    // normalised by nonces*repeats, so it stays comparable).
    constexpr uint64_t bigGrid{ 1ull << 20 };
    constexpr uint64_t smallGrid{ 1ull << 15 };

    std::vector<Variant> const variants{
        { "baseline: private-d, hw %  ", 0u, 0u, 1u, 64u, smallGrid, 2u },
        { "  + barrett (private-d)     ", 0u, 1u, 1u, 64u, smallGrid, 2u },
        { "  + coop LDS-d (hw %)       ", 1u, 0u, 1u, 128u, bigGrid, 4u },
        { "  + coop + barrett (IL1)    ", 1u, 1u, 1u, 128u, bigGrid, 4u },
        { "  + IL2                     ", 1u, 1u, 2u, 128u, bigGrid, 4u },
        { "  + IL8                     ", 1u, 1u, 8u, 128u, bigGrid, 4u },
        { "  + IL16                    ", 1u, 1u, 16u, 128u, bigGrid, 4u },
        { "  IL8  gs64                 ", 1u, 1u, 8u, 64u, bigGrid, 4u },
        { "  IL16 gs64                 ", 1u, 1u, 16u, 64u, bigGrid, 4u },
        { "  IL8  gs256                ", 1u, 1u, 8u, 256u, bigGrid, 4u },
    };

    ////////////////////////////////////////////////////////////////////////////
    algo::octopus::LightCache const light{ algo::octopus::buildLightCache(0ull) };
    uint32_t const                  numFullPages{ 4096u };
    uint64_t const                  pageSize{ 4ull * (algo::octopus::MIX_BYTES / 4u) };
    uint64_t const                  fullSizeBytes{ static_cast<uint64_t>(numFullPages) * pageSize };
    uint32_t const                  dagNodes{ numFullPages * 4u };

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
    cl_ulong4 winBoundary{};
    for (uint32_t w{ 0u }; w < 4u; ++w)
    {
        winBoundary.s[w] = swapU64(minHash.word64[w]);
    }
    cl_ulong4 const zeroBoundary{};

    auto const setArgs{ [&](cl::Kernel& k, uint64_t const nonce, cl_ulong4 const boundary)
    {
        for (uint32_t i{ 0u }; i < 16u; ++i)
        {
            k.setArg(i, dagBuf);
        }
        k.setArg(16u, resultBuf);
        k.setArg(17u, headerBuf);
        k.setArg(18u, nonce);
        k.setArg(19u, numFullPages);
        k.setArg(20u, dagNodes);
        k.setArg(21u, boundary);
    } };

    std::cout << "\n  octopus_search variant sweep (RX 9070 XT, synthetic DAG -> compute-bound)\n";
    std::cout << "  +-----------------------------+------+-----+----+-----+----------+---------+------+\n";
    std::cout << "  | variant                     | coop | bar | IL |  GS |   MH/s   | speedup | bit  |\n";
    std::cout << "  +-----------------------------+------+-----+----+-----+----------+---------+------+\n";

    double baselineMhs{ 0.0 };
    for (Variant const& v : variants)
    {
        common::KernelGeneratorOpenCL gen{};
        gen.setKernelName("octopus_search");
        gen.addDefine("GROUP_SIZE", v.groupSize);
        gen.addDefine("OCT_COOP_D", v.coop);
        gen.addDefine("OCT_USE_BARRETT", v.barrett);
        gen.addDefine("OCT_INTERLEAVE", v.interleave);
        if (false == gen.appendFile("kernel/octopus/octopus_search.cl")
            || false == gen.build(&properties.clDevice, &properties.clContext))
        {
            std::cout << "  | " << v.label << " |  build FAILED\n";
            continue;
        }

        // Correctness: one warp, the minimum-hash nonce must be the sole winner.
        algo::octopus::Result zero{};
        properties.clQueue.enqueueWriteBuffer(resultBuf, CL_TRUE, 0u, sizeof(zero), &zero);
        setArgs(gen.clKernel, startNonce, winBoundary);
        properties.clQueue.enqueueNDRangeKernel(gen.clKernel, cl::NullRange, cl::NDRange(WARP), cl::NDRange(WARP));
        properties.clQueue.finish();
        algo::octopus::Result corr{};
        properties.clQueue.enqueueReadBuffer(resultBuf, CL_TRUE, 0u, sizeof(corr), &corr);
        bool const bitOk{ corr.found && corr.nonces[0] == targetNonce };

        // Throughput: full grid, boundary 0 (no winners), warmup + timed repeats.
        setArgs(gen.clKernel, startNonce, zeroBoundary);
        properties.clQueue.enqueueNDRangeKernel(
            gen.clKernel, cl::NullRange, cl::NDRange(v.nonces), cl::NDRange(v.groupSize));
        properties.clQueue.finish();

        auto const t0{ std::chrono::high_resolution_clock::now() };
        for (uint32_t r{ 0u }; r < v.repeats; ++r)
        {
            properties.clQueue.enqueueNDRangeKernel(
                gen.clKernel, cl::NullRange, cl::NDRange(v.nonces), cl::NDRange(v.groupSize));
        }
        properties.clQueue.finish();
        auto const t1{ std::chrono::high_resolution_clock::now() };

        double const seconds{ std::chrono::duration<double>(t1 - t0).count() };
        double const mhs{ (static_cast<double>(v.nonces) * v.repeats) / seconds / 1.0e6 };
        if (baselineMhs == 0.0)
        {
            baselineMhs = mhs;
        }

        std::cout << "  | " << v.label << " |  " << v.coop << "   |  " << v.barrett << "  | " << std::setw(2)
                  << v.interleave << " | " << std::setw(3) << v.groupSize << " | " << std::setw(8) << std::fixed
                  << std::setprecision(2) << mhs << " | " << std::setw(6) << std::setprecision(1)
                  << (mhs / baselineMhs) << "x | " << (bitOk ? " OK " : "BAD ") << " |\n";
        EXPECT_TRUE(bitOk) << "variant '" << v.label << "' is not bit-exact";
    }
    std::cout << "  +-----------------------------+------+-----+----+-----+----------+---------+------+\n\n";
}
