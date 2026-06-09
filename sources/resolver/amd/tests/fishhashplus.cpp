// On-GPU coverage for KarlsenHashV2 (FishHashPlus). Mirrors the FishHash AMD test: it builds
// the same 4.83 GB DAG on the device, compiles fishhash_search.cl with FISHHASH_PLUS via the
// real device compiler, and confirms the search kernel reproduces the CPU oracle
// (algo::fishhash::hashPlus) for a planted winning nonce. This is the bit-exact GPU-vs-CPU gate.
//
// The shipped .cl files are loaded at runtime by the kernel generator (relative to the working
// directory), so this test must run from the directory that holds the kernel/ tree.

#include <cstdint>
#include <cstring>

#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/fishhash/fishhash.hpp>
#include <algo/hash.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/fishhashplus.hpp>
#include <resolver/tests/amd.hpp>


namespace
{
    // 80-byte Karlsen header: prePowHash[0..31] = i, timestamp little-endian at [32..40],
    // zero[40..72], nonce little-endian at [72..80] (written by the search kernel).
    void buildBootstrapHeader(uint8_t* const header)
    {
        std::memset(header, 0, algo::fishhash::HEADER_SIZE_PLUS);
        for (uint8_t i{ 0u }; i < 32u; ++i)
        {
            header[i] = i;
        }
        uint64_t const timestamp{ 0x1122334455667788ull };
        for (size_t i{ 0u }; i < 8u; ++i)
        {
            header[32u + i] = static_cast<uint8_t>((timestamp >> (8u * i)) & 0xffu);
        }
    }


    void setRandomnessLE(uint8_t* const header, uint64_t const randomness)
    {
        for (size_t i{ 0u }; i < 8u; ++i)
        {
            header[algo::fishhash::RANDOMNESS_OFFSET_PLUS + i] =
                static_cast<uint8_t>((randomness >> (8u * i)) & 0xffu);
        }
    }
}


struct ResolverFishhashPlusAmdTest : public testing::Test
{
    stratum::StratumJobInfo           jobInfo{};
    resolver::ResolverAmdFishhashPlus resolver{};
    resolver::tests::Properties       properties{};
    common::mocker::MockerStratum     stratum{};

    ResolverFishhashPlusAmdTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);

        if (false == resolver::tests::initializeOpenCL(properties))
        {
            logErr() << "fail init opencl";
        }

        resolver.setDevice(&properties.clDevice);
        resolver.setQueue(&properties.clQueue);
        resolver.setContext(&properties.clContext);
    }

    ~ResolverFishhashPlusAmdTest()
    {
        properties.clDevice = nullptr;
        properties.clContext = nullptr;
        properties.clQueue = nullptr;
    }

    // Plant the bootstrap header and make `randomness` the winning nonce: the boundary is its
    // exact CPU-reference digest (so it passes <=), and the search starts at that nonce.
    void initializeJob(uint64_t const randomness)
    {
        uint8_t header[algo::fishhash::HEADER_SIZE_PLUS]{};
        buildBootstrapHeader(header);
        setRandomnessLE(header, randomness);

        algo::fishhash::Context* ctx{ algo::fishhash::getContext(false) };
        ASSERT_NE(nullptr, ctx);
        uint8_t digest[32]{};
        algo::fishhash::hashPlus(digest, ctx, header, algo::fishhash::HEADER_SIZE_PLUS);
        algo::fishhash::freeContext(ctx);

        std::memcpy(jobInfo.headerBlob.ubytes, header, algo::fishhash::HEADER_SIZE_PLUS);
        std::memcpy(jobInfo.boundary.ubytes, digest, 32u);
        jobInfo.nonce = randomness;
        jobInfo.jobIDStr = "fishhashplus-amd-test";
    }
};


TEST_F(ResolverFishhashPlusAmdTest, findNonce)
{
    initializeJob(45u);
    resolver.updateJobId(jobInfo.jobIDStr); // align jobId so the share is not flagged stale

    // updateMemory builds the DAG on the GPU, integrity-checks it against the CPU reference,
    // and compiles the DAG + search kernels (buildSearch() with the FISHHASH_PLUS define).
    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));

    // One work-item: thread_id 0 maps to start_nonce, so exactly the planted nonce is tested.
    resolver.setBlocks(1u);
    resolver.setThreads(1u);

    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmitObject.empty());

    std::string const nonceStr{ stratum.paramSubmitObject.at("nonce").as_string().c_str() };
    EXPECT_EQ("000000000000002d", nonceStr); // 45
}
