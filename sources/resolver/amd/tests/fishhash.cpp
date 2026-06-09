// On-GPU coverage for FishHash (Iron Fish). The CPU reference test
// (sources/algo/fishhash/tests/fishhash.cpp) validates the algorithm in software; this
// builds the 4.83 GB DAG on the device, compiles the OpenCL kernels with the real device
// compiler (the only thing that exercises fishhash_search.cl + the shared crypto/blake3.cl
// on hardware), and confirms the search kernel finds the known winning nonce.
//
// The shipped .cl files are loaded at runtime by the kernel generator (relative to the
// working directory), so this test must run from the directory that holds the kernel/ tree.

#include <cstdint>
#include <cstring>

#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/fishhash/fishhash.hpp>
#include <algo/hash.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/fishhash.hpp>
#include <resolver/tests/amd.hpp>


namespace
{
    // Bootstrap KAT header from ironfish-rust mining::mine: bytes [0..9] repeated 18x
    // (180 bytes). Randomness lives at [172..180] big-endian; the search kernel writes it.
    void buildBootstrapHeader(uint8_t* const header)
    {
        size_t pos{ 0u };
        for (int i{ 0 }; i < 18; ++i)
        {
            for (uint8_t b{ 0u }; b < 10u; ++b)
            {
                header[pos++] = b;
            }
        }
    }


    void setRandomnessBE(uint8_t* const header, uint64_t const randomness)
    {
        for (size_t i{ 0u }; i < 8u; ++i)
        {
            header[algo::fishhash::RANDOMNESS_OFFSET + i] =
                static_cast<uint8_t>((randomness >> (8u * (7u - i))) & 0xffu);
        }
    }
}


struct ResolverFishhashAmdTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    resolver::ResolverAmdFishhash resolver{};
    resolver::tests::Properties   properties{};
    common::mocker::MockerStratum stratum{};

    ResolverFishhashAmdTest()
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

    ~ResolverFishhashAmdTest()
    {
        properties.clDevice = nullptr;
        properties.clContext = nullptr;
        properties.clQueue = nullptr;
    }

    // Plant the bootstrap header and make `randomness` the winning nonce: the boundary is its
    // exact CPU-reference digest (so it passes <=), and the search starts at that nonce.
    void initializeJob(uint64_t const randomness)
    {
        uint8_t header[algo::fishhash::HEADER_SIZE]{};
        buildBootstrapHeader(header);
        setRandomnessBE(header, randomness);

        algo::fishhash::Context* ctx{ algo::fishhash::getContext(false) };
        ASSERT_NE(nullptr, ctx);
        uint8_t digest[32]{};
        algo::fishhash::hash(digest, ctx, header, algo::fishhash::HEADER_SIZE);
        algo::fishhash::freeContext(ctx);

        std::memcpy(jobInfo.headerBlob.ubytes, header, algo::fishhash::HEADER_SIZE);
        std::memcpy(jobInfo.boundary.ubytes, digest, 32u);
        jobInfo.nonce = randomness;
        jobInfo.jobIDStr = "fishhash-amd-test";
    }
};


TEST_F(ResolverFishhashAmdTest, findNonce)
{
    initializeJob(45u); // bootstrap KAT winner
    resolver.updateJobId(jobInfo.jobIDStr); // align jobId so the share is not flagged stale

    // updateMemory builds the DAG on the GPU, integrity-checks it against the CPU reference,
    // and compiles the DAG + search kernels with the device compiler (buildSearch()).
    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));

    // One work-item: thread_id 0 maps to start_nonce, so exactly the planted nonce is tested.
    resolver.setBlocks(1u);
    resolver.setThreads(1u);

    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmitObject.empty());

    // Iron Fish submits the randomness as the full 8-byte big-endian nonce (16 hex chars).
    std::string const nonceStr{ stratum.paramSubmitObject.at("nonce").as_string().c_str() };
    EXPECT_EQ("000000000000002d", nonceStr); // 45
}
