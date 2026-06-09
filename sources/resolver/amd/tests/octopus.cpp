#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/octopus/light.hpp>
#include <algo/octopus/octopus.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/octopus.hpp>
#include <resolver/tests/amd.hpp>


namespace
{
    uint64_t parseHexU64(std::string const& s)
    {
        return std::stoull(s, nullptr, 16);
    }

    // hash <= boundary as 32-byte big-endian integers (octopus_check_difficulty).
    bool bytesLte(algo::hash256 const& hash, algo::hash256 const& boundary)
    {
        for (uint32_t i{ 0u }; i < 32u; ++i)
        {
            if (hash.ubytes[i] != boundary.ubytes[i])
            {
                return hash.ubytes[i] < boundary.ubytes[i];
            }
        }
        return true;
    }
}


struct ResolverOctopusAmdTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    resolver::ResolverAmdOctopus  resolver{};
    resolver::tests::Properties   properties{};
    common::mocker::MockerStratum stratum{};

    ResolverOctopusAmdTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
        resolver::tests::initializeOpenCL(properties);
        resolver.setDevice(&properties.clDevice);
        resolver.setQueue(&properties.clQueue);
        resolver.setContext(&properties.clContext);
    }

    ~ResolverOctopusAmdTest()
    {
        properties.clDevice = nullptr;
        properties.clContext = nullptr;
        properties.clQueue = nullptr;
    }
};


// End-to-end: build the real epoch-0 DAG (4 GiB -> two 2 GiB chunks; integrity-checked
// across both chunks), then prove the chunked search finds a nonce the CPU oracle says
// is a winner. Validates chunked DAG build, chunk selection, and boundary packing.
TEST_F(ResolverOctopusAmdTest, buildDagAndFindNonce)
{
    // 32-aligned: the warp-cooperative search kernel requires a 32-aligned start nonce.
    uint64_t const targetNonce{ 0x0102030405060700ull };

    // CPU oracle hash for the target nonce (epoch 0, full dataset).
    algo::octopus::LightCache const light{ algo::octopus::buildLightCache(0ull) };
    algo::hash256 header{};
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        header.ubytes[i] = static_cast<uint8_t>(i);
    }
    algo::hash256 const winner{ algo::octopus::octopusLightCompute(header, targetNonce, light) };

    // Boundary == winner hash (big-endian) so the target nonce is exactly on target.
    jobInfo.blockNumber = 0ull;
    jobInfo.headerHash = header;
    jobInfo.boundary = winner;
    jobInfo.nonce = targetNonce;
    jobInfo.extraNonceSize = 0u;
    jobInfo.jobIDStr.assign("octopus-test");
    resolver.updateJobId("octopus-test");  // match resultShare.jobId so the share isn't treated as stale

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));

    // Keep the search range small; the target is the first work-item (nonce == startNonce).
    resolver.setBlocks(64u);
    resolver.setThreads(64u);

    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(static_cast<stratum::Stratum*>(&stratum));

    // A winner must be found and submitted. The boundary equals the target's hash, so
    // the target (or any lower-hashing nonce in range) qualifies; verify the SUBMITTED
    // nonce is a genuine winner the CPU oracle agrees on — full end-to-end validation
    // through the real chunked DAG.
    ASSERT_FALSE(stratum.paramSubmit.empty());
    uint64_t const      foundNonce{ parseHexU64(std::string{ stratum.paramSubmit[1].as_string().c_str() }) };
    algo::hash256 const cpuHash{ algo::octopus::octopusLightCompute(header, foundNonce, light) };
    EXPECT_TRUE(bytesLte(cpuHash, winner)) << "submitted nonce " << std::hex << foundNonce << " is not a winner";
}
