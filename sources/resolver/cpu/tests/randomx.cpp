#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/cpu/randomx.hpp>
#include <stratum/job_info.hpp>


struct ResolverRandomXCpuTest : public testing::Test
{
    stratum::StratumJobInfo       jobInfo{};
    common::mocker::MockerStratum stratum{};
    resolver::ResolverCpuRandomX  resolver{};

    void initializeJob()
    {
        // Arbitrary 32-byte RandomX key (epoch seed) and a Monero-sized hashing blob. RandomX
        // hashes arbitrary bytes, so the exact blob content is irrelevant here -- the target is
        // all-ones below, making every hash a winner, which is what exercises the double-buffer
        // harvest/submit sequencing rather than the (KAT-tested) hash itself.
        jobInfo.seedHash = algo::toHash256("0101010101010101010101010101010101010101010101010101010101010101");
        for (uint32_t i{ 0u }; i < 76u; ++i)
        {
            jobInfo.headerBlob.ubytes[i] = static_cast<uint8_t>(i);
        }
        jobInfo.blobLength = 76u;
        jobInfo.jobIDStr.assign("rx-1");
        // Device::updateJob calls resolver->updateJobId before submit; mirror it so submit()
        // does not treat the share as stale.
        resolver.updateJobId("rx-1");
        // Scan exactly one nonce per execute call: blocks*threads == 1.
        resolver.setBlocks(1u);
        resolver.setThreads(1u);
    }
};


// executeAsync double-buffers: a call harvests the batch launched by the *previous* call. With
// an all-ones target every hash is a winner, so the first call only launches (nothing to submit
// yet) and the second drains the first batch into a submit echoing the little-endian nonce.
TEST_F(ResolverRandomXCpuTest, executeAsyncDoubleBuffersThenSubmits)
{
    initializeJob();
    jobInfo.boundaryU64 = 0xffffffffffffffffull;
    jobInfo.nonce = 0x11223344ull;

    ASSERT_TRUE(resolver.updateMemory(jobInfo)); // builds the cache + keyed VMs
    ASSERT_TRUE(resolver.updateConstants(jobInfo));

    // First call only launches a batch: nothing is in flight to harvest, so no share yet.
    ASSERT_TRUE(resolver.executeAsync(jobInfo));
    resolver.submit(&stratum);
    EXPECT_TRUE(stratum.paramSubmitObject.empty());

    // Second call waits on the first batch, harvests it, then submit emits the found nonce
    // (the 4 nonce bytes in little-endian blob order: 0x11223344 -> 44 33 22 11).
    ASSERT_TRUE(resolver.executeAsync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmitObject.empty());
    EXPECT_EQ("44332211", std::string{ stratum.paramSubmitObject.at("nonce").as_string().c_str() });
    EXPECT_EQ("rx-1", std::string{ stratum.paramSubmitObject.at("jobId").as_string().c_str() });
}


// executeSync is the blocking path tests/debug rely on: a single call finds and submits.
TEST_F(ResolverRandomXCpuTest, executeSyncFindsThenSubmits)
{
    initializeJob();
    jobInfo.boundaryU64 = 0xffffffffffffffffull;
    jobInfo.nonce = 0x00000005ull;

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmitObject.empty());
    EXPECT_EQ("05000000", std::string{ stratum.paramSubmitObject.at("nonce").as_string().c_str() });
}
