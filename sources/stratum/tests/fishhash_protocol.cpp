// Offline protocol coverage for StratumFishhash (Iron Fish stratum v3). No network: the
// handlers parse a JSON message and mutate plain state (or invoke a callback), so we feed
// messages directly and assert the result. Locks in the Iron-Fish-specific quirks:
//   - mining.notify carries the work under "body" (header hex + miningRequestId), epoch fixed;
//   - mining.set_target left-pads a short target so the value is right-aligned (low bytes);
//   - mining.subscribed with a zero xnonce must start the randomness at 1 (isValidJob rejects 0);
//   - mining.submitted drives the share-status callback.

#include <cstdint>
#include <cstring>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <algo/fishhash/fishhash.hpp>
#include <algo/hash.hpp>
#include <stratum/fishhash.hpp>
#include <stratum/stratum.hpp>


namespace
{
    struct ProbeStratumFishhash final : public stratum::StratumFishhash
    {
        bool authed() const
        {
            return authenticated;
        }
    };


    boost::json::object withBody(boost::json::object body)
    {
        boost::json::object root;
        root["body"] = std::move(body);
        return root;
    }
}


struct StratumFishhashProtocolTest : public testing::Test
{
};


TEST_F(StratumFishhashProtocolTest, NotifyDecodesHeaderAndJobId)
{
    ProbeStratumFishhash stratum{};

    boost::json::object body;
    body["miningRequestId"] = 7u;
    body["header"] = "aabbcc"; // decoded natural order into headerBlob[0..]
    stratum.onMiningNotify(withBody(body));

    EXPECT_EQ("7", stratum.jobInfo.jobIDStr);
    EXPECT_EQ(0, stratum.jobInfo.epoch); // fixed: DAG built once
    EXPECT_EQ(0xaau, stratum.jobInfo.headerBlob.ubytes[0]);
    EXPECT_EQ(0xbbu, stratum.jobInfo.headerBlob.ubytes[1]);
    EXPECT_EQ(0xccu, stratum.jobInfo.headerBlob.ubytes[2]);
}


TEST_F(StratumFishhashProtocolTest, SetTargetLeftPadsShortHex)
{
    ProbeStratumFishhash stratum{};

    boost::json::object body;
    body["target"] = "ff"; // short: must become 00..00ff (value in the low byte)
    stratum.onMiningSetTarget(withBody(body));

    EXPECT_EQ(0x00u, stratum.jobInfo.boundary.ubytes[0]);                          // high byte padded
    EXPECT_EQ(0xffu, stratum.jobInfo.boundary.ubytes[algo::LEN_HASH_256 - 1u]);    // value right-aligned
}


TEST_F(StratumFishhashProtocolTest, SubscribedZeroXnonceStartsAtOne)
{
    ProbeStratumFishhash stratum{};

    boost::json::object body;
    body["xn"] = "00"; // zero xnonce -> startNonce would be 0, which isValidJob rejects

    boost::json::object root;
    root["method"] = "mining.subscribed";
    root["body"] = body;
    stratum.onUnknownMethod(root);

    EXPECT_TRUE(stratum.authed());
    EXPECT_EQ(1ull, stratum.jobInfo.startNonce);
    EXPECT_EQ(1ull, stratum.jobInfo.nonce);
}


TEST_F(StratumFishhashProtocolTest, SubmittedDrivesShareStatusCallback)
{
    ProbeStratumFishhash stratum{};

    bool     gotValid{ false };
    uint32_t gotRequestId{ 0u };
    stratum.setCallbackShareStatus(
        [&](bool const isValid, uint32_t const requestID, uint32_t const)
        {
            gotValid = isValid;
            gotRequestId = requestID;
        });

    boost::json::object body;
    body["result"] = true;
    body["id"] = 1000u; // our submit id ((deviceId+1)*OVERCOM_NONCE)

    boost::json::object root;
    root["method"] = "mining.submitted";
    root["body"] = body;
    stratum.onUnknownMethod(root);

    EXPECT_TRUE(gotValid);
    EXPECT_EQ(1000u, gotRequestId);
}
