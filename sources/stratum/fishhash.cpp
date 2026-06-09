#include <cstring>
#include <stdexcept>
#include <string>

#include <algo/fishhash/fishhash.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/app.hpp>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/fishhash.hpp>


namespace
{
    inline uint8_t hexNibble(char c)
    {
        if (c >= '0' && c <= '9')
        {
            return static_cast<uint8_t>(c - '0');
        }
        if (c >= 'a' && c <= 'f')
        {
            return static_cast<uint8_t>(c - 'a' + 10);
        }
        if (c >= 'A' && c <= 'F')
        {
            return static_cast<uint8_t>(c - 'A' + 10);
        }
        return 0u;
    }

    // Decode a hex string into dst (up to maxBytes), in natural byte order.
    size_t hexToBytes(std::string const& hex, uint8_t* dst, size_t maxBytes)
    {
        size_t const bytes{ hex.size() / 2u };
        size_t const n{ bytes < maxBytes ? bytes : maxBytes };
        for (size_t i{ 0u }; i < n; ++i)
        {
            dst[i] = static_cast<uint8_t>((hexNibble(hex[i * 2u]) << 4) | hexNibble(hex[i * 2u + 1u]));
        }
        return n;
    }
}


void stratum::StratumFishhash::onConnect()
{
    logInfo() << "Stratum connected!";

    common::Config const& config{ common::Config::instance() };
    if (true == config.mining.wallet.empty())
    {
        logErr() << "Cannot connect: empty wallet (Iron Fish public address).";
        return;
    }

    // Iron Fish has no password; subscribe directly.
    miningSubscribe();
}


void stratum::StratumFishhash::miningSubscribe()
{
    auto const softwareName{ "luminousminer/" + std::to_string(common::VERSION_MAJOR) + "."
                             + std::to_string(common::VERSION_MINOR) };

    boost::json::object body;
    body["version"] = 3;
    body["name"] = workerName;
    body["publicAddress"] = wallet;
    body["agent"] = softwareName;

    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_SUBSCRIBE;
    root["method"] = "mining.subscribe";
    root["body"] = body;

    send(root);
}


void stratum::StratumFishhash::onResponse(boost::json::object const& root)
{
    if (true == root.contains("error") && false == root.at("error").is_null())
    {
        logErr() << "Stratum error: " << root;
    }
}


void stratum::StratumFishhash::onMiningNotify(boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    boost::json::object const& body{ root.at("body").as_object() };

    uint32_t const    miningRequestId{ common::boostJsonGetNumber<uint32_t>(body.at("miningRequestId")) };
    std::string const headerHex{ common::boostGetString(body, "header") };

    ////////////////////////////////////////////////////////////////////////////
    std::memset(jobInfo.headerBlob.ubytes, 0, sizeof(jobInfo.headerBlob.ubytes));
    hexToBytes(headerHex, jobInfo.headerBlob.ubytes, algo::fishhash::HEADER_SIZE);

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobIDStr.assign(std::to_string(miningRequestId));
    jobInfo.jobID = algo::toHash256(jobInfo.jobIDStr);

    // FishHash has no epochs: keep the epoch fixed so the DAG is built only once.
    jobInfo.epoch = 0;

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumFishhash::onMiningSetTarget(boost::json::object const& root)
{
    boost::json::object const& body{ root.at("body").as_object() };
    std::string                targetHex{ common::boostGetString(body, "target") };

    // The target is a 32-byte (64-hex) big-endian value. If a pool omits leading zeros,
    // left-pad so the value lands in the high bytes (right-aligned), not the low bytes.
    constexpr size_t fullHex{ algo::LEN_HASH_256 * 2u };
    if (targetHex.size() < fullHex)
    {
        targetHex.insert(0u, fullHex - targetHex.size(), '0');
    }

    std::memset(jobInfo.boundary.ubytes, 0, sizeof(jobInfo.boundary.ubytes));
    hexToBytes(targetHex, jobInfo.boundary.ubytes, algo::LEN_HASH_256);

    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);

    logInfo() << "Target: " << targetHex;
}


void stratum::StratumFishhash::onMiningSetDifficulty([[maybe_unused]] boost::json::object const& root)
{
    // Iron Fish uses mining.set_target; difficulty messages are not used.
}


void stratum::StratumFishhash::onUnknownMethod(boost::json::object const& root)
{
    std::string const method{ common::boostGetString(root, "method") };

    if ("mining.subscribed" == method)
    {
        boost::json::object const& body{ root.at("body").as_object() };
        std::string const          xn{ common::boostGetString(body, "xn") };
        setExtraNonce(xn);

        // The xnonce only fixes the leading byte(s) of the 8-byte randomness; pools
        // such as unmineable send xn="00", giving startNonce==0. isValidJob() rejects
        // a zero nonce, so start the iterated range at 1 (the leading xn byte is
        // unchanged, so the submitted randomness still starts with xn).
        if (0ull == jobInfo.startNonce)
        {
            jobInfo.startNonce = 1ull;
            jobInfo.nonce = 1ull;
        }

        authenticated = true;
        logInfo() << "Subscribed. xnonce=" << xn;
    }
    else if ("mining.wait_for_work" == method)
    {
        logInfo() << "Pool: wait for work.";
    }
    else if ("mining.submitted" == method)
    {
        boost::json::object const& body{ root.at("body").as_object() };
        bool const                 result{ body.contains("result") && body.at("result").as_bool() };
        // The request id used for device attribution is our submit id ((deviceId+1)*1000),
        // echoed in body.id. The outer "id" is the pool's own message sequence.
        uint32_t const requestID{ body.contains("id") ? common::boostJsonGetNumber<uint32_t>(body.at("id"))
                                                       : common::boostJsonGetNumber<uint32_t>(root.at("id")) };
        if (nullptr != shareStatus)
        {
            shareStatus(result, requestID, uuid);
        }
        if (false == result && true == body.contains("message"))
        {
            logWarn() << "Share rejected: " << common::boostGetString(body, "message");
        }
    }
    else if ("mining.disconnect" == method)
    {
        logWarn() << "Pool disconnect: " << root;
    }
}


void stratum::StratumFishhash::miningSubmit(
    [[maybe_unused]] uint32_t const             deviceId,
    [[maybe_unused]] boost::json::object const& params)
{
    UNIQUE_LOCK(mtxSubmit);

    std::string const jobId{ common::boostGetString(params, "jobId") };
    uint32_t          miningRequestId{ 0u };
    try
    {
        miningRequestId = static_cast<uint32_t>(std::stoul(jobId));
    }
    catch (std::exception const& e)
    {
        logErr() << "FishHash submit: invalid jobId '" << jobId << "' (" << e.what() << "); dropping share.";
        return;
    }

    boost::json::object body;
    body["miningRequestId"] = miningRequestId;
    body["randomness"] = common::boostGetString(params, "nonce");

    boost::json::object root;
    root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "mining.submit";
    root["body"] = body;

    send(root);
}
