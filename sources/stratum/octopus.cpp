#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/octopus/octopus.hpp>
#include <common/boost_utils.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/octopus.hpp>


void stratum::StratumOctopus::onResponse(boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch (miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            if (false == root.contains("error") || true == root.at("error").is_null())
            {
                // EthereumStratum/1.0.0 subscribe reply: result = [[notify, id, proto], xn].
                // The second element is the extranonce prefix that fixes the high nonce
                // byte(s). Parse it defensively; always authorize on a non-error reply so a
                // parse hiccup cannot leave the worker unauthenticated (pool rejects all).
                if (true == root.contains("result") && true == root.at("result").is_array())
                {
                    boost::json::array const& result{ root.at("result").as_array() };
                    if (result.size() >= 2u && true == result.at(1).is_string())
                    {
                        setExtraNonce(std::string{ result.at(1).as_string().c_str() });
                    }
                }
                miningAuthorize();
            }
            else
            {
                logErr() << "Subscribe failed : " << root;
            }
            break;
        }
        case stratum::Stratum::ID_MINING_AUTHORIZE:
        {
            if (false == root.contains("error") || true == root.at("error").is_null())
            {
                authenticated = root.at("result").as_bool();
                if (true == authenticated)
                {
                    logInfo() << "Successful login!";
                }
                else
                {
                    logErr() << "Fail to login : " << root;
                }
            }
            else
            {
                logErr() << "Authorize failed : " << root;
            }
            break;
        }
        default:
        {
            onShare(root, miningRequestID);
            break;
        }
    }
}


void stratum::StratumOctopus::setBoundary(std::string const& boundaryHex)
{
    jobInfo.boundary = algo::toHash256(boundaryHex);
    // boundaryU64 is only used by Stratum::isValidJob (must be non-zero). The Octopus
    // resolver compares the full 256-bit boundary, so a high word of 0 (common for low
    // share targets) is fine -- just keep isValidJob happy.
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    if (0ull == jobInfo.boundaryU64)
    {
        jobInfo.boundaryU64 = 1ull;
    }
}


void stratum::StratumOctopus::onMiningNotify(boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    // Some pools carry the extranonce prefix as a top-level "xn" field on the notify.
    if (true == root.contains("xn") && true == root.at("xn").is_string())
    {
        std::string const xn{ root.at("xn").as_string().c_str() };
        if (false == xn.empty())
        {
            setExtraNonce(xn);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    boost::json::array const& params(root.at("params").as_array());

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobIDStr.assign(params.at(0).as_string().c_str());
    jobInfo.blockNumber = std::strtoull(params.at(1).as_string().c_str(), nullptr, 10);
    jobInfo.headerHash = algo::toHash256(params.at(2).as_string().c_str());
    setBoundary(std::string{ params.at(3).as_string().c_str() });

    if (true == root.contains("clean"))
    {
        jobInfo.cleanJob = root.at("clean").as_bool();
    }

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobID = algo::toHash256(jobInfo.jobIDStr);
    jobInfo.epoch = castU32(algo::octopus::getEpoch(jobInfo.blockNumber));

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumOctopus::onMiningSetDifficulty(boost::json::object const& root)
{
    // Octopus pools set the boundary directly via notify/set_target; difficulty, if sent,
    // is converted as a fallback so isValidJob has a target before the first notify.
    boost::json::array const& params(root.at("params").as_array());
    double const              difficulty{ common::boostJsonGetNumber<double>(params.at(0)) };

    jobInfo.boundary = algo::toHash256(difficulty);
    jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
    if (0ull == jobInfo.boundaryU64)
    {
        jobInfo.boundaryU64 = 1ull;
    }

    logInfo() << "Difficulty: " << difficulty;
}


void stratum::StratumOctopus::onMiningSetTarget(boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    setBoundary(std::string{ params.at(0).as_string().c_str() });

    logInfo() << "Target: " << params.at(0).as_string().c_str();
}


void stratum::StratumOctopus::onMiningSetExtraNonce(boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    setExtraNonce(std::string{ params.at(0).as_string().c_str() });
}


void stratum::StratumOctopus::miningSubmit(uint32_t const deviceId, boost::json::array const& params)
{
    using namespace std::string_literals;

    UNIQUE_LOCK(mtxSubmit);

    // params from the resolver = [jobId, nonce(16-hex, full 8-byte, xn prefix included)].
    std::string const nonceHex{ params.at(1).as_string().c_str() };

    boost::json::object root;
    root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "mining.submit";
    root["params"] = boost::json::array{
        wallet + "."s + workerName,        // login.workername
        params.at(0),                      // jobId
        "0x"s + nonceHex,                  // 8-byte nonce
        "0x"s + algo::toHex(jobInfo.headerHash) // header hash
    };

    send(root);
}
