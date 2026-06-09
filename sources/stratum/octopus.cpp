#include <random>
#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/octopus/octopus.hpp>
#include <common/boost_utils.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/octopus.hpp>


// HeroMiners-style Conflux pools assign NO extranonce: the login IS the subscribe and
// the full 8-byte nonce space belongs to the miner. Pick a random non-zero start nonce
// once per session so this rig searches its own region (and isValidJob's nonce != 0
// check passes). The device advances from here each batch.
void stratum::StratumOctopus::ensureStartNonce()
{
    if (0ull != jobInfo.startNonce)
    {
        return;
    }
    std::random_device                      rd{};
    std::mt19937_64                         gen{ rd() };
    std::uniform_int_distribution<uint64_t> dist{ 1ull, 0x00ffffffffffffffull };
    // 32-align: the warp-cooperative search kernel groups 32 consecutive nonces that
    // share one polynomial-coefficient vector (warp base = nonce / 32).
    uint64_t const start{ (dist(gen) & ~0x1full) | 0x20ull };

    jobInfo.startNonce = start;
    jobInfo.nonce = start;
    jobInfo.gapNonce = 0x1ull;
}


void stratum::StratumOctopus::miningSubscribe()
{
    // HeroMiners CFX: mining.subscribe params = [wallet.worker, password] IS the login
    // (returns {result:true}; no extranonce, no separate authorize). Sending the default
    // EthereumStratum [agent, protocol] is rejected with "Invalid address".
    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_SUBSCRIBE;
    root["method"] = "mining.subscribe";
    root["params"] = boost::json::array{ wallet + "." + workerName, password.empty() ? std::string{ "x" } : password };

    send(root);
}


void stratum::StratumOctopus::onResponse(boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch (miningRequestID)
    {
        case stratum::Stratum::ID_MINING_AUTHORIZE:  // unused for CFX; kept for safety
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            // HeroMiners CFX: subscribe with [wallet.worker, password] IS the login and
            // replies {result:true} (no extranonce, no separate authorize). Mining notifies
            // follow immediately. A non-error reply authenticates the worker.
            bool const ok{ (false == root.contains("error") || true == root.at("error").is_null())
                           && true == root.contains("result") && true == root.at("result").is_bool()
                           && true == root.at("result").as_bool() };
            if (true == ok)
            {
                authenticated = true;
                ensureStartNonce();
                logInfo() << "Successful login!";
            }
            else
            {
                logErr() << "Subscribe/login failed : " << root;
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
    // CFX job ids are small decimals ("0", "1", ...); toHash256("0") is all-zero, which
    // isValidJob treats as empty. jobID (the hash) is only used for logging/validation
    // (shares match on jobIDStr), so fall back to the always-non-empty header hash.
    jobInfo.jobID = algo::toHash256(jobInfo.jobIDStr);
    if (true == algo::isHashEmpty(jobInfo.jobID))
    {
        jobInfo.jobID = jobInfo.headerHash;
    }
    jobInfo.epoch = castU32(algo::octopus::getEpoch(jobInfo.blockNumber));
    ensureStartNonce();  // no extranonce on CFX; keep nonce != 0 for isValidJob

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
