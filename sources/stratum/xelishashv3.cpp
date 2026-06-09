#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/boost_utils.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/xelishashv3.hpp>


namespace
{
    // Parse up to `outLen` bytes from a hex string into out (zero-padded if shorter).
    bool hexToBytes(std::string const& hex, uint8_t* out, size_t const outLen)
    {
        size_t const avail{ hex.size() / 2u };
        size_t const n{ (avail < outLen) ? avail : outLen };
        for (size_t i{ 0u }; i < outLen; ++i)
        {
            out[i] = 0u;
        }
        for (size_t i{ 0u }; i < n; ++i)
        {
            out[i] = static_cast<uint8_t>(std::stoul(hex.substr(i * 2u, 2u), nullptr, 16));
        }
        return true;
    }


    // target = floor((2^256 - 1) / diff), written big-endian (MSB-first) into out[32].
    // This is the Xelis difficulty->target conversion (U256::MAX / diff). Implemented as a
    // bitwise long division so it needs no 128-bit integer (the Windows cross-link has no
    // compiler-rt __udivti3): the remainder always stays below diff, hence within 64 bits.
    void difficultyToTargetBe(uint64_t diff, uint8_t out[32])
    {
        if (0ull == diff)
        {
            diff = 1ull;
        }
        for (int i{ 0 }; i < 32; ++i)
        {
            out[i] = 0u;
        }
        uint64_t rem{ 0ull };
        for (int bit{ 0 }; bit < 256; ++bit)  // numerator is 256 one-bits, MSB first
        {
            uint64_t const high{ rem >> 63 };  // bit shifted out by rem<<1
            rem = (rem << 1) | 1ull;
            if (0ull != high || rem >= diff)
            {
                rem -= diff;  // u64 wrap is exact when high == 1 (true value is 2^64 + rem)
                out[bit >> 3] |= static_cast<uint8_t>(0x80u >> (bit & 7));
            }
        }
    }
}


void stratum::StratumXelisHashV3::onResponse(boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch (miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            // result = [ sessionId, extranonceHex, extranonceLen, pubKeyHex ]
            if (true == root.contains("result") && true == root.at("result").is_array())
            {
                boost::json::array const& result(root.at("result").as_array());
                UNIQUE_LOCK(mtxDispatchJob);
                if (2u <= result.size() && true == result.at(1).is_string())
                {
                    uint8_t bytes[32]{};
                    hexToBytes(result.at(1).as_string().c_str(), bytes, 32u);
                    for (uint32_t i{ 0u }; i < 32u; ++i)
                    {
                        jobInfo.xelisExtraNonce.ubytes[i] = bytes[i];
                    }
                }
                if (4u <= result.size() && true == result.at(3).is_string())
                {
                    uint8_t bytes[32]{};
                    hexToBytes(result.at(3).as_string().c_str(), bytes, 32u);
                    for (uint32_t i{ 0u }; i < 32u; ++i)
                    {
                        jobInfo.xelisPubKey.ubytes[i] = bytes[i];
                    }
                }
            }
            break;
        }
        case stratum::Stratum::ID_MINING_AUTHORIZE:
        {
            if (false == root.contains("error") || true == root.at("error").is_null())
            {
                authenticated = true;
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


void stratum::StratumXelisHashV3::onUnknownMethod(boost::json::object const& root)
{
    std::string const method{ common::boostGetString(root, "method") };

    if ("mining.authorize" == method)
    {
        onResponse(root);
    }
    else if ("set_extranonce" == method || "mining.set_extranonce" == method)
    {
        onMiningSetExtraNonce(root);
    }
    else if ("mining.ping" == method)
    {
        boost::json::object pong;
        pong["id"] = root.contains("id") ? root.at("id") : boost::json::value(nullptr);
        pong["method"] = "mining.pong";
        send(pong);
    }
    else if ("mining.print" == method && true == root.contains("params"))
    {
        boost::json::array const& params(root.at("params").as_array());
        if (2u <= params.size() && true == params.at(1).is_string())
        {
            logInfo() << "Pool: " << params.at(1).as_string().c_str();
        }
    }
}


void stratum::StratumXelisHashV3::onMiningNotify(boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    // params = [ jobId, timestampHex, headerWorkHashHex(32B), algo, clean ]
    boost::json::array const& params(root.at("params").as_array());
    if (3u > params.size())
    {
        logErr() << "mining.notify: malformed params " << root;
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobIDStr.assign(params.at(0).as_string().c_str());
    jobInfo.timestamp = std::stoull(params.at(1).as_string().c_str(), nullptr, 16);

    uint8_t header[32]{};
    hexToBytes(params.at(2).as_string().c_str(), header, 32u);
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        jobInfo.headerHash.ubytes[i] = header[i];
    }

    ////////////////////////////////////////////////////////////////////////////
    // header_work_hash uniquely identifies the job; use it as the non-empty jobID gate
    // (jobIDStr remains the wire value used for submit + staleness).
    algo::copyHash(jobInfo.jobID, jobInfo.headerHash);

    ////////////////////////////////////////////////////////////////////////////
    // Seed a non-zero starting nonce (isValidJob() rejects nonce == 0) derived from the
    // pool extranonce so reconnects/peers explore different ranges. The 8-byte search
    // nonce at MinerWork offset 40 is independent of the extra_nonce field at [48..80].
    if (0ull == jobInfo.startNonce)
    {
        uint64_t seed{ 0ull };
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            seed = (seed << 8) | jobInfo.xelisExtraNonce.ubytes[i];
        }
        jobInfo.startNonce = (0ull != seed) ? seed : 1ull;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Restart the nonce sweep; not memory-hard per-job (no DAG/epoch), so pin epoch
    // to a constant: updateMemory (buffer alloc + kernel build) runs once, subsequent
    // jobs only re-run updateConstants (blob/target re-upload).
    jobInfo.nonce = jobInfo.startNonce;
    jobInfo.epoch = 1;

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumXelisHashV3::onMiningSetDifficulty(boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    double const              diffD{ common::boostJsonGetNumber<double>(params.at(0)) };
    uint64_t const            diff{ (diffD < 1.0) ? 1ull : static_cast<uint64_t>(diffD) };

    uint8_t target[32]{};
    difficultyToTargetBe(diff, target);
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        jobInfo.boundary.ubytes[i] = target[i];
    }

    // isValidJob() requires boundaryU64 != 0; fill it from the target MSBs (guarded).
    uint64_t high{ 0ull };
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        high = (high << 8) | jobInfo.boundary.ubytes[i];
    }
    jobInfo.boundaryU64 = (0ull != high) ? high : 1ull;

    logInfo() << "Difficulty: " << diff;
}


void stratum::StratumXelisHashV3::onMiningSetExtraNonce(boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    if (false == params.empty() && true == params.at(0).is_string())
    {
        UNIQUE_LOCK(mtxDispatchJob);
        uint8_t bytes[32]{};
        hexToBytes(params.at(0).as_string().c_str(), bytes, 32u);
        for (uint32_t i{ 0u }; i < 32u; ++i)
        {
            jobInfo.xelisExtraNonce.ubytes[i] = bytes[i];
        }
        logInfo() << "New extra nonce set";
    }
}


void stratum::StratumXelisHashV3::miningSubscribe()
{
    using namespace std::string_literals;

    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_SUBSCRIBE;
    root["method"] = "mining.subscribe";
    root["params"] = boost::json::array{ "LuminousMiner/1.0.0"s, boost::json::array{ "xel/v3"s } };
    send(root);

    miningAuthorize();
}


void stratum::StratumXelisHashV3::miningAuthorize()
{
    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_AUTHORIZE;
    root["method"] = "mining.authorize";
    root["params"] = boost::json::array{ wallet, workerName, password };
    send(root);
}


void stratum::StratumXelisHashV3::miningSubmit(uint32_t const deviceId, boost::json::array const& params)
{
    UNIQUE_LOCK(mtxSubmit);

    // params from the resolver = [ jobIdStr, nonceHex ]; Xelis submit = [ worker, jobId, nonce ].
    boost::json::object root;
    root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "mining.submit";
    root["params"] = boost::json::array{ workerName, params.at(0), params.at(1) };

    send(root);
}
