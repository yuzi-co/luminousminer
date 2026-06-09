#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include <algo/fishhash/fishhash.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/boost_utils.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/fishhashplus.hpp>


namespace
{
    // Reconstruct the 32-byte pre-pow hash from the 4 little-endian u64 words a Kaspa pool
    // sends in mining.notify (word w -> bytes[w*8 .. w*8+7], little-endian).
    void prePowFromWords(uint64_t const words[4], uint8_t out[32])
    {
        for (uint32_t w{ 0u }; w < 4u; ++w)
        {
            for (uint32_t b{ 0u }; b < 8u; ++b)
            {
                out[w * 8u + b] = static_cast<uint8_t>((words[w] >> (8u * b)) & 0xffu);
            }
        }
    }


    // Convert a stratum difficulty into the 256-bit target the kernel compares against
    // (little-endian: out[0] is the least-significant byte). maxTarget = 2^224 - 1, per
    // kaspa-stratum-bridge; target = floor(maxTarget / diff). NOTE: maxTarget and any
    // per-pool scaling are the single most pool-specific value in the protocol and MUST be
    // confirmed against the chosen Karlsen pool before trusting share acceptance.
    void difficultyToTargetLe(double const diff, uint8_t out[32])
    {
        for (uint32_t i{ 0u }; i < 32u; ++i)
        {
            out[i] = 0u;
        }
        if (diff <= 0.0)
        {
            return;
        }

        // Quantise diff to Q16.16 so fractional difficulties are honoured while the divisor
        // stays <= 2^32 (keeps the long-division remainder in 64 bits).
        double const dScaled{ std::round(diff * 65536.0) };
        if (dScaled < 1.0)
        {
            return;
        }
        uint64_t divisor{ static_cast<uint64_t>(dScaled) };
        if (divisor > 0xffffffffull)
        {
            divisor = 0xffffffffull;
        }

        // numerator = (2^224 - 1) << 16, as 8 little-endian 32-bit limbs.
        uint32_t const maxTarget[8]{ 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                                     0xffffffffu, 0xffffffffu, 0xffffffffu, 0x00000000u };
        uint32_t num[8]{};
        num[0] = maxTarget[0] << 16;
        for (uint32_t i{ 1u }; i < 8u; ++i)
        {
            num[i] = static_cast<uint32_t>((maxTarget[i] << 16) | (maxTarget[i - 1u] >> 16));
        }

        // quotient = numerator / divisor (schoolbook long division, high limb first).
        uint32_t quotient[8]{};
        uint64_t remainder{ 0ull };
        for (int32_t i{ 7 }; i >= 0; --i)
        {
            uint64_t const cur{ (remainder << 32) | static_cast<uint64_t>(num[i]) };
            quotient[i] = static_cast<uint32_t>(cur / divisor);
            remainder = cur % divisor;
        }

        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            for (uint32_t b{ 0u }; b < 4u; ++b)
            {
                out[i * 4u + b] = static_cast<uint8_t>((quotient[i] >> (8u * b)) & 0xffu);
            }
        }
    }
}


void stratum::StratumFishhashPlus::onResponse(boost::json::object const& root)
{
    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    switch (miningRequestID)
    {
        case stratum::Stratum::ID_MINING_SUBSCRIBE:
        {
            // Some bridges return the extranonce as a subscribe result element; adopt it if
            // present, otherwise it arrives via mining.set_extranonce.
            if (true == root.contains("result") && true == root.at("result").is_array())
            {
                boost::json::array const& result(root.at("result").as_array());
                if (false == result.empty() && true == result.at(0).is_string())
                {
                    setExtraNonce(result.at(0).as_string().c_str());
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


void stratum::StratumFishhashPlus::onUnknownMethod(boost::json::object const& root)
{
    std::string const method{ common::boostGetString(root, "method") };

    if ("mining.authorize" == method)
    {
        onResponse(root);
    }
    // Some bridges (NiceHash dialect) deliver the extranonce with the bare method name
    // "set_extranonce" (no "mining." prefix), which the base router misses.
    else if ("set_extranonce" == method)
    {
        onMiningSetExtraNonce(root);
    }
}


void stratum::StratumFishhashPlus::onMiningNotify(boost::json::object const& root)
{
    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    // Two job encodings (both Kaspa gostratum dialect):
    //   NORMAL: [ jobIdStr, [u64_0..u64_3], timestamp ]   (4 LE u64 pre_pow words)
    //   BIG:    [ jobIdStr, "<80 hex>" ]                  pre_pow_hash[32] || timestamp_u64_LE[8]
    boost::json::array const& params(root.at("params").as_array());
    if (2u > params.size())
    {
        logErr() << "mining.notify: malformed params " << root;
        return;
    }

    jobInfo.jobIDStr.assign(params.at(0).as_string().c_str());

    uint8_t  prePow[32]{};
    uint64_t timestamp{ 0ull };

    boost::json::value const& jobField(params.at(1));
    if (true == jobField.is_array())
    {
        boost::json::array const& words(jobField.as_array());
        if (4u != words.size() || 3u > params.size())
        {
            logErr() << "mining.notify: malformed NORMAL job " << root;
            return;
        }
        uint64_t prePowWords[4]{};
        for (uint32_t i{ 0u }; i < 4u; ++i)
        {
            prePowWords[i] = common::boostJsonGetNumber<uint64_t>(words.at(i));
        }
        prePowFromWords(prePowWords, prePow);
        timestamp = common::boostJsonGetNumber<uint64_t>(params.at(2));
    }
    else if (true == jobField.is_string())
    {
        std::string const blob{ jobField.as_string().c_str() };
        if (80u > blob.size()) // 32-byte header + 8-byte timestamp = 80 hex chars
        {
            logErr() << "mining.notify: short BIG job blob " << root;
            return;
        }
        uint8_t bytes[40]{};
        for (uint32_t i{ 0u }; i < 40u; ++i)
        {
            bytes[i] = static_cast<uint8_t>(std::stoul(blob.substr(i * 2u, 2u), nullptr, 16));
        }
        for (uint32_t i{ 0u }; i < 32u; ++i)
        {
            prePow[i] = bytes[i];
        }
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            timestamp |= static_cast<uint64_t>(bytes[32u + i]) << (8u * i);
        }
    }
    else
    {
        logErr() << "mining.notify: unexpected job field type " << root;
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Assemble the 80-byte KarlsenHashV2 header the search kernel consumes:
    //   prePowHash[32] || timestamp_le[8] || zero[32] || nonce[8]   (kernel writes the nonce).
    std::memset(jobInfo.headerBlob.ubytes, 0, sizeof(jobInfo.headerBlob.ubytes));
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        jobInfo.headerBlob.ubytes[i] = prePow[i];
        jobInfo.headerHash.ubytes[i] = prePow[i];
    }
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        jobInfo.headerBlob.ubytes[32u + i] = static_cast<uint8_t>((timestamp >> (8u * i)) & 0xffu);
    }

    ////////////////////////////////////////////////////////////////////////////
    // The pre-pow header uniquely identifies the job; use it as the non-empty jobID gate so
    // isValidJob() accepts tiny/zero jobId strings. jobIDStr stays the wire value for submit.
    algo::copyHash(jobInfo.jobID, jobInfo.headerHash);

    // Restart the per-job nonce sweep from the (extranonce) base.
    jobInfo.nonce = jobInfo.startNonce;

    // KarlsenHashV2 has a fixed seed and no epochs: pin the epoch so the DAG is built once.
    jobInfo.epoch = 0;

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumFishhashPlus::onMiningSetDifficulty(boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    double const              difficulty{ common::boostJsonGetNumber<double>(params.at(0)) };

    uint8_t target[32]{};
    difficultyToTargetLe(difficulty, target);
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        jobInfo.boundary.ubytes[i] = target[i];
    }

    // isValidJob() requires boundaryU64 != 0; fill it from the target's most-significant
    // 64 bits (little-endian buffer => top bytes are the high end).
    uint64_t high{ 0ull };
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        high = (high << 8) | jobInfo.boundary.ubytes[31u - i];
    }
    jobInfo.boundaryU64 = (0ull != high) ? high : 1ull;

    logInfo() << "Difficulty: " << difficulty;
}


void stratum::StratumFishhashPlus::onMiningSetExtraNonce(boost::json::object const& root)
{
    boost::json::array const& params(root.at("params").as_array());
    if (false == params.empty() && true == params.at(0).is_string())
    {
        setExtraNonce(params.at(0).as_string().c_str());
        logInfo() << "Nonce start: " << std::hex << jobInfo.startNonce << std::dec;
    }
}


void stratum::StratumFishhashPlus::miningSubscribe()
{
    // Subscribe (so the bridge keeps the NORMAL job encoding), then authorize.
    stratum::Stratum::miningSubscribe();
    miningAuthorize();
}


void stratum::StratumFishhashPlus::miningSubmit(uint32_t const deviceId, boost::json::object const& params)
{
    using namespace std::string_literals;

    UNIQUE_LOCK(mtxSubmit);

    // params from the FishHash resolver = { jobId, nonce }. Kaspa wire form is the array
    // [ wallet.worker, jobIdStr, nonceHex ].
    std::string const jobId{ common::boostGetString(params, "jobId") };
    std::string const nonce{ common::boostGetString(params, "nonce") };

    boost::json::object root;
    root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "mining.submit";
    root["params"] = boost::json::array{ wallet + "."s + workerName, jobId, nonce };

    send(root);
}
