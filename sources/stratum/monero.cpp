#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <common/app.hpp>
#include <common/boost_utils.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <stratum/monero.hpp>


namespace
{
    // Parse up to 8 little-endian bytes from a Monero target hex string.
    uint64_t hexToUintLE(std::string const& hex)
    {
        uint64_t     value{ 0ull };
        size_t const bytes{ (hex.size() / 2u > 8u) ? 8u : hex.size() / 2u };
        for (size_t i{ 0u }; i < bytes; ++i)
        {
            uint64_t const byte{ std::strtoul(hex.substr(2u * i, 2u).c_str(), nullptr, 16) };
            value |= byte << (8u * i);
        }
        return value;
    }


    // Expand a Monero stratum target to the full 64-bit boundary (XMRig semantics):
    // a 4-byte compact target T means boundary = 2^64-1 / (2^32-1 / T); an 8-byte
    // target is the boundary directly.
    uint64_t expandTarget(std::string const& hex)
    {
        uint64_t const raw{ hexToUintLE(hex) };
        if (0ull == raw)
        {
            return 0ull;
        }
        if (hex.size() <= 8u)
        {
            return 0xFFFFFFFFFFFFFFFFull / (0xFFFFFFFFull / raw);
        }
        return raw;
    }


    // Build the 256-bit boundary from the expanded 64-bit target, big-endian in the high
    // 8 bytes, so that algo::toUINT64(boundary) == target (the boundary/boundaryU64 pair
    // stays consistent, matching how the other stratums derive boundaryU64 from boundary).
    algo::hash256 boundaryFromTarget(uint64_t const target)
    {
        algo::hash256 boundary{};
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            boundary.ubytes[i] = static_cast<uint8_t>((target >> (56u - 8u * i)) & 0xffull);
        }
        return boundary;
    }
}


void stratum::StratumMonero::onConnect()
{
    logInfo() << "Stratum connected!";

    common::Config const& config{ common::Config::instance() };
    if (true == config.mining.wallet.empty() || true == config.mining.password.empty())
    {
        logErr() << "Cannot connect wallet[" << config.mining.wallet << "]"
                 << " password[" << config.mining.password << "]";
        return;
    }

    // Always the Monero login handshake, independent of config.mining.stratumType.
    miningSubscribe();
}


void stratum::StratumMonero::miningSubscribe()
{
    common::Config const& config{ common::Config::instance() };

    auto const softwareName{ "luminousminer/" + std::to_string(common::VERSION_MAJOR) + "."
                             + std::to_string(common::VERSION_MINOR) };

    boost::json::object root;
    root["id"] = stratum::Stratum::ID_MINING_SUBSCRIBE;
    root["method"] = "login";
    root["params"] = boost::json::object{ { "login", config.mining.wallet },
                                          { "pass", config.mining.password },
                                          { "agent", softwareName },
                                          { "rigid", config.mining.workerName } };

    send(root);
}


void stratum::StratumMonero::parseJob(boost::json::object const& job)
{
    ////////////////////////////////////////////////////////////////////////////
    // Required fields: reject the job rather than silently degrading to empty blob /
    // zero target (which would mine uselessly against a zero boundary).
    std::string const jobId{ common::boostGetString(job, "job_id") };
    std::string const blob{ common::boostGetString(job, "blob") };
    std::string const seedHash{ common::boostGetString(job, "seed_hash") };
    std::string const target{ common::boostGetString(job, "target") };
    if (true == jobId.empty() || true == blob.empty() || true == seedHash.empty() || true == target.empty())
    {
        logErr() << "Malformed job (missing job_id/blob/seed_hash/target): " << job;
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    UNIQUE_LOCK(mtxDispatchJob);

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.jobIDStr.assign(jobId);
    jobInfo.jobID = algo::toHash256(jobInfo.jobIDStr);

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.blobLength = castU32(blob.size() / 2u);
    jobInfo.headerBlob = algo::toHash<algo::hash3072>(blob, algo::HASH_SHIFT::LEFT);

    ////////////////////////////////////////////////////////////////////////////
    // RandomX re-keys when the epoch seed_hash rotates. Bump the epoch on change so the
    // device/resolver layer gets an explicit re-key signal (the RandomX VM/dataset must
    // be rebuilt for the new key).
    algo::hash256 const newSeed{ algo::toHash256(seedHash) };
    if (0 != std::memcmp(newSeed.ubytes, jobInfo.seedHash.ubytes, sizeof(newSeed.ubytes)))
    {
        jobInfo.seedHash = newSeed;
        ++jobInfo.epoch;
    }

    ////////////////////////////////////////////////////////////////////////////
    jobInfo.boundaryU64 = expandTarget(target);
    jobInfo.boundary = boundaryFromTarget(jobInfo.boundaryU64);

    ////////////////////////////////////////////////////////////////////////////
    if (true == job.contains("height"))
    {
        jobInfo.blockNumber = common::boostJsonGetNumber<uint64_t>(job.at("height"));
    }

    ////////////////////////////////////////////////////////////////////////////
    // RandomX has no pool-assigned extranonce: the worker owns the full 4-byte nonce at
    // blob offset 39. Seed a non-zero start so the job validates; the CPU resolver (added
    // separately) owns the real per-device nonce partitioning.
    if (0ull == jobInfo.nonce)
    {
        jobInfo.startNonce = 1ull;
        jobInfo.nonce = 1ull;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Job: " << jobInfo.jobIDStr << " height " << jobInfo.blockNumber << " target " << std::hex
              << jobInfo.boundaryU64 << std::dec;

    ////////////////////////////////////////////////////////////////////////////
    updateJob();
}


void stratum::StratumMonero::onResponse(boost::json::object const& root)
{
    using namespace std::string_literals;

    auto const miningRequestID{ common::boostJsonGetNumber<uint32_t>(root.at("id")) };

    ////////////////////////////////////////////////////////////////////////////
    // login response: { result: { id, job, status } }
    if (stratum::Stratum::ID_MINING_SUBSCRIBE == miningRequestID)
    {
        bool const hasError{ true == common::boostJsonContains(root, "error") && false == root.at("error").is_null() };
        if (true == hasError || false == common::boostJsonContains(root, "result")
            || true == root.at("result").is_null())
        {
            logErr() << "Login failed: " << root;
            return;
        }

        boost::json::object const& result{ root.at("result").as_object() };

        // rpcId is read by miningSubmit on the device threads under mtxSubmit; take the
        // same lock here (network I/O thread) so the session-id write is not a data race.
        {
            UNIQUE_LOCK(mtxSubmit);
            rpcId = common::boostGetString(result, "id");
        }
        authenticated = true;

        if (true == result.contains("job") && true == result.at("job").is_object())
        {
            parseJob(result.at("job").as_object());
        }
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    // submit response: { result: { status: "OK" } }
    bool const isErrResult{ false == common::boostJsonContains(root, "result") || true == root.at("result").is_null()
                            || false == root.at("result").is_object()
                            || "OK"s != common::boostGetString(root.at("result").as_object(), "status") };
    bool const isErrError{ true == common::boostJsonContains(root, "error") && false == root.at("error").is_null() };
    bool const isValid{ false == isErrResult && false == isErrError };

    if (false == isValid)
    {
        logErr() << root;
    }
    if (nullptr != shareStatus)
    {
        shareStatus(isValid, miningRequestID, uuid);
    }
}


void stratum::StratumMonero::onUnknownMethod(boost::json::object const& root)
{
    std::string const method{ common::boostGetString(root, "method") };

    if ("job" == method && true == root.contains("params") && true == root.at("params").is_object())
    {
        parseJob(root.at("params").as_object());
        return;
    }

    logErr() << "Unknown[" << method << "] " << root;
}


void stratum::StratumMonero::onMiningNotify([[maybe_unused]] boost::json::object const& root)
{
    // Monero delivers jobs via the `job` method (onUnknownMethod), not mining.notify.
}


void stratum::StratumMonero::onMiningSetDifficulty([[maybe_unused]] boost::json::object const& root)
{
    // Monero carries difficulty inside each job's `target`, not a separate method.
}


void stratum::StratumMonero::miningSubmit(uint32_t const deviceId, boost::json::object const& params)
{
    UNIQUE_LOCK(mtxSubmit);

    boost::json::object root;
    root["id"] = (deviceId + 1u) * stratum::Stratum::OVERCOM_NONCE;
    root["method"] = "submit";
    root["params"] = boost::json::object{ { "id", rpcId },
                                          { "job_id", common::boostGetString(params, "jobId") },
                                          { "nonce", common::boostGetString(params, "nonce") },
                                          { "result", common::boostGetString(params, "result") } };

    send(root);
}
