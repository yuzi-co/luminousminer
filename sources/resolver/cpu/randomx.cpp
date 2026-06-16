#if defined(CPU_ENABLE)

#include <cstddef>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>

#include <boost/json.hpp>
#include <randomx.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/randomx/result.hpp>
#include <common/cast.hpp>
#include <common/config.hpp>
#include <common/log/log.hpp>
#include <resolver/cpu/cpu_params.hpp>
#include <resolver/cpu/randomx.hpp>


// RandomX is a ~kH/s CPU PoW (Monero rx/0), not the MH/s the GPU/Blake3 batches assume. A
// small per-batch nonce count keeps each executeSync() well under a second, so the device
// loop stays responsive to new jobs and the hashrate displays (see
// DeviceCpu::getMinimumKernelExecuted). Overridden by --threads / --blocks.
namespace
{
    constexpr uint32_t DEFAULT_THREADS{ 64u };
    constexpr uint32_t DEFAULT_BLOCKS{ 16u };

    // Monero rx/0 hashing blob: the 4-byte little-endian nonce sits at this byte offset.
    constexpr size_t NONCE_OFFSET{ 39u };


    // XMRig share rule: the last 8 bytes of the 32-byte RandomX hash, read little-endian,
    // must be strictly below the expanded 64-bit Monero target.
    uint64_t tail64LittleEndian(algo::hash256 const& hash)
    {
        uint64_t value{ 0ull };
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            value |= static_cast<uint64_t>(hash.ubytes[24u + i]) << (8u * i);
        }
        return value;
    }


    // Hex of the 4 nonce bytes in blob order (offset 39..42), i.e. little-endian -- the form
    // a Monero pool expects back in the `nonce` submit field.
    std::string nonceToHexLittleEndian(uint32_t const nonce)
    {
        std::stringstream stream;
        for (uint32_t i{ 0u }; i < 4u; ++i)
        {
            uint32_t const byte{ (nonce >> (8u * i)) & 0xffu };
            stream << std::setw(2) << std::setfill('0') << std::hex << byte;
        }
        return stream.str();
    }
}


resolver::ResolverCpuRandomX::PoolConfig resolver::ResolverCpuRandomX::resolvePoolConfig()
{
    common::Config const& config{ common::Config::instance() };

    // Parse the affinity mask exactly once: the worker-count resolution needs it (popcount
    // when --cpu_threads is unset) and the pool needs it for pinning.
    uint64_t const mask{ config.cpu.affinity.has_value() ? resolver::cpu::parseHexMask(*config.cpu.affinity) : 0ull };
    uint32_t const workers{
        resolver::cpu::resolveWorkerCount(config.cpu.threads, mask, std::thread::hardware_concurrency())
    };
    return PoolConfig{ workers, mask };
}


resolver::ResolverCpuRandomX::ResolverCpuRandomX(PoolConfig const poolConfig)
    : resolver::ResolverCpu(), threadPool{ poolConfig.workerCount, poolConfig.affinityMask },
      workerCount{ (0u < poolConfig.workerCount) ? poolConfig.workerCount : 1u }
{
}


resolver::ResolverCpuRandomX::ResolverCpuRandomX() : ResolverCpuRandomX(resolvePoolConfig())
{
    algorithm = algo::ALGORITHM::RANDOMX;
    overrideOccupancy(DEFAULT_THREADS, DEFAULT_BLOCKS);
}


resolver::ResolverCpuRandomX::~ResolverCpuRandomX()
{
    // Drain any in-flight async batch before freeing the VMs it is hashing on: threadPool is a
    // member destroyed only after this body runs, so its workers can still be live here.
    if (true == inFlight)
    {
        threadPool.wait();
        inFlight = false;
    }
    releaseRandomX();
}


void resolver::ResolverCpuRandomX::releaseRandomX()
{
    for (::randomx_vm* const vm : vms)
    {
        if (nullptr != vm)
        {
            randomx_destroy_vm(vm);
        }
    }
    vms.clear();

    if (nullptr != cache)
    {
        randomx_release_cache(cache);
        cache = nullptr;
    }

    seedReady = false;
    seedKey.clear();
}


bool resolver::ResolverCpuRandomX::ensureSeed(stratum::StratumJobInfo const& jobInfo)
{
    std::string const newKey{ reinterpret_cast<char const*>(jobInfo.seedHash.ubytes), algo::LEN_HASH_256_WORD_8 };

    // Same key, cache + VMs already built: nothing to do (RandomX re-keys only per Monero
    // epoch). The hot loop calls this every batch, so the unchanged path must stay cheap.
    if (true == seedReady && newKey == seedKey)
    {
        return true;
    }

    // Mirror the flag selection of the KAT-validated host hasher so a mined hash is
    // bit-identical to algo::randomx::calculateHash for the same key + input.
    randomx_flags const flags{ randomx_get_flags() };

    ::randomx_cache* const newCache{ randomx_alloc_cache(flags) };
    if (nullptr == newCache)
    {
        logErr() << "RandomX: cannot allocate cache";
        return false;
    }
    randomx_init_cache(newCache, jobInfo.seedHash.ubytes, algo::LEN_HASH_256_WORD_8);

    if (true == vms.empty())
    {
        // First key: one light-mode VM per worker, all sharing newCache (read-only during
        // hashing). A failure mid-way tears everything down so we never mine on a partial set.
        vms.resize(workerCount, nullptr);
        for (uint32_t i{ 0u }; i < workerCount; ++i)
        {
            vms[i] = randomx_create_vm(flags, newCache, nullptr);
            if (nullptr == vms[i])
            {
                logErr() << "RandomX: cannot create VM " << i;
                releaseRandomX();
                randomx_release_cache(newCache);
                return false;
            }
        }
    }
    else
    {
        // Re-key: point existing VMs at the new cache before freeing the old one, so no VM
        // ever references released memory.
        for (::randomx_vm* const vm : vms)
        {
            randomx_vm_set_cache(vm, newCache);
        }
    }

    if (nullptr != cache)
    {
        randomx_release_cache(cache);
    }
    cache = newCache;
    seedKey = newKey;
    seedReady = true;
    return true;
}


bool resolver::ResolverCpuRandomX::updateMemory(stratum::StratumJobInfo const& jobInfo)
{
    // A Monero epoch rotation (new seed_hash) lands here as an updateMemory: rebuild the
    // ~256 MiB cache once, off the per-batch path. Drain any in-flight async batch first --
    // ensureSeed re-keys the worker VMs, and re-keying a VM a worker is still hashing on would
    // race a cache swap mid-hash. The dropped batch's results are discarded (the epoch rotated,
    // so they would be stale anyway).
    if (true == inFlight)
    {
        threadPool.wait();
        inFlight = false;
    }
    return ensureSeed(jobInfo);
}


bool resolver::ResolverCpuRandomX::updateConstants([[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    // Blob and target are read straight from jobInfo at execute time; the device tracks the
    // job id for staleness. Nothing to precompute.
    return true;
}


void resolver::ResolverCpuRandomX::prepareBatch(
    Batch&                         batch,
    stratum::StratumJobInfo const& jobInfo,
    size_t const                   blobLength)
{
    // Copy by value: the worker closure for an async batch runs after executeAsync() returns,
    // so it cannot reference the caller's jobInfo.
    batch.header = jobInfo.headerBlob;
    batch.base = jobInfo.nonce;
    batch.target = jobInfo.boundaryU64;
    batch.blobLength = blobLength;
    batch.result.found = false;
}


void resolver::ResolverCpuRandomX::hashChunk(
    uint64_t const lo,
    uint64_t const hi,
    uint32_t const workerIndex,
    Batch&         batch)
{
    // Each worker owns its VM (by workerIndex) and a private blob copy patched per nonce.
    ::randomx_vm* const vm{ vms[workerIndex] };

    uint8_t blob[algo::LEN_HASH_3072_WORD_8];
    std::memcpy(blob, batch.header.ubytes, batch.blobLength);

    for (uint64_t i{ lo }; i < hi; ++i)
    {
        uint64_t const candidate{ batch.base + i };
        uint32_t const nonce{ castU32(candidate) };

        blob[NONCE_OFFSET + 0u] = static_cast<uint8_t>(nonce & 0xffu);
        blob[NONCE_OFFSET + 1u] = static_cast<uint8_t>((nonce >> 8u) & 0xffu);
        blob[NONCE_OFFSET + 2u] = static_cast<uint8_t>((nonce >> 16u) & 0xffu);
        blob[NONCE_OFFSET + 3u] = static_cast<uint8_t>((nonce >> 24u) & 0xffu);

        algo::hash256 digest{};
        randomx_calculate_hash(vm, blob, batch.blobLength, digest.ubytes);

        // First hit wins (single-share PoW). Hits are rare, so the lock barely contends.
        if (tail64LittleEndian(digest) < batch.target)
        {
            std::scoped_lock<std::mutex> const guard{ batch.hitMutex };
            if (false == batch.result.found)
            {
                batch.result.found = true;
                batch.result.nonce = candidate;
                batch.result.hash = digest;
            }
        }
    }
}


void resolver::ResolverCpuRandomX::harvest(Batch& batch, stratum::StratumJobInfo const& jobInfo)
{
    if (true == batch.result.found)
    {
        resultShare.found = true;
        resultShare.nonce = batch.result.nonce;
        resultShare.jobId = jobInfo.jobIDStr;
        resultShare.nonceHex = nonceToHexLittleEndian(castU32(batch.result.nonce));
        resultShare.resultHex = algo::toHex(batch.result.hash);

        batch.result.found = false;
    }
}


bool resolver::ResolverCpuRandomX::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Drain any async batch left in flight, then guarantee a keyed VM set. ensureSeed must run
    // with no batch in flight because it can re-key the worker VMs.
    if (true == inFlight)
    {
        threadPool.wait();
        inFlight = false;
    }
    if (false == ensureSeed(jobInfo))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const count{ castU64(getBlocks()) * castU64(getThreads()) };

    size_t blobLength{ jobInfo.blobLength };
    if (blobLength > algo::LEN_HASH_3072_WORD_8)
    {
        blobLength = algo::LEN_HASH_3072_WORD_8;
    }
    if (blobLength < NONCE_OFFSET + 4u)
    {
        // Not a usable Monero blob; advance without hashing rather than read out of range.
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    Batch& current{ batch[currentIndex] };
    prepareBatch(current, jobInfo, blobLength);

    // Fan the nonce batch across the pinned worker pool and block until it finishes: the
    // synchronous path is the one tests and debugging rely on.
    threadPool.setCallback(
        [this](uint64_t const lo, uint64_t const hi, uint32_t const workerIndex)
        {
            hashChunk(lo, hi, workerIndex, batch[currentIndex]);
        });
    threadPool.run(count);

    ////////////////////////////////////////////////////////////////////////////
    harvest(current, jobInfo);

    return true;
}


bool resolver::ResolverCpuRandomX::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // CPU mirror of the GPU/Blake3 double-buffer: wait for the batch dispatched on the previous
    // call, harvest it, then launch the next batch into the idle buffer and return immediately.
    // The wait() must precede ensureSeed() -- ensureSeed() can re-key the worker VMs, and
    // re-keying a VM an in-flight worker is hashing on would race a cache swap mid-hash.
    if (true == inFlight)
    {
        threadPool.wait();
        harvest(batch[currentIndex], jobInfo);
        inFlight = false;
    }
    if (false == ensureSeed(jobInfo))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const count{ castU64(getBlocks()) * castU64(getThreads()) };

    size_t blobLength{ jobInfo.blobLength };
    if (blobLength > algo::LEN_HASH_3072_WORD_8)
    {
        blobLength = algo::LEN_HASH_3072_WORD_8;
    }
    if (blobLength < NONCE_OFFSET + 4u)
    {
        // Not a usable Monero blob; nothing launched this call (a previous batch, if any, was
        // already harvested above).
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    currentIndex ^= 1u;
    Batch& next{ batch[currentIndex] };
    prepareBatch(next, jobInfo, blobLength);

    uint32_t const launchIndex{ currentIndex };
    threadPool.setCallback(
        [this, launchIndex](uint64_t const lo, uint64_t const hi, uint32_t const workerIndex)
        {
            hashChunk(lo, hi, workerIndex, batch[launchIndex]);
        });
    threadPool.runAsync(count);
    inFlight = true;

    return true;
}


void resolver::ResolverCpuRandomX::submit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            boost::json::object params{};
            params["jobId"] = resultShare.jobId;
            params["nonce"] = resultShare.nonceHex;
            params["result"] = resultShare.resultHex;

            stratum->miningSubmit(deviceId, params);
        }
    }

    resultShare.found = false;
}


void resolver::ResolverCpuRandomX::submit(stratum::StratumSmartMining* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            boost::json::object params{};
            params["jobId"] = resultShare.jobId;
            params["nonce"] = resultShare.nonceHex;
            params["result"] = resultShare.resultHex;

            stratum->miningSubmit(deviceId, params);
        }
    }

    resultShare.found = false;
}

#endif
