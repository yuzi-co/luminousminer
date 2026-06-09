#include <iomanip>
#include <sstream>

#include <CL/opencl.hpp>

#include <algo/xelishashv3/types.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/log/log.hpp>
#include <resolver/amd/xelishashv3.hpp>


resolver::ResolverAmdXelisHashV3::ResolverAmdXelisHashV3() : resolver::ResolverAmd()
{
    if (algorithm == algo::ALGORITHM::UNKNOWN)
    {
        algorithm = algo::ALGORITHM::XELISHASHV3;
    }
}


resolver::ResolverAmdXelisHashV3::~ResolverAmdXelisHashV3()
{
    parameters.headerCache.free();
    parameters.targetCache.free();
    parameters.scratchCache.free();
    parameters.resultCache.free();
}


bool resolver::ResolverAmdXelisHashV3::updateMemory(stratum::StratumJobInfo const& jobInfo)
{
    (void)jobInfo;

    if (nullptr == clContext) [[unlikely]]
    {
        return false;
    }
    if (nullptr == clQueue[0] || nullptr == clQueue[1]) [[unlikely]]
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // XelisHash V3 is memory-hard: one ~531 KiB scratchpad per in-flight nonce.
    // Pick a modest occupancy so the single scratch buffer stays under AMD's 4 GiB
    // per-allocation limit (96*64 = 6144 nonces ~ 3.1 GiB). The CLI --blocks/--threads
    // override this; bump it on cards with headroom (cap ~7800 nonces / <4 GiB).
    overrideOccupancy(64u, 96u);

    size_t const inFlight{ static_cast<size_t>(getBlocks()) * static_cast<size_t>(getThreads()) };
    parameters.scratchCache.setSize(inFlight * ::xelishashv3::MEMSIZE_BYTES);

    ////////////////////////////////////////////////////////////////////////////
    parameters.headerCache.free();
    parameters.targetCache.free();
    parameters.scratchCache.free();
    parameters.resultCache.free();

    if (   false == parameters.headerCache.alloc(*clContext)
        || false == parameters.targetCache.alloc(*clContext)
        || false == parameters.scratchCache.alloc(*clContext)
        || false == parameters.resultCache.alloc(clQueue[currentIndexStream], *clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == buildSearch())
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdXelisHashV3::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // Assemble the 112-byte MinerWork template (nonce bytes [40..48] left zero; the
    // search kernel fills them per work-item, big-endian):
    //   headerHash[0:32] | timestamp[32:40 BE] | nonce[40:48] | extraNonce[48:80] | pubKey[80:112].
    uint8_t blob[112]{};
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        blob[i] = jobInfo.headerHash.ubytes[i];
    }
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        blob[32u + i] = static_cast<uint8_t>(jobInfo.timestamp >> (8u * (7u - i))); // big-endian
    }
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        blob[48u + i] = jobInfo.xelisExtraNonce.ubytes[i];
        blob[80u + i] = jobInfo.xelisPubKey.ubytes[i];
    }

    if (false == parameters.headerCache.write(blob, sizeof(blob), clQueue[currentIndexStream]))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Target is the 32-byte big-endian boundary (U256::MAX / difficulty) the stratum
    // already laid out MSB-first in jobInfo.boundary. Copy out of the const job into a
    // mutable buffer for the (non-const) device write.
    uint8_t target[32];
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        target[i] = jobInfo.boundary.ubytes[i];
    }
    if (false == parameters.targetCache.write(target, 32u, clQueue[currentIndexStream]))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    overrideOccupancy(64u, 96u);

    return true;
}


bool resolver::ResolverAmdXelisHashV3::buildSearch()
{
    ////////////////////////////////////////////////////////////////////////////
    kernelGenerator.clear();
    kernelGenerator.setKernelName("search");
    kernelGenerator.addDefine("MAX_RESULT", algo::xelishashv3::MAX_RESULT);
    // Stage-3 128/64 divmod implementation. v3 (native fold + base-2^32 division, no per-bit loop)
    // benchmarked 1.65x over the bit-serial baseline on gfx1201 — stage 3 is divmod-ALU-bound.
    kernelGenerator.addDefine("XV3_DIVMOD_IMPL", static_cast<uint32_t>(3));

    ////////////////////////////////////////////////////////////////////////////
    // The shared BLAKE3 primitive must precede the algorithm kernel (stage 1 & 4).
    if (false == kernelGenerator.appendFile("kernel/crypto/blake3.cl"))
    {
        return false;
    }
    if (false == kernelGenerator.appendFile("kernel/xelishashv3/xelishashv3.cl"))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == kernelGenerator.build(clDevice, clContext))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdXelisHashV3::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    auto& clKernel{ kernelGenerator.clKernel };
    OPENCL_ER(clKernel.setArg(0u, *(parameters.headerCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(1u, *(parameters.targetCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(2u, jobInfo.nonce));
    OPENCL_ER(clKernel.setArg(3u, *(parameters.scratchCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(4u, *(parameters.resultCache.getBuffer())));

    ////////////////////////////////////////////////////////////////////////////
    size_t const globalSize{ static_cast<size_t>(getBlocks()) * static_cast<size_t>(getThreads()) };
    size_t const localSize{ static_cast<size_t>(getThreads()) };
    OPENCL_ER(clQueue[currentIndexStream]
                  ->enqueueNDRangeKernel(clKernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize)));
    OPENCL_ER(clQueue[currentIndexStream]->finish());

    ////////////////////////////////////////////////////////////////////////////
    if (false == getResultCache(jobInfo.jobIDStr))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdXelisHashV3::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    return executeSync(jobInfo);
}


bool resolver::ResolverAmdXelisHashV3::getResultCache(std::string const& _jobId)
{
    algo::xelishashv3::Result data{};

    if (false == parameters.resultCache.getBufferHost(clQueue[currentIndexStream], &data))
    {
        return false;
    }

    if (true == data.found)
    {
        uint32_t const count{ common::max_limit(data.count, algo::xelishashv3::MAX_RESULT) };

        resultShare.found = true;
        resultShare.count = count;
        resultShare.extraNonceSize = 0u;
        resultShare.jobId.assign(_jobId);

        for (uint32_t i{ 0u }; i < count; ++i)
        {
            resultShare.nonces[i] = data.nonces[i];
        }

        if (false == parameters.resultCache.resetBufferHost(clQueue[currentIndexStream]))
        {
            return false;
        }
    }

    return true;
}


void resolver::ResolverAmdXelisHashV3::submit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::setw(16) << std::setfill('0') << std::hex << resultShare.nonces[i];

                boost::json::array params{ resultShare.jobId, nonceHexa.str() };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}


void resolver::ResolverAmdXelisHashV3::submit(stratum::StratumSmartMining* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::setw(16) << std::setfill('0') << std::hex << resultShare.nonces[i];

                boost::json::array params{ resultShare.jobId, nonceHexa.str() };

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}
