#include <cstring>
#include <iomanip>
#include <sstream>

#include <CL/opencl.hpp>

#include <algo/octopus/light.hpp>
#include <algo/octopus/octopus.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/log/log.hpp>
#include <resolver/amd/octopus.hpp>


resolver::ResolverAmdOctopus::ResolverAmdOctopus() : resolver::ResolverAmd()
{
    if (algorithm == algo::ALGORITHM::UNKNOWN)
    {
        algorithm = algo::ALGORITHM::OCTOPUS;
    }
}


resolver::ResolverAmdOctopus::~ResolverAmdOctopus()
{
    parameters.lightCache.free();
    parameters.headerCache.free();
    parameters.resultCache.free();
    dagChunks.clear();
}


uint32_t resolver::ResolverAmdOctopus::chunkBase(uint32_t const chunkIndex) const
{
    return chunkIndex * chunkItems;
}


uint32_t resolver::ResolverAmdOctopus::chunkCount(uint32_t const chunkIndex) const
{
    uint32_t const base{ chunkBase(chunkIndex) };
    return common::max_limit(chunkItems, dagNumberItem - base);  // min(chunkItems, remaining)
}


// Bind the (up to 8) chunk buffers to consecutive kernel args starting at firstArg.
// Unused slots are bound to chunk 0 so the kernel always has valid pointers.
bool resolver::ResolverAmdOctopus::bindDagChunks(cl::Kernel& clKernel, uint32_t const firstArg)
{
    for (uint32_t i{ 0u }; i < MAX_DAG_CHUNKS; ++i)
    {
        cl::Buffer& buf{ i < numChunks ? dagChunks[i] : dagChunks[0] };
        OPENCL_ER(clKernel.setArg(firstArg + i, buf));
    }
    return true;
}


bool resolver::ResolverAmdOctopus::updateMemory(stratum::StratumJobInfo const& jobInfo)
{
    if (nullptr == clContext) [[unlikely]]
    {
        return false;
    }
    if (nullptr == clQueue[0] || nullptr == clQueue[1]) [[unlikely]]
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // The DAG is valid for an entire epoch (~3 days); rebuild only when it changes.
    uint64_t const epoch{ algo::octopus::getEpoch(jobInfo.blockNumber) };
    if (true == dagBuilt && epoch == builtEpoch)
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build the CPU light cache for this epoch (also used to verify the GPU DAG).
    lightHost = algo::octopus::buildLightCache(jobInfo.blockNumber);
    if (lightHost.nodes.empty())
    {
        resolverErr() << "Cannot build Octopus light cache.";
        return false;
    }

    uint64_t const dataSize{ algo::octopus::getDataSize(jobInfo.blockNumber) };
    uint64_t const cacheSize{ lightHost.numNodes * algo::octopus::HASH_BYTES };

    dagNumberItem = castU32(dataSize / algo::octopus::HASH_BYTES);
    numFullPages = castU32(dataSize / (4ull * (algo::octopus::MIX_BYTES / 4u)));  // dataSize / 256
    cacheNumberItem = castU32(lightHost.numNodes);

    chunkItems = DAG_CHUNK_ITEMS;
    numChunks = (dagNumberItem + chunkItems - 1u) / chunkItems;
    if (numChunks > MAX_DAG_CHUNKS)
    {
        resolverErr() << "Octopus DAG needs " << numChunks << " chunks but only " << MAX_DAG_CHUNKS
                      << " are supported.";
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const totalMemoryNeeded{ dataSize + cacheSize };
    if (0ull != deviceMemoryAvailable && totalMemoryNeeded >= deviceMemoryAvailable)
    {
        resolverErr() << "Device has not enough memory for Octopus."
                      << " Needed " << totalMemoryNeeded << ", available " << deviceMemoryAvailable;
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    parameters.lightCache.free();
    parameters.headerCache.free();
    parameters.resultCache.free();
    dagChunks.clear();
    dagBuilt = false;

    parameters.lightCache.setSize(cacheSize);

    if (false == parameters.lightCache.alloc(*clContext)
        || false == parameters.headerCache.alloc(clQueue[currentIndexStream], *clContext)
        || false == parameters.resultCache.alloc(clQueue[currentIndexStream], *clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false
        == parameters.lightCache.write(reinterpret_cast<algo::u_hash512*>(lightHost.nodes.data()), cacheSize,
                                       clQueue[currentIndexStream]))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build the DAG-gen program once, then allocate + fill each chunk buffer (< 4 GiB).
    if (false == buildDagProgram())
    {
        return false;
    }
    for (uint32_t c{ 0u }; c < numChunks; ++c)
    {
        size_t const bytes{ static_cast<size_t>(chunkCount(c)) * algo::octopus::HASH_BYTES };
        cl_int       err{ CL_SUCCESS };
        cl::Buffer   buffer{ *clContext, CL_MEM_READ_WRITE, bytes, nullptr, &err };
        if (CL_SUCCESS != err)
        {
            resolverErr() << "Cannot allocate Octopus DAG chunk " << c << " (" << bytes << " B), err " << err;
            return false;
        }
        dagChunks.push_back(buffer);
        if (false == enqueueDagChunk(c))
        {
            return false;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == buildSearch())
    {
        return false;
    }
    if (false == checkDag())
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    parameters.lightCache.free();
    lightHost.nodes.clear();
    lightHost.nodes.shrink_to_fit();

    builtEpoch = epoch;
    builtBlockNumber = jobInfo.blockNumber;
    dagBuilt = true;

    return true;
}


bool resolver::ResolverAmdOctopus::buildDagProgram()
{
    kernelGenerator.clear();
    kernelGenerator.setKernelName("octopus_build_dag");

    kernelGenerator.addDefine("GROUP_SIZE", getMaxGroupSize());
    kernelGenerator.addDefine("DAG_LOOP", algo::octopus::DAG_ITEM_PARENTS / 4u / 4u);

    if (false == kernelGenerator.appendFile("kernel/octopus/octopus_dag.cl"))
    {
        return false;
    }
    return kernelGenerator.build(clDevice, clContext);
}


bool resolver::ResolverAmdOctopus::enqueueDagChunk(uint32_t const chunkIndex)
{
    uint32_t const maxGroupSize{ getMaxGroupSize() };
    uint32_t const base{ chunkBase(chunkIndex) };
    uint32_t const count{ chunkCount(chunkIndex) };

    auto& clKernel{ kernelGenerator.clKernel };
    OPENCL_ER(clKernel.setArg(0u, dagChunks[chunkIndex]));
    OPENCL_ER(clKernel.setArg(1u, *(parameters.lightCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(2u, algo::octopus::DAG_ITEM_PARENTS));
    OPENCL_ER(clKernel.setArg(3u, base));
    OPENCL_ER(clKernel.setArg(4u, count));
    OPENCL_ER(clKernel.setArg(5u, cacheNumberItem));

    uint32_t const threadKernel{ (count + maxGroupSize - 1u) / maxGroupSize };
    OPENCL_ER(clQueue[currentIndexStream]->enqueueNDRangeKernel(
        clKernel,
        cl::NullRange,
        cl::NDRange(maxGroupSize, threadKernel, 1),
        cl::NDRange(maxGroupSize, 1, 1)));
    OPENCL_ER(clQueue[currentIndexStream]->finish());

    return true;
}


// Read back a few nodes from the GPU DAG and compare to the CPU oracle. A mismatch
// means the on-GPU DAG is corrupt (e.g. a per-buffer size limit / 32-bit overflow)
// and every share would read "above target", so fail loudly rather than mine garbage.
bool resolver::ResolverAmdOctopus::checkDag()
{
    uint32_t const checkIndices[]{ 0u, 4095u, dagNumberItem / 2u, dagNumberItem - 1u };
    for (uint32_t const idx : checkIndices)
    {
        uint32_t const   chunk{ idx / chunkItems };
        uint32_t const   off{ idx - chunk * chunkItems };
        algo::u_hash512   gpuItem{};
        OPENCL_ER(clQueue[currentIndexStream]->enqueueReadBuffer(
            dagChunks[chunk],
            CL_TRUE,
            static_cast<size_t>(off) * algo::octopus::HASH_BYTES,
            algo::octopus::HASH_BYTES,
            &gpuItem));
        algo::hash512 const refItem{ algo::octopus::calcDatasetItem(lightHost, idx) };
        if (0 != std::memcmp(gpuItem.ubytes, refItem.ubytes, algo::octopus::HASH_BYTES))
        {
            resolverErr() << "Octopus DAG integrity check FAILED at node " << idx << " (chunk " << chunk
                          << "). The on-GPU dataset is corrupt; aborting.";
            return false;
        }
    }
    resolverInfo() << "Octopus DAG integrity check passed (" << numChunks << " chunk(s), " << dagNumberItem
                   << " nodes).";
    return true;
}


bool resolver::ResolverAmdOctopus::buildSearch()
{
    kernelGenerator.clear();
    kernelGenerator.setKernelName("octopus_search");

    kernelGenerator.addDefine("GROUP_SIZE", getMaxGroupSize());

    if (false == kernelGenerator.appendFile("kernel/octopus/octopus_search.cl"))
    {
        return false;
    }
    return kernelGenerator.build(clDevice, clContext);
}


bool resolver::ResolverAmdOctopus::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    uint32_t const* const header{ jobInfo.headerHash.word32 };
    if (false == parameters.headerCache.setBufferDevice(clQueue[currentIndexStream], header))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Boundary is a 32-byte big-endian target (byte 0 = most significant). Pack it
    // into four 64-bit words (s0 most significant) for the kernel's U256 compare.
    for (uint32_t w{ 0u }; w < 4u; ++w)
    {
        uint64_t v{ 0ull };
        for (uint32_t k{ 0u }; k < 8u; ++k)
        {
            v = (v << 8) | static_cast<uint64_t>(jobInfo.boundary.ubytes[w * 8u + k]);
        }
        boundaryWords.s[w] = v;
    }

    ////////////////////////////////////////////////////////////////////////////
    overrideOccupancy(8192u, getMaxGroupSize());

    return true;
}


bool resolver::ResolverAmdOctopus::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    auto& clKernel{ kernelGenerator.clKernel };
    if (false == bindDagChunks(clKernel, 0u))  // args 0..15
    {
        return false;
    }
    OPENCL_ER(clKernel.setArg(16u, *(parameters.resultCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(17u, *(parameters.headerCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(18u, jobInfo.nonce));
    OPENCL_ER(clKernel.setArg(19u, numFullPages));
    OPENCL_ER(clKernel.setArg(20u, chunkItems));
    OPENCL_ER(clKernel.setArg(21u, boundaryWords));

    OPENCL_ER(clQueue[currentIndexStream]->enqueueNDRangeKernel(
        clKernel,
        cl::NullRange,
        cl::NDRange(blocks, threads, 1),
        cl::NDRange(blocks, 1, 1)));
    OPENCL_ER(clQueue[currentIndexStream]->finish());

    if (false == getResultCache(jobInfo.jobIDStr, jobInfo.extraNonceSize))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdOctopus::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    return executeSync(jobInfo);
}


bool resolver::ResolverAmdOctopus::getResultCache(std::string const& jobId, uint32_t const extraNonceSize)
{
    algo::octopus::Result data{};

    if (false == parameters.resultCache.getBufferHost(clQueue[currentIndexStream], &data))
    {
        return false;
    }

    if (true == data.found)
    {
        uint32_t const count{ common::max_limit(data.count, algo::octopus::MAX_RESULT) };

        resultShare.found = true;
        resultShare.count = count;
        resultShare.extraNonceSize = extraNonceSize;
        resultShare.jobId.assign(jobId);

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


void resolver::ResolverAmdOctopus::doSubmit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                // Octopus pools take the full 8-byte nonce as 16-char hex (xnonce prefix
                // included, not stripped); StratumOctopus adds worker + header hash.
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


void resolver::ResolverAmdOctopus::submit(stratum::Stratum* const stratum)
{
    doSubmit(stratum);
}


void resolver::ResolverAmdOctopus::submit([[maybe_unused]] stratum::StratumSmartMining* const stratum)
{
    // Smart-mining is not supported for Octopus yet.
}
