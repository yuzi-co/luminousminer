#include <cstring>
#include <iomanip>
#include <sstream>

#include <CL/opencl.hpp>

#include <algo/fishhash/fishhash.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/log/log.hpp>
#include <resolver/amd/fishhash.hpp>


resolver::ResolverAmdFishhash::ResolverAmdFishhash() : resolver::ResolverAmd()
{
    if (algorithm == algo::ALGORITHM::UNKNOWN)
    {
        algorithm = algo::ALGORITHM::FISHHASH;
    }
}


resolver::ResolverAmdFishhash::~ResolverAmdFishhash()
{
    parameters.lightCache.free();
    parameters.headerCache.free();
    parameters.boundaryCache.free();
    parameters.resultCache.free();
    dagChunks.clear();
}


uint32_t resolver::ResolverAmdFishhash::chunkBase(uint32_t const chunkIndex) const
{
    return chunkIndex * chunkItems;
}


uint32_t resolver::ResolverAmdFishhash::chunkCount(uint32_t const chunkIndex) const
{
    uint32_t const base{ chunkBase(chunkIndex) };
    return common::max_limit(chunkItems, dagNumberItem - base); // min(chunkItems, remaining)
}


// Bind the (up to 8) chunk buffers to consecutive kernel args starting at firstArg.
// Unused slots are bound to chunk 0 so the kernel always has valid pointers.
bool resolver::ResolverAmdFishhash::bindDagChunks(cl::Kernel& clKernel, uint32_t const firstArg)
{
    for (uint32_t i{ 0u }; i < MAX_DAG_CHUNKS; ++i)
    {
        cl::Buffer& buf{ i < numChunks ? dagChunks[i] : dagChunks[0] };
        OPENCL_ER(clKernel.setArg(firstArg + i, buf));
    }
    return true;
}


bool resolver::ResolverAmdFishhash::updateMemory([[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    ////////////////////////////////////////////////////////////////////////////
    // FishHash has a fixed seed and no epochs: build the DAG exactly once.
    if (true == dagBuilt)
    {
        return true;
    }

    if (nullptr == clContext) [[unlikely]]
    {
        return false;
    }
    if (nullptr == clQueue[0] || nullptr == clQueue[1]) [[unlikely]]
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    context = algo::fishhash::getContext(false);
    if (nullptr == context)
    {
        resolverErr() << "Cannot build FishHash light cache.";
        return false;
    }

    dagNumberItem = castU32(algo::fishhash::FULL_DATASET_NUM_ITEMS);
    uint32_t const lightCacheNumberItem{ castU32(algo::fishhash::lightCacheNumItems(context)) };

    chunkItems = DAG_CHUNK_ITEMS;
    numChunks = (dagNumberItem + chunkItems - 1u) / chunkItems;
    if (numChunks > MAX_DAG_CHUNKS)
    {
        resolverErr() << "FishHash DAG needs " << numChunks << " chunks but only " << MAX_DAG_CHUNKS
                      << " are supported.";
        return false;
    }

    uint64_t const lightCacheSize{ static_cast<uint64_t>(lightCacheNumberItem) * sizeof(algo::u_hash512) };
    uint64_t const dagCacheSize{ static_cast<uint64_t>(dagNumberItem) * sizeof(algo::u_hash1024) };

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const totalMemoryNeeded{ dagCacheSize + lightCacheSize };
    if (0ull != deviceMemoryAvailable && totalMemoryNeeded >= deviceMemoryAvailable)
    {
        resolverErr() << "Device has not enough memory for FishHash."
                      << " Needed " << totalMemoryNeeded << " (DAG ~4.83 GB), available " << deviceMemoryAvailable;
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    parameters.lightCache.free();
    parameters.headerCache.free();
    parameters.boundaryCache.free();
    parameters.resultCache.free();
    dagChunks.clear();

    parameters.lightCache.setSize(lightCacheSize);

    if (false == parameters.lightCache.alloc(*clContext)
        || false == parameters.headerCache.alloc(clQueue[currentIndexStream], *clContext)
        || false == parameters.boundaryCache.alloc(clQueue[currentIndexStream], *clContext)
        || false == parameters.resultCache.alloc(clQueue[currentIndexStream], *clContext))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Upload the light cache (read by the DAG-gen kernel).
    algo::u_hash512* const lightCacheData{ const_cast<algo::u_hash512*>(
        reinterpret_cast<algo::u_hash512 const*>(algo::fishhash::lightCache(context))) };
    if (false == parameters.lightCache.write(lightCacheData, lightCacheSize, clQueue[currentIndexStream]))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build the DAG-gen program once, then allocate + fill each chunk buffer (< 4 GB).
    if (false == buildDagProgram())
    {
        return false;
    }
    for (uint32_t c{ 0u }; c < numChunks; ++c)
    {
        size_t const bytes{ static_cast<size_t>(chunkCount(c)) * sizeof(algo::u_hash1024) };
        cl_int       err{ CL_SUCCESS };
        cl::Buffer   buffer{ *clContext, CL_MEM_READ_WRITE, bytes, nullptr, &err };
        if (CL_SUCCESS != err)
        {
            resolverErr() << "Cannot allocate FishHash DAG chunk " << c << " (" << bytes << " B), err " << err;
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
    algo::fishhash::freeContext(context);
    context = nullptr;
    dagBuilt = true;

    return true;
}


bool resolver::ResolverAmdFishhash::buildDagProgram()
{
    kernelGenerator.clear();
    kernelGenerator.setKernelName("fishhash_build_dag");

    kernelGenerator.addDefine("GROUP_SIZE", getMaxGroupSize());
    kernelGenerator.addDefine("FISHHASH_DAG_ITEM_PARENTS", castU32(algo::fishhash::FULL_DATASET_ITEM_PARENTS));

    if (false == kernelGenerator.appendFile("kernel/fishhash/fishhash_dag.cl"))
    {
        return false;
    }
    return kernelGenerator.build(clDevice, clContext);
}


bool resolver::ResolverAmdFishhash::enqueueDagChunk(uint32_t const chunkIndex)
{
    uint32_t const maxGroupSize{ getMaxGroupSize() };
    uint32_t const base{ chunkBase(chunkIndex) };
    uint32_t const count{ chunkCount(chunkIndex) };

    auto& clKernel{ kernelGenerator.clKernel };
    OPENCL_ER(clKernel.setArg(0u, dagChunks[chunkIndex]));
    OPENCL_ER(clKernel.setArg(1u, *(parameters.lightCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(2u, base));
    OPENCL_ER(clKernel.setArg(3u, count));
    OPENCL_ER(clKernel.setArg(4u, castU32(algo::fishhash::lightCacheNumItems(context))));

    uint32_t const threadKernel{ (count + maxGroupSize - 1u) / maxGroupSize };
    OPENCL_ER(clQueue[currentIndexStream]->enqueueNDRangeKernel(
        clKernel,
        cl::NullRange,
        cl::NDRange(maxGroupSize, threadKernel, 1),
        cl::NDRange(maxGroupSize, 1, 1)));
    OPENCL_ER(clQueue[currentIndexStream]->finish());

    return true;
}


// Read back a few items from the GPU DAG and compare to the CPU reference. A mismatch
// means the on-GPU DAG is corrupt (e.g. a per-buffer size limit) and every share would
// be rejected, so fail loudly rather than mine garbage. Requires the light context.
bool resolver::ResolverAmdFishhash::checkDag()
{
    uint32_t const checkIndices[]{ 0u, 4095u, dagNumberItem / 2u, dagNumberItem - 1u };
    for (uint32_t const idx : checkIndices)
    {
        uint32_t const   chunk{ idx / chunkItems };
        uint32_t const   off{ idx - chunk * chunkItems };
        algo::u_hash1024 gpuItem{};
        OPENCL_ER(clQueue[currentIndexStream]->enqueueReadBuffer(
            dagChunks[chunk],
            CL_TRUE,
            static_cast<size_t>(off) * sizeof(algo::u_hash1024),
            sizeof(algo::u_hash1024),
            &gpuItem));
        algo::hash1024 const refItem{ algo::fishhash::datasetItem(context, idx) };
        if (0 != std::memcmp(gpuItem.ubytes, refItem.ubytes, sizeof(gpuItem.ubytes)))
        {
            resolverErr() << "FishHash DAG integrity check FAILED at item " << idx
                          << " (chunk " << chunk << "). The on-GPU dataset is corrupt; aborting.";
            return false;
        }
    }
    resolverInfo() << "FishHash DAG integrity check passed.";
    return true;
}


bool resolver::ResolverAmdFishhash::buildSearch()
{
    kernelGenerator.clear();
    kernelGenerator.setKernelName("fishhash_search");

    kernelGenerator.addDefine("GROUP_SIZE", getMaxGroupSize());

    // KarlsenHashV2 shares this kernel; the FISHHASH_PLUS path swaps the header layout,
    // seed construction, index derivation and final hash (see fishhash_search.cl).
    if (algo::ALGORITHM::FISHHASHPLUS == algorithm)
    {
        kernelGenerator.declareDefine("FISHHASH_PLUS");
    }

    if (false == kernelGenerator.appendFile("kernel/fishhash/fishhash_search.cl"))
    {
        return false;
    }
    if (false == kernelGenerator.build(clDevice, clContext))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdFishhash::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    if (false == parameters.headerCache.setBufferDevice(clQueue[currentIndexStream], &jobInfo.headerBlob))
    {
        return false;
    }
    if (false == parameters.boundaryCache.setBufferDevice(clQueue[currentIndexStream], &jobInfo.boundary))
    {
        return false;
    }

    overrideOccupancy(8192u, getMaxGroupSize());

    return true;
}


bool resolver::ResolverAmdFishhash::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    auto& clKernel{ kernelGenerator.clKernel };
    if (false == bindDagChunks(clKernel, 0u)) // args 0..7
    {
        return false;
    }
    OPENCL_ER(clKernel.setArg(8u, *(parameters.resultCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(9u, *(parameters.headerCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(10u, *(parameters.boundaryCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(11u, jobInfo.nonce));
    OPENCL_ER(clKernel.setArg(12u, dagNumberItem));
    OPENCL_ER(clKernel.setArg(13u, chunkItems));

    OPENCL_ER(clQueue[currentIndexStream]->enqueueNDRangeKernel(
        clKernel,
        cl::NullRange,
        cl::NDRange(blocks, threads, 1),
        cl::NDRange(blocks, 1, 1)));
    OPENCL_ER(clQueue[currentIndexStream]->finish());

    if (false == getResultCache(jobInfo.jobIDStr))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdFishhash::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    return executeSync(jobInfo);
}


bool resolver::ResolverAmdFishhash::getResultCache(std::string const& jobId)
{
    algo::fishhash::Result data{};

    if (false == parameters.resultCache.getBufferHost(clQueue[currentIndexStream], &data))
    {
        return false;
    }

    if (true == data.found)
    {
        uint32_t const count{ common::max_limit(data.count, algo::fishhash::MAX_RESULT) };

        resultShare.found = true;
        resultShare.count = count;
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


void resolver::ResolverAmdFishhash::doSubmit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                // Iron Fish randomness = the full 8-byte big-endian nonce as 16-char hex
                // (must start with the xnonce prefix; not stripped).
                std::stringstream nonceHexa;
                nonceHexa << std::setw(16) << std::setfill('0') << std::hex << resultShare.nonces[i];

                boost::json::object params;
                params["jobId"] = resultShare.jobId;
                params["nonce"] = nonceHexa.str();

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }

        resultShare.count = 0u;
        resultShare.found = false;
    }
}


void resolver::ResolverAmdFishhash::submit(stratum::Stratum* const stratum)
{
    doSubmit(stratum);
}


void resolver::ResolverAmdFishhash::submit([[maybe_unused]] stratum::StratumSmartMining* const stratum)
{
    // Smart-mining is not supported for FishHash yet.
}
