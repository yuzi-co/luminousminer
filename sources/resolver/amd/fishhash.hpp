#pragma once

#if defined(AMD_ENABLE)

#include <vector>

#include <CL/opencl.hpp>

#include <algo/fishhash/fishhash.hpp>
#include <algo/fishhash/result.hpp>
#include <algo/hash.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/fishhash_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdFishhash : public resolver::ResolverAmd
    {
      public:
        // Max chunk buffers the search kernel accepts (8 args). 4.83 GB / 1 GiB = 5.
        static constexpr uint32_t MAX_DAG_CHUNKS{ 8u };
        static constexpr uint32_t DAG_CHUNK_ITEMS{ 8388608u }; // 1 GiB / 128 B
      public:
        ResolverAmdFishhash();
        ~ResolverAmdFishhash();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::fishhash::ResultShare               resultShare{};
        resolver::amd::fishhash::KernelParameters parameters{};
        common::KernelGeneratorOpenCL             kernelGenerator{};

        algo::fishhash::Context* context{ nullptr };
        uint32_t                 dagNumberItem{ 0u };
        uint32_t                 chunkItems{ 0u };
        uint32_t                 numChunks{ 0u };
        std::vector<cl::Buffer>  dagChunks{};
        bool                     dagBuilt{ false };

        // Geometry of DAG chunk c: first global item index and item count.
        uint32_t chunkBase(uint32_t const chunkIndex) const;
        uint32_t chunkCount(uint32_t const chunkIndex) const;

        bool buildDagProgram();
        bool enqueueDagChunk(uint32_t const chunkIndex);
        bool checkDag();
        bool buildSearch();
        bool bindDagChunks(cl::Kernel& clKernel, uint32_t const firstArg);
        bool getResultCache(std::string const& jobId);
        void doSubmit(stratum::Stratum* const stratum);
    };
}

#endif
