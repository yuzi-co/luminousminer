#pragma once

#if defined(AMD_ENABLE)

#include <vector>

#include <CL/opencl.hpp>

#include <algo/hash.hpp>
#include <algo/octopus/light.hpp>
#include <algo/octopus/result.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/octopus_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdOctopus : public resolver::ResolverAmd
    {
      public:
        // Max chunk buffers the search kernel accepts (16 args). Chunks are 1 GiB to
        // stay well under the AMD OpenCL ~2 GiB (2^31 byte) signed buffer-addressing
        // limit; a single >= 2 GiB buffer corrupts at its top. 16 GiB capacity covers
        // the live 8.45 GiB dataset (9 chunks) with headroom (FishHash-proven pattern).
        static constexpr uint32_t MAX_DAG_CHUNKS{ 16u };
        static constexpr uint32_t DAG_CHUNK_ITEMS{ 1u << 24 };  // 1 GiB / 64 B = 16,777,216 nodes

      public:
        ResolverAmdOctopus();
        ~ResolverAmdOctopus();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::octopus::ResultShare               resultShare{};
        resolver::amd::octopus::KernelParameters parameters{};
        common::KernelGeneratorOpenCL            kernelGenerator{};

        // CPU light cache for the currently built epoch (kept until the DAG is verified).
        algo::octopus::LightCache lightHost{};
        uint64_t                  builtBlockNumber{ 0ull };
        uint64_t                  builtEpoch{ 0ull };
        bool                      dagBuilt{ false };

        uint32_t dagNumberItem{ 0u };  // total DAG nodes (64 B each)
        uint32_t numFullPages{ 0u };   // dagNumberItem / MIX_NODES = full_size / 256
        uint32_t cacheNumberItem{ 0u };
        uint32_t chunkItems{ 0u };
        uint32_t numChunks{ 0u };
        std::vector<cl::Buffer> dagChunks{};

        cl_ulong4 boundaryWords{};  // 256-bit target as big-endian words (s0 most significant)

        // Geometry of DAG chunk c: first global node index and node count.
        uint32_t chunkBase(uint32_t const chunkIndex) const;
        uint32_t chunkCount(uint32_t const chunkIndex) const;

        bool buildDagProgram();
        bool enqueueDagChunk(uint32_t const chunkIndex);
        bool checkDag();
        bool buildSearch();
        bool bindDagChunks(cl::Kernel& clKernel, uint32_t const firstArg);
        bool getResultCache(std::string const& jobId, uint32_t const extraNonceSize);
        void doSubmit(stratum::Stratum* const stratum);
    };
}

#endif
