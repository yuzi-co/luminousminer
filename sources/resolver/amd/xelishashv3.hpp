#pragma once

#if defined(AMD_ENABLE)

#include <algo/xelishashv3/result.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/xelishashv3_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdXelisHashV3 : public resolver::ResolverAmd
    {
      public:
        ResolverAmdXelisHashV3();
        ~ResolverAmdXelisHashV3();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::xelishashv3::ResultShare               resultShare{};
        resolver::amd::xelishashv3::KernelParameters parameters{};
        common::KernelGeneratorOpenCL                kernelGenerator{};

        bool buildSearch();
        bool getResultCache(std::string const& _jobId);
    };
}

#endif
