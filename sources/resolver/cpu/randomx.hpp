#pragma once

#if defined(CPU_ENABLE)


#include <string>
#include <vector>

#include <algo/randomx/result.hpp>
#include <resolver/cpu/cpu.hpp>
#include <resolver/cpu/thread_pool.hpp>


// Opaque vendored RandomX handles, so this header does not pull in <randomx.h> (kept
// private to randomx.cpp). These match the struct tags in the vendored randomx.h.
struct randomx_cache;
struct randomx_vm;


namespace resolver
{
    // CPU RandomX (Monero rx/0) resolver. Light-mode: one shared ~256 MiB cache plus one
    // RandomX VM per pinned worker (VMs are not thread-safe, the cache is shared read-only
    // during hashing -- the standard multi-thread layout). The pool fans the per-batch
    // nonce range across workers; each worker patches the 4-byte nonce into its own blob
    // copy and tests one hash at a time against the 64-bit Monero target.
    class ResolverCpuRandomX : public resolver::ResolverCpu
    {
      public:
        ResolverCpuRandomX();
        ~ResolverCpuRandomX();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::randomx::ResultShare resultShare{};

      private:
        // Resolved CPU pool sizing, computed once from Config so the affinity mask is parsed
        // a single time and fed to both the worker-count resolution and the pool itself.
        struct PoolConfig
        {
            uint32_t workerCount{ 1u };
            uint64_t affinityMask{ 0ull };
        };

        static PoolConfig resolvePoolConfig();
        explicit ResolverCpuRandomX(PoolConfig const poolConfig);

        // (Re)build the cache and re-point every worker VM at the new key when `seed`
        // changes (a Monero epoch rotation). Cheap no-op when the seed is unchanged.
        // Returns false on allocation failure.
        bool ensureSeed(stratum::StratumJobInfo const& jobInfo);

        void releaseRandomX();

        // The pinned worker pool: RandomX is the second CPU resolver to parallelize its scan.
        resolver::CpuThreadPool    threadPool;
        uint32_t                   workerCount{ 1u };
        ::randomx_cache*           cache{ nullptr };
        std::vector<::randomx_vm*> vms{};
        std::string                seedKey{};
        bool                       seedReady{ false };
    };
}

#endif
