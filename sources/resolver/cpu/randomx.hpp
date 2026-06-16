#pragma once

#if defined(CPU_ENABLE)


#include <cstddef>
#include <mutex>
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
        // One output buffer of the double-buffer pair. executeAsync() launches the pool into the
        // idle buffer while the device reads/submits the other, so the blob/base/target the
        // workers scan are copied here by value: the worker closure outlives the executeAsync()
        // call that dispatched it. The first hit wins (single-share PoW), guarded by a mutex with
        // negligible contention. Mirrors ResolverCpuBlake3::Batch.
        struct Batch
        {
            algo::hash3072        header{};
            uint64_t              base{ 0ull };
            uint64_t              target{ 0ull };
            size_t                blobLength{ 0u };
            algo::randomx::Result result{};
            std::mutex            hitMutex{};
        };

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

        // Load a buffer from a job (copy blob/base/target by value, reset its result) before
        // dispatching it; hash one [lo, hi) nonce slice into it on worker `workerIndex`'s VM;
        // drain a completed buffer into resultShare. Labeling uses the live jobInfo like the
        // GPU/Blake3 executeAsync(), and submit() drops anything gone stale.
        void prepareBatch(Batch& batch, stratum::StratumJobInfo const& jobInfo, size_t blobLength);
        void hashChunk(uint64_t lo, uint64_t hi, uint32_t workerIndex, Batch& batch);
        void harvest(Batch& batch, stratum::StratumJobInfo const& jobInfo);

        // The pinned worker pool: RandomX is the second CPU resolver to parallelize its scan.
        resolver::CpuThreadPool    threadPool;
        uint32_t                   workerCount{ 1u };
        ::randomx_cache*           cache{ nullptr };
        std::vector<::randomx_vm*> vms{};
        std::string                seedKey{};
        bool                       seedReady{ false };

        // Double-buffer state: executeAsync() launches into batch[currentIndex] and harvests the
        // other; inFlight tracks whether a previous async batch is still pending its wait().
        Batch    batch[2]{};
        uint32_t currentIndex{ 0u };
        bool     inFlight{ false };
    };
}

#endif
