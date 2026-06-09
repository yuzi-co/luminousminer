#pragma once

#include <cstdint>
#if !defined(__LIB_CUDA)
#include <string>
#endif


namespace algo
{
    namespace xelishashv3
    {
        constexpr uint32_t MAX_RESULT{ 4u };

        // GPU-side result buffer. Layout must match the `Result` struct in
        // sources/algo/xelishashv3/opencl/xelishashv3.cl (found | count | nonces).
        struct alignas(8) Result
        {
            bool     found{ false };
            uint32_t count{ 0u };
            uint64_t nonces[algo::xelishashv3::MAX_RESULT]{ 0ull, 0ull, 0ull, 0ull };
        };

#if !defined(__LIB_CUDA)
        struct ResultShare
        {
            std::string jobId{};
            bool        found{ false };
            uint32_t    extraNonceSize{ 0u };
            uint32_t    count{ 0u };
            uint64_t    nonces[algo::xelishashv3::MAX_RESULT]{ 0ull, 0ull, 0ull, 0ull };
        };
#endif
    }
}
