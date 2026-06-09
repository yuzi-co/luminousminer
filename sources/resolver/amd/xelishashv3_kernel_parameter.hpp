#pragma once

#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/xelishashv3/result.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>


namespace resolver
{
    namespace amd
    {
        namespace xelishashv3
        {
            // XelisHash V3 is memory-hard: each in-flight nonce needs its own ~531 KiB
            // scratchpad, so the scratch buffer is sized blocks*threads*MEMSIZE_BYTES and
            // capped by VRAM (the resolver picks a modest occupancy). The per-job state is
            // just the 112-byte MinerWork template and the 32-byte big-endian target.
            struct KernelParameters
            {
                common::opencl::Buffer<uint8_t> headerCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 112u };
                common::opencl::Buffer<uint8_t> targetCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 32u };
                common::opencl::Buffer<uint8_t> scratchCache{ CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 0u };
                common::opencl::BufferMapped<algo::xelishashv3::Result> resultCache{ CL_MEM_READ_WRITE
                                                                                     | CL_MEM_ALLOC_HOST_PTR };
            };
        }
    }
}

#endif
