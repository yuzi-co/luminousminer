#pragma once

#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/hash.hpp>
#include <algo/octopus/result.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>


namespace resolver
{
    namespace amd
    {
        namespace octopus
        {
            struct KernelParameters
            {
                // Light cache (~35 MB now) uploaded from host, freed after the DAG is built.
                common::opencl::Buffer<algo::u_hash512> lightCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY };
                // The full dataset (>= 4 GiB, 8.45 GiB now) is split into < 4 GiB chunk
                // buffers held in the resolver (ResolverAmdOctopus::dagChunks), not here.
                // 32-byte header hash, read by the search kernel as __constant ulong[4].
                common::opencl::BufferMapped<uint32_t> headerCache{ CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                                                    algo::LEN_HASH_256 };
                common::opencl::BufferMapped<algo::octopus::Result> resultCache{ CL_MEM_READ_WRITE
                                                                                 | CL_MEM_ALLOC_HOST_PTR };
            };
        }
    }
}

#endif
