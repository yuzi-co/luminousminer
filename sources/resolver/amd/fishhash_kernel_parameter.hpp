#pragma once

#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/fishhash/fishhash.hpp>
#include <algo/fishhash/result.hpp>
#include <algo/hash.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>


namespace resolver
{
    namespace amd
    {
        namespace fishhash
        {
            struct KernelParameters
            {
                // Light cache (~75 MB) uploaded from host, freed after the DAG is built.
                common::opencl::Buffer<algo::u_hash512> lightCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY };
                // The full dataset (~4.83 GB) is split into < 4 GB chunk buffers held in
                // the resolver (ResolverAmdFishhash::dagChunks), not here.
                // 180-byte Iron Fish header (stored in a hash3072-sized mapped buffer).
                common::opencl::BufferMapped<algo::u_hash3072> headerCache{ CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                                                            algo::fishhash::HEADER_SIZE };
                // 32-byte big-endian target.
                common::opencl::BufferMapped<algo::u_hash256> boundaryCache{ CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR };
                common::opencl::BufferMapped<algo::fishhash::Result> resultCache{ CL_MEM_READ_WRITE
                                                                                  | CL_MEM_ALLOC_HOST_PTR };
            };
        }
    }
}

#endif
