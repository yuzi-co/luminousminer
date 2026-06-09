#pragma once

#if defined(AMD_ENABLE)

#include <resolver/amd/fishhash.hpp>


namespace resolver
{
    // KarlsenHashV2 (FishHashPlus). Reuses the entire FishHash AMD resolver — same 4.83 GB
    // DAG, same DAG-gen kernel — and only flips the algorithm so buildSearch() compiles the
    // FISHHASH_PLUS path of fishhash_search.cl. See algo::fishhash::hashPlus.
    class ResolverAmdFishhashPlus : public resolver::ResolverAmdFishhash
    {
      public:
        ResolverAmdFishhashPlus();
        ~ResolverAmdFishhashPlus() = default;
    };
}

#endif
