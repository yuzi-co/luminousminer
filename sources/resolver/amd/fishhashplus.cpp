#include <algo/algo_type.hpp>
#include <resolver/amd/fishhashplus.hpp>


resolver::ResolverAmdFishhashPlus::ResolverAmdFishhashPlus() : resolver::ResolverAmdFishhash()
{
    // Base ctor defaults the algorithm to FISHHASH; promote it so buildSearch() takes the
    // KarlsenHashV2 kernel path and doSubmit()/stratum use the right wire format.
    algorithm = algo::ALGORITHM::FISHHASHPLUS;
}
