// Octopus search variant lm2 — baseline + Barrett reduction (still private/scratch d[]).
#define OCT_KERNEL_NAME octopus_lm2
#define OCT_COOP_D      0
#define OCT_USE_BARRETT 1
#define OCT_INTERLEAVE  1u
#include "kernel/octopus/octopus_search.cl"
