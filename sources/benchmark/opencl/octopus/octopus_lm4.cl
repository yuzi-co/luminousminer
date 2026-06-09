// Octopus search variant lm4 — cooperative LDS-shared d[] + Barrett, no interleave.
#define OCT_KERNEL_NAME octopus_lm4
#define OCT_COOP_D      1
#define OCT_USE_BARRETT 1
#define OCT_INTERLEAVE  1u
#include "kernel/octopus/octopus_search.cl"
