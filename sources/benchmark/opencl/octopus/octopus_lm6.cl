// Octopus search variant lm6 — cooperative + Barrett + interleave 16 (production default).
#define OCT_KERNEL_NAME octopus_lm6
#define OCT_COOP_D      1
#define OCT_USE_BARRETT 1
#define OCT_INTERLEAVE  16u
#include "kernel/octopus/octopus_search.cl"
