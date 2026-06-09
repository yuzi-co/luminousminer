// Octopus search variant lm3 — cooperative LDS-shared d[] (hardware % MOD), no interleave.
#define OCT_KERNEL_NAME octopus_lm3
#define OCT_COOP_D      1
#define OCT_USE_BARRETT 0
#define OCT_INTERLEAVE  1u
#define OCT_LAZY_HORNER 0
#include "kernel/octopus/octopus_search.cl"
