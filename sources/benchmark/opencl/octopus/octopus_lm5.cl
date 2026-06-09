// Octopus search variant lm5 — cooperative + Barrett + interleave 8.
#define OCT_KERNEL_NAME octopus_lm5
#define OCT_COOP_D      1
#define OCT_USE_BARRETT 1
#define OCT_INTERLEAVE  8u
#define OCT_LAZY_HORNER 0
#include "kernel/octopus/octopus_search.cl"
