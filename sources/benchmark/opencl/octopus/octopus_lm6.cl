// Octopus search variant lm6 — cooperative + Barrett + interleave 16 (pre-lazy baseline; see lm7).
#define OCT_KERNEL_NAME octopus_lm6
#define OCT_COOP_D      1
#define OCT_USE_BARRETT 1
#define OCT_INTERLEAVE  16u
#define OCT_LAZY_HORNER 0
#define OCT_USE_MONT    0
#include "kernel/octopus/octopus_search.cl"
