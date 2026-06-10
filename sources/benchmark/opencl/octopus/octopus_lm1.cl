// Octopus search variant lm1 — UNOPTIMIZED BASELINE.
// Each work-item builds its own d[OCT_N] in private memory (spills to scratch) and uses
// the hardware 64-bit `% MOD` divide; no Horner interleave. Reference point for the sweep.
#define OCT_KERNEL_NAME octopus_lm1
#define OCT_COOP_D      0
#define OCT_USE_BARRETT 0
#define OCT_INTERLEAVE  1u
#define OCT_LAZY_HORNER 0
#define OCT_USE_MONT    0
#include "kernel/octopus/octopus_search.cl"
