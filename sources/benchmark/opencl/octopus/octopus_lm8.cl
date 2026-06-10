// Octopus search variant lm8 — production: cooperative + Barrett (setup) + interleave 16 +
// LAZY Horner + MONTGOMERY modmul (R = 2^32). ~+27% over lm7 (Barrett) on gfx1201; the
// current default. The d[] coefficients and xs are carried in Montgomery form.
#define OCT_KERNEL_NAME octopus_lm8
#define OCT_COOP_D      1
#define OCT_USE_BARRETT 1
#define OCT_INTERLEAVE  16u
#define OCT_LAZY_HORNER 1
#define OCT_USE_MONT    1
#include "kernel/octopus/octopus_search.cl"