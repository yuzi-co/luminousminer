// Octopus search variant lm7 — cooperative + Barrett + interleave 16 + LAZY Horner
// (current production default). Drops the per-step canonical conditional subtraction in the
// 32*1024 modmuls/nonce Horner loop, carrying a partially reduced accumulator in [0, 2*MOD)
// and canonicalising only on store.
#define OCT_KERNEL_NAME octopus_lm7
#define OCT_COOP_D      1
#define OCT_USE_BARRETT 1
#define OCT_INTERLEAVE  16u
#define OCT_LAZY_HORNER 1
#include "kernel/octopus/octopus_search.cl"
