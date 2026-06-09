#pragma once

#include <algo/xelishashv3/types.hpp>


namespace xelishashv3
{
    // Stage 1: derive 4 ChaCha8 keys via BLAKE3 and fill the scratchpad byte image with
    // their keystreams (each key/nonce chained from the previous chunk). `scratch` must
    // point at MEMSIZE_BYTES of storage.
    void stage1(uint8_t const input[INPUT_LEN], uint8_t* scratch);

    // Stage 3: the memory-hard, branchy, power-hungry mixing pass over the scratchpad
    // (two halves A/B, ITERS outer × BUFSIZE inner, 16-way ALU switch). `scratch` is the
    // MEMSIZE-word buffer viewed as u64.
    void stage3(uint64_t* scratch);

    // Full XelisHash V3: stage1 -> stage3 -> BLAKE3 over the whole scratchpad. `pad` is a
    // reusable scratchpad (fully overwritten each call, so it may be shared across hashes).
    Hash256 hash(uint8_t const input[INPUT_LEN], ScratchPad& pad);
}
