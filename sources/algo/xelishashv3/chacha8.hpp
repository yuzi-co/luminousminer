#pragma once

#include <cstddef>
#include <cstdint>


namespace xelishashv3
{
    // ChaCha8 keystream generator, RFC-8439 (IETF) layout: 256-bit key, 96-bit (12-byte)
    // nonce, 32-bit block counter starting at 0, 8 rounds (4 double-rounds). XELIS stage 1
    // calls this with in == 0, so the output is the raw keystream. `len` is exercised only
    // with multiples of 64 (a whole stage-1 chunk), but partial tails are handled too.
    void chacha8Keystream(uint8_t const key[32], uint8_t const nonce[12], uint8_t* out, size_t len);
}
