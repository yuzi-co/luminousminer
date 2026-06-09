#pragma once

#include <cstdint>


namespace xelishashv3
{
    // One AES encryption round, bit-identical to Intel's _mm_aesenc_si128(block, key):
    //   block <- MixColumns(ShiftRows(SubBytes(block))) XOR key
    // The 16 bytes are the AES state in column-major order (bytes 0..3 = column 0, the
    // byte index within a column = the row), matching the little-endian _mm_loadu_si128
    // the reference uses. Operates in place on `block`.
    void aesSingleRound(uint8_t block[16], uint8_t const key[16]);
}
