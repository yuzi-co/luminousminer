#pragma once

#include <array>
#include <cstdint>


// Known-answer vectors lifted verbatim from the upstream reference
// (xelis-project/xelis-hash): C/xelis_hash_v3.c timing_test() gold and go/v3/v3_test.go
// (TestZeroHash, TestVerifyOutput).
namespace xelishashv3::kat
{
    // 112 zero bytes -> gold (C reference `gold`, Go TestZeroHash `expected`).
    inline constexpr std::array<uint8_t, 112> ZERO_INPUT{};

    inline constexpr std::array<uint8_t, 32> ZERO_EXPECTED{
        105, 172, 103, 40,  94,  253, 92,  162, 42,  252, 5,   196, 236, 238, 91,  218,
        22,  157, 228, 233, 239, 8,   250, 57,  212, 166, 121, 132, 148, 205, 103, 163
    };

    // Go TestVerifyOutput: arbitrary 112-byte input -> expected.
    inline constexpr std::array<uint8_t, 112> VERIFY_INPUT{
        172, 236, 108, 212, 181, 31,  109, 45,  44,  242, 54,  225, 143, 133, 89,  44,
        179, 108, 39,  191, 32,  116, 229, 33,  63,  130, 33,  120, 185, 89,  146, 141,
        10,  79,  183, 107, 238, 122, 92,  222, 25,  134, 90,  107, 116, 110, 236, 53,
        255, 5,   214, 126, 24,  216, 97,  199, 148, 239, 253, 102, 199, 184, 232, 253,
        158, 145, 86,  187, 112, 81,  78,  70,  80,  110, 33,  37,  159, 233, 198, 1,
        178, 108, 210, 100, 109, 155, 106, 124, 124, 83,  89,  50,  197, 115, 231, 32,
        74,  2,   92,  47,  25,  220, 135, 249, 122, 172, 220, 137, 143, 234, 68,  188
    };

    inline constexpr std::array<uint8_t, 32> VERIFY_EXPECTED{
        242, 8,   176, 222, 203, 27,  104, 187, 22,  40,  68,  73,  79,  79,  65,  83,
        138, 101, 10,  116, 194, 41,  153, 21,  92,  163, 12,  206, 231, 156, 70,  83
    };
}
