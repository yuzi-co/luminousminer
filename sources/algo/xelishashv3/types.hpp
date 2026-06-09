#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>


namespace xelishashv3
{
    // ── XelisHash V3 parameters (xelis-project/xelis-hash, C/xelis_hash_v3.c) ──
    // The scratchpad is MEMSIZE u64 words (≈ 531 KiB). Stage 1 fills it as a byte
    // stream (ChaCha8 keystream), stage 3 mixes it as u64 words, stage 4 hashes the
    // whole byte image with BLAKE3. The u64 view is the little-endian reading of the
    // bytes — matches every target we build for (x86/ARM/RDNA are all LE), which is
    // also how the upstream gold vectors were produced.
    constexpr size_t INPUT_LEN{ 112 };
    constexpr size_t HASH_SIZE{ 32 };
    constexpr size_t MEMSIZE{ 531 * 128 }; // u64 words = 67968
    constexpr size_t MEMSIZE_BYTES{ MEMSIZE * 8 };
    constexpr size_t BUFSIZE{ MEMSIZE / 2 }; // 33984 — half-buffer word count
    constexpr size_t ITERS{ 2 };
    constexpr size_t CHUNK_SIZE{ 32 };
    constexpr size_t CHUNKS{ 4 };
    constexpr size_t NONCE_SIZE{ 12 };

    // Stage 3 AES round key.
    constexpr char KEY[16]{ 'x', 'e', 'l', 'i', 's', 'h', 'a', 's', 'h', '-', 'p', 'o', 'w', '-', 'v', '3' };

    using Hash256 = std::array<uint8_t, HASH_SIZE>;

    // Reusable per-thread scratchpad. Heap-backed so the ~531 KiB does not land on the
    // stack; the reference reuses one allocation across hashes (see TestReusedScratchpad).
    struct ScratchPad
    {
        std::vector<uint64_t> words;

        ScratchPad() : words(MEMSIZE, 0u)
        {
        }

        uint64_t* u64()
        {
            return words.data();
        }
        uint8_t* bytes()
        {
            return reinterpret_cast<uint8_t*>(words.data());
        }
        uint8_t const* bytes() const
        {
            return reinterpret_cast<uint8_t const*>(words.data());
        }
    };
}
