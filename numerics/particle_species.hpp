#pragma once

#include <cstdint>
#include <random>

// The published runs drew three equiprobable species with GCC 14 libstdc++.
// For mt19937_64 and this range its mapping discards zero and divides the
// remaining 2^64 - 1 outputs into three equally sized, consecutive intervals.
inline int species_index(std::mt19937_64& generator) {
    constexpr auto interval_size = std::mt19937_64::max() / 3;
    std::uint64_t value;
    do {
        value = generator();
    } while (value == 0);
    return static_cast<int>((value - 1) / interval_size);
}
