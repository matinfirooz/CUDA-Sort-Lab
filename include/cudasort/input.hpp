#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace cudasort {

inline constexpr std::array<std::string_view, 8> patterns{
    "random", "sorted", "reverse", "equal", "few-unique", "extremes",
    "nearly-sorted", "organ-pipe"
};

inline std::vector<int> make_input(std::size_t n, std::string_view pattern,
                                   std::uint32_t seed, int minimum, int maximum) {
    if (minimum > maximum) throw std::invalid_argument("minimum exceeds maximum");
    if (std::find(patterns.begin(), patterns.end(), pattern) == patterns.end())
        throw std::invalid_argument("Unknown input pattern");
    std::vector<int> a(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(minimum, maximum);
    for (int& value : a) value = dist(rng);
    const auto span = static_cast<long long>(maximum) - minimum;
    if (pattern == "sorted" || pattern == "nearly-sorted") {
        std::sort(a.begin(), a.end());
        if (pattern == "nearly-sorted" && n > 1) {
            std::uniform_int_distribution<std::size_t> position(0, n - 1);
            for (std::size_t i = 0; i < std::max<std::size_t>(1, n / 100); ++i)
                std::swap(a[position(rng)], a[position(rng)]);
        }
    } else if (pattern == "reverse") {
        std::sort(a.rbegin(), a.rend());
    } else if (pattern == "equal") {
        std::fill(a.begin(), a.end(), static_cast<int>(minimum + span / 2));
    } else if (pattern == "few-unique") {
        std::uniform_int_distribution<int> choose(0, 4);
        for (int& value : a) value = static_cast<int>(minimum + span * choose(rng) / 4);
    } else if (pattern == "extremes") {
        constexpr int keys[]{INT_MIN, INT_MAX, 0, -1, 1};
        for (std::size_t i = 0; i < n; ++i) a[i] = keys[i % 5];
    } else if (pattern == "organ-pipe") {
        for (std::size_t i = 0; i < n; ++i) {
            const auto height = std::min(i, n - 1 - i);
            a[i] = static_cast<int>(minimum + span * static_cast<long long>(height) /
                                   static_cast<long long>(std::max<std::size_t>(1, n / 2)));
        }
    }
    return a;
}

}  // namespace cudasort
