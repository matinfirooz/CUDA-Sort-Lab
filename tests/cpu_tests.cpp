#include "cudasort/algorithms.hpp"
#include "cudasort/core.hpp"
#include "cudasort/input.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array classic{
    cudasort::Algorithm::Bubble, cudasort::Algorithm::Selection,
    cudasort::Algorithm::Insertion, cudasort::Algorithm::Heap,
    cudasort::Algorithm::Cocktail, cudasort::Algorithm::Comb,
    cudasort::Algorithm::Gnome, cudasort::Algorithm::Cycle,
    cudasort::Algorithm::Pancake
};

std::size_t checks = 0;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
    ++checks;
}

void check_classics(const std::vector<int>& input) {
    auto expected = input;
    std::sort(expected.begin(), expected.end());
    for (auto algorithm : classic) {
        auto actual = input;
        cudasort::detail::classic_sort(actual.data(), static_cast<int>(actual.size()), algorithm);
        require(actual == expected, "Classic mismatch: " + std::string(cudasort::info(algorithm).name));
    }
}

}  // namespace

int main() {
    try {
        // Exhaustive duplicate-heavy inputs catch cycle-sort and exchange bugs.
        std::size_t combinations = 1;
        for (int n = 0; n <= 8; ++n) {
            for (std::size_t code = 0; code < combinations; ++code) {
                std::vector<int> input(n);
                auto digits = code;
                for (int& value : input) { value = static_cast<int>(digits % 3) - 1; digits /= 3; }
                check_classics(input);
            }
            combinations *= 3;
        }
        for (auto n : {0u, 1u, 2u, 3u, 31u, 32u, 33u, 127u, 255u, 256u, 257u, 513u, 1025u})
            for (auto pattern : cudasort::patterns) {
                auto input = cudasort::make_input(n, pattern, 42, -71, 73);
                check_classics(input);
                auto sorted = input;
                std::sort(sorted.begin(), sorted.end());
                for (int value : {INT_MIN, -72, -1, 0, 1, 74, INT_MAX}) {
                    const auto lower = std::lower_bound(sorted.begin(), sorted.end(), value) - sorted.begin();
                    const auto upper = std::upper_bound(sorted.begin(), sorted.end(), value) - sorted.begin();
                    require(cudasort::detail::lower_bound(sorted.data(), 0, static_cast<int>(n), value) == lower,
                            "lower_bound mismatch");
                    require(cudasort::detail::upper_bound(sorted.data(), 0, static_cast<int>(n), value) == upper,
                            "upper_bound mismatch");
                }
            }
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> values(INT_MIN, INT_MAX);
        for (int test = 0; test < 500; ++test) {
            std::vector<int> input(rng() % 129);
            for (int& value : input) value = values(rng);
            check_classics(input);
        }
        const int boundary[]{INT_MIN, INT_MIN + 1, -1, 0, 1, INT_MAX - 1, INT_MAX};
        for (std::size_t i = 1; i < std::size(boundary); ++i)
            require(cudasort::detail::ordered_key(boundary[i - 1]) <
                    cudasort::detail::ordered_key(boundary[i]), "Signed-key transform mismatch");
        for (int a : boundary) for (int b : boundary) for (int c : boundary) {
            std::array keys{a, b, c};
            std::sort(keys.begin(), keys.end());
            require(cudasort::detail::median3(a, b, c) == keys[1], "median3 mismatch");
        }
        for (const auto& entry : cudasort::algorithms)
            require(cudasort::parse_algorithm(entry.name) == entry.id, "Algorithm name mismatch");
        bool rejected = false;
        try { (void)cudasort::parse_algorithm("not-a-sort"); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Missing unknown-algorithm rejection");
        std::cout << "PASS: " << checks << " CPU checks (classic routines and shared helpers).\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
