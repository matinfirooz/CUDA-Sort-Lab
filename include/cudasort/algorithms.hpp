#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cudasort {

enum class Algorithm {
    Bitonic, OddEven, Merge, Radix, Counting, Bucket, Rank, Quick, Shell,
    Bubble, Selection, Insertion, Heap, Cocktail, Comb, Gnome, Cycle, Pancake,
    Thrust, Cub
};

struct AlgorithmInfo {
    Algorithm id;
    std::string_view name;
    std::string_view execution;
    std::size_t suggested_limit;  // CLI guard; 0 means no educational limit.
};

inline constexpr std::array<AlgorithmInfo, 20> algorithms{{
    {Algorithm::Bitonic, "bitonic", "parallel compare-exchange network", 0},
    {Algorithm::OddEven, "odd-even", "parallel adjacent compare-exchange", 4096},
    {Algorithm::Merge, "merge", "parallel binary-search rank merge", 0},
    {Algorithm::Radix, "radix", "parallel stable binary LSD + custom scan", 0},
    {Algorithm::Counting, "counting", "parallel histogram + scan + expansion", 0},
    {Algorithm::Bucket, "bucket", "parallel scatter; serial heap per bucket", 262144},
    {Algorithm::Rank, "rank", "parallel all-pairs enumeration", 8192},
    {Algorithm::Quick, "quick", "one block per partition + shared tile leaves", 0},
    {Algorithm::Shell, "shell", "parallel residue classes; serial final gap", 262144},
    {Algorithm::Bubble, "bubble", "one GPU thread (educational)", 4096},
    {Algorithm::Selection, "selection", "one GPU thread (educational)", 4096},
    {Algorithm::Insertion, "insertion", "one GPU thread (educational)", 4096},
    {Algorithm::Heap, "heap", "one GPU thread (educational)", 65536},
    {Algorithm::Cocktail, "cocktail", "one GPU thread (educational)", 4096},
    {Algorithm::Comb, "comb", "one GPU thread (educational)", 65536},
    {Algorithm::Gnome, "gnome", "one GPU thread (educational)", 4096},
    {Algorithm::Cycle, "cycle", "one GPU thread (educational)", 4096},
    {Algorithm::Pancake, "pancake", "one GPU thread (educational)", 4096},
    {Algorithm::Thrust, "thrust", "NVIDIA Thrust library", 0},
    {Algorithm::Cub, "cub", "NVIDIA CUB radix library", 0},
}};

inline Algorithm parse_algorithm(std::string_view name) {
    for (const auto& info : algorithms)
        if (info.name == name) return info.id;
    throw std::invalid_argument("Unknown algorithm: " + std::string(name));
}

inline const AlgorithmInfo& info(Algorithm algorithm) {
    for (const auto& entry : algorithms)
        if (entry.id == algorithm) return entry;
    throw std::invalid_argument("Invalid algorithm enum");
}

inline constexpr std::size_t max_items = std::size_t{1} << 26;

}  // namespace cudasort
