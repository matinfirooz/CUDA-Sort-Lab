#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include "algorithms.hpp"

namespace cudasort {

struct Options {
    // Counting sort only: an inclusive range. Out-of-range keys throw.
    int minimum = -4096;
    int maximum = 4095;
    std::size_t max_counting_bins = std::size_t{1} << 20;
};

// Sort signed 32-bit keys in ascending order in device memory.
// Blocking: synchronizes the supplied stream before returning. Temporary
// allocations and host queue management belong to this operation's cost.
// n == 0 permits nullptr; n > 0 requires a valid device pointer.
// Key-only API: no promise of payload stability or key/value support.
void sort(int* device_values, std::size_t n, Algorithm algorithm,
          const Options& options = {}, cudaStream_t stream = nullptr);

}  // namespace cudasort
