#include "cudasort/sort.cuh"
#include "cudasort/core.hpp"
#include "cudasort/cuda_utils.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <climits>
#include <cstdint>
#include <utility>

namespace cudasort {
namespace {

constexpr int threads = 256;
constexpr int scan_tile = 2 * threads;
constexpr int buckets = 256;
inline int blocks(int n) { return (n + threads - 1) / threads; }

__global__ void pad_keys(const int* in, int* out, int n, int padded) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < padded) out[i] = i < n ? in[i] : INT_MAX;
}

__global__ void bitonic_step(int* a, int n, int distance, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = i ^ distance;
    // Exactly one thread owns each pair, so there are no conflicting writes.
    if (i < n && j > i && j < n) {
        const bool ascending = (i & size) == 0;
        const int left = a[i], right = a[j];
        if ((ascending && left > right) || (!ascending && left < right)) {
            a[i] = right;
            a[j] = left;
        }
    }
}

__global__ void odd_even_step(int* a, int n, int parity) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = 2 * pair + parity;
    if (i + 1 < n && a[i] > a[i + 1]) detail::exchange(a[i], a[i + 1]);
}

__global__ void merge_pass(const int* in, int* out, int n, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int base = (i / (2 * width)) * (2 * width);
    const int mid = base + width < n ? base + width : n;
    const int end = base + 2 * width < n ? base + 2 * width : n;
    // Lower bound for left keys, upper bound for right keys: equal keys
    // occupy distinct output slots, with the left run preceding the right.
    const int dest = i < mid
        ? base + (i - base) + detail::lower_bound(in, mid, end, in[i]) - mid
        : base + (i - mid) + detail::upper_bound(in, base, mid, in[i]) - base;
    out[dest] = in[i];
}

// Work-efficient Blelloch scan. Every thread reaches every barrier, including
// threads corresponding to padding in the final partial tile.
__global__ void scan_blocks(const int* in, int* out, int* sums, int n) {
    __shared__ int tile[scan_tile];
    const int t = threadIdx.x;
    const int base = blockIdx.x * scan_tile;
    tile[t] = base + t < n ? in[base + t] : 0;
    tile[t + threads] = base + t + threads < n ? in[base + t + threads] : 0;
    int offset = 1;
    for (int active = scan_tile / 2; active > 0; active >>= 1) {
        __syncthreads();
        if (t < active) {
            const int left = offset * (2 * t + 1) - 1;
            const int right = offset * (2 * t + 2) - 1;
            tile[right] += tile[left];
        }
        offset <<= 1;
    }
    __syncthreads();
    if (t == 0) {
        sums[blockIdx.x] = tile[scan_tile - 1];
        tile[scan_tile - 1] = 0;
    }
    for (int active = 1; active < scan_tile; active <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (t < active) {
            const int left = offset * (2 * t + 1) - 1;
            const int right = offset * (2 * t + 2) - 1;
            const int temp = tile[left];
            tile[left] = tile[right];
            tile[right] += temp;
        }
    }
    __syncthreads();
    if (base + t < n) out[base + t] = tile[t];
    if (base + t + threads < n) out[base + t + threads] = tile[t + threads];
}

__global__ void add_scan_offsets(int* data, const int* offsets, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += offsets[i / scan_tile];
}

void exclusive_scan(const int* in, int* out, int n, cudaStream_t stream) {
    const int count = (n + scan_tile - 1) / scan_tile;
    DeviceBuffer<int> sums(count);
    scan_blocks<<<count, threads, 0, stream>>>(in, out, sums.get(), n);
    CS_CUDA(cudaGetLastError());
    if (count > 1) {
        DeviceBuffer<int> offsets(count);
        exclusive_scan(sums.get(), offsets.get(), count, stream);
        add_scan_offsets<<<blocks(n), threads, 0, stream>>>(out, offsets.get(), n);
        CS_CUDA(cudaGetLastError());
    }
}

__global__ void radix_flags(const int* in, int* flags, int n, int bit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) flags[i] = ((detail::ordered_key(in[i]) >> bit) & 1u) == 0;
}

__global__ void radix_scatter(const int* in, int* out, const int* flags,
                              const int* prefix, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int total_zeros = prefix[n - 1] + flags[n - 1];
    const int dest = flags[i] ? prefix[i] : total_zeros + i - prefix[i];
    out[dest] = in[i];
}

__global__ void counting_histogram(const int* in, int* hist, int n,
                                   int minimum, int maximum, int* invalid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int value = in[i];
        if (value < minimum || value > maximum) atomicExch(invalid, 1);
        else atomicAdd(hist + static_cast<int>(static_cast<long long>(value) - minimum), 1);
    }
}

__global__ void expand_histogram(const int* prefix, int* out, int n,
                                 int bin_count, int minimum) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Find the last bin whose starting offset is <= i. Repeated offsets
    // (empty bins) must be skipped with upper_bound, not lower_bound.
    const int bin = detail::upper_bound(prefix, 0, bin_count, i) - 1;
    out[i] = static_cast<int>(static_cast<long long>(minimum) + bin);
}

__global__ void bucket_histogram(const int* in, int* counts, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counts + (detail::ordered_key(in[i]) >> 24), 1);
}

__global__ void bucket_scatter(const int* in, int* out, int* cursors, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const unsigned bucket = detail::ordered_key(in[i]) >> 24;
        const int dest = atomicAdd(cursors + bucket, 1);
        out[dest] = in[i];
    }
}

__global__ void sort_buckets(int* values, const int* counts, const int* starts) {
    const int bucket = threadIdx.x;
    detail::heap_sort(values + starts[bucket], counts[bucket]);
}

__global__ void rank_sort(const int* in, int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int key = in[i];
    int rank = 0;
    for (int j = 0; j < n; ++j)
        rank += in[j] < key || (in[j] == key && j < i);
    out[rank] = key;
}

struct Range { int begin, end; };

__global__ void quick_pass(int* values, int* scratch, const Range* current,
                            Range* next, int* next_count) {
    const Range range = current[blockIdx.x];
    const int t = threadIdx.x;
    const int n = range.end - range.begin;
    __shared__ int tile[threads];
    __shared__ int pivot;
    __shared__ int counts[3];
    __shared__ int cursors[3];

    // Uniform branch per block; shared-memory bitonic sorting for leaves.
    if (n <= threads) {
        tile[t] = t < n ? values[range.begin + t] : INT_MAX;
        __syncthreads();
        for (int size = 2; size <= threads; size <<= 1)
            for (int distance = size >> 1; distance; distance >>= 1) {
                const int j = t ^ distance;
                if (j > t) {
                    const bool ascending = (t & size) == 0;
                    if ((ascending && tile[t] > tile[j]) ||
                        (!ascending && tile[t] < tile[j]))
                        detail::exchange(tile[t], tile[j]);
                }
                __syncthreads();
            }
        if (t < n) values[range.begin + t] = tile[t];
        return;
    }

    if (t == 0) {
        pivot = detail::median3(values[range.begin], values[range.begin + n / 2],
                                values[range.end - 1]);
        counts[0] = counts[1] = counts[2] = 0;
    }
    __syncthreads();
    for (int i = range.begin + t; i < range.end; i += threads) {
        const int value = values[i];
        const int part = value < pivot ? 0 : (value == pivot ? 1 : 2);
        atomicAdd(counts + part, 1);
    }
    __syncthreads();
    if (t == 0) {
        cursors[0] = range.begin;
        cursors[1] = range.begin + counts[0];
        cursors[2] = range.begin + counts[0] + counts[1];
    }
    __syncthreads();
    for (int i = range.begin + t; i < range.end; i += threads) {
        const int value = values[i];
        const int part = value < pivot ? 0 : (value == pivot ? 1 : 2);
        scratch[atomicAdd(cursors + part, 1)] = value;
    }
    // Only this block owns the range: this barrier makes its scratch writes
    // visible before threads copy them back. Ranges never overlap.
    __syncthreads();
    for (int i = range.begin + t; i < range.end; i += threads) values[i] = scratch[i];
    __syncthreads();
    if (t == 0) {
        if (counts[0] > 1) {
            const int slot = atomicAdd(next_count, 1);
            next[slot] = {range.begin, range.begin + counts[0]};
        }
        if (counts[2] > 1) {
            const int slot = atomicAdd(next_count, 1);
            next[slot] = {range.end - counts[2], range.end};
        }
    }
}

__global__ void shell_pass(int* a, int n, int gap) {
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    if (start >= gap) return;
    for (int i = start + gap; i < n; i += gap) {
        const int value = a[i];
        int j = i;
        while (j >= gap && a[j - gap] > value) {
            a[j] = a[j - gap];
            j -= gap;
        }
        a[j] = value;
    }
}

__global__ void classic_kernel(int* a, int n, Algorithm algorithm) {
    if (blockIdx.x == 0 && threadIdx.x == 0) detail::classic_sort(a, n, algorithm);
}

}  // namespace

void sort(int* values, std::size_t size, Algorithm algorithm,
          const Options& options, cudaStream_t stream) {
    (void)info(algorithm);  // Reject an invalid enum, even for an empty input.
    if (size > max_items) throw std::invalid_argument("n exceeds max_items (2^26)");
    if (size && !values) throw std::invalid_argument("Null device pointer for nonempty input");
    // Counting sort validates its range and keys even for a singleton.
    if (algorithm == Algorithm::Counting) {
        const auto bin_count64 = static_cast<long long>(options.maximum) - options.minimum + 1;
        if (bin_count64 <= 0 || static_cast<unsigned long long>(bin_count64) >
                options.max_counting_bins || bin_count64 > static_cast<long long>(max_items))
            throw std::invalid_argument("Counting range invalid or exceeds max_counting_bins");
        if (size == 0) {
            CS_CUDA(cudaStreamSynchronize(stream));
            return;
        }
        const int n = static_cast<int>(size);
        const int bin_count = static_cast<int>(bin_count64);
        DeviceBuffer<int> hist(bin_count), prefix(bin_count), invalid(1);
        CS_CUDA(cudaMemsetAsync(hist.get(), 0, bin_count * sizeof(int), stream));
        CS_CUDA(cudaMemsetAsync(invalid.get(), 0, sizeof(int), stream));
        counting_histogram<<<blocks(n), threads, 0, stream>>>(
            values, hist.get(), n, options.minimum, options.maximum, invalid.get());
        CS_CUDA(cudaGetLastError());
        int bad = 0;
        CS_CUDA(cudaMemcpyAsync(&bad, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
        CS_CUDA(cudaStreamSynchronize(stream));
        if (bad) throw std::invalid_argument("Counting input contains an out-of-range key");
        exclusive_scan(hist.get(), prefix.get(), bin_count, stream);
        expand_histogram<<<blocks(n), threads, 0, stream>>>(
            prefix.get(), values, n, bin_count, options.minimum);
        CS_CUDA(cudaGetLastError());
        CS_CUDA(cudaStreamSynchronize(stream));
        return;
    }
    if (size < 2) {
        CS_CUDA(cudaStreamSynchronize(stream));
        return;
    }
    const int n = static_cast<int>(size);
    const auto bytes = size * sizeof(int);
    switch (algorithm) {
    case Algorithm::Bitonic: {
        int padded = 1;
        while (padded < n) padded <<= 1;
        DeviceBuffer<int> buffer(padded);
        pad_keys<<<blocks(padded), threads, 0, stream>>>(values, buffer.get(), n, padded);
        CS_CUDA(cudaGetLastError());
        for (int length = 2; length <= padded; length <<= 1)
            for (int distance = length >> 1; distance; distance >>= 1) {
                bitonic_step<<<blocks(padded), threads, 0, stream>>>(
                    buffer.get(), padded, distance, length);
                CS_CUDA(cudaGetLastError());
            }
        CS_CUDA(cudaMemcpyAsync(values, buffer.get(), bytes, cudaMemcpyDeviceToDevice, stream));
        break;
    }
    case Algorithm::OddEven:
        for (int pass = 0; pass < n; ++pass) {
            odd_even_step<<<blocks((n + 1) / 2), threads, 0, stream>>>(values, n, pass & 1);
            CS_CUDA(cudaGetLastError());
        }
        break;
    case Algorithm::Merge: {
        DeviceBuffer<int> scratch(size);
        int* in = values;
        int* out = scratch.get();
        for (int width = 1; width < n; width <<= 1) {
            merge_pass<<<blocks(n), threads, 0, stream>>>(in, out, n, width);
            CS_CUDA(cudaGetLastError());
            std::swap(in, out);
        }
        if (in != values) CS_CUDA(cudaMemcpyAsync(values, in, bytes, cudaMemcpyDeviceToDevice, stream));
        break;
    }
    case Algorithm::Radix: {
        DeviceBuffer<int> scratch(size), flags(size), prefix(size);
        int* in = values;
        int* out = scratch.get();
        for (int bit = 0; bit < 32; ++bit) {
            radix_flags<<<blocks(n), threads, 0, stream>>>(in, flags.get(), n, bit);
            CS_CUDA(cudaGetLastError());
            exclusive_scan(flags.get(), prefix.get(), n, stream);
            radix_scatter<<<blocks(n), threads, 0, stream>>>(in, out, flags.get(), prefix.get(), n);
            CS_CUDA(cudaGetLastError());
            std::swap(in, out);
        }
        if (in != values) CS_CUDA(cudaMemcpyAsync(values, in, bytes, cudaMemcpyDeviceToDevice, stream));
        break;
    }
    case Algorithm::Bucket: {
        DeviceBuffer<int> counts(buckets), starts(buckets), cursors(buckets), scratch(size);
        CS_CUDA(cudaMemsetAsync(counts.get(), 0, buckets * sizeof(int), stream));
        bucket_histogram<<<blocks(n), threads, 0, stream>>>(values, counts.get(), n);
        CS_CUDA(cudaGetLastError());
        exclusive_scan(counts.get(), starts.get(), buckets, stream);
        CS_CUDA(cudaMemcpyAsync(cursors.get(), starts.get(), buckets * sizeof(int),
                                cudaMemcpyDeviceToDevice, stream));
        bucket_scatter<<<blocks(n), threads, 0, stream>>>(values, scratch.get(), cursors.get(), n);
        CS_CUDA(cudaGetLastError());
        sort_buckets<<<1, buckets, 0, stream>>>(scratch.get(), counts.get(), starts.get());
        CS_CUDA(cudaGetLastError());
        CS_CUDA(cudaMemcpyAsync(values, scratch.get(), bytes, cudaMemcpyDeviceToDevice, stream));
        break;
    }
    case Algorithm::Rank: {
        DeviceBuffer<int> scratch(size);
        rank_sort<<<blocks(n), threads, 0, stream>>>(values, scratch.get(), n);
        CS_CUDA(cudaGetLastError());
        CS_CUDA(cudaMemcpyAsync(values, scratch.get(), bytes, cudaMemcpyDeviceToDevice, stream));
        break;
    }
    case Algorithm::Quick: {
        DeviceBuffer<int> scratch(size), next_count(1);
        // Every queued range has >= 2 elements, so at most floor(n/2)
        // disjoint ranges can exist at any level.
        DeviceBuffer<Range> queue_a(size / 2), queue_b(size / 2);
        Range* current = queue_a.get();
        Range* next = queue_b.get();
        const Range initial{0, n};
        CS_CUDA(cudaMemcpyAsync(current, &initial, sizeof(Range), cudaMemcpyHostToDevice, stream));
        int count = 1;
        while (count > 0) {
            CS_CUDA(cudaMemsetAsync(next_count.get(), 0, sizeof(int), stream));
            quick_pass<<<count, threads, 0, stream>>>(values, scratch.get(), current, next, next_count.get());
            CS_CUDA(cudaGetLastError());
            CS_CUDA(cudaMemcpyAsync(&count, next_count.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
            CS_CUDA(cudaStreamSynchronize(stream));
            std::swap(current, next);
        }
        break;
    }
    case Algorithm::Shell:
        for (int gap = n / 2; gap > 0; gap /= 2) {
            shell_pass<<<blocks(gap), threads, 0, stream>>>(values, n, gap);
            CS_CUDA(cudaGetLastError());
        }
        break;
    case Algorithm::Thrust:
        thrust::sort(thrust::cuda::par.on(stream), values, values + n);
        break;
    case Algorithm::Cub: {
        DeviceBuffer<int> output(size);
        std::size_t temp_bytes = 0;
        CS_CUDA(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, values,
                                              output.get(), n, 0, 32, stream));
        DeviceBuffer<unsigned char> temp(temp_bytes);
        CS_CUDA(cub::DeviceRadixSort::SortKeys(temp.get(), temp_bytes, values,
                                              output.get(), n, 0, 32, stream));
        CS_CUDA(cudaMemcpyAsync(values, output.get(), bytes, cudaMemcpyDeviceToDevice, stream));
        break;
    }
    default:
        classic_kernel<<<1, 1, 0, stream>>>(values, n, algorithm);
        CS_CUDA(cudaGetLastError());
        break;
    }
    CS_CUDA(cudaStreamSynchronize(stream));
}

}  // namespace cudasort
