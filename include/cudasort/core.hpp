#pragma once

#include <cstdint>
#include <limits>
#include "algorithms.hpp"

#ifdef __CUDACC__
#define CS_HD __host__ __device__
#else
#define CS_HD
#endif

namespace cudasort::detail {

static_assert(sizeof(int) == 4 && std::numeric_limits<int>::digits == 31,
              "CUDA Sort Lab requires 32-bit int");

// Flipping the sign bit maps signed integer order to unsigned integer order.
CS_HD inline std::uint32_t ordered_key(int value) {
    return static_cast<std::uint32_t>(value) ^ 0x80000000u;
}

CS_HD inline void exchange(int& a, int& b) {
    const int temp = a;
    a = b;
    b = temp;
}

CS_HD inline int median3(int a, int b, int c) {
    if (a > b) exchange(a, b);
    if (b > c) exchange(b, c);
    if (a > b) exchange(a, b);
    return b;
}

CS_HD inline int lower_bound(const int* a, int lo, int hi, int value) {
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (a[mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

CS_HD inline int upper_bound(const int* a, int lo, int hi, int value) {
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (a[mid] <= value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

CS_HD inline void insertion_sort(int* a, int n) {
    for (int i = 1; i < n; ++i) {
        const int value = a[i];
        int j = i;
        while (j > 0 && a[j - 1] > value) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = value;
    }
}

CS_HD inline void sift_down(int* a, int n, int root) {
    while (2 * root + 1 < n) {
        int child = 2 * root + 1;
        if (child + 1 < n && a[child] < a[child + 1]) ++child;
        if (a[root] >= a[child]) break;
        exchange(a[root], a[child]);
        root = child;
    }
}

CS_HD inline void heap_sort(int* a, int n) {
    for (int i = n / 2; i > 0; --i) sift_down(a, n, i - 1);
    for (int end = n - 1; end > 0; --end) {
        exchange(a[0], a[end]);
        sift_down(a, end, 0);
    }
}

CS_HD inline void flip_prefix(int* a, int count) {
    for (int left = 0, right = count - 1; left < right; ++left, --right)
        exchange(a[left], a[right]);
}

// Shared host/device routines make the classic algorithms CPU-testable.
// CUDA launches these on one thread; this does not imply GPU acceleration.
CS_HD inline void classic_sort(int* a, int n, Algorithm algorithm) {
    if (n < 2) return;
    switch (algorithm) {
    case Algorithm::Bubble:
        for (int end = n - 1; end > 0; --end) {
            bool changed = false;
            for (int i = 0; i < end; ++i)
                if (a[i] > a[i + 1]) {
                    exchange(a[i], a[i + 1]);
                    changed = true;
                }
            if (!changed) break;
        }
        break;
    case Algorithm::Selection:
        for (int i = 0; i < n - 1; ++i) {
            int best = i;
            for (int j = i + 1; j < n; ++j)
                if (a[j] < a[best]) best = j;
            exchange(a[i], a[best]);
        }
        break;
    case Algorithm::Insertion:
        insertion_sort(a, n);
        break;
    case Algorithm::Heap:
        heap_sort(a, n);
        break;
    case Algorithm::Cocktail: {
        int left = 0, right = n - 1;
        while (left < right) {
            bool changed = false;
            for (int i = left; i < right; ++i)
                if (a[i] > a[i + 1]) {
                    exchange(a[i], a[i + 1]);
                    changed = true;
                }
            if (!changed) break;
            --right;
            changed = false;
            for (int i = right; i > left; --i)
                if (a[i - 1] > a[i]) {
                    exchange(a[i - 1], a[i]);
                    changed = true;
                }
            ++left;
            if (!changed) break;
        }
        break;
    }
    case Algorithm::Comb: {
        int gap = n;
        bool changed = true;
        while (gap > 1 || changed) {
            gap = gap > 1 ? (gap * 10) / 13 : 1;
            if (gap < 1) gap = 1;
            changed = false;
            for (int i = 0; i + gap < n; ++i)
                if (a[i] > a[i + gap]) {
                    exchange(a[i], a[i + gap]);
                    changed = true;
                }
        }
        break;
    }
    case Algorithm::Gnome: {
        int i = 1;
        while (i < n) {
            if (i == 0 || a[i - 1] <= a[i]) ++i;
            else {
                exchange(a[i - 1], a[i]);
                --i;
            }
        }
        break;
    }
    case Algorithm::Cycle:
        for (int start = 0; start < n - 1; ++start) {
            int item = a[start];
            int pos = start;
            for (int i = start + 1; i < n; ++i)
                if (a[i] < item) ++pos;
            if (pos == start) continue;
            while (item == a[pos]) ++pos;
            exchange(item, a[pos]);
            while (pos != start) {
                pos = start;
                for (int i = start + 1; i < n; ++i)
                    if (a[i] < item) ++pos;
                while (item == a[pos]) ++pos;
                exchange(item, a[pos]);
            }
        }
        break;
    case Algorithm::Pancake:
        for (int count = n; count > 1; --count) {
            int best = 0;
            for (int i = 1; i < count; ++i)
                if (a[i] > a[best]) best = i;
            if (best == count - 1) continue;
            if (best > 0) flip_prefix(a, best + 1);
            flip_prefix(a, count);
        }
        break;
    default:
        break;  // Only classic enum values are dispatched here.
    }
}

}  // namespace cudasort::detail

#undef CS_HD
