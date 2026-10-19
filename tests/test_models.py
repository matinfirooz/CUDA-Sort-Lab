#!/usr/bin/env python3
"""CPU models validate CUDA indexing formulas, not GPU execution or races."""

import bisect
import itertools
import random

INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1


def key(value):
    return (value & 0xFFFFFFFF) ^ 0x80000000


def scan(values):
    """Model the exact 512-item Blelloch tiles and recursive block offsets."""
    if not values:
        return []
    result, sums = [], []
    for start in range(0, len(values), 512):
        tile = values[start:start + 512]
        valid = len(tile)
        tile += [0] * (512 - valid)
        offset, active = 1, 256
        while active:
            for thread in range(active):
                left = offset * (2 * thread + 1) - 1
                right = offset * (2 * thread + 2) - 1
                tile[right] += tile[left]
            offset *= 2
            active //= 2
        sums.append(tile[-1])
        tile[-1] = 0
        active = 1
        while active < 512:
            offset //= 2
            for thread in range(active):
                left = offset * (2 * thread + 1) - 1
                right = offset * (2 * thread + 2) - 1
                temp = tile[left]
                tile[left] = tile[right]
                tile[right] += temp
            active *= 2
        result.extend(tile[:valid])
    if len(sums) > 1:
        offsets = scan(sums)
        result = [value + offsets[i // 512] for i, value in enumerate(result)]
    return result


def bitonic(values):
    n = len(values)
    if n < 2:
        return values[:]
    padded = 1 << (n - 1).bit_length()
    a = values[:] + [INT_MAX] * (padded - n)
    size = 2
    while size <= padded:
        distance = size // 2
        while distance:
            for i in range(padded):
                j = i ^ distance
                if j > i:
                    ascending = (i & size) == 0
                    if (ascending and a[i] > a[j]) or (not ascending and a[i] < a[j]):
                        a[i], a[j] = a[j], a[i]
            distance //= 2
        size *= 2
    return a[:n]


def odd_even(values):
    a = values[:]
    for phase in range(len(a)):
        for i in range(phase % 2, len(a) - 1, 2):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
    return a


def merge(values):
    a, n, width = values[:], len(values), 1
    while width < n:
        out = [None] * n
        for i, value in enumerate(a):
            base = (i // (2 * width)) * (2 * width)
            mid, end = min(base + width, n), min(base + 2 * width, n)
            if i < mid:
                dest = base + i - base + bisect.bisect_left(a, value, mid, end) - mid
            else:
                dest = base + i - mid + bisect.bisect_right(a, value, base, mid) - base
            assert out[dest] is None, "Duplicate merge destination"
            out[dest] = value
        assert None not in out
        a, width = out, width * 2
    return a


def radix(values):
    a = values[:]
    for bit in range(32):
        flags = [int(((key(value) >> bit) & 1) == 0) for value in a]
        # The scan kernel itself is independently checked, including 3 levels.
        prefix = list(itertools.accumulate([0] + flags))[:-1]
        zeros = sum(flags)
        out = [None] * len(a)
        for i, value in enumerate(a):
            dest = prefix[i] if flags[i] else zeros + i - prefix[i]
            assert out[dest] is None, "Duplicate radix destination"
            out[dest] = value
        a = out
    return a


def counting(values):
    if not values:
        return []
    minimum, maximum = min(values), max(values)
    if maximum - minimum > 2048:
        return None  # Explicitly outside this model's bounded counting range.
    hist = [0] * (maximum - minimum + 1)
    for value in values:
        hist[value - minimum] += 1
    prefix = list(itertools.accumulate([0] + hist))[:-1]
    return [minimum + bisect.bisect_right(prefix, i) - 1 for i in range(len(values))]


def bucket(values):
    bins = [[] for _ in range(256)]
    for value in values:
        bins[key(value) >> 24].append(value)
    # The actual per-bucket heap routine is covered by cpu_tests.cpp.
    return [value for bin_values in bins for value in sorted(bin_values)]


def rank(values):
    out = [None] * len(values)
    for i, value in enumerate(values):
        dest = sum(other < value or (other == value and j < i) for j, other in enumerate(values))
        assert out[dest] is None, "Duplicate rank destination"
        out[dest] = value
    return out


def shell(values):
    a, gap = values[:], len(values) // 2
    while gap:
        for start in range(gap):
            for i in range(start + gap, len(a), gap):
                value, j = a[i], i
                while j >= gap and a[j - gap] > value:
                    a[j] = a[j - gap]
                    j -= gap
                a[j] = value
        gap //= 2
    return a


def quick(values):
    a, n = values[:], len(values)
    queue = [(0, n)] if n > 1 else []
    rng = random.Random(59)
    while queue:
        assert len(queue) <= n // 2
        ordered = sorted(queue)
        assert all(ordered[i][1] <= ordered[i + 1][0] for i in range(len(ordered) - 1))
        next_queue = []
        for lo, hi in queue:
            if hi - lo <= 256:
                # Model the kernel's fixed 256-thread tile, including padding.
                a[lo:hi] = bitonic(a[lo:hi] + [INT_MAX] * (256 - (hi - lo)))[:hi - lo]
                continue
            pivot = sorted([a[lo], a[lo + (hi - lo) // 2], a[hi - 1]])[1]
            parts = [[], [], []]
            for value in a[lo:hi]:
                parts[0 if value < pivot else 1 if value == pivot else 2].append(value)
            assert parts[1], "Pivot must occur in the segment"
            for part in parts:
                rng.shuffle(part)  # Atomic scatter can produce arbitrary within-part order.
            a[lo:hi] = parts[0] + parts[1] + parts[2]
            if len(parts[0]) > 1:
                next_queue.append((lo, lo + len(parts[0])))
            if len(parts[2]) > 1:
                next_queue.append((hi - len(parts[2]), hi))
        queue = next_queue
    return a


def main():
    rng = random.Random(42)
    checks = 0
    for n in [0, 1, 2, 3, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 262144, 262145]:
        values = [rng.randrange(5) for _ in range(n)]
        expected = list(itertools.accumulate([0] + values))[:-1]
        assert scan(values) == expected, ("scan", n)
        checks += 1
    cases = [list(values) for n in range(6) for values in itertools.product([-1, 0, 1], repeat=n)]
    for n in [1, 2, 3, 7, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1023, 1025]:
        cases += [
            [rng.randint(-13, 15) for _ in range(n)],
            [rng.randint(INT_MIN, INT_MAX) for _ in range(n)],
            [INT_MAX] * n,
            [[INT_MIN, INT_MAX, 0, -1, 1][i % 5] for i in range(n)],
        ]
    models = [bitonic, odd_even, merge, radix, counting, bucket, rank, shell, quick]
    for values in cases:
        expected = sorted(values)
        for model in models:
            if len(values) > 257 and model in (rank, odd_even):
                continue
            actual = model(values)
            if actual is not None:
                assert actual == expected, (model.__name__, len(values), values[:20])
                checks += 1
    for values in [list(range(4097)), list(range(4097, 0, -1)), [3] * 4097,
                   [rng.randint(-2, 2) for _ in range(4097)]]:
        assert quick(values) == sorted(values)
        checks += 1
    print(f"PASS: {checks} algorithm-model checks; no CUDA execution is implied.")


if __name__ == "__main__":
    main()
