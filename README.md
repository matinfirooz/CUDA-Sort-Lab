# CUDA Sort Lab

**20 sorting implementations for ascending signed 32-bit integer keys:**
18 custom CUDA implementations, plus NVIDIA Thrust and CUB baselines.

Study GPU sorting networks, parallel scans, rank-based merging, histograms,
partition queues, and the limits of moving classic sequential sorts to a GPU.
The parallel algorithms contain their own CUDA kernels; only the explicitly
named `thrust` and `cub` entries call library sorting routines.

| Family | Algorithms |
| --- | --- |
| Sorting network / adjacent exchange | Bitonic, odd-even transposition |
| Divide and conquer | Merge, three-way quicksort with bitonic leaves |
| Distribution | Binary LSD radix, counting, bucket |
| All-pairs ranking | Enumeration / rank sort |
| Gapped insertion | Shell |
| Classic educational sorts | Bubble, selection, insertion, heap, cocktail, comb, gnome, cycle, pancake |
| NVIDIA library baselines | Thrust sort, CUB device radix sort |

This collection covers major families and representative variants. Its scope
is key-only `int32` sorting; floats, payloads, custom comparators, external
sorting, and distributed sorting would need additional implementations.

## Build and run

Requirements: NVIDIA GPU, a compatible driver, CUDA Toolkit 12 or newer,
and a toolkit-supported C++17 host compiler. Use GNU Make or CMake 3.24+;
Python 3 is needed for the model tests and benchmark sweep script.
Compilation was verified with CUDA 13.0; see [VALIDATION.md](VALIDATION.md)
for the exact tests performed and the GPU execution limitation.

```bash
make ARCH=sm_75
./build/cuda_sorts --list
./build/cuda_sorts --algo all --n 1024
./build/cuda_sorts --algo radix,merge,quick,cub,thrust --n 1048576
./build/cuda_sorts --self-test
```

The default target is `sm_75`. Choose the architecture supported by your GPU
and toolkit, for example `ARCH=sm_86` or `ARCH=sm_89`. After changing `ARCH`,
run `make clean` before rebuilding so an existing binary is recompiled.

CMake also builds a reusable library and the API example:

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cmake --parallel
ctest --test-dir build-cmake --output-on-failure
./build-cmake/cuda_sorts --self-test
./build-cmake/sort_example
```

GPU regression is a separate explicit command by default. To register it with
CTest, configure with `-DCUDA_SORT_REGISTER_GPU_TESTS=ON`. Without a CUDA
compiler, CMake builds only the CPU checks and emits a warning. On a CPU-only
machine, you can explicitly request that configuration:

```bash
cmake -S . -B build-cpu -DCUDA_SORT_ENABLE_CUDA=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
# Or, without CMake:
make cpu-test
```

## Implementations and cost

`N` is the number of keys, `P` the next power of two, `B` the number of bins,
and `m_b` a bucket population. Work is the total algorithmic work across GPU
threads; it does not predict elapsed time. The complexity descriptions apply
to these implementations, including their educational tradeoffs.

| CLI name | Implementation | Work | Extra device state |
| --- | --- | --- | --- |
| `bitonic` | Global compare-exchange stages; pad with `INT_MAX` | O(P log² P) | O(P) |
| `odd-even` | N alternating disjoint adjacent-pair phases | O(N²) | O(1) |
| `merge` | Bottom-up runs; each key binary-searches the opposite run | O(N log² N) | O(N) |
| `radix` | 32 stable binary passes; custom hierarchical Blelloch scan | O(32N) | O(N) |
| `counting` | Atomic histogram, scan, binary-search output expansion | O(N log(B+1) + B) | O(B) |
| `bucket` | 256 monotonic buckets; one serial heap sort per bucket | O(N + Σ m_b log(m_b+1)) | O(N+B) |
| `rank` | One thread per key; all-pairs comparisons with index tie-break | O(N²) | O(N) |
| `quick` | Breadth-first three-way partitions; shared bitonic leaves | Expected O(N log N), worst O(N²) | O(N) |
| `shell` | One thread per gap residue; halve the gap each pass | Worst O(N²) | O(1) |
| `bubble` | One GPU thread; early exit on unchanged pass | Worst O(N²) | O(1) |
| `selection` | One GPU thread; repeated minimum selection | O(N²) | O(1) |
| `insertion` | One GPU thread; ordered-prefix insertion | Worst O(N²) | O(1) |
| `heap` | One GPU thread; in-place max heap | O(N log N) | O(1) |
| `cocktail` | One GPU thread; bidirectional bubble passes | Worst O(N²) | O(1) |
| `comb` | One GPU thread; shrinking compare-exchange gaps | Worst O(N²) | O(1) |
| `gnome` | One GPU thread; adjacent swaps and backtracking | Worst O(N²) | O(1) |
| `cycle` | One GPU thread; duplicate-aware cycle placement | O(N²) comparisons | O(1) |
| `pancake` | One GPU thread; maximum selection and prefix reversal | O(N²) | O(1) |
| `thrust` | `thrust::sort(thrust::cuda::par.on(stream), ...)` | Backend dependent | Backend dependent |
| `cub` | `cub::DeviceRadixSort::SortKeys` | Linear for fixed-width keys | Library workspace + O(N) output |

Bubble, selection, insertion, heap, cocktail, comb, gnome, cycle, and pancake
run on the GPU but use a single thread for the complete array. They are
educational translations. Shell's final gap is also serial. Bucket performance
depends strongly on the populations: narrow ranges can occupy just one or two
of its 256 fixed buckets. Quicksort starts with one active block and copies one
queue-count integer to the host per level. The custom binary radix makes 32
passes; it is intended to expose the algorithm and scan mechanics.

Use the baselines to measure library performance on your own workloads.
The implementation notes above explain why the custom versions can have
substantial overhead. No GPU performance measurements are bundled.

## API

```cpp
#include "cudasort/sort.cuh"

// d_values already contains N keys in device memory.
cudasort::sort(d_values, N, cudasort::Algorithm::Radix);

// Counting sort requires a known inclusive key range.
cudasort::Options options;
options.minimum = -100;
options.maximum = 100;
cudasort::sort(d_values, N, cudasort::Algorithm::Counting, options, stream);
```

See [examples/basic.cu](examples/basic.cu) for a complete upload/sort/download
example, and [include/cudasort/sort.cuh](include/cudasort/sort.cuh) for the API.

- All algorithms sort ascending signed 32-bit integer keys in place at the API
  boundary. Internal scratch storage varies by algorithm.
- Lengths from 0 through 2^26 are accepted, including arbitrary non-power-of-two
  lengths. Allocation must still fit the device's memory.
- Empty input permits a null pointer; nonempty input requires a valid device
  pointer. A singleton and an empty input are valid cases.
- The function blocks until its stream's work completes. Internal allocations,
  frees, and host queue work are part of the call. It is not a stream-capture
  or allocation-free interface; `cudaFree` can also introduce wider device
  synchronization.
- Counting sort checks its declared range and rejects an out-of-range key.
  The default maximum is 2^20 bins. Range arithmetic uses 64 bits.
- This API stores keys only. There is no key/value stability guarantee. Extending
  it to payloads requires carrying payloads through every exchange and scatter.
- Each parallel merge/scatter writes each output location exactly once. Kernel
  boundaries in the same stream order global phases. Block barriers are used
  only where a complete block owns the shared state or partition.

## Correctness checks

Every warm-up and timed run compares the complete output against CPU
`std::sort`. This checks ordering and preservation of the input multiset,
including duplicates. Sorting an already sorted result repeatedly would bias
adaptive algorithms, so the original input is uploaded before every run.

```bash
make cpu-test
./build/cuda_sorts --self-test
```

The CPU tests exhaustively exercise classic routines on small duplicate-heavy
arrays, test signed extremes and shared helpers, and validate independent
models of the parallel indexing formulas. Scan models include lengths around
512 and 512². CPU models cannot establish GPU race freedom or execution
correctness. The GPU suite runs all 20 implementations across small boundary
sizes and eight patterns, plus larger tests for the scalable algorithms.
Oversized counting ranges are skipped explicitly; invalid API inputs are
tested for rejection.

On a machine with NVIDIA Compute Sanitizer installed:

```bash
compute-sanitizer --tool memcheck ./build/cuda_sorts --self-test
compute-sanitizer --tool racecheck ./build/cuda_sorts --algo quick --n 4097
compute-sanitizer --tool racecheck ./build/cuda_sorts --algo radix --n 1025
compute-sanitizer --tool synccheck ./build/cuda_sorts --algo quick --n 4097
```

Racecheck focuses on shared-memory hazards; it does not establish global-memory
race freedom. These commands are provided for local GPU validation and were
not executed in the creation environment.

## Benchmarks

```bash
./build/cuda_sorts --algo all --n 4096 --pattern few-unique --csv run.csv
./build/cuda_sorts --algo radix,cub,thrust --n 1048576 \
  --min -2147483648 --max 2147483647 --csv full_range.csv
python3 scripts/benchmark.py --exe build/cuda_sorts \
  --sizes 256 1024 4096 65536 1048576 --output results/benchmark.csv
```

Input patterns: `random`, `sorted`, `reverse`, `equal`, `few-unique`, `extremes`,
`nearly-sorted`, and `organ-pipe`. `--min` / `--max` control generated values;
the `extremes` pattern deliberately uses `INT_MIN`, `INT_MAX`, 0, -1, and 1.
The CLI derives counting-sort bounds from the actual input.

CSV measurements are arithmetic means after one verified warm-up:

| Column | Measurement |
| --- | --- |
| `sort_ms` | Host wall time of the blocking sort call, including temporary allocations, frees, launches, library work, and quicksort's host control |
| `end_to_end_ms` | Upload + synchronization + sort + download + synchronization |
| `cpu_sort_ms` | One CPU `std::sort` on a separate copy, for context |
| `million_keys_s` | N / (1000 × `sort_ms`) |
| `status`, `reason` | Verified PASS, or an explicit SKIP and explanation |

Host buffers are pageable. The outer device input allocation, host input
generation, host result-buffer allocation, CPU reference generation, and result
comparison are excluded from both GPU timing columns. GPU context creation is
outside timing. These timings measure whole operations; they are not pure
kernel times or a controlled CPU-versus-GPU speedup study. Use larger repeat
counts and record hardware, clocks, driver, toolkit, compiler, and architecture
for comparisons. The executable prints GPU and CUDA version metadata to stderr.

Slow educational variants have CLI size guards. `--allow-slow` overrides those
guards; it does not override the counting-bin cap or the 2^26 input cap. A
completely skipped request exits with code 2; incorrect output or runtime errors
exit with code 1; successful verification exits with code 0.

## Project map

| File | Purpose |
| --- | --- |
| `src/sorts.cu` | Custom kernels, hierarchical scan, dispatch, Thrust/CUB baselines |
| `src/main.cu` | CLI, regression suite, verification, benchmark CSV |
| `include/cudasort/core.hpp` | Host/device classic routines and shared arithmetic |
| `include/cudasort/algorithms.hpp` | Algorithm registry and educational limits |
| `include/cudasort/input.hpp` | Reproducible input patterns |
| `examples/basic.cu` | Complete API usage |
| `tests/cpu_tests.cpp` | Actual shared-routine checks against `std::sort` |
| `tests/test_models.py` | Parallel algorithm and scan index models |
| `scripts/benchmark.py` | Real GPU benchmark sweep and CSV collection |
| `docs/ALGORITHMS.md` | Walkthroughs and correctness invariants |
| `VALIDATION.md` | Validation actually performed for this package |

## NVIDIA references

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUB DeviceRadixSort source and API comments](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh)
- [Thrust execution systems and stream policies](https://nvidia.github.io/cccl/unstable/thrust/developer/systems.html)
- [CUDA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

MIT license; see [LICENSE](LICENSE).
