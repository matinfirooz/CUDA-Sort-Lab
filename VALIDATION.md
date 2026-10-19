# Validation performed

Date: 2026-10-19.

## Environment

| Component | Version / setting |
| --- | --- |
| Host | Linux x86-64 |
| Host compiler | GNU g++ 13.3.0 |
| CUDA compiler | NVCC 13.0.88 |
| CUDA runtime package | 13.0.96 |
| CUDA CCCL package | 13.0.85 |
| Language | C++17 / CUDA C++17 |
| CUDA target | sm_75 |
| Make optimization | -O3 |
| CMake | 4.4.4, Release build |
| GPU / usable CUDA driver | Unavailable |

The CUDA compiler and headers were installed for compile-only validation.
The downloadable archive contains project source and documentation; users
build it against their own CUDA installation and chosen GPU architecture.

## Results

| Check | Result |
| --- | --- |
| Make CUDA executable build | PASS: all 20 dispatch entries compile and link |
| CMake static library build | PASS |
| CMake benchmark/CLI executable build | PASS |
| CMake API example build | PASS |
| Actual shared host/device classic routines + helpers | PASS: 95,831 CPU checks |
| Parallel algorithm / scan CPU models | PASS: 3,767 checks |
| AddressSanitizer + UndefinedBehaviorSanitizer on shared CPU routines | PASS: 95,831 checks, leak detection disabled |
| Compiled CUDA CLI argument/help/list checks | PASS: 10 checks; list contains 20 entries |
| CTest integration | PASS: 3/3 registered tests |
| Python script syntax and benchmark help | PASS |
| GPU regression suite | NOT RUN: no usable CUDA driver/device |
| NVIDIA Compute Sanitizer | NOT RUN: requires GPU execution |
| GPU timings / speedups | NOT MEASURED |

CTest ran `cpu_reference_checks`, `algorithm_models`, and `cuda_cli_list`.
The optional GPU CTest was not enabled. The CMake CLI check links the static
CUDA runtime, so listing algorithms does not require a runtime shared-library
search path or a CUDA device.

LeakSanitizer could not inspect `/proc` in this execution environment.
The host sanitizer test was rerun with `ASAN_OPTIONS=detect_leaks=0`;
address and undefined-behavior instrumentation remained enabled. This result
does not claim a leak check or any device-memory instrumentation.

A real GPU execution probe (`--algo radix --n 16`) stopped at
`cudaGetDeviceCount` with:

```text
CUDA driver version is insufficient for CUDA runtime version
```

No sorting kernel executed during that probe. Successful compilation and CPU
models do not establish GPU runtime correctness, race freedom, or performance.

## Reproduce and finish validation on a GPU machine

```bash
make ARCH=sm_75
make cpu-test
./build/cuda_sorts --self-test

cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCUDA_SORT_REGISTER_GPU_TESTS=ON
cmake --build build-cmake --parallel
ctest --test-dir build-cmake --output-on-failure

compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build/cuda_sorts --self-test
compute-sanitizer --tool racecheck --error-exitcode 1 \
  ./build/cuda_sorts --algo quick --n 4097
compute-sanitizer --tool racecheck --error-exitcode 1 \
  ./build/cuda_sorts --algo radix --n 1025
compute-sanitizer --tool synccheck --error-exitcode 1 \
  ./build/cuda_sorts --algo quick --n 4097

python3 scripts/benchmark.py --exe build/cuda_sorts \
  --output results/benchmark.csv
```

Choose your GPU's architecture and a compatible toolkit/driver. The commands
above are provided for reproduction and do not describe tests already run here.
