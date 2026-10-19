NVCC ?= nvcc
CXX ?= g++
ARCH ?= sm_75
CUDAFLAGS ?= -O3 -std=c++17
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: all cpu-test test clean
all: build/cuda_sorts

build:
	mkdir -p build

build/cuda_sorts: src/main.cu src/sorts.cu $(wildcard include/cudasort/*) | build
	$(NVCC) $(CUDAFLAGS) -arch=$(ARCH) -Iinclude src/main.cu src/sorts.cu -o $@

build/cpu_tests: tests/cpu_tests.cpp $(wildcard include/cudasort/*.hpp) | build
	$(CXX) $(CXXFLAGS) -Iinclude tests/cpu_tests.cpp -o $@

cpu-test: build/cpu_tests
	./build/cpu_tests
	python3 tests/test_models.py

test: all cpu-test
	./build/cuda_sorts --self-test

clean:
	rm -rf build
