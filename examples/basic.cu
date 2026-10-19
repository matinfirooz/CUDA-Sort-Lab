#include "cudasort/sort.cuh"
#include "cudasort/cuda_utils.cuh"

#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    try {
        std::vector<int> input{7, -4, 0, 7, 3, -19, 8, 1, 2};
        auto expected = input;
        std::sort(expected.begin(), expected.end());
        cudasort::DeviceBuffer<int> device(input.size());
        cudasort::Stream stream;
        CS_CUDA(cudaMemcpyAsync(device.get(), input.data(), input.size() * sizeof(int),
                                cudaMemcpyHostToDevice, stream));
        cudasort::sort(device.get(), input.size(), cudasort::Algorithm::Radix, {}, stream);
        CS_CUDA(cudaMemcpyAsync(input.data(), device.get(), input.size() * sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
        CS_CUDA(cudaStreamSynchronize(stream));
        if (input != expected) throw std::runtime_error("Result mismatches std::sort");
        for (int value : input) std::cout << value << ' ';
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
