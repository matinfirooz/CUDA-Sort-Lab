#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace cudasort {

inline void cuda_check(cudaError_t result, const char* expression,
                       const char* file, int line) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
            " " + expression + ": " + cudaGetErrorString(result));
}

#define CS_CUDA(expr) ::cudasort::cuda_check((expr), #expr, __FILE__, __LINE__)

template<class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        if (count) CS_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) (void)cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

class Stream {
public:
    Stream() { CS_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { (void)cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    operator cudaStream_t() const { return stream_; }
private:
    cudaStream_t stream_{};
};

}  // namespace cudasort
