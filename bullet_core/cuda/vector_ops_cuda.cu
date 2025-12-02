// Vector operations CUDA kernels
// Element-wise operations

#include "cuda_utils.cuh"

// Vector addition: out = a + b
__global__ void vector_add_kernel_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" void vector_add_cuda_f32(
    const float* a, const float* b, float* out, int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    vector_add_kernel_f32<<<grid_size, block_size>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
}

// Vector multiplication: out = a * b
__global__ void vector_mul_kernel_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

extern "C" void vector_mul_cuda_f32(
    const float* a, const float* b, float* out, int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    vector_mul_kernel_f32<<<grid_size, block_size>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
}

// Scalar multiplication: out = a * scale
__global__ void vector_scale_kernel_f32(
    const float* __restrict__ a,
    float scale,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scale;
    }
}

extern "C" void vector_scale_cuda_f32(
    const float* a, float scale, float* out, int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    vector_scale_kernel_f32<<<grid_size, block_size>>>(a, scale, out, size);
    CUDA_CHECK(cudaGetLastError());
}

// ReLU activation: out = max(0, x)
__global__ void vector_relu_kernel_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

extern "C" void vector_relu_cuda_f32(
    const float* input, float* output, int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    vector_relu_kernel_f32<<<grid_size, block_size>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}
