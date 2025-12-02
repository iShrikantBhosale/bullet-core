// LayerNorm CUDA kernel
// Standard layer normalization

#include "cuda_utils.cuh"

__global__ void layernorm_kernel_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int dim, float eps
) {
    int b = blockIdx.x;
    if (b >= batch) return;
    
    const float* in_row = input + b * dim;
    float* out_row = output + b * dim;
    
    __shared__ float shared_data[NORM_BLOCK_SIZE];
    
    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += in_row[i];
    }
    
    // Reduce sum
    shared_data[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = shared_data[0] / dim;
    
    // Compute variance
    float sum_sq_diff = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float diff = in_row[i] - mean;
        sum_sq_diff += diff * diff;
    }
    
    // Reduce variance
    shared_data[threadIdx.x] = sum_sq_diff;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    float variance = shared_data[0] / dim;
    
    // Normalize, scale, and shift
    float inv_std = 1.0f / sqrtf(variance + eps);
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float normalized = (in_row[i] - mean) * inv_std;
        out_row[i] = normalized * weight[i] + bias[i];
    }
}

extern "C" void layernorm_cuda_f32(
    const float* input, const float* weight, const float* bias, float* output,
    int batch, int dim, float eps
) {
    dim3 block_size(NORM_BLOCK_SIZE);
    dim3 grid_size(batch);
    
    layernorm_kernel_f32<<<grid_size, block_size>>>(input, weight, bias, output, batch, dim, eps);
    CUDA_CHECK(cudaGetLastError());
}
