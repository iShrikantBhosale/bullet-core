// RMSNorm CUDA kernel
// Root Mean Square Normalization for transformers

#include "cuda_utils.cuh"

__global__ void rmsnorm_kernel_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch, int dim, float eps
) {
    int b = blockIdx.x;
    if (b >= batch) return;
    
    const float* in_row = input + b * dim;
    float* out_row = output + b * dim;
    
    // Shared memory for reduction
    __shared__ float shared_sum[NORM_BLOCK_SIZE];
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = in_row[i];
        sum_sq += val * val;
    }
    
    // Reduce sum across block
    shared_sum[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_sq = shared_sum[0];
    
    // Compute RMS and normalize
    float rms = sqrtf(sum_sq / dim + eps);
    float inv_rms = 1.0f / rms;
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = (in_row[i] * inv_rms) * weight[i];
    }
}

extern "C" void rmsnorm_cuda_f32(
    const float* input, const float* weight, float* output,
    int batch, int dim, float eps
) {
    dim3 block_size(NORM_BLOCK_SIZE);
    dim3 grid_size(batch);
    
    rmsnorm_kernel_f32<<<grid_size, block_size>>>(input, weight, output, batch, dim, eps);
    CUDA_CHECK(cudaGetLastError());
}
