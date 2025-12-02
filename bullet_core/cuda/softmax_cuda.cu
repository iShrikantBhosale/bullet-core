// Softmax CUDA kernel
// Numerically stable implementation with reduction

#include "cuda_utils.cuh"

// Softmax kernel for each row
__global__ void softmax_kernel_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int dim
) {
    int b = blockIdx.x;
    if (b >= batch) return;
    
    const float* in_row = input + b * dim;
    float* out_row = output + b * dim;
    
    // Shared memory for reductions
    __shared__ float shared_data[SOFTMAX_BLOCK_SIZE];
    
    // Step 1: Find max (for numerical stability)
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    
    // Reduce max across block
    shared_data[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = shared_data[0];
    
    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float exp_val = expf(in_row[i] - max_val);
        out_row[i] = exp_val;
        sum += exp_val;
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
    sum = shared_data[0];
    
    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] *= inv_sum;
    }
}

extern "C" void softmax_cuda_f32(
    const float* input, float* output,
    int batch, int dim
) {
    dim3 block_size(SOFTMAX_BLOCK_SIZE);
    dim3 grid_size(batch);
    
    softmax_kernel_f32<<<grid_size, block_size>>>(input, output, batch, dim);
    CUDA_CHECK(cudaGetLastError());
}
