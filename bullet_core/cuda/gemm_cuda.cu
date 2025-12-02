// GEMM (Matrix Multiplication) CUDA kernel
// Optimized for GT 730 (CC 3.5, 384 CUDA cores, 2GB VRAM)
// Uses tiled algorithm with shared memory

#include "cuda_utils.cuh"

// Tiled matrix multiplication kernel
__global__ void gemm_kernel_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tiles (16x16 = 256 floats = 1KB per tile)
    __shared__ float As[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    __shared__ float Bs[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Output position
    int row = blockIdx.y * GEMM_TILE_SIZE + ty;
    int col = blockIdx.x * GEMM_TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    int num_tiles = (K + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A into shared memory
        int a_col = t * GEMM_TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B into shared memory
        int b_row = t * GEMM_TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < GEMM_TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch GEMM kernel
extern "C" void gemm_cuda_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    // Configure kernel launch
    dim3 block_size(GEMM_TILE_SIZE, GEMM_TILE_SIZE);
    dim3 grid_size(
        (N + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE,
        (M + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE
    );
    
    // Launch kernel
    gemm_kernel_f32<<<grid_size, block_size>>>(A, B, C, M, N, K);
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
}

// Transposed variants
__global__ void gemm_tn_kernel_f32(
    const float* __restrict__ A,  // Transposed
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    __shared__ float Bs[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * GEMM_TILE_SIZE + ty;
    int col = blockIdx.x * GEMM_TILE_SIZE + tx;
    
    float sum = 0.0f;
    int num_tiles = (K + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A^T (swap indices)
        int a_row = t * GEMM_TILE_SIZE + tx;
        if (row < M && a_row < K) {
            As[ty][tx] = A[a_row * M + row];  // Transposed access
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        int b_row = t * GEMM_TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < GEMM_TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void gemm_tn_cuda_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    dim3 block_size(GEMM_TILE_SIZE, GEMM_TILE_SIZE);
    dim3 grid_size(
        (N + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE,
        (M + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE
    );
    
    gemm_tn_kernel_f32<<<grid_size, block_size>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void gemm_nt_kernel_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,  // Transposed
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    __shared__ float Bs[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * GEMM_TILE_SIZE + ty;
    int col = blockIdx.x * GEMM_TILE_SIZE + tx;
    
    float sum = 0.0f;
    int num_tiles = (K + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A
        int a_col = t * GEMM_TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B^T (swap indices)
        int b_col = t * GEMM_TILE_SIZE + ty;
        if (col < N && b_col < K) {
            Bs[ty][tx] = B[col * K + b_col];  // Transposed access
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < GEMM_TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void gemm_nt_cuda_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    dim3 block_size(GEMM_TILE_SIZE, GEMM_TILE_SIZE);
    dim3 grid_size(
        (N + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE,
        (M + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE
    );
    
    gemm_nt_cuda_f32<<<grid_size, block_size>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
