// Common CUDA utilities for Bullet-Core
// Optimized for GT 730 (CC 3.5, 2GB VRAM)

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while(0)

// GT 730 constraints (CC 3.5)
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int SHARED_MEMORY_SIZE = 48 * 1024;  // 48KB
constexpr int WARP_SIZE = 32;

// Tile sizes for different operations
constexpr int GEMM_TILE_SIZE = 16;  // 16x16 tiles for matrix multiply
constexpr int SOFTMAX_BLOCK_SIZE = 256;
constexpr int NORM_BLOCK_SIZE = 256;

// Helper: Get optimal grid size
inline dim3 get_grid_size(int total_threads, int block_size) {
    int num_blocks = (total_threads + block_size - 1) / block_size;
    return dim3(num_blocks);
}

// Helper: Get 2D grid size for matrices
inline dim3 get_grid_2d(int rows, int cols, int tile_size) {
    int grid_x = (cols + tile_size - 1) / tile_size;
    int grid_y = (rows + tile_size - 1) / tile_size;
    return dim3(grid_x, grid_y);
}

// Device query
inline bool is_cuda_available() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

// Get device properties
inline cudaDeviceProp get_device_properties(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}
