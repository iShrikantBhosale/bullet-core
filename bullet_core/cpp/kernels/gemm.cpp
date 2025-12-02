// GEMM (General Matrix Multiply) - SIMD-optimized
// C = A @ B (with optional transposes)
// Optimized for CPU with AVX/SSE4.2 and OpenMP

#include "../utils.h"
#include <immintrin.h>  // AVX/SSE intrinsics
#include <omp.h>
#include <cstring>

namespace bullet {

// Cache blocking parameters (tuned for typical L1/L2 cache)
constexpr int BLOCK_SIZE = 64;
constexpr int SIMD_WIDTH = 8;  // AVX processes 8 floats at once

// Kernel for small matrix blocks (cache-friendly)
static void gemm_kernel_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    for (int i = 0; i < M; ++i) {
        int j = 0;
        for (; j <= N - SIMD_WIDTH; j += SIMD_WIDTH) {
            __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);
            
            for (int k = 0; k < K; ++k) {
                __m256 a_vec = _mm256_set1_ps(A[i * lda + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * ldb + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            }
            
            _mm256_storeu_ps(&C[i * ldc + j], c_vec);
        }
        
        // Handle remaining elements
        for (int j = (N / SIMD_WIDTH) * SIMD_WIDTH; j < N; ++j) {
            float sum = C[i * ldc + j];
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// Main GEMM function with cache blocking
void gemm_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    bool transpose_A, bool transpose_B
) {
    // Initialize C to zero
    std::memset(C, 0, M * N * sizeof(float));
    
    if (!transpose_A && !transpose_B) {
        // C = A @ B (most common case)
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i += BLOCK_SIZE) {
            for (int j = 0; j < N; j += BLOCK_SIZE) {
                for (int k = 0; k < K; k += BLOCK_SIZE) {
                    int block_m = std::min(BLOCK_SIZE, M - i);
                    int block_n = std::min(BLOCK_SIZE, N - j);
                    int block_k = std::min(BLOCK_SIZE, K - k);
                    
                    gemm_kernel_f32(
                        &A[i * K + k], &B[k * N + j], &C[i * N + j],
                        block_m, block_n, block_k,
                        K, N, N
                    );
                }
            }
        }
    } else if (transpose_A && !transpose_B) {
        // C = A.T @ B
        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < K; ++k) {
                    sum += A[k * M + i] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    } else if (!transpose_A && transpose_B) {
        // C = A @ B.T
        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[j * K + k];
                }
                C[i * N + j] = sum;
            }
        }
    } else {
        // C = A.T @ B.T
        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < K; ++k) {
                    sum += A[k * M + i] * B[j * K + k];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

} // namespace bullet
