// Vector Operations - Element-wise SIMD operations
// Basic building blocks for neural networks

#include "../utils.h"
#include <immintrin.h>
#include <omp.h>

namespace bullet {

// Vector addition: out = a + b
void vector_add_f32(const float* a, const float* b, float* out, int size) {
    int i = 0;
    
    // Process 8 elements at a time with AVX
    for (; i + 7 < size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(&out[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        out[i] = a[i] + b[i];
    }
}

// Vector multiplication: out = a * b
void vector_mul_f32(const float* a, const float* b, float* out, int size) {
    int i = 0;
    
    // Process 8 elements at a time with AVX
    for (; i + 7 < size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(&out[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        out[i] = a[i] * b[i];
    }
}

// Scalar multiplication: out = a * scale
void vector_scale_f32(const float* a, float scale, float* out, int size) {
    int i = 0;
    __m256 scale_vec = _mm256_set1_ps(scale);
    
    // Process 8 elements at a time with AVX
    for (; i + 7 < size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 result = _mm256_mul_ps(a_vec, scale_vec);
        _mm256_storeu_ps(&out[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        out[i] = a[i] * scale;
    }
}

// Vector subtraction: out = a - b
void vector_sub_f32(const float* a, const float* b, float* out, int size) {
    int i = 0;
    
    // Process 8 elements at a time with AVX
    for (; i + 7 < size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(&out[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        out[i] = a[i] - b[i];
    }
}

// ReLU activation: out = max(0, x)
void vector_relu_f32(const float* input, float* output, int size) {
    int i = 0;
    __m256 zero = _mm256_setzero_ps();
    
    // Process 8 elements at a time with AVX
    for (; i + 7 < size; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 result = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

} // namespace bullet
