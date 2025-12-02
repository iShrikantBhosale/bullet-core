#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

// Common utilities for Bullet-Core CPU kernels

namespace bullet {

// Alignment for SIMD operations
constexpr size_t SIMD_ALIGN = 32;

// Helper to align pointers
template<typename T>
inline bool is_aligned(const T* ptr, size_t alignment = SIMD_ALIGN) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

// Safe division
inline float safe_div(float a, float b, float eps = 1e-8f) {
    return a / (b + eps);
}

// Fast approximate exp (for softmax)
inline float fast_exp(float x) {
    // Clamp to prevent overflow
    x = std::max(-88.0f, std::min(88.0f, x));
    return std::exp(x);
}

// Vector sum (used in normalization)
inline float vector_sum(const float* data, int size) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

// Vector sum of squares (used in RMSNorm)
inline float vector_sum_squares(const float* data, int size) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += data[i] * data[i];
    }
    return sum;
}

// Find max value (for numerical stability in softmax)
inline float vector_max(const float* data, int size) {
    float max_val = data[0];
    #pragma omp simd reduction(max:max_val)
    for (int i = 1; i < size; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    return max_val;
}

} // namespace bullet
