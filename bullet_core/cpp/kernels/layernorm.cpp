// LayerNorm (Standard Layer Normalization)
// output = ((input - mean) / sqrt(var + eps)) * weight + bias

#include "../utils.h"
#include <immintrin.h>
#include <omp.h>
#include <cmath>

namespace bullet {

// LayerNorm for a single row
static void layernorm_row_f32(
    const float* input, const float* weight, const float* bias,
    float* output, int dim, float eps
) {
    // Step 1: Compute mean
    float sum = vector_sum(input, dim);
    float mean = sum / dim;
    
    // Step 2: Compute variance
    float sum_sq_diff = 0.0f;
    #pragma omp simd reduction(+:sum_sq_diff)
    for (int i = 0; i < dim; ++i) {
        float diff = input[i] - mean;
        sum_sq_diff += diff * diff;
    }
    float variance = sum_sq_diff / dim;
    
    // Step 3: Normalize, scale, and shift
    float inv_std = 1.0f / std::sqrt(variance + eps);
    
    #pragma omp simd
    for (int i = 0; i < dim; ++i) {
        float normalized = (input[i] - mean) * inv_std;
        output[i] = normalized * weight[i] + bias[i];
    }
}

// Batch LayerNorm
void layernorm_f32(
    const float* input, const float* weight, const float* bias,
    float* output, int batch, int dim, float eps
) {
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        const float* in_row = input + b * dim;
        float* out_row = output + b * dim;
        layernorm_row_f32(in_row, weight, bias, out_row, dim, eps);
    }
}

} // namespace bullet
