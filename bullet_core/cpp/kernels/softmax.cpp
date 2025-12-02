// Softmax - Numerically stable implementation with SIMD
// output = exp(input - max(input)) / sum(exp(input - max(input)))

#include "../utils.h"
#include <immintrin.h>
#include <omp.h>
#include <cmath>

namespace bullet {

// Softmax for a single row (SIMD-optimized)
static void softmax_row_f32(const float* input, float* output, int dim) {
    // Step 1: Find max for numerical stability
    float max_val = vector_max(input, dim);
    
    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < dim; ++i) {
        float exp_val = fast_exp(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    #pragma omp simd
    for (int i = 0; i < dim; ++i) {
        output[i] *= inv_sum;
    }
}

// Batch softmax (process multiple rows)
void softmax_f32(const float* input, float* output, int batch, int dim) {
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        const float* in_row = input + b * dim;
        float* out_row = output + b * dim;
        softmax_row_f32(in_row, out_row, dim);
    }
}

} // namespace bullet
