// RMSNorm (Root Mean Square Normalization) - Used in LLaMA/BULLET models
// output = (input / rms(input)) * weight
// where rms(input) = sqrt(mean(input^2) + eps)

#include "../utils.h"
#include <immintrin.h>
#include <omp.h>
#include <cmath>

namespace bullet {

// RMSNorm for a single row
static void rmsnorm_row_f32(
    const float* input, const float* weight,
    float* output, int dim, float eps
) {
    // Step 1: Compute sum of squares
    float sum_sq = vector_sum_squares(input, dim);
    
    // Step 2: Compute RMS
    float rms = std::sqrt(sum_sq / dim + eps);
    float inv_rms = 1.0f / rms;
    
    // Step 3: Normalize and scale by weight
    #pragma omp simd
    for (int i = 0; i < dim; ++i) {
        output[i] = (input[i] * inv_rms) * weight[i];
    }
}

// Batch RMSNorm
void rmsnorm_f32(
    const float* input, const float* weight,
    float* output, int batch, int dim, float eps
) {
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        const float* in_row = input + b * dim;
        float* out_row = output + b * dim;
        rmsnorm_row_f32(in_row, weight, out_row, dim, eps);
    }
}

} // namespace bullet
