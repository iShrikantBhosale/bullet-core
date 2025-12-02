// bq4_inference.h
// Fused BQ4 Inference Kernels (MatMul + Dequant)
// Bullet OS - 4-bit Quantization System

#ifndef BQ4_INFERENCE_H
#define BQ4_INFERENCE_H

#include "bq4_kernels.h"
#include "bq4_format.h"
#include <vector>
#include <cstring>

namespace bullet {
namespace bq4 {

// ============================================================================
// Fused MatMul + Dequant (Single Row)
// ============================================================================
// Computes: out = dot(activation, weight_row_bq4)
// This is the core kernel for all linear layers

inline float matmul_row_bq4(
    const float* act,              // FP32 activation [hidden_dim]
    const uint8_t* w_blocks,       // packed int4 weights for this row
    const float* scales,           // block scales for this row
    int hidden_dim,                // input dimension
    int block_size                 // BQ4 block size (usually 32)
) {
    float sum = 0.0f;
    int num_blocks = hidden_dim / block_size;
    
    for (int b = 0; b < num_blocks; b++) {
        const uint8_t* block_data = w_blocks + b * (block_size / 2);
        float scale = scales[b];
        
        // Process this block
        for (int i = 0; i < block_size / 2; i++) {
            uint8_t packed = block_data[i];
            
            // Extract two signed int4 values
            int8_t lo = (packed & 0x0F);
            int8_t hi = (packed >> 4);
            
            // Convert to signed
            if (lo > 7) lo -= 16;
            if (hi > 7) hi -= 16;
            
            int idx = b * block_size + 2*i;
            
            // Fused: dequant + dot product
            sum += act[idx + 0] * (lo * scale);
            sum += act[idx + 1] * (hi * scale);
        }
    }
    
    return sum;
}

// ============================================================================
// Full MatMul (Matrix × Vector)
// ============================================================================
// Computes: out = act * W^T
// where W is stored in BQ4 format

inline void matmul_bq4(
    const float* act,              // input [hidden_dim]
    const Tensor& W,               // BQ4 weight matrix
    float* out,                    // output [out_dim]
    const float* bias = nullptr    // optional bias [out_dim]
) {
    uint32_t hidden_dim = W.dims[1];  // input dim
    uint32_t out_dim = W.dims[0];     // output dim
    int block_size = W.block_size;
    int blocks_per_row = hidden_dim / block_size;
    
    for (uint32_t row = 0; row < out_dim; row++) {
        // Get blocks for this row
        const uint8_t* row_data = nullptr;
        float* row_scales = new float[blocks_per_row];
        
        // Extract blocks for this row
        uint32_t start_block = row * blocks_per_row;
        
        // Temporary buffer for packed data
        std::vector<uint8_t> row_packed(blocks_per_row * (block_size / 2));
        
        for (int b = 0; b < blocks_per_row; b++) {
            const Block& block = W.blocks[start_block + b];
            row_scales[b] = block.scale;
            
            // Copy packed data
            memcpy(row_packed.data() + b * (block_size / 2),
                   block.data.data(),
                   block_size / 2);
        }
        
        // Compute dot product
        out[row] = matmul_row_bq4(
            act,
            row_packed.data(),
            row_scales,
            hidden_dim,
            block_size
        );
        
        // Add bias if provided
        if (bias) out[row] += bias[row];
        
        delete[] row_scales;
    }
}

// ============================================================================
// Softmax
// ============================================================================

inline void softmax(float* x, int n) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Exp and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// ============================================================================
// LayerNorm
// ============================================================================

inline void layernorm(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int dim,
    float eps = 1e-5f
) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean += x[i];
    }
    mean /= dim;
    
    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= dim;
    
    // Normalize
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// ============================================================================
// GELU Activation
// ============================================================================

inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

inline void gelu_vec(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = gelu(x[i]);
    }
}

} // namespace bq4
} // namespace bullet

#endif // BQ4_INFERENCE_H
