// bq4_attention.h
// Single-Head Attention Kernel for BQ4
// Bullet OS - Minimum Viable Transformer

#ifndef BQ4_ATTENTION_H
#define BQ4_ATTENTION_H

#include "bq4_inference.h"
#include <cmath>
#include <algorithm>

namespace bullet {
namespace bq4 {

// ============================================================================
// Single-Head Attention
// ============================================================================
// Computes: output = softmax(Q·K^T / sqrt(d)) · V
// This is the core of transformer self-attention

struct AttentionWeights {
    Tensor Wq;  // Query projection
    Tensor Wk;  // Key projection
    Tensor Wv;  // Value projection
};

inline void attention_single_head(
    const float* x,              // input [hidden_dim]
    const AttentionWeights& W,   // Q/K/V weights (BQ4)
    float* output,               // output [hidden_dim]
    float* Q_buf,                // scratch [head_dim]
    float* K_buf,                // scratch [head_dim]
    float* V_buf,                // scratch [head_dim]
    float* scores_buf,           // scratch [seq_len]
    int hidden_dim,
    int head_dim,
    int seq_pos = 0              // current position in sequence
) {
    // 1. Project to Q, K, V
    matmul_bq4(x, W.Wq, Q_buf);
    matmul_bq4(x, W.Wk, K_buf);
    matmul_bq4(x, W.Wv, V_buf);
    
    // 2. Compute attention scores: Q·K^T / sqrt(d)
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // For single position (simplified - no KV cache yet)
    scores_buf[0] = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        scores_buf[0] += Q_buf[i] * K_buf[i];
    }
    scores_buf[0] *= scale;
    
    // 3. Softmax (trivial for single position)
    scores_buf[0] = 1.0f;
    
    // 4. Weighted sum: output = scores·V
    for (int i = 0; i < head_dim; i++) {
        output[i] = scores_buf[0] * V_buf[i];
    }
}

// ============================================================================
// Multi-Position Attention (with KV cache)
// ============================================================================

struct KVCache {
    std::vector<float> K;  // [max_seq][head_dim]
    std::vector<float> V;  // [max_seq][head_dim]
    int max_seq;
    int head_dim;
    int current_pos;
    
    KVCache(int max_seq, int head_dim) 
        : max_seq(max_seq), head_dim(head_dim), current_pos(0) {
        K.resize(max_seq * head_dim);
        V.resize(max_seq * head_dim);
    }
    
    void add(const float* k, const float* v) {
        if (current_pos >= max_seq) return;
        
        float* k_slot = K.data() + current_pos * head_dim;
        float* v_slot = V.data() + current_pos * head_dim;
        
        memcpy(k_slot, k, head_dim * sizeof(float));
        memcpy(v_slot, v, head_dim * sizeof(float));
        
        current_pos++;
    }
    
    void reset() {
        current_pos = 0;
    }
};

inline void attention_with_cache(
    const float* x,              // input [hidden_dim]
    const AttentionWeights& W,   // Q/K/V weights (BQ4)
    KVCache& cache,              // KV cache
    float* output,               // output [head_dim]
    float* Q_buf,                // scratch [head_dim]
    float* K_buf,                // scratch [head_dim]
    float* V_buf,                // scratch [head_dim]
    float* scores_buf,           // scratch [max_seq]
    int head_dim
) {
    // 1. Project to Q, K, V
    matmul_bq4(x, W.Wq, Q_buf);
    matmul_bq4(x, W.Wk, K_buf);
    matmul_bq4(x, W.Wv, V_buf);
    
    // 2. Add K, V to cache
    cache.add(K_buf, V_buf);
    
    // 3. Compute attention scores for all cached positions
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int t = 0; t < cache.current_pos; t++) {
        const float* k_t = cache.K.data() + t * head_dim;
        
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += Q_buf[i] * k_t[i];
        }
        scores_buf[t] = score * scale;
    }
    
    // 4. Softmax over positions
    softmax(scores_buf, cache.current_pos);
    
    // 5. Weighted sum of values
    memset(output, 0, head_dim * sizeof(float));
    
    for (int t = 0; t < cache.current_pos; t++) {
        const float* v_t = cache.V.data() + t * head_dim;
        float weight = scores_buf[t];
        
        for (int i = 0; i < head_dim; i++) {
            output[i] += weight * v_t[i];
        }
    }
}

} // namespace bq4
} // namespace bullet

#endif // BQ4_ATTENTION_H
