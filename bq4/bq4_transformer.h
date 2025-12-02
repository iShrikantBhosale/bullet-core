// bq4_transformer.h
// Full Transformer Block (Attention + MLP)
// Bullet OS - Complete GPT Architecture

#ifndef BQ4_TRANSFORMER_H
#define BQ4_TRANSFORMER_H

#include "bq4_attention.h"
#include "bq4_inference.h"

namespace bullet {
namespace bq4 {

// ============================================================================
// Transformer Block Weights
// ============================================================================

struct TransformerBlock {
    // Layer Norm 1
    std::vector<float> ln1_weight;
    std::vector<float> ln1_bias;
    
    // Attention
    Tensor Wq, Wk, Wv;  // Q/K/V projections (BQ4)
    Tensor Wo;          // Output projection (BQ4)
    std::vector<float> attn_bias;
    
    // Layer Norm 2
    std::vector<float> ln2_weight;
    std::vector<float> ln2_bias;
    
    // MLP
    Tensor W_gate;      // Gate projection (BQ4)
    Tensor W_up;        // Up projection (BQ4)
    Tensor W_down;      // Down projection (BQ4)
    std::vector<float> mlp_bias;
};

// ============================================================================
// Full Transformer Block Forward Pass
// ============================================================================

inline void transformer_block_forward(
    float* x,                      // input/output [hidden_dim]
    const TransformerBlock& block,
    KVCache& kv_cache,
    float* scratch1,               // [hidden_dim]
    float* scratch2,               // [hidden_dim]
    float* scratch3,               // [hidden_dim]
    float* Q_buf,                  // [head_dim]
    float* K_buf,                  // [head_dim]
    float* V_buf,                  // [head_dim]
    float* scores_buf,             // [max_seq]
    int hidden_dim,
    int head_dim
) {
    // 1. Layer Norm 1
    layernorm(x, block.ln1_weight.data(), block.ln1_bias.data(), 
              scratch1, hidden_dim);
    
    // 2. Self-Attention
    AttentionWeights attn_w = {block.Wq, block.Wk, block.Wv};
    
    attention_with_cache(
        scratch1,
        attn_w,
        kv_cache,
        scratch2,      // attention output
        Q_buf, K_buf, V_buf, scores_buf,
        head_dim
    );
    
    // 3. Attention output projection
    matmul_bq4(scratch2, block.Wo, scratch3, block.attn_bias.data());
    
    // 4. Residual connection 1
    for (int i = 0; i < hidden_dim; i++) {
        x[i] += scratch3[i];
    }
    
    // 5. Layer Norm 2
    layernorm(x, block.ln2_weight.data(), block.ln2_bias.data(),
              scratch1, hidden_dim);
    
    // 6. MLP: Gate
    matmul_bq4(scratch1, block.W_gate, scratch2);
    gelu_vec(scratch2, hidden_dim);
    
    // 7. MLP: Up
    matmul_bq4(scratch1, block.W_up, scratch3);
    
    // 8. Element-wise multiply (SwiGLU-style)
    for (int i = 0; i < hidden_dim; i++) {
        scratch2[i] *= scratch3[i];
    }
    
    // 9. MLP: Down
    matmul_bq4(scratch2, block.W_down, scratch3, block.mlp_bias.data());
    
    // 10. Residual connection 2
    for (int i = 0; i < hidden_dim; i++) {
        x[i] += scratch3[i];
    }
}

// ============================================================================
// Full Model Structure
// ============================================================================

struct BulletModel {
    // Embeddings
    std::vector<float> token_embedding;  // [vocab_size x hidden_dim]
    std::vector<float> pos_embedding;    // [max_seq x hidden_dim]
    
    // Transformer blocks
    std::vector<TransformerBlock> blocks;
    
    // Final layer norm
    std::vector<float> ln_final_weight;
    std::vector<float> ln_final_bias;
    
    // LM head
    Tensor lm_head;  // [vocab_size x hidden_dim] (BQ4)
    
    // Model config
    int vocab_size;
    int hidden_dim;
    int num_layers;
    int num_heads;
    int head_dim;
    int max_seq;
    
    BulletModel() {}
};

// ============================================================================
// Full Forward Pass (Single Token)
// ============================================================================

inline void forward_pass(
    int token_id,
    int pos,
    const BulletModel& model,
    std::vector<KVCache>& kv_caches,
    float* logits,                 // output [vocab_size]
    float* hidden,                 // scratch [hidden_dim]
    float* scratch1,
    float* scratch2,
    float* scratch3,
    float* Q_buf,
    float* K_buf,
    float* V_buf,
    float* scores_buf
) {
    int hidden_dim = model.hidden_dim;
    int head_dim = model.head_dim;
    
    // 1. Token + Position Embedding
    const float* tok_emb = model.token_embedding.data() + token_id * hidden_dim;
    const float* pos_emb = model.pos_embedding.data() + pos * hidden_dim;
    
    for (int i = 0; i < hidden_dim; i++) {
        hidden[i] = tok_emb[i] + pos_emb[i];
    }
    
    // 2. Transformer Blocks
    for (int layer = 0; layer < model.num_layers; layer++) {
        transformer_block_forward(
            hidden,
            model.blocks[layer],
            kv_caches[layer],
            scratch1, scratch2, scratch3,
            Q_buf, K_buf, V_buf, scores_buf,
            hidden_dim, head_dim
        );
    }
    
    // 3. Final Layer Norm
    layernorm(hidden, model.ln_final_weight.data(), 
              model.ln_final_bias.data(), scratch1, hidden_dim);
    
    // 4. LM Head
    matmul_bq4(scratch1, model.lm_head, logits);
}

} // namespace bq4
} // namespace bullet

#endif // BQ4_TRANSFORMER_H
