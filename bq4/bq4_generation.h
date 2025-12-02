// bq4_generation.h
// Token-by-Token Generation Loop
// Bullet OS - Autoregressive Text Generation

#ifndef BQ4_GENERATION_H
#define BQ4_GENERATION_H

#include "bq4_transformer.h"
#include <random>

namespace bullet {
namespace bq4 {

// ============================================================================
// Sampling Functions
// ============================================================================

inline int sample_token(
    const float* logits,
    int vocab_size,
    float temperature = 0.7f,
    int top_k = 40,
    float top_p = 0.9f
) {
    // Copy logits for modification
    std::vector<float> probs(logits, logits + vocab_size);
    
    // Apply temperature
    if (temperature > 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= temperature;
        }
    }
    
    // Softmax
    softmax(probs.data(), vocab_size);
    
    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
        std::vector<std::pair<float, int>> sorted;
        for (int i = 0; i < vocab_size; i++) {
            sorted.push_back({probs[i], i});
        }
        std::sort(sorted.begin(), sorted.end(), 
                  [](auto& a, auto& b) { return a.first > b.first; });
        
        float cutoff = sorted[top_k].first;
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] < cutoff) probs[i] = 0.0f;
        }
        
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += probs[i];
        for (int i = 0; i < vocab_size; i++) probs[i] /= sum;
    }
    
    // Sample from distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float r = dist(gen);
    float cumsum = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    
    return vocab_size - 1;  // fallback
}

// ============================================================================
// Generation Loop
// ============================================================================

inline std::vector<int> generate(
    const std::vector<int>& prompt_tokens,
    const BulletModel& model,
    int max_new_tokens = 50,
    float temperature = 0.7f,
    int top_k = 40,
    float top_p = 0.9f,
    int eos_token = 0
) {
    std::vector<int> tokens = prompt_tokens;
    
    // Allocate buffers
    std::vector<float> logits(model.vocab_size);
    std::vector<float> hidden(model.hidden_dim);
    std::vector<float> scratch1(model.hidden_dim);
    std::vector<float> scratch2(model.hidden_dim);
    std::vector<float> scratch3(model.hidden_dim);
    std::vector<float> Q_buf(model.head_dim);
    std::vector<float> K_buf(model.head_dim);
    std::vector<float> V_buf(model.head_dim);
    std::vector<float> scores_buf(model.max_seq);
    
    // Initialize KV caches
    std::vector<KVCache> kv_caches;
    for (int i = 0; i < model.num_layers; i++) {
        kv_caches.emplace_back(model.max_seq, model.head_dim);
    }
    
    // Process prompt
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        forward_pass(
            prompt_tokens[i], i, model, kv_caches,
            logits.data(), hidden.data(),
            scratch1.data(), scratch2.data(), scratch3.data(),
            Q_buf.data(), K_buf.data(), V_buf.data(),
            scores_buf.data()
        );
    }
    
    // Generate new tokens
    for (int i = 0; i < max_new_tokens; i++) {
        int pos = tokens.size();
        
        if (pos >= model.max_seq) break;
        
        // Forward pass
        forward_pass(
            tokens.back(), pos, model, kv_caches,
            logits.data(), hidden.data(),
            scratch1.data(), scratch2.data(), scratch3.data(),
            Q_buf.data(), K_buf.data(), V_buf.data(),
            scores_buf.data()
        );
        
        // Sample next token
        int next_token = sample_token(
            logits.data(), model.vocab_size,
            temperature, top_k, top_p
        );
        
        tokens.push_back(next_token);
        
        // Stop on EOS
        if (next_token == eos_token) break;
    }
    
    return tokens;
}

} // namespace bq4
} // namespace bullet

#endif // BQ4_GENERATION_H
