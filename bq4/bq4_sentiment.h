// bq4_sentiment.h
// Sentiment Classification Head
// Bullet OS - Multi-Task Hybrid AI

#ifndef BQ4_SENTIMENT_H
#define BQ4_SENTIMENT_H

#include "bq4_transformer.h"
#include "bq4_tokenizer.h"

namespace bullet {
namespace bq4 {

// Forward declaration not needed if we include header
// class Tokenizer;

// ============================================================================
// Sentiment Head
// ============================================================================

struct SentimentHead {
    Tensor W_sentiment;  // [num_labels x hidden_dim] (BQ4)
    std::vector<float> bias;  // [num_labels]
    int num_labels;  // typically 3: negative, neutral, positive
};

// ============================================================================
// Sentiment Classification
// ============================================================================
// Returns: 0 = negative, 1 = neutral, 2 = positive

inline int classify_sentiment(
    const std::vector<int>& tokens,
    const BulletModel& model,
    const SentimentHead& sentiment_head,
    float* hidden_buf,
    float* scratch1,
    float* scratch2,
    float* scratch3,
    float* Q_buf,
    float* K_buf,
    float* V_buf,
    float* scores_buf,
    float* logits
) {
    // Initialize KV caches
    std::vector<KVCache> kv_caches;
    for (int i = 0; i < model.num_layers; i++) {
        kv_caches.emplace_back(model.max_seq, model.head_dim);
    }
    
    // 1. Run transformer on all tokens
    for (size_t i = 0; i < tokens.size(); i++) {
        forward_pass(
            tokens[i], i, model, kv_caches,
            logits,  // temp buffer
            hidden_buf,
            scratch1, scratch2, scratch3,
            Q_buf, K_buf, V_buf, scores_buf
        );
    }
    
    // 2. Use final hidden state for classification
    // (hidden_buf now contains the last token's representation)
    
    // 3. Apply sentiment head
    std::vector<float> sentiment_logits(sentiment_head.num_labels);
    matmul_bq4(hidden_buf, sentiment_head.W_sentiment, 
               sentiment_logits.data(), sentiment_head.bias.data());
    
    // 4. Argmax to get predicted class
    int predicted_class = 0;
    float max_logit = sentiment_logits[0];
    
    for (int i = 1; i < sentiment_head.num_labels; i++) {
        if (sentiment_logits[i] > max_logit) {
            max_logit = sentiment_logits[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}

// ============================================================================
// Convenience wrapper with tokenizer
// ============================================================================

inline std::string sentiment_analysis(
    const std::string& text,
    const BulletModel& model,
    const SentimentHead& sentiment_head,
    const Tokenizer& tokenizer
) {
    // Encode text
    std::vector<int> tokens = tokenizer.encode(text);
    
    // Allocate buffers
    int hidden_dim = model.hidden_dim;
    int vocab_size = model.vocab_size;
    
    std::vector<float> hidden(hidden_dim);
    std::vector<float> scratch1(hidden_dim);
    std::vector<float> scratch2(hidden_dim);
    std::vector<float> scratch3(hidden_dim);
    std::vector<float> Q_buf(model.head_dim);
    std::vector<float> K_buf(model.head_dim);
    std::vector<float> V_buf(model.head_dim);
    std::vector<float> scores_buf(model.max_seq);
    std::vector<float> logits(vocab_size);
    
    // Classify
    int sentiment = classify_sentiment(
        tokens, model, sentiment_head,
        hidden.data(), scratch1.data(), scratch2.data(), scratch3.data(),
        Q_buf.data(), K_buf.data(), V_buf.data(),
        scores_buf.data(), logits.data()
    );
    
    // Map to label
    const char* labels[] = {"NEGATIVE", "NEUTRAL", "POSITIVE"};
    return labels[sentiment];
}

} // namespace bq4
} // namespace bullet

#endif // BQ4_SENTIMENT_H
