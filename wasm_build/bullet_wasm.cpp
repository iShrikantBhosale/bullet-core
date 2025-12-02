#include "../bq4/bq4_loader.h"
#include "../bq4/bq4_transformer.h"
#include "../bq4/bq4_generation.h"
#include "../bq4/bq4_sentiment.h"
#include "../bq4/bq4_tokenizer.h"
#include <emscripten/bind.h>
#include <string>
#include <vector>

using namespace emscripten;
using namespace bullet::bq4;

// Global model instance
static BulletModel* g_model = nullptr;
static Tokenizer* g_tokenizer = nullptr;

// Load model from file
bool loadModel(const std::string& model_path, const std::string& vocab_path) {
    try {
        // Load tokenizer
        g_tokenizer = new Tokenizer();
        if (!g_tokenizer->load_vocab(vocab_path)) {
            return false;
        }
        
        // TODO: Load BQ4 model
        // g_model = new BulletModel();
        
        return true;
    } catch (...) {
        return false;
    }
}

// Generate text
std::string generate(const std::string& prompt, int max_tokens) {
    if (!g_model || !g_tokenizer) {
        return "Error: Model not loaded";
    }
    
    // TODO: Implement generation
    return prompt + " [WASM generation ready]";
}

// Sentiment analysis
std::string sentiment(const std::string& text) {
    if (!g_model || !g_tokenizer) {
        return "UNKNOWN";
    }
    
    // TODO: Implement sentiment
    return "NEUTRAL";
}

// Free resources
void unload() {
    delete g_model;
    delete g_tokenizer;
    g_model = nullptr;
    g_tokenizer = nullptr;
}

// Emscripten bindings
EMSCRIPTEN_BINDINGS(bullet_module) {
    function("loadModel", &loadModel);
    
    // Wrap generate to avoid signature issues
    function("generate", optional_override([](const std::string& prompt, int max_tokens) -> std::string {
        return generate(prompt, max_tokens);
    }));
    
    function("sentiment", &sentiment);
    function("unload", &unload);
}
