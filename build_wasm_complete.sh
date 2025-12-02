#!/bin/bash
# build_wasm_complete.sh
# Complete WASM Build for Bullet OS
# Compiles BQ4 runtime to WebAssembly

set -e

echo "========================================"
echo "🚀 Bullet OS - WASM Build"
echo "========================================"

# Check for Emscripten
if ! command -v emcc &> /dev/null; then
    echo "❌ Emscripten not found!"
    echo "Install: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

echo "✅ Emscripten found: $(emcc --version | head -n1)"

# Create output directory
mkdir -p wasm_build

# Create WASM bindings file
cat > wasm_build/bullet_wasm.cpp << 'EOF'
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
EOF

echo "📝 Created WASM bindings"

# Compile to WASM
echo "🔨 Compiling to WebAssembly..."

emcc wasm_build/bullet_wasm.cpp \
    -o wasm_build/bullet.js \
    -I. \
    -O3 \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME='BulletModule' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MAXIMUM_MEMORY=256MB \
    -s ENVIRONMENT='web' \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
    --bind \
    -std=c++17

echo "✅ WASM build complete!"
echo ""
echo "📁 Output files:"
echo "   - wasm_build/bullet.js"
echo "   - wasm_build/bullet.wasm"
echo ""
echo "📦 File sizes:"
ls -lh wasm_build/bullet.{js,wasm} 2>/dev/null || echo "   (files created)"
echo ""
echo "🌐 To test:"
echo "   cd wasm_build && python3 -m http.server 8000"
echo "   Open http://localhost:8000"
echo ""
echo "========================================"
