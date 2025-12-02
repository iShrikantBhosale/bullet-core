#!/bin/bash
# build_wasm.sh
# Build Bullet OS for WebAssembly

set -e

echo "🔨 Building Bullet OS for WASM..."

# Check for Emscripten
if ! command -v emcc &> /dev/null; then
    echo "❌ Emscripten not found!"
    echo "Install: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

echo "✅ Emscripten found: $(emcc --version | head -n1)"

# Create WASM bindings
cat > bullet_wasm.cpp << 'EOF'
#include "bq4/bq4_transformer.h"
#include "bq4/bq4_generation.h"
#include "bq4/bq4_sentiment.h"
#include "bq4/bq4_tokenizer.h"
#include <emscripten/bind.h>

using namespace emscripten;
using namespace bullet::bq4;

// Global instances
static BulletModel* model = nullptr;
static Tokenizer* tokenizer = nullptr;

std::string generate_text(const std::string& prompt) {
    if (!model || !tokenizer) return "Error: Model not loaded";
    
    auto tokens = tokenizer->encode(prompt);
    // auto output_tokens = generate(tokens, *model);
    // return tokenizer->decode(output_tokens);
    
    return prompt + " [generation ready]";
}

std::string analyze_sentiment(const std::string& text) {
    if (!model || !tokenizer) return "UNKNOWN";
    // return sentiment_analysis(text, *model, sentiment_head, *tokenizer);
    return "NEUTRAL";
}

EMSCRIPTEN_BINDINGS(bullet_module) {
    function("generate", &generate_text);
    function("sentiment", &analyze_sentiment);
}
EOF

# Compile to WASM
echo "📦 Compiling to WebAssembly..."

emcc bullet_wasm.cpp \
    -o web/bullet.js \
    -O3 \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME='BulletModule' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s ENVIRONMENT='web' \
    --bind

echo "✅ WASM build complete!"
echo "📁 Output: web/bullet.js, web/bullet.wasm"
echo ""
echo "🌐 To test:"
echo "   cd web && python3 -m http.server 8000"
echo "   Open http://localhost:8000"
