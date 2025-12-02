#!/bin/bash
# build-wasm.sh
# Compile Bullet-Core to WebAssembly using Emscripten

set -e

echo "🔨 Building Bullet-Core for WebAssembly..."

# Check if emcc is available
if ! command -v emcc &> /dev/null; then
    echo "❌ Emscripten not found!"
    echo "Install it with:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk"
    echo "  ./emsdk install latest"
    echo "  ./emsdk activate latest"
    echo "  source ./emsdk_env.sh"
    exit 1
fi

echo "✅ Emscripten found: $(emcc --version | head -n1)"

# Compile to WASM
echo "📦 Compiling bullet-api.cpp to WASM..."

emcc bullet-api.cpp \
    -o web/bullet.js \
    -O3 \
    -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_bullet_load_model","_bullet_generate","_bullet_free_model","_bullet_free_string"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME='BulletModule' \
    -s ENVIRONMENT='web' \
    --no-entry

echo "✅ WASM build complete!"
echo "📁 Output files:"
echo "   - web/bullet.js"
echo "   - web/bullet.wasm"
echo ""
echo "🌐 To test, run:"
echo "   cd web && python3 -m http.server 8000"
echo "   Then open http://localhost:8000"
