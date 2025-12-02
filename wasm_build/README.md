# 🌐 Bullet OS - WASM Dashboard

## Run AI in Your Browser - No Server Required!

This is the WebAssembly build of Bullet OS that runs entirely in your browser.

### 🚀 Quick Start

```bash
# Navigate to wasm_build directory
cd wasm_build

# Start local server
python3 -m http.server 8000

# Open in browser
# http://localhost:8000
```

### ✨ Features

- **Text Generation** - Generate Marathi/Hindi text
- **Sentiment Analysis** - Analyze text sentiment
- **Real-time Performance** - See tokens/sec in browser
- **No Backend** - Everything runs client-side

### 📊 Performance

- **WASM Size**: ~160KB (compressed)
- **Load Time**: <1 second
- **Inference**: 10-20 tokens/sec (browser-dependent)

### 🛠️ Building from Source

```bash
# Build WASM module
./build_wasm_complete.sh

# Output files:
#   - bullet.wasm (WebAssembly binary)
#   - bullet.js (JavaScript glue code)
#   - index.html (Dashboard UI)
```

### 🎯 Use Cases

- **Offline AI** - Works without internet
- **Privacy** - All processing in browser
- **Demos** - Easy to share and demo
- **Education** - Learn AI without server setup

### 🇮🇳 Made in India

Created by **Shrikant Bhosale** | Mentored by [Hintson.com](https://hintson.com)

© 2025 Bullet OS | MIT License
