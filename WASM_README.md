# Bullet OS - WebAssembly (WASM) Build Guide

This directory contains the source code and build scripts to compile the Bullet OS runtime to WebAssembly using Emscripten.

## Prerequisites

You need the **Emscripten SDK (emsdk)** installed and active.

1. **Install Emscripten**:
   ```bash
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh
   ```

## Build Instructions

Once Emscripten is active, run the build script:

```bash
cd /home/shri/Desktop/bulletOs
./build_wasm_complete.sh
```

## Output

The build script will create a `wasm_build/` directory containing:

- **`bullet.js`**: The JavaScript glue code.
- **`bullet.wasm`**: The compiled binary runtime.

## Testing

To test the WASM build locally:

```bash
cd wasm_build
python3 -m http.server 8000
```

Open `http://localhost:8000` in your browser.

## Integration

To use Bullet OS in your web app:

```javascript
const bullet = require('./bullet.js');

bullet.onRuntimeInitialized = () => {
    // Load model
    bullet.loadModel('model.bq4', 'vocab.txt');
    
    // Generate text
    const result = bullet.generate("Hello world", 50);
    console.log(result);
    
    // Sentiment analysis
    const sentiment = bullet.sentiment("This is great!");
    console.log(sentiment);
};
```
