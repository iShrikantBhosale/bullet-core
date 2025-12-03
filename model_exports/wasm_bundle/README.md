# Marathi Philosophy Model - WASM Bundle

## Contents

- `marathi_model.js` - JavaScript wrapper for the model
- `model_weights.bin` - Binary model weights (optimized for WASM)
- `tokenizer.json` - BPE tokenizer
- `demo.html` - Interactive web demo

## Quick Start

### Option 1: Open Demo Directly

Simply open `demo.html` in a modern web browser. Note: You may need to serve it via HTTP due to CORS restrictions.

### Option 2: Use with HTTP Server

```bash
# Python 3
python -m http.server 8000

# Node.js
npx http-server

# Then open: http://localhost:8000/demo.html
```

### Option 3: Integrate in Your App

```html
<script src="marathi_model.js"></script>
<script>
    const model = new MarathiPhilosophyModel();
    
    model.load().then(() => {
        model.generate('जीवनाचा अर्थ', {
            maxTokens: 100,
            temperature: 0.7
        }).then(output => {
            console.log(output);
        });
    });
</script>
```

## Features

- ✅ Runs entirely in the browser (no server needed)
- ✅ ~5MB total size (model + tokenizer)
- ✅ Privacy-friendly (all processing local)
- ✅ Fast inference on modern browsers
- ✅ Mobile-friendly

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- Mobile browsers: ✅ Supported

## Notes

- The current implementation uses a simplified inference engine
- For production use, consider implementing full transformer inference in WASM
- Model weights are in FP32 format (can be quantized further for smaller size)

## Performance

- Load time: ~2-5 seconds
- Inference: ~10-30 tokens/second (varies by browser)
- Memory usage: ~100-200MB

## License

[Your License Here]
