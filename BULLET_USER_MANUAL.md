# 📘 Bullet Model User Manual

## How to Use Your Trained .bullet File

**Congratulations!** You've trained a Bullet OS model. This guide shows you how to use it.

---

## 🎯 What is a .bullet File?

A `.bullet` file is a **production-ready AI model** that contains:
- ✅ Quantized weights (BQ4 format - 4-bit)
- ✅ Model architecture configuration
- ✅ Tokenizer vocabulary
- ✅ Everything needed for inference

**Size:** Typically 1-5MB (vs 100MB+ for traditional models)

---

## 🚀 Quick Start

### Option 1: Python Inference (Easiest)

```python
import sys
sys.path.append('bullet_core')

from python.transformer import GPT
from python.tokenizer import BPETokenizer
from python.tensor import Tensor
from utils.bullet_io import BulletReader
import numpy as np

# 1. Load the .bullet file
reader = BulletReader('your_model.bullet')
reader.load()

# 2. Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load('bullet_core/marathi_tokenizer.json')

# 3. Create model
model = GPT(
    vocab_size=1511,
    d_model=256,
    n_head=4,
    n_layer=8,
    max_len=128
)

# 4. Load weights from .bullet file
model_params = list(model.parameters())
for i, param in enumerate(model_params):
    key = f"param_{i}"
    if key in reader.tensors:
        param.data = reader.tensors[key]

# 5. Generate text!
prompt = "जीवनाचा अर्थ"
tokens = tokenizer.encode(prompt)
x = Tensor(np.array([tokens], dtype=np.int32), requires_grad=False)

model.eval()
logits = model(x)

# Get next token
next_token = np.argmax(logits.data[0, -1, :])
result = tokenizer.decode(tokens + [next_token])
print(result)
```

### Option 2: C++ Inference (Fastest)

```bash
# Compile the runtime
g++ -O3 -std=c++17 -o bullet bullet-core.cpp

# Run inference
./bullet your_model.bullet -p "Your prompt here"
```

### Option 3: WASM (Browser)

```bash
# Navigate to WASM build
cd wasm_build

# Start server
python3 -m http.server 8000

# Open http://localhost:8000
# Upload your .bullet file in the dashboard
```

---

## 📊 Common Use Cases

### 1. Text Generation

```python
def generate_text(model, tokenizer, prompt, max_tokens=20):
    tokens = tokenizer.encode(prompt)
    generated = tokens.copy()
    
    for _ in range(max_tokens):
        x = Tensor(np.array([generated], dtype=np.int32), requires_grad=False)
        logits = model(x)
        
        # Apply repetition penalty
        last_logits = logits.data[0, -1, :].copy()
        for token_id in set(generated):
            if last_logits[token_id] > 0:
                last_logits[token_id] /= 1.2
            else:
                last_logits[token_id] *= 1.2
        
        next_token = np.argmax(last_logits)
        generated.append(next_token)
    
    return tokenizer.decode(generated)

# Use it
result = generate_text(model, tokenizer, "जीवनाचा अर्थ", max_tokens=50)
print(result)
```

### 2. Batch Processing

```python
# Process multiple prompts
prompts = [
    "कृत्रिम बुद्धिमत्ता",
    "मशीन लर्निंग",
    "डीप लर्निंग"
]

for prompt in prompts:
    result = generate_text(model, tokenizer, prompt, max_tokens=20)
    print(f"{prompt} → {result}")
```

### 3. API Server

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model once at startup
model, tokenizer = load_model('your_model.bullet')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 20)
    
    result = generate_text(model, tokenizer, prompt, max_tokens)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 🔧 Advanced Usage

### Fine-tuning Your Model

```bash
# Resume training from your .bullet file
python bullet_core/train_production.py \
  --config bullet_core/configs/your_config.yaml \
  --resume-from your_model.bullet \
  --steps 5000
```

### Model Inspection

```python
from utils.bullet_io import BulletReader

reader = BulletReader('your_model.bullet')
reader.load()

# Check model info
print(f"Config: {reader.config}")
print(f"Vocab size: {len(reader.vocab)}")
print(f"Number of tensors: {len(reader.tensors)}")

# Inspect weights
for name, tensor in reader.tensors.items():
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
```

### Export to Other Formats

```python
# Export to ONNX (if needed)
# Export to TensorFlow Lite
# Export to CoreML
# (Implementation depends on your target platform)
```

---

## 📱 Deployment Options

### 1. **Mobile (Android)**

```kotlin
// Use the Android SDK
val model = BulletModel("your_model.bullet")
val result = model.generate("Your prompt", maxTokens = 20)
```

### 2. **Web (WASM)**

```javascript
// Load in browser
const module = await BulletModule();
module.loadModel('your_model.bullet');
const result = module.generate('Your prompt', 20);
```

### 3. **Edge Devices (Raspberry Pi)**

```bash
# Just copy the .bullet file and run
./bullet your_model.bullet -p "Your prompt"
```

### 4. **Cloud (Docker)**

```dockerfile
FROM ubuntu:22.04
COPY your_model.bullet /app/
COPY bullet-core.cpp /app/
RUN g++ -O3 -o /app/bullet /app/bullet-core.cpp
CMD ["/app/bullet", "/app/your_model.bullet"]
```

---

## ⚡ Performance Tips

### 1. **Optimize Batch Size**

```python
# Process multiple prompts together
batch_prompts = ["prompt1", "prompt2", "prompt3"]
# Batch inference is faster than sequential
```

### 2. **Use Repetition Penalty**

```python
# Prevents repetitive output
for token_id in set(generated):
    if last_logits[token_id] > 0:
        last_logits[token_id] /= 1.2  # Penalty factor
```

### 3. **Cache Model Loading**

```python
# Load once, use many times
model = load_model_once()  # Cache this

# Reuse for all requests
for prompt in prompts:
    generate(model, prompt)
```

---

## 🐛 Troubleshooting

### Problem: "Model generates gibberish"

**Solutions:**
- Train longer (5000+ steps)
- Use more diverse data
- Apply repetition penalty
- Check tokenizer vocabulary

### Problem: "Slow inference"

**Solutions:**
- Use C++ runtime (10x faster than Python)
- Reduce max_tokens
- Use smaller model
- Enable CPU optimizations (`-O3` flag)

### Problem: "Out of memory"

**Solutions:**
- Reduce batch size
- Use smaller max_seq_len
- Close other applications
- Use BQ4 quantization (already applied)

### Problem: "Can't load .bullet file"

**Solutions:**
- Check file path is correct
- Verify file is not corrupted
- Ensure you have read permissions
- Use absolute path instead of relative

---

## 📚 Additional Resources

- **Education Manual:** [BULLET_EDUCATION_MANUAL.md](BULLET_EDUCATION_MANUAL.md)
- **File Format Spec:** [BULLET_SPEC_v1.0.md](BULLET_SPEC_v1.0.md)
- **BQ4 Quantization:** [BQ4_PAPER.md](BQ4_PAPER.md)
- **GitHub Repo:** [github.com/iShrikantBhosale/bullet-core](https://github.com/iShrikantBhosale/bullet-core)
- **Community:** [GitHub Discussions](https://github.com/iShrikantBhosale/bullet-core/discussions)

---

## 🎓 Example Projects

### 1. **Marathi Chatbot**

```python
while True:
    user_input = input("You: ")
    response = generate_text(model, tokenizer, user_input)
    print(f"Bot: {response}")
```

### 2. **Text Completion API**

```python
@app.route('/complete')
def complete():
    text = request.args.get('text')
    return generate_text(model, tokenizer, text, max_tokens=10)
```

### 3. **Batch Translation**

```python
# Translate multiple sentences
sentences = load_sentences('input.txt')
translations = [generate_text(model, tokenizer, s) for s in sentences]
save_translations('output.txt', translations)
```

---

## 🇮🇳 Support

**Created by:** Shrikant Bhosale  
**Mentored by:** [Hintson.com](https://hintson.com)  

**Questions?** Open an issue on [GitHub](https://github.com/iShrikantBhosale/bullet-core/issues)

**© 2025 Bullet OS | MIT License**
