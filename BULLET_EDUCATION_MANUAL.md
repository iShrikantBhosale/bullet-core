# 🎓 Bullet OS Education Manual
## Turn Your College Computer Lab into an AI Research Lab

**No GPU. No Cloud. No Cost.**

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Understanding Bullet OS](#understanding-bullet-os)
4. [Training Your First Model](#training-your-first-model)
5. [BQ4 Quantization Explained](#bq4-quantization-explained)
6. [Classroom Exercises](#classroom-exercises)
7. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

### What is Bullet OS?

Bullet OS is a revolutionary AI training system designed for **resource-constrained environments**. It enables students and researchers to train real Transformer models on ordinary computers without expensive GPUs or cloud credits.

### Why Bullet OS for Education?

✅ **Zero Cost** - Runs on any CPU, no cloud required
✅ **Real AI** - Train actual Transformer models, not toy examples
✅ **Production Ready** - Same tech used in real deployments
✅ **Learn by Doing** - Hands-on experience with quantization, training, inference

### System Requirements

- **CPU**: Any modern processor (Intel/AMD/ARM)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows (with WSL)
- **Python**: 3.8 or higher

---

## 2. Quick Start

### Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/iShrikantBhosale/bullet-core.git
cd bullet-core

# Install dependencies
pip install numpy

# Verify installation
python -c "import numpy; print('✅ Ready to go!')"
```

### Your First Training Run (10 minutes)

```bash
# Train a tiny demo model
python bullet_core/train_production.py \
  --config bullet_core/configs/marathi_small.yaml \
  --steps 100

# Watch the magic happen!
# You'll see loss decreasing in real-time
```

### Test the Model

```bash
# Convert to .bullet format
python test_checkpoints.py

# The model is now ready for deployment!
```

---

## 3. Understanding Bullet OS

### The Mother-Child Architecture

Instead of one massive model, Bullet OS uses **specialized micro-models**:

```
Mother Model (Router)
    ├── Summarizer (2MB)
    ├── Translator (3MB)
    ├── Sentiment Analyzer (1MB)
    └── Question Answerer (2MB)
```

**Why?**
- Each model is tiny (1-15MB)
- Faster inference
- Easier to train
- More interpretable

### Key Components

1. **Training Engine** (`bullet_core/`)
   - Custom autograd (no PyTorch dependency!)
   - FlashAttention-inspired kernels
   - Automatic checkpointing

2. **BQ4 Quantization** (`bullet_core/utils/bullet_io.py`)
   - 4-bit signed weights
   - 6.4x compression
   - Minimal accuracy loss

3. **`.bullet` Format**
   - Memory-mapped for instant loading
   - Cross-platform compatible
   - Production-ready

---

## 4. Training Your First Model

### Step 1: Prepare Your Dataset

Create a file `my_dataset.jsonl`:

```json
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "Machine learning is transforming the world."}
{"text": "Python is a powerful programming language."}
```

**Tips:**
- Use diverse, clean text
- Aim for 1000+ examples for meaningful results
- One sentence per line works best

### Step 2: Configure Training

Edit `bullet_core/configs/my_config.yaml`:

```yaml
# Model Architecture
hidden_size: 128        # Smaller = faster training
num_heads: 4
num_layers: 4
vocab_size: 5000

# Training
learning_rate: 0.0002
batch_size: 4
max_seq_len: 64
max_steps: 1000

# Data
dataset_path: "my_dataset.jsonl"
checkpoint_dir: "my_checkpoints"
```

### Step 3: Start Training

```bash
python bullet_core/train_production.py --config bullet_core/configs/my_config.yaml
```

**What to Watch:**
- **Loss** should decrease (e.g., 8.0 → 4.0 → 2.5)
- **Speed** ~50-100 steps/min on CPU
- **Checkpoints** saved every 100 steps

### Step 4: Monitor Progress

```bash
# Watch the log file
tail -f logs/training.log

# You'll see:
# Step 100 | Loss: 6.234 | LR: 0.0002 | Speed: 87 steps/min
# Step 200 | Loss: 4.891 | LR: 0.0002 | Speed: 92 steps/min
```

### Step 5: Export to .bullet

```bash
# Convert checkpoint to production format
python test_checkpoints.py

# Your model is now in my_checkpoints/checkpoint_step_1000.bullet
```

---

## 5. BQ4 Quantization Explained

### What is Quantization?

**Before (FP32)**: Each weight = 32 bits = 4 bytes
**After (BQ4)**: Each weight = 4 bits = 0.5 bytes

**Result**: 8x smaller models!

### How BQ4 Works

1. **Group weights into blocks** (32 weights per block)
2. **Find scale factor** (max absolute value / 7)
3. **Quantize to 4-bit** (-8 to +7 range)
4. **Pack into bytes** (2 weights per byte)

### Example

```python
# Original weights (FP32)
weights = [-0.046, 0.028, -0.008, -0.033, ...]

# After BQ4
scale = 0.0077  # max_abs / 7
quantized = [-6, 4, -1, -4, ...]  # 4-bit signed
packed = [0x4A, 0xF1, ...]  # 2 weights per byte
```

### Why It Matters

- **Smaller models** → Faster downloads
- **Less memory** → Run on phones/edge devices
- **Same accuracy** → Minimal loss (<1%)

---

## 6. Classroom Exercises

### Exercise 1: Train a Sentiment Classifier (Beginner)

**Goal**: Build a model that classifies text as positive/negative

**Dataset**: Create `sentiment.jsonl`
```json
{"text": "I love this product! Amazing quality."}
{"text": "Terrible experience. Would not recommend."}
{"text": "Pretty good, worth the price."}
```

**Task**:
1. Train for 500 steps
2. Test on new sentences
3. Measure accuracy

**Expected Time**: 30 minutes

### Exercise 2: Multilingual Model (Intermediate)

**Goal**: Train a model on mixed English-Hindi text

**Dataset**: Mix languages in `multilingual.jsonl`
```json
{"text": "Hello दुनिया! Welcome to AI."}
{"text": "मशीन लर्निंग is fascinating."}
```

**Task**:
1. Build BPE tokenizer for both scripts
2. Train for 1000 steps
3. Test code-switching ability

**Expected Time**: 1 hour

### Exercise 3: Optimize for Speed (Advanced)

**Goal**: Make training 2x faster

**Challenges**:
1. Profile the code (find bottlenecks)
2. Optimize data loading
3. Tune batch size and sequence length
4. Implement gradient accumulation

**Expected Time**: 2 hours

### Exercise 4: Deploy to Web (Advanced)

**Goal**: Run your model in a browser

**Tasks**:
1. Convert model to .bullet
2. Create simple HTML interface
3. Load model with JavaScript
4. Implement text generation UI

**Expected Time**: 2 hours

---

## 7. Troubleshooting

### Problem: "Out of Memory"

**Solution**:
```yaml
# Reduce these in config:
batch_size: 2          # Was 4
max_seq_len: 32        # Was 128
hidden_size: 64        # Was 256
```

### Problem: "Loss Not Decreasing"

**Checklist**:
- ✅ Is learning rate too high? Try 0.0001
- ✅ Is dataset too small? Need 500+ examples
- ✅ Is data clean? Remove duplicates/noise
- ✅ Is model too small? Try more layers

### Problem: "Training Too Slow"

**Solutions**:
1. Reduce `max_seq_len` (biggest impact)
2. Reduce `batch_size`
3. Use smaller `hidden_size`
4. Enable gradient accumulation

### Problem: "Model Generates Gibberish"

**Fixes**:
- Train longer (5000+ steps minimum)
- Use repetition penalty (1.2)
- Check tokenizer vocabulary
- Verify dataset quality

---

## 📖 Additional Resources

### Documentation
- [BULLET_SPEC_v1.0.md](BULLET_SPEC_v1.0.md) - File format details
- [BQ4_PAPER.md](BQ4_PAPER.md) - Quantization research
- [COLAB_README.md](COLAB_README.md) - Google Colab guide

### Community
- GitHub: https://github.com/iShrikantBhosale/bullet-core
- Issues: Report bugs and ask questions
- Discussions: Share your models and results

### Citation

If you use Bullet OS in your research or teaching, please cite:

```bibtex
@software{bullet_os_2025,
  title = {Bullet OS: Nano AI Agent System},
  author = {Bhosale, Shrikant},
  year = {2025},
  url = {https://github.com/iShrikantBhosale/bullet-core}
}
```

---

## 🎯 Learning Outcomes

After completing this manual, students will:

✅ Understand Transformer architecture
✅ Train real language models from scratch
✅ Implement quantization techniques
✅ Deploy models to production
✅ Optimize for resource-constrained environments

---

**🇮🇳 Made in India | Democratizing AI Education**

© 2025 Bullet OS. MIT License.
