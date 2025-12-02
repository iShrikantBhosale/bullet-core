# Training Speed Optimization Guide

## Why Training Was Slow
- **500k tokens** × **256 dimensions** × **8 layers** = Very large computation
- **CPU-only** with 4 threads (no GPU acceleration)
- **Full batch processing** without gradient accumulation

## Optimizations Applied

### 1. Gradient Accumulation
- Split batches into 4 mini-batches
- Accumulate gradients before updating weights
- **Result**: 4x less memory, smoother training

### 2. Reduced Logging Frequency
- Log every 5 batches instead of every batch
- **Result**: Less I/O overhead, faster training

### 3. Preset Configurations
- **Quick Train**: 128 dim, 4 layers, 32 seq_len, 1 epoch
  - ~5-10 minutes on 4-core CPU
  - Good for testing the pipeline
- **Full Train**: 256 dim, 8 layers, 64 seq_len, 5 epochs
  - ~30-60 minutes on 4-core CPU
  - Better quality model

## Recommended Settings for Your Hardware (4-core CPU)

### For Testing (Fast)
```
Vocab: 1000
Dim: 128
Heads: 4
Layers: 4
Seq Len: 32
Epochs: 1
Batch Size: 4
```
**Time**: ~5-10 minutes

### For Production (Quality)
```
Vocab: 5000
Dim: 256
Heads: 4
Layers: 6 (reduced from 8)
Seq Len: 64
Epochs: 3 (reduced from 5)
Batch Size: 8
```
**Time**: ~20-30 minutes

## Further Optimizations (If Needed)

1. **Reduce Dataset Size**
   - Use first 1000 entries instead of 3000
   - `head -n 1000 marathi_philosophy_dataset.jsonl > marathi_small.jsonl`

2. **Use GPU** (if available)
   - Install CUDA-enabled PyTorch
   - Training will be 10-50x faster

3. **Reduce Vocabulary**
   - Lower vocab_size to 1000-2000
   - Faster embedding lookups

4. **Shorter Sequences**
   - Reduce max_seq_len to 32
   - Quadratic attention complexity reduction

## How to Use Presets in Dashboard

1. Click "⚡ Quick Train" for fast testing
2. Click "🎯 Full Train" for quality model
3. Upload your dataset
4. Click "Start Training"
5. Watch the stats panel for real-time progress
