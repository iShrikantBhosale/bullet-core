# Quick Start - Test Training in 2 Minutes

## Problem
Training with 3000 entries (500k tokens) takes 20-30 minutes on 4-core CPU.

## Solution
Use the quick test dataset with 100 entries (~16k tokens).

## Steps

### 1. Use the Quick Test Dataset
I've created `dashboard/marathi_quick_test.jsonl` with 100 entries.

### 2. Open Dashboard
`http://localhost:8000/dashboard/index.html`

### 3. Click "⚡ Quick Train (Fast)"
This loads:
- 128 dimensions (instead of 256)
- 4 layers (instead of 8)
- 1 epoch
- Batch size 4

### 4. Upload Quick Test File
Upload: `dashboard/marathi_quick_test.jsonl`

### 5. Start Training
Click "Start Training"

**Expected Time**: ~2-3 minutes

## What You'll See

```
[Time] Uploading dataset and starting training...
[Time] Training started with config: {...}
[Time] Vocab size: ~300. Total tokens: ~16000
[Time] Using CPU with 4 threads.
[Time] Initializing model architecture...
[Time] Model initialized: 0.5M parameters
[Time] Setting up optimizer...
[Time] Preparing training data...
[Time] Total mini-batches: ~400
[Time] Epoch 1 | Batch 5/400 | Loss: 8.5234
[Time] Epoch 1 | Batch 10/400 | Loss: 7.2341
...
[Time] Epoch 1/1 Completed in 120s - Avg Loss: 5.1234
[Time] Training complete. Exporting to .bullet...
[Time] Model exported to model_XXXXX.bullet
```

## For Production Model (BULLET_SPEC)

Once you verify the quick test works:

1. Click "🎯 Full Train (Quality)"
2. Upload full `marathi_philosophy_dataset.jsonl` (3000 entries)
3. Be patient - it will take 20-30 minutes
4. Watch the stats panel for progress

## Why It's Slow

**Math**: 
- 500k tokens × 256 dim × 8 layers = ~1 billion operations per batch
- 4 CPU cores at ~3 GHz = ~12 billion ops/sec
- First batch: ~0.1 seconds
- Total batches: ~1000
- Total time: ~100 seconds per epoch × 2 epochs = ~3-5 minutes

But with Python overhead, PyTorch CPU operations, and memory transfers, it's actually 20-30 minutes.

## Recommendation

**For testing**: Use `marathi_quick_test.jsonl` (100 entries)
**For production**: Use full dataset but be patient, or consider:
- Using a GPU (10-50x faster)
- Training on a more powerful CPU
- Reducing to 6 layers instead of 8
