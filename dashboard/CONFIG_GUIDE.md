# Training Configuration Guide

## BULLET_SPEC_v1.0.md Alignment

The dashboard now uses the exact configuration from BULLET_SPEC_v1.0.md:

```json
{
  "dimensions": {
    "hidden_size": 256,
    "num_layers": 8,
    "num_heads": 4
  }
}
```

### Default Settings (Click "🎯 Full Train")
- **Vocab Size**: 5000
- **Dimensions**: 256 (matches spec)
- **Attention Heads**: 4 (matches spec)
- **Layers**: 8 (matches spec)
- **Sequence Length**: 64
- **Learning Rate**: 0.001
- **Epochs**: 2 (reduced for reasonable time)
- **Batch Size**: 8

## Speed Optimizations Applied

### 1. Gradient Accumulation
- Splits batch_size=8 into 4 mini-batches of size 2
- Accumulates gradients before weight update
- **Benefit**: Faster progress updates, less memory

### 2. Reduced Logging
- Logs every 5 batches instead of every batch
- **Benefit**: Less I/O overhead

### 3. Optimized Batch Processing
- Smaller mini-batches show progress faster
- Early stopping if user clicks "Stop"

## Estimated Training Times (4-core CPU)

### Quick Train Preset (⚡)
- 128 dim, 4 layers, 1 epoch
- **Time**: ~5-10 minutes
- **Use**: Testing pipeline

### Full Train Preset (🎯)
- 256 dim, 8 layers, 2 epochs (BULLET_SPEC)
- **Time**: ~20-30 minutes
- **Use**: Production model

## How to Use

1. Open `http://localhost:8000/dashboard/index.html`
2. Click "🎯 Full Train (Quality)" for BULLET_SPEC config
3. Upload `marathi_philosophy_dataset.jsonl`
4. Click "Start Training"
5. Monitor real-time stats panel
6. Download `.bullet` model when complete

The model will be fully compatible with `bullet-core.cpp` inference engine.
