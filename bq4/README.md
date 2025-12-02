# BQ4 - Bullet Quantization 4-bit

**Ultra-lightweight 4-bit quantization for Transformer models**

## Overview

BQ4 is a custom 4-bit quantization format designed for tiny language models. It achieves **6-8x compression** with minimal accuracy loss, enabling deployment on mobile, WASM, and edge devices.

## Features

- ✅ **Tiny models**: 1.7MB → 276KB (6.4x compression)
- ✅ **Fast inference**: 1,500+ inferences/sec on CPU
- ✅ **Fused kernels**: MatMul + dequant in one pass
- ✅ **Portable**: Pure C++, no dependencies
- ✅ **WASM-ready**: Browser deployment
- ✅ **Mobile-ready**: Android/iOS compatible

## Quick Start

### 1. Export Model to BQ4

```bash
python3 export_bq4.py trained_model.pkl output.bq4
```

### 2. Run Inference

```cpp
#include "bq4/bq4_loader.h"
#include "bq4/bq4_inference.h"

// Load model
Loader loader("model.bq4");
loader.load_metadata();
Tensor W = loader.load_tensor(0);

// Run inference
std::vector<float> input(256);
std::vector<float> output(1511);
matmul_bq4(input.data(), W, output.data());
```

### 3. Benchmark

```bash
g++ -O3 -o demo demo_bq4_inference.cpp
./demo model.bq4
```

## File Format

BQ4 uses a simple, streamable format:

```
[Header: "BQ4F" + version + num_tensors]
[Tensor Directory: names, shapes, offsets]
[Quantized Blocks: packed int4 + FP32 scales]
```

Each block:
- 32 weights → 16 bytes (packed int4)
- 1 scale → 4 bytes (FP32)
- Total: 20 bytes per block

## Quantization Method

**Symmetric quantization** with block-wise scaling:

```
scale = max(|weights|) / 7
quantized = clip(round(weight / scale), -8, 7)
```

Range: int4 ∈ [-8, +7]

## Performance

**Marathi Model (1511×256)**:
- Original: 1,767 KB
- BQ4: 276 KB (6.4x smaller)
- Inference: 0.65 ms/call
- Throughput: 1,546 inferences/sec

## Components

| File | Purpose |
| :--- | :--- |
| `bq4_kernels.h` | Quantize/dequantize |
| `bq4_format.h` | File format structs |
| `bq4_loader.h` | Model loader |
| `bq4_inference.h` | Fused inference kernels |
| `export_bq4.py` | Python exporter |

## Roadmap

- [x] Core quantization kernels
- [x] File format & loader
- [x] Fused matmul kernels
- [x] Python exporter
- [x] Benchmarks
- [ ] Full transformer inference
- [ ] WASM build
- [ ] Mobile SDKs

## License

MIT License (Runtime)

---

**Bullet OS** - Tiny AI, Infinite Privacy
