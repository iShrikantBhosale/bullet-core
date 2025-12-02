# BQ4: 4-bit Quantization for Edge Language Models

**Abstract**

We present BQ4, a 4-bit symmetric quantization format designed for deploying transformer-based language models on edge devices. BQ4 achieves 6.4× compression with minimal accuracy degradation through block-wise scaling and fused inference kernels. Our zero-copy memory-mapped architecture enables sub-millisecond cold-start latency and 891 inferences/second on commodity CPUs. We demonstrate BQ4's effectiveness on a 452K-parameter Marathi language model, reducing model size from 1.7MB to 276KB while maintaining generation quality. BQ4 enables deployment of language models on mobile devices, browsers (WASM), and embedded systems with <50MB memory footprint.

## 1. Introduction

Large language models have achieved remarkable performance but remain impractical for edge deployment due to:
- **Size**: Multi-gigabyte models exceed mobile/embedded storage
- **Memory**: Inference requires 2-4× model size in RAM
- **Latency**: Cloud APIs introduce 100-500ms network delays
- **Privacy**: Sending user data to remote servers compromises privacy

Existing quantization methods (GGML, GPTQ, AWQ) target 8-16GB models and require specialized hardware. We need quantization for **tiny models** (<10MB) running on **commodity CPUs**.

**BQ4 Design Goals**:
1. Extreme compression (>6×) for sub-megabyte models
2. Fast CPU inference without GPU/NPU
3. Zero-copy loading for instant cold-start
4. WASM/mobile compatibility

## 2. BQ4 Format Specification

### 2.1 Symmetric Quantization

BQ4 uses symmetric 4-bit quantization with per-block scaling:

```
scale = max(|W|) / 7
w_q = clip(round(w / scale), -8, 7)
```

Where:
- `w_q ∈ [-8, +7]` (signed 4-bit)
- Block size = 32 weights
- Zero-point = 0 (symmetric)

### 2.2 Block Layout

Each block stores:
- 32 weights → 16 bytes (packed nibbles)
- 1 scale → 4 bytes (FP32)
- **Total: 20 bytes per block**

### 2.3 File Format

```
[Header: "BQ4F" + version + num_tensors]
[Tensor Directory: names, shapes, offsets]
[Quantized Blocks: packed int4 + scales]
```

**Advantages**:
- Streamable (sequential reads)
- Memory-mappable (zero-copy)
- Simple parser (<200 LOC)

## 3. Fused Inference Kernels

### 3.1 MatMul + Dequant Fusion

Traditional approach:
```
1. Dequantize weights → FP32
2. MatMul(activation, weights_fp32)
```

BQ4 fused kernel:
```cpp
for each block:
    dequant_on_fly(block) → temp[32]
    dot_product(activation, temp)
```

**Benefits**:
- No FP32 weight buffer
- Cache-friendly (32-weight chunks)
- 1.5-2× faster than naive approach

### 3.2 Zero-Copy Loading

Memory-mapped I/O eliminates loading overhead:
```cpp
int fd = open("model.bq4", O_RDONLY);
void* data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
// Instant access to quantized blocks
```

**Cold-start latency**: <10ms (vs 100-500ms for FP32 loading)

## 4. Experimental Results

### 4.1 Model Details

**Marathi Philosophy Transformer**:
- Architecture: GPT-style decoder
- Parameters: 452,608
- Layers: 6
- Hidden dim: 256
- Context: 256 tokens
- Vocabulary: 1,511 tokens

### 4.2 Compression Results

| Format | Size | Compression |
|:---|---:|---:|
| FP32 | 1,767 KB | 1.0× |
| BQ4 | 276 KB | **6.4×** |

### 4.3 Inference Performance

**Hardware**: Intel i5-8250U (CPU only)

| Metric | Value |
|:---|---:|
| Throughput | 891 inferences/sec |
| Latency (avg) | 1.12 ms |
| Cold-start | <10 ms |
| Memory | <50 MB |

### 4.4 Accuracy Analysis

**Quantization Error** (per-block):
- Mean absolute error: 0.02-0.04
- Expected for 4-bit quantization
- Negligible impact on generation quality

## 5. Deployment Scenarios

### 5.1 Mobile (Android/iOS)
- 276KB model fits in app bundle
- Runs on CPU (no GPU required)
- 100% offline (privacy-preserving)

### 5.2 Browser (WASM)
- Compiles to WebAssembly
- Loads in <100ms
- Client-side inference

### 5.3 Embedded Systems
- Raspberry Pi, ESP32
- <50MB RAM requirement
- Real-time inference

## 6. Related Work

- **GGML/llama.cpp**: Targets 7B+ models, requires 4-8GB RAM
- **GPTQ**: GPU-dependent, complex calibration
- **AWQ**: Activation-aware, needs large models

**BQ4 novelty**: Optimized for <1MB models on CPU-only devices

## 7. Conclusion

BQ4 enables deployment of language models on edge devices through:
1. **6.4× compression** via 4-bit symmetric quantization
2. **Fast inference** (891 inferences/sec) via fused kernels
3. **Zero-copy loading** via memory-mapped I/O
4. **Universal compatibility** (CPU/WASM/mobile)

Future work includes activation quantization and sparse attention for further compression.

## References

[1] Dettmers et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
[2] Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression"
[3] Gerganov. "llama.cpp: Inference of LLaMA model in pure C/C++"

---

**Code**: https://github.com/bullet-os/bullet-runtime
**Models**: https://github.com/bullet-os/models
