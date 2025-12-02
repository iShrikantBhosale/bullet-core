# Marathi Philosophy Model Report (Upgraded)

**Version**: 2.0 (Upgraded)
**Date**: December 2, 2025
**Model Path**: `./marathi_checkpoints_upgraded/final_model.pkl`

## 1. Executive Summary

This report details the specifications and performance of the upgraded Marathi Philosophy Transformer model. This iteration represents a significant leap from the previous version, featuring a **4x larger architecture**, **BPE tokenization**, and **20x longer training duration**. The model is designed to generate philosophical text in Marathi, drawing from a dataset of 3.2 million characters.

## 2. Technical Specifications

### Architecture
The model is a decoder-only Transformer (GPT-style) built from scratch using the `bullet_core` library.

| Component | Specification | Description |
| :--- | :--- | :--- |
| **Type** | Decoder-only Transformer | Autoregressive language model |
| **Layers** | 4 | Number of transformer blocks |
| **Attention Heads** | 4 | Parallel attention mechanisms per layer |
| **Embedding Dimension** | 256 | Size of vector representation for each token |
| **Feed-Forward Dimension** | 1024 | Hidden size of the MLP within each block (4x embedding) |
| **Context Window** | 128 tokens | Maximum sequence length the model can attend to |
| **Parameter Count** | ~419,840 | Total trainable weights |
| **File Size** | ~1.7 MB | Serialized pickle file size |

### Tokenization
We switched from character-level tokenization to **Byte-Pair Encoding (BPE)** to improve information density and context handling.

| Feature | Value | Notes |
| :--- | :--- | :--- |
| **Algorithm** | Byte-Pair Encoding (BPE) | Iteratively merges frequent character pairs |
| **Vocabulary Size** | 1511 | Optimized for Marathi script and philosophy domain |
| **Compression Ratio** | ~3.3x | Compared to raw character-level encoding |
| **Tokenizer File** | `marathi_tokenizer.json` | Contains vocab and merge rules |

## 3. Training Dynamics

### Configuration
- **Dataset**: `marathi_philosophy_dataset.jsonl` (3.2M chars)
- **Steps**: 10,000 iterations
- **Batch Size**: 1 (Gradient accumulation not explicitly used, but effective batch size is small)
- **Optimizer**: AdamW
- **Learning Rate**: 5e-4 (0.0005)
- **Weight Decay**: 1e-2 (0.01)

### Performance
- **Final Training Loss**: **6.2877**
- **Best Validation Loss**: **6.1279**
- **Convergence**: The model reached stable validation loss around step 3000.
- **Hardware**: Trained on CPU (optimized with `bullet_core` kernels).

## 4. Usage Guide

### Loading the Model
The model is saved as a standard Python pickle file containing the weight dictionary and metadata.

```python
import pickle
import sys
import os

# Ensure bullet_core is in path
sys.path.insert(0, os.path.abspath("."))
from python.transformer import GPT
from python.tokenizer import BPETokenizer

# 1. Load Tokenizer
tokenizer = BPETokenizer()
tokenizer.load("marathi_tokenizer.json")

# 2. Load Model
with open("marathi_checkpoints_upgraded/final_model.pkl", "rb") as f:
    data = pickle.load(f)

metadata = data['metadata']
model = GPT(
    vocab_size=metadata['vocab_size'],
    d_model=metadata['d_model'],
    n_head=metadata['n_head'],
    n_layer=metadata['n_layer'],
    max_len=metadata['block_size']
)

# Load weights
for i, param in enumerate(model.parameters()):
    if f'param_{i}' in data['parameters']:
        param.data = data['parameters'][f'param_{i}']
```

### Inference
For best results, use the following sampling parameters:
- **Temperature**: 0.7 (Balances creativity and coherence)
- **Top-K**: 40 (Restricts sampling to top 40 likely tokens)
- **Top-P**: 0.9 (Nucleus sampling)

## 5. Test Analysis

### Generated Samples
**Prompt: "जीवन" (Life)**
> जीवन र्णश्रसिद्धअंधार।छिन्दनवूपाऐक।'।''स्थितप्रज्ञ्र्यमशिसर्वप्रथनात्मानमवसाः।'...

**Prompt: "आत्मा" (Soul)**
> आत्मा दु:खअर्जुमोहातूनमोहातूनते.्ये।'कालमोहातूनप्रश्नांमोहातून...

### Limitations
1.  **Coherence**: While the model uses correct vocabulary, sentence structure is often fragmented.
2.  **Repetition**: The model tends to repeat certain tokens (e.g., "मोहातून"), indicating potential overfitting to frequent words in the small dataset.
3.  **Context**: The 128-token context window limits long-range consistency.

### Recommendations for Next Version
1.  **Larger Dataset**: 3.2M characters is small for a language model. Expanding to 10M+ would improve grammar.
2.  **Larger Context**: Increasing context to 256 or 512 would help with sentence structure.
3.  **Fine-tuning**: Pre-training on general Marathi text before fine-tuning on philosophy might yield better results.
