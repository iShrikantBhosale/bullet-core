# Marathi Philosophy Model - Deployment Package

## Model Information
- **Name**: marathi_philosophy_v1
- **Version**: 1.0
- **Language**: Marathi
- **Training Steps**: 3000
- **Validation Loss**: 3.9662

## Files Included
- `model.bullet` - Full precision model (FP32)
- `tokenizer.json` - BPE tokenizer
- `config.json` - Model configuration
- `inference.py` - Inference script (see below)

## Quick Start

```python
import pickle
import json
from tokenizer import BPETokenizer

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load('tokenizer.json')

# Load model
with open('model.bullet', 'rb') as f:
    model_data = pickle.load(f)

# Generate text
# (Use the inference script provided)
```

## Model Specifications
- Architecture: GPT-style Transformer
- Layers: 8
- Dimension: 256
- Attention Heads: 4
- Context Length: 512 tokens
- Parameters: 518,144
- Size: ~2MB

## Performance
- Inference Speed: 20-50 tokens/sec (CPU)
- Memory Usage: ~2GB RAM
- Load Time: <5 seconds

## License
[Your License Here]

## Contact
[Your Contact Info]
