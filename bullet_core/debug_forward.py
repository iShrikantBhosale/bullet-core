"""
Debug training - trace where zeros appear
"""

import json
import numpy as np
from bullet_core.tensor import Tensor
from bullet_core import nn

# Load small sample
data_path = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset.jsonl"
text_data = ""
with open(data_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        item = json.loads(line)
        text_data += item['instruction'] + " " + item['response'] + "\n"

# Tokenizer
chars = sorted(list(set(text_data)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }

def encode(text):
    return [stoi[c] for c in text]

data_ids = np.array(encode(text_data), dtype=np.int32)

print(f"Vocab size: {vocab_size}")

# Model components
BLOCK_SIZE = 64
D_MODEL = 128

# Test embedding
tok_emb = nn.Embedding(vocab_size, D_MODEL)
pos_emb = nn.Embedding(BLOCK_SIZE, D_MODEL)

x = data_ids[:BLOCK_SIZE].reshape(1, BLOCK_SIZE)
pos = np.arange(0, BLOCK_SIZE, dtype=np.int32).reshape(1, BLOCK_SIZE)

tok_out = tok_emb(x)
pos_out = pos_emb(pos)

print(f"\nToken embedding output:")
print(f"  Shape: {tok_out.shape}")
print(f"  Range: [{tok_out.data.min():.4f}, {tok_out.data.max():.4f}]")
print(f"  Mean: {tok_out.data.mean():.4f}")

print(f"\nPosition embedding output:")
print(f"  Shape: {pos_out.shape}")
print(f"  Range: [{pos_out.data.min():.4f}, {pos_out.data.max():.4f}]")
print(f"  Mean: {pos_out.data.mean():.4f}")

# Test addition
x_emb = tok_out + pos_out
print(f"\nAfter adding embeddings:")
print(f"  Range: [{x_emb.data.min():.4f}, {x_emb.data.max():.4f}]")
print(f"  Mean: {x_emb.data.mean():.4f}")

# Test RMSNorm
ln = nn.RMSNorm(D_MODEL)
x_norm = ln(x_emb)
print(f"\nAfter RMSNorm:")
print(f"  Range: [{x_norm.data.min():.4f}, {x_norm.data.max():.4f}]")
print(f"  Mean: {x_norm.data.mean():.4f}")

# Test Linear
linear = nn.Linear(D_MODEL, D_MODEL)
x_linear = linear(x_norm)
print(f"\nAfter Linear:")
print(f"  Range: [{x_linear.data.min():.4f}, {x_linear.data.max():.4f}]")
print(f"  Mean: {x_linear.data.mean():.4f}")

# Test attention components
print(f"\n--- Testing Attention ---")
# Reshape to (T, C) for attention
x_2d = x_norm.reshape(BLOCK_SIZE, D_MODEL)
print(f"Reshaped to 2D: {x_2d.shape}")
print(f"  Range: [{x_2d.data.min():.4f}, {x_2d.data.max():.4f}]")

q_proj = nn.Linear(D_MODEL, D_MODEL)
q = q_proj(x_2d)
print(f"\nQuery projection:")
print(f"  Range: [{q.data.min():.4f}, {q.data.max():.4f}]")

# Test transpose
q_t = q.transpose()
print(f"\nAfter transpose:")
print(f"  Shape: {q_t.shape}")
print(f"  Range: [{q_t.data.min():.4f}, {q_t.data.max():.4f}]")

# Test matmul
att = q @ q_t
print(f"\nAttention scores (Q @ K.T):")
print(f"  Shape: {att.shape}")
print(f"  Range: [{att.data.min():.4f}, {att.data.max():.4f}]")
print(f"  Mean: {att.data.mean():.4f}")
