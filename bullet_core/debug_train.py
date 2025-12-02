"""
Debug training to see what's happening
"""

import json
import numpy as np
from bullet_core.tensor import Tensor
from bullet_core import nn
from bullet_core.transformer import GPT

# Load small sample
data_path = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset.jsonl"
text_data = ""
with open(data_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:  # Just 10 examples
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
print(f"Data length: {len(data_ids)}")

# Model
BLOCK_SIZE = 64
D_MODEL = 128
N_HEAD = 4
N_LAYER = 2

model = GPT(vocab_size, D_MODEL, N_HEAD, N_LAYER, max_len=BLOCK_SIZE)
print(f"Parameters: {sum(p.data.size for p in model.parameters())}")

# Get a batch
x = data_ids[:BLOCK_SIZE].reshape(1, BLOCK_SIZE)
y = data_ids[1:BLOCK_SIZE+1].reshape(1, BLOCK_SIZE)

x_tensor = Tensor(x, requires_grad=False)
y_tensor = Tensor(y, requires_grad=False)

# Forward
logits = model(x_tensor)
print(f"\nLogits shape: {logits.shape}")
print(f"Logits range: [{logits.data.min():.4f}, {logits.data.max():.4f}]")
print(f"Logits mean: {logits.data.mean():.4f}")
print(f"Logits std: {logits.data.std():.4f}")

# Loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, y_tensor)

print(f"\nLoss: {loss.data}")
print(f"Loss is finite: {np.isfinite(loss.data)}")

# Check softmax
V = logits.shape[-1]
logits_flat = logits.reshape(-1, V)
probs = logits_flat.softmax(axis=-1)
print(f"\nProbs shape: {probs.shape}")
print(f"Probs sum (should be ~1): {probs.data.sum(axis=1)[:5]}")
print(f"Probs range: [{probs.data.min():.6f}, {probs.data.max():.6f}]")

# Check selected probs
y_onehot = np.zeros((64, vocab_size), dtype=np.float32)
y_onehot[np.arange(64), y.flatten()] = 1.0
selected_probs_data = (y_onehot * probs.data).sum(axis=1)
print(f"\nSelected probs range: [{selected_probs_data.min():.6f}, {selected_probs_data.max():.6f}]")
print(f"Selected probs (first 5): {selected_probs_data[:5]}")
