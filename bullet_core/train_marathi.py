"""
Train Marathi Philosophy Model using Bullet-Core Transformer
"""

import json
import numpy as np
import random
from bullet_core.tensor import Tensor
from bullet_core import nn
from bullet_core.transformer import GPT
from bullet_core.optim import AdamW
from bullet_core.scheduler import WarmupCosineAnnealing
from bullet_core.trainer import Trainer

print("=" * 60)
print("Marathi Philosophy Transformer Training")
print("=" * 60)

# 1. Load Data & Build Tokenizer
print("\n📁 Loading dataset...")
data_path = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset.jsonl"

text_data = ""
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        text_data += item['instruction'] + " " + item['response'] + "\n"

# Character-level tokenizer
chars = sorted(list(set(text_data)))
vocab_size = len(chars)
print(f"✅ Vocab size: {vocab_size}")
print(f"✅ Total characters: {len(text_data)}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(text):
    return [stoi[c] for c in text]

def decode(indices):
    return ''.join([itos[i] for i in indices])

# 2. Prepare Data
print("\n🔄 Tokenizing data...")
data_ids = np.array(encode(text_data), dtype=np.int32)

# Split train/val
n = int(0.9 * len(data_ids))
train_data = data_ids[:n]
val_data = data_ids[n:]

print(f"Train tokens: {len(train_data)}")
print(f"Val tokens: {len(val_data)}")

# 3. Model Config
BLOCK_SIZE = 64 # Context length
BATCH_SIZE = 1  # Forced to 1 due to engine limitations
D_MODEL = 128
N_HEAD = 4
N_LAYER = 2

print("\n🏗️  Building Transformer...")
model = GPT(vocab_size, D_MODEL, N_HEAD, N_LAYER, max_len=BLOCK_SIZE)

print(f"Parameters: {sum(p.data.size for p in model.parameters())}")

# 4. Training Setup
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
# scheduler = WarmupCosineAnnealing(optimizer, warmup_steps=100, T_max=1000)

criterion = nn.CrossEntropyLoss()

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = random.randint(0, len(data) - BLOCK_SIZE - 1)
    x = data[ix:ix+BLOCK_SIZE]
    y = data[ix+1:ix+BLOCK_SIZE+1]
    
    # Reshape for batch size 1
    x = x.reshape(1, BLOCK_SIZE)
    y = y.reshape(1, BLOCK_SIZE)
    
    return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)

# Custom training loop because Trainer expects iterator
print("\n🚀 Starting training...")
MAX_STEPS = 500
EVAL_INTERVAL = 50

for step in range(MAX_STEPS):
    # Get batch
    x, y = get_batch('train')
    
    # Forward
    logits = model(x)
    loss = criterion(logits, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Update
    optimizer.step()
    
    # Log
    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.data:.4f}")
        
    # Generate sample
    if step % EVAL_INTERVAL == 0:
        print("\nGenerating...")
        # Simple generation
        ctx = x.data[0].tolist()
        for _ in range(20):
            ctx_tensor = Tensor(np.array([ctx[-BLOCK_SIZE:]], dtype=np.int32), requires_grad=False)
            logits = model(ctx_tensor)
            # Greedy decode
            next_token = np.argmax(logits.data[0, -1])
            ctx.append(next_token)
        print(f"Output: {decode(ctx[-20:])}\n")

print("\n✅ Training Complete!")
