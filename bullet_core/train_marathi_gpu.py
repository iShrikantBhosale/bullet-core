"""
UPGRADED Marathi Philosophy Transformer Training - GPU ENABLED
Automatically uses CUDA if available, falls back to CPU
"""

import json
import numpy as np
import random
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bullet_core.tensor import Tensor
from bullet_core import nn
from bullet_core.transformer import GPT
from bullet_core.optim import AdamW

# Import from python subdirectory
from python.tokenizer import BPETokenizer
from python.checkpoint import (
    save_checkpoint, load_checkpoint, save_model_only,
    get_best_checkpoint, cleanup_old_checkpoints
)

print("=" * 70)
print("MARATHI PHILOSOPHY TRANSFORMER - GPU-ENABLED TRAINING")
print("=" * 70)

# ============================================================================
# GPU DETECTION AND SETUP
# ============================================================================

def detect_cuda():
    """Detect if CUDA is available"""
    try:
        import bullet_core
        has_cuda = hasattr(bullet_core, 'bullet_core_cuda')
        if has_cuda:
            print("\n🎮 CUDA Status: AVAILABLE")
            print("   GPU acceleration: ENABLED")
            return True
        else:
            print("\n💻 CUDA Status: NOT AVAILABLE")
            print("   GPU acceleration: DISABLED (using CPU)")
            return False
    except:
        print("\n💻 CUDA Status: NOT AVAILABLE")
        print("   GPU acceleration: DISABLED (using CPU)")
        return False

USE_CUDA = detect_cuda()
DEVICE = 'cuda' if USE_CUDA else 'cpu'

if USE_CUDA:
    # Try to get GPU info
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                               '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"   GPU: {gpu_info}")
    except:
        pass

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR GT 730 (2GB VRAM)
# ============================================================================

# Model Architecture
BLOCK_SIZE = 128
D_MODEL = 256
N_HEAD = 4
N_LAYER = 4
FFN_HIDDEN = 1024

# Training
MAX_STEPS = 10000
BATCH_SIZE = 1  # Keep at 1 for GT 730's 2GB VRAM
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-2

# Evaluation & Checkpointing
EVAL_INTERVAL = 500
SAVE_INTERVAL = 1000
LOG_INTERVAL = 10

# Sampling
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.9

# Paths
DATA_PATH = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset.jsonl"
TOKENIZER_PATH = "/home/shri/Desktop/bulletOs/bullet_core/marathi_tokenizer.json"
CHECKPOINT_DIR = "./marathi_checkpoints_gpu" if USE_CUDA else "./marathi_checkpoints_cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\n📋 Configuration:")
print(f"  Device: {DEVICE.upper()}")
print(f"  Model: {N_LAYER} layers, {D_MODEL} dim, {N_HEAD} heads")
print(f"  Context: {BLOCK_SIZE} tokens")
print(f"  Training: {MAX_STEPS} steps, LR={LEARNING_RATE}")
print(f"  Checkpoints: {CHECKPOINT_DIR}/")

# ============================================================================
# LOAD TOKENIZER
# ============================================================================

print(f"\n📁 Loading BPE tokenizer...")
tokenizer = BPETokenizer()
tokenizer.load(TOKENIZER_PATH)

vocab_size = len(tokenizer.vocab)
print(f"  Vocabulary size: {vocab_size}")

# ============================================================================
# LOAD & TOKENIZE DATA
# ============================================================================

print(f"\n📁 Loading dataset...")
text_data = ""
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        text_data += item['instruction'] + " " + item['response'] + "\n"

print(f"  Total characters: {len(text_data):,}")

print(f"\n🔄 Tokenizing with BPE...")
data_ids = np.array(tokenizer.encode(text_data), dtype=np.int32)

# Split train/val
n = int(0.9 * len(data_ids))
train_data = data_ids[:n]
val_data = data_ids[n:]

print(f"  Train tokens: {len(train_data):,}")
print(f"  Val tokens: {len(val_data):,}")

# ============================================================================
# BUILD MODEL
# ============================================================================

print(f"\n🏗️  Building Transformer...")
model = GPT(vocab_size, D_MODEL, N_HEAD, N_LAYER, max_len=BLOCK_SIZE)

total_params = sum(p.data.size for p in model.parameters())
print(f"  Parameters: {total_params:,}")
print(f"  Size: ~{total_params * 4 / 1024:.1f} KB")

if USE_CUDA:
    print(f"  ⚡ Model will use GPU acceleration")

# ============================================================================
# TRAINING SETUP
# ============================================================================

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

def get_batch(split):
    """Get a random batch"""
    data = train_data if split == 'train' else val_data
    ix = random.randint(0, len(data) - BLOCK_SIZE - 1)
    x = data[ix:ix+BLOCK_SIZE]
    y = data[ix+1:ix+BLOCK_SIZE+1]
    
    x = x.reshape(1, BLOCK_SIZE)
    y = y.reshape(1, BLOCK_SIZE)
    
    return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)

# ============================================================================
# IMPROVED SAMPLING
# ============================================================================

def sample_with_strategy(logits, temperature=0.7, top_k=40, top_p=0.9):
    """Sample from logits with temperature, top-k, and top-p filtering"""
    logits = logits / temperature
    
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    
    if top_k > 0:
        top_k_indices = np.argsort(probs)[-top_k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
        return top_k_indices[sampled_idx]
    
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)
        
        cutoff_idx = np.searchsorted(cumsum_probs, top_p) + 1
        nucleus_indices = sorted_indices[:cutoff_idx]
        nucleus_probs = probs[nucleus_indices]
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
        
        sampled_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
        return nucleus_indices[sampled_idx]
    
    return np.random.choice(len(probs), p=probs)

def generate(prompt="", max_new_tokens=100):
    """Generate text with improved sampling"""
    context = tokenizer.encode(prompt) if prompt else [0]
    generated = context.copy()
    
    for _ in range(max_new_tokens):
        ctx = generated[-BLOCK_SIZE:] if len(generated) >= BLOCK_SIZE else generated
        ctx = ctx + [0] * (BLOCK_SIZE - len(ctx))
        
        ctx_tensor = Tensor(np.array([ctx], dtype=np.int32), requires_grad=False)
        logits = model(ctx_tensor)
        
        logits_last = logits.data[0, len(generated) if len(generated) < BLOCK_SIZE else -1, :]
        
        next_token = sample_with_strategy(
            logits_last, 
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P
        )
        
        generated.append(int(next_token))
    
    return tokenizer.decode(generated)

# ============================================================================
# TRAINING LOOP
# ============================================================================

print(f"\n🚀 Starting training...")
print("=" * 70)

best_val_loss = float('inf')
train_losses = []
val_losses = []

start_time = time.time()
step_times = []

for step in range(MAX_STEPS):
    step_start = time.time()
    
    # Training step
    x, y = get_batch('train')
    
    logits = model(x)
    loss = criterion(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(float(loss.data))
    
    step_time = time.time() - step_start
    step_times.append(step_time)
    
    # Logging
    if step % LOG_INTERVAL == 0:
        elapsed = time.time() - start_time
        avg_step_time = np.mean(step_times[-100:]) if step_times else step_time
        steps_remaining = MAX_STEPS - step
        eta_seconds = avg_step_time * steps_remaining
        eta_minutes = eta_seconds / 60
        
        print(f"Step {step:5d} | Loss: {loss.data:.4f} | "
              f"Time: {elapsed:.1f}s | Step: {step_time:.3f}s | ETA: {eta_minutes:.1f}min")
    
    # Evaluation
    if step % EVAL_INTERVAL == 0 and step > 0:
        print(f"\n{'='*70}")
        print(f"EVALUATION AT STEP {step}")
        print(f"{'='*70}")
        
        # Compute validation loss
        val_loss_sum = 0.0
        num_val_batches = 10
        for _ in range(num_val_batches):
            x_val, y_val = get_batch('val')
            logits_val = model(x_val)
            loss_val = criterion(logits_val, y_val)
            val_loss_sum += float(loss_val.data)
        
        val_loss = val_loss_sum / num_val_batches
        val_losses.append(val_loss)
        
        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Train Loss (avg last 100): {np.mean(train_losses[-100:]):.4f}")
        print(f"Avg step time: {np.mean(step_times[-100:]):.3f}s")
        
        # Generate sample
        print(f"\n📝 Generated Sample:")
        print("-" * 70)
        sample = generate("", max_new_tokens=100)
        print(sample[:200])
        print("-" * 70)
        
        # Save if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"\n✅ New best validation loss: {val_loss:.4f}")
        
        print(f"{'='*70}\n")
    
    # Checkpointing
    if step % SAVE_INTERVAL == 0 and step > 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{step}.pkl")
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val_loss,
            save_path=checkpoint_path,
            metadata={
                'vocab_size': vocab_size,
                'd_model': D_MODEL,
                'n_layer': N_LAYER,
                'n_head': N_HEAD,
                'block_size': BLOCK_SIZE,
                'device': DEVICE
            }
        )
        
        cleanup_old_checkpoints(CHECKPOINT_DIR, keep_best=3, keep_latest=2)

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*70}")

total_time = time.time() - start_time
print(f"\nTotal time: {total_time / 60:.1f} minutes")
print(f"Average step time: {np.mean(step_times):.3f}s")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final training loss: {train_losses[-1]:.4f}")

if USE_CUDA:
    speedup = 0.5 / np.mean(step_times)  # Assuming CPU is 0.5s/step
    print(f"\n⚡ GPU Speedup: ~{speedup:.1f}x faster than CPU")

# Save final model
final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pkl")
save_model_only(
    model=model,
    save_path=final_model_path,
    metadata={
        'vocab_size': vocab_size,
        'd_model': D_MODEL,
        'n_layer': N_LAYER,
        'n_head': N_HEAD,
        'block_size': BLOCK_SIZE,
        'total_steps': MAX_STEPS,
        'final_train_loss': float(train_losses[-1]),
        'best_val_loss': float(best_val_loss),
        'device': DEVICE,
        'training_time_minutes': total_time / 60
    }
)

best_checkpoint_path = get_best_checkpoint(CHECKPOINT_DIR)
if best_checkpoint_path:
    print(f"\n📊 Best checkpoint: {best_checkpoint_path}")
    print(f"   Validation loss: {best_val_loss:.4f}")

print(f"\n💾 Models saved to: {CHECKPOINT_DIR}/")
print(f"   - final_model.pkl (last step)")
print(f"   - checkpoint_step_*.pkl (periodic saves)")

# Final generation tests
print(f"\n{'='*70}")
print("FINAL GENERATION TESTS")
print(f"{'='*70}")

test_prompts = ["", "जीवन", "आत्मा", "ध्यान"]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 70)
    generated = generate(prompt, max_new_tokens=150)
    print(generated[:300])
    print("-" * 70)

print(f"\n✅ Training session complete!")
print(f"Device used: {DEVICE.upper()}")
