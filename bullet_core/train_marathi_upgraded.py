"""
UPGRADED Marathi Philosophy Transformer Training
Implements production-grade stability improvements
"""

import json
import numpy as np
import random
import time
import os
import sys

# Add parent directory to path
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.tensor import Tensor
from python import nn
from python.transformer import GPT
from python.optim import AdamW

# Import training utilities
try:
    from python.training_utils import (
        set_seed, validate_config, CheckpointManager, 
        TrainingLogger, sanitize_text
    )
    UTILS_AVAILABLE = True
except ImportError:
    print("⚠️  Training utilities not available, using basic mode")
    UTILS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set deterministic seed for reproducible results
if UTILS_AVAILABLE:
    set_seed(42)

# Model Architecture
D_MODEL = 256
N_HEAD = 4
N_LAYER = 8
FFN_HIDDEN = 1024
BLOCK_SIZE = 256  # Reduced for stability

# Training Parameters
MAX_STEPS = 10000
BATCH_SIZE = 1
LEARNING_RATE = 5e-5  # Conservative
WEIGHT_DECAY = 1e-2

# Import from python subdirectory
from python.tokenizer import BPETokenizer
from python.checkpoint import (
    save_checkpoint, load_checkpoint, save_model_only,
    get_best_checkpoint, cleanup_old_checkpoints, get_checkpoint_info
)

print("=" * 70)
print("MARATHI PHILOSOPHY TRANSFORMER - UPGRADED TRAINING")
print("=" * 70)

# ============================================================================
# CONFIGURATION - UPGRADED SETTINGS
# ============================================================================

# Model Architecture (V3.0: Larger context, deeper)
BLOCK_SIZE = 512      # was 256 (2x larger context)
D_MODEL = 256         # same
N_HEAD = 4            # same
N_LAYER = 8           # was 6 (deeper for more data)
FFN_HIDDEN = 1024     # same

# Training (V3.0: More steps for convergence)
MAX_STEPS = 20000     # was 15000
BATCH_SIZE = 1        # same
LEARNING_RATE = 1e-4  # was 5e-4 (REDUCED for stability)
WEIGHT_DECAY = 1e-2

# Evaluation & Checkpointing
EVAL_INTERVAL = 500   # same
SAVE_INTERVAL = 1000  # same
LOG_INTERVAL = 10

# Sampling (same)
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.9

# Paths
DATA_PATH = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset_v2.jsonl"  # V2 dataset
TOKENIZER_PATH = "/home/shri/Desktop/bulletOs/bullet_core/marathi_tokenizer.json"
CHECKPOINT_DIR = "./marathi_checkpoints_v3"  # New checkpoint dir for V3

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\n📋 Configuration:")
print(f"  Model: {N_LAYER} layers, {D_MODEL} dim, {N_HEAD} heads")
print(f"  Context: {BLOCK_SIZE} tokens")
print(f"  Training: {MAX_STEPS} steps, LR={LEARNING_RATE}")
print(f"  Sampling: temp={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}")

# ============================================================================
# LOAD TOKENIZER (UPGRADED: BPE instead of char-level)
# ============================================================================

print(f"\n📁 Loading BPE tokenizer...")
tokenizer = BPETokenizer()
tokenizer.load(TOKENIZER_PATH)

vocab_size = len(tokenizer.vocab)
print(f"  Vocabulary size: {vocab_size}")
print(f"  Compression: ~3.3x vs character-level")

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
# BUILD MODEL (UPGRADED: 4 layers, 256 dim)
# ============================================================================

print(f"\n🏗️  Building Transformer...")
model = GPT(vocab_size=vocab_size, 
            d_model=D_MODEL, 
            n_head=N_HEAD, 
            n_layer=N_LAYER, 
            max_len=BLOCK_SIZE)

total_params = sum(p.data.size for p in model.parameters())
print(f"  Parameters: {total_params:,}")
print(f"  Size: ~{total_params * 4 / 1024:.1f} KB")

# ============================================================================
# CRITICAL FIX: Conservative Embedding Initialization
# ============================================================================
print("\n🔧 Applying embedding gradient fix...")

# Get embedding parameters (first two in the model)
params = list(model.parameters())
token_embedding = params[0]  # (vocab_size, d_model)
pos_embedding = params[1]     # (block_size, d_model)

# Reinitialize with much smaller std to prevent gradient explosion
embedding_std = 0.01  # Very conservative (default is ~0.02)
token_embedding.data = np.random.normal(0.0, embedding_std, token_embedding.data.shape).astype(np.float32)
pos_embedding.data = np.random.normal(0.0, embedding_std, pos_embedding.data.shape).astype(np.float32)

print(f"  ✓ Token embedding: std={embedding_std}")
print(f"  ✓ Position embedding: std={embedding_std}")
print()

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
# IMPROVED SAMPLING (UPGRADED: Top-k, Top-p, Temperature)
# ============================================================================

def sample_with_strategy(logits, temperature=0.7, top_k=40, top_p=0.9):
    """
    Sample from logits with temperature, top-k, and top-p filtering
    """
    # Apply temperature
    logits = logits / temperature
    
    # Softmax
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    
    # Top-k filtering
    if top_k > 0:
        top_k_indices = np.argsort(probs)[-top_k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        # Sample from top-k
        sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
        return top_k_indices[sampled_idx]
    
    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)
        
        # Find cutoff
        cutoff_idx = np.searchsorted(cumsum_probs, top_p) + 1
        nucleus_indices = sorted_indices[:cutoff_idx]
        nucleus_probs = probs[nucleus_indices]
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
        
        # Sample from nucleus
        sampled_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
        return nucleus_indices[sampled_idx]
    
    # Standard sampling
    return np.random.choice(len(probs), p=probs)

def generate(prompt="", max_new_tokens=100):
    """Generate text with improved sampling"""
    context = tokenizer.encode(prompt) if prompt else [0]
    generated = context.copy()
    
    for _ in range(max_new_tokens):
        # Get context
        ctx = generated[-BLOCK_SIZE:] if len(generated) >= BLOCK_SIZE else generated
        ctx = ctx + [0] * (BLOCK_SIZE - len(ctx))
        
        # Forward
        ctx_tensor = Tensor(np.array([ctx], dtype=np.int32), requires_grad=False)
        logits = model(ctx_tensor)
        
        # Get logits for last position
        logits_last = logits.data[0, len(generated) if len(generated) < BLOCK_SIZE else -1, :]
        
        # Sample with strategy
        next_token = sample_with_strategy(
            logits_last, 
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P
        )
        
        generated.append(int(next_token))
    
    return tokenizer.decode(generated)

# ============================================================================
# TRAINING LOOP (UPGRADED: Checkpointing, Better logging)
# ============================================================================

print(f"\n🚀 Starting training...")
print("=" * 70)

best_val_loss = float('inf')
train_losses = []
val_losses = []

start_time = time.time()
start_step = 0

# Check for existing checkpoints to resume
checkpoints = get_checkpoint_info(CHECKPOINT_DIR)
if checkpoints:
    latest_ckpt = max(checkpoints, key=lambda x: x['step'])
    print(f"\n🔄 Resuming from checkpoint: {latest_ckpt['filename']}")
    
    checkpoint = load_checkpoint(model, optimizer, latest_ckpt['path'])
    start_step = checkpoint['step'] + 1
    best_val_loss = checkpoint['best_val_loss']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    print(f"   Resuming at step {start_step}")

for step in range(start_step, MAX_STEPS):
    # Training step
    x, y = get_batch('train')
    
    logits = model(x)
    loss = criterion(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # ========================================================================
    # DIAGNOSTIC MONITORING (Phase 1: Diagnosis)
    # ========================================================================
    
    # 1. Gradient Analysis - Identify exploding layers
    if step % 10 == 0:  # Log every 10 steps
        print(f"\n📊 Gradient Analysis (Step {step}):")
        suspicious_grads = False
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = np.linalg.norm(param.grad)
                grad_max = np.abs(param.grad).max()
                if grad_norm > 10.0 or grad_max > 5.0:
                    print(f"  ⚠️  Param shape {param.data.shape}: norm={grad_norm:.2f}, max={grad_max:.2f}")
                    suspicious_grads = True
        if not suspicious_grads:
            print(f"  ✓ All gradients stable")
    
    # 2. Weight Statistics - Detect weight explosion
    if step % 10 == 0:
        print(f"\n📈 Weight Statistics (Step {step}):")
        suspicious_weights = False
        for param in model.parameters():
            weight_norm = np.linalg.norm(param.data)
            weight_max = np.abs(param.data).max()
            weight_mean = np.abs(param.data).mean()
            if weight_max > 100.0 or weight_norm > 1000.0:
                print(f"  ⚠️  Param shape {param.data.shape}: norm={weight_norm:.2f}, max={weight_max:.2f}, mean={weight_mean:.2f}")
                suspicious_weights = True
        if not suspicious_weights:
            print(f"  ✓ All weights stable")
    
    # ========================================================================
    # CRITICAL FIX: Embedding-Aware Gradient Clipping
    # ========================================================================
    
    # Clip embedding gradients MUCH more aggressively (they accumulate over sequence)
    embedding_max_norm = 0.1  # Very strict for embeddings
    general_max_norm = 1.0     # Normal for other layers
    
    params_list = list(model.parameters())
    
    for i, param in enumerate(params_list):
        if param.grad is not None:
            # First two parameters are embeddings
            max_norm = embedding_max_norm if i < 2 else general_max_norm
            
            grad_norm = np.linalg.norm(param.grad)
            if grad_norm > max_norm:
                param.grad = param.grad * (max_norm / grad_norm)
            
            # Additional safety: clip individual gradient values
            param.grad = np.clip(param.grad, -10.0, 10.0)
    
    optimizer.step()
    
    train_losses.append(float(loss.data))
    
    # Logging
    if step % LOG_INTERVAL == 0:
        elapsed = time.time() - start_time
        print(f"Step {step:5d} | Loss: {loss.data:.4f} | Time: {elapsed:.1f}s")
    
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
        
        # Save checkpoint
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
                'block_size': BLOCK_SIZE
            }
        )
        
        # Cleanup old checkpoints (keep best 3 and latest 2)
        cleanup_old_checkpoints(CHECKPOINT_DIR, keep_best=3, keep_latest=2)

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} minutes")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final training loss: {train_losses[-1]:.4f}")

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
        'best_val_loss': float(best_val_loss)
    }
)

# Save best model (if different from final)
best_checkpoint_path = get_best_checkpoint(CHECKPOINT_DIR)
if best_checkpoint_path:
    print(f"\n📊 Best checkpoint: {best_checkpoint_path}")
    print(f"   Validation loss: {best_val_loss:.4f}")

print(f"\n💾 Models saved to: {CHECKPOINT_DIR}/")
print(f"   - final_model.pkl (last step)")
print(f"   - checkpoint_step_*.pkl (periodic saves)")
print(f"   - Use get_best_checkpoint() to find best model")

# Final generation test
print(f"\n{'='*70}")
print("FINAL GENERATION TESTS")
print(f"{'='*70}")

test_prompts = [
    "",
    "जीवन",
    "आत्मा",
    "ध्यान"
]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 70)
    generated = generate(prompt, max_new_tokens=150)
    print(generated[:300])
    print("-" * 70)

print(f"\n✅ Training session complete!")
print(f"Model saved to: {CHECKPOINT_DIR}")
