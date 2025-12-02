"""
Bullet OS - Production Model Training (PyTorch)
Stable, proven training stack for real results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import math
from pathlib import Path
import time

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Model Architecture
    d_model = 256
    n_heads = 4
    n_layers = 8
    ffn_dim = 1024
    max_seq_len = 256  # Conservative for stability
    vocab_size = 1511  # From tokenizer
    
    # Training
    batch_size = 4
    learning_rate = 5e-5  # Conservative
    max_steps = 15000
    warmup_steps = 500
    weight_decay = 0.01
    grad_clip = 1.0
    
    # Data
    data_path = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset_v2.jsonl"
    tokenizer_path = "/home/shri/Desktop/bulletOs/bullet_core/marathi_tokenizer.json"
    
    # Checkpointing
    checkpoint_dir = "./pytorch_checkpoints"
    save_every = 1000
    eval_every = 500
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Tokenizer
# ============================================================================

class BPETokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = data.get('merges', [])
        self.vocab_size = len(self.vocab)
        
        # Create reverse mapping
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
    
    def encode(self, text):
        # Simple word-level tokenization for now
        tokens = text.split()
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id.get('<unk>', 0))
        return ids
    
    def decode(self, ids):
        return ' '.join([self.vocab[i] for i in ids if i < len(self.vocab)])

# ============================================================================
# Dataset
# ============================================================================

class MarathiDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load all text
        print(f"Loading data from {data_path}...")
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data['text'])
        
        # Tokenize everything
        print("Tokenizing...")
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens):,}")
    
    def __len__(self):
        return len(self.tokens) - self.max_seq_len - 1
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.max_seq_len]
        y = self.tokens[idx + 1:idx + self.max_seq_len + 1]
        return x, y

# ============================================================================
# Model Architecture
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = 1.0 / math.sqrt(self.d_head)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )
    
    def forward(self, x, mask=None):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class BulletTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.ffn_dim)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)
        
        x = tok_emb + pos_emb
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

# ============================================================================
# Training
# ============================================================================

def train():
    config = Config()
    
    # Setup
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer(config.tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    
    # Create dataset
    dataset = MarathiDataset(config.data_path, tokenizer, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    
    # Create model
    print("Creating model...")
    model = BulletTransformer(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler
    def get_lr(step):
        if step < config.warmup_steps:
            return config.learning_rate * step / config.warmup_steps
        return config.learning_rate * 0.5 * (1 + math.cos(math.pi * step / config.max_steps))
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    model.train()
    step = 0
    data_iter = iter(dataloader)
    losses = []
    start_time = time.time()
    
    while step < config.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        
        x, y = x.to(device), y.to(device)
        
        # Forward
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Update
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # Logging
        if step % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")
        
        # Checkpointing
        if step % config.save_every == 0 and step > 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'config': config.__dict__
            }, checkpoint_path)
            print(f"✅ Saved checkpoint: {checkpoint_path}")
        
        step += 1
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Save final model
    final_path = Path(config.checkpoint_dir) / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__
    }, final_path)
    print(f"✅ Saved final model: {final_path}")

if __name__ == "__main__":
    train()
