"""
Bullet OS - FactBullet Model Training
Standard, proven training stack adapted for FactBullet data.
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
    # Model Architecture (Standard Bullet Config)
    d_model = 256
    n_heads = 4
    n_layers = 8
    ffn_dim = 1024
    max_seq_len = 256
    vocab_size = 0  # Will be loaded from tokenizer
    
    # Training
    batch_size = 4
    learning_rate = 5e-5
    max_steps = 1000  # Reduced for demo speed, but standard loop
    warmup_steps = 100
    weight_decay = 0.01
    grad_clip = 1.0
    
    # Data
    data_path = "factbullet_dataset.jsonl"
    tokenizer_path = "factbullet_tokenizer.json"
    
    # Checkpointing
    checkpoint_dir = "./factbullet_checkpoints"
    save_every = 200
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Tokenizer
# ============================================================================

class SimpleTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.vocab_size = len(self.vocab)
        self.token_to_id = self.vocab # It's already a map in our simple builder
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def encode(self, text):
        # Character level for this simple demo tokenizer
        ids = []
        for char in text:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            else:
                ids.append(self.token_to_id.get('<UNK>', 0))
        return ids
    
    def decode(self, ids):
        return ''.join([self.id_to_token.get(i, '') for i in ids])

# ============================================================================
# Dataset
# ============================================================================

class FactBulletDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        print(f"Loading data from {data_path}...")
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data['text'])
        
        print("Tokenizing...")
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            # Add EOS equivalent if needed, or just concat
            
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens):,}")
    
    def __len__(self):
        # Simple sliding window
        if len(self.tokens) <= self.max_seq_len:
            return 0
        return len(self.tokens) - self.max_seq_len - 1
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.max_seq_len]
        y = self.tokens[idx + 1:idx + self.max_seq_len + 1]
        return x, y

# ============================================================================
# Model Architecture (Standard BulletTransformer)
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
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
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
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class BulletTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.ffn_dim)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        for block in self.blocks: x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)

# ============================================================================
# Training Loop
# ============================================================================

def train():
    config = Config()
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = SimpleTokenizer(config.tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {config.vocab_size}")
    
    dataset = FactBulletDataset(config.data_path, tokenizer, config.max_seq_len)
    # Use num_workers=0 for maximum compatibility/safety
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    print("Creating model...")
    model = BulletTransformer(config).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print("Starting training...")
    model.train()
    step = 0
    data_iter = iter(dataloader)
    
    while step < config.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
        step += 1
    
    print("Training Complete.")
    torch.save(model.state_dict(), f"{config.checkpoint_dir}/final_model.pt")
    print("Model saved.")

if __name__ == "__main__":
    train()
