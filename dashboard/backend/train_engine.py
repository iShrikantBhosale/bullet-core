import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import time
import threading
import queue
import json
import math
from typing import Dict, Any, List

# Add integration folder to path to import bullet_bindings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../integration")))

try:
    import bullet_bindings
except ImportError:
    print("Warning: bullet_bindings not found. Model export will fail.")

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    print("Warning: sentencepiece not found. Will use simple tokenizer.")
    SENTENCEPIECE_AVAILABLE = False

# ===== BULLET-Spec Compliant Architecture =====
# Matches BULLET_SPEC_v1.0.md and BULLET_CORE_ARCHITECTURE.md

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """Precompute RoPE (Rotary Positional Embedding) frequencies"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs).float()
    # Return as complex numbers for rotation
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary positional embeddings to queries and keys"""
    # Reshape to complex: (batch, seq, heads, head_dim) -> (batch, seq, heads, head_dim/2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis to match shape
    freqs_cis = freqs_cis[:xq.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim/2)
    
    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, freqs_cis):
        batch, seq_len, dim = x.shape
        
        # Project to Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape for multi-head: (batch, seq, dim) -> (batch, seq, heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask (for autoregressive generation)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        
        return self.wo(out)

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (better than standard FFN)"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim  # Standard 4x expansion
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
    
    def forward(self, x):
        # SwiGLU: (SiLU(W1(x)) * W3(x)) @ W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerLayer(nn.Module):
    """BULLET-spec compliant transformer layer with RMSNorm, RoPE, and SwiGLU"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Pre-normalization (RMSNorm)
        self.norm_attn = RMSNorm(dim)
        self.attention = SelfAttention(dim, num_heads)
        
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim)
    
    def forward(self, x, freqs_cis):
        # Attention block with residual
        h = x + self.attention(self.norm_attn(x), freqs_cis)
        
        # FFN block with residual
        out = h + self.ffn(self.norm_ffn(h))
        
        return out

class BulletTransformer(nn.Module):
    """BULLET-spec compliant transformer model"""
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Precompute RoPE frequencies
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(dim // num_heads, max_seq_len)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads) for _ in range(num_layers)
        ])
        
        # Final normalization and output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying (share embeddings with output)
        self.output.weight = self.tok_embeddings.weight
    
    def forward(self, tokens):
        batch, seqlen = tokens.shape
        
        # Embed tokens
        h = self.tok_embeddings(tokens)
        
        # Get position-specific RoPE frequencies
        freqs_cis = self.freqs_cis[:seqlen]
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis)
        
        # Final normalization
        h = self.norm(h)
        
        # Project to vocabulary
        logits = self.output(h)
        
        
        return logits

class TrainingManager:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.is_training = False
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = 0
        self.model_path = ""
        self.should_stop = False
        
        # Production Dashboard: Metrics tracking for graphs
        self.metrics = {
            'steps': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'perplexity': [],
            'best_val_loss': float('inf'),
            'tokens_per_sec': 0
        }
    
    def log(self, message):
        self.log_queue.put(message)
        print(message)  # Also print to console

    def get_logs(self):
        logs = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get())
        return logs

    def stop_training(self):
        if self.is_training:
            self.is_training = False
            self.log("Stopping training...")

    def train_and_export(self, config, text_file_path, output_dir):
        try:
            self.is_training = True
            self.progress = 0.0
            self.model_path = ""
            self.log_queue.queue.clear()
            
            self.log("Starting training pipeline...")
            
            # ===== CRITICAL: CPU OPTIMIZATION MUST BE FIRST =====
            # Set thread count BEFORE any PyTorch operations (including tokenization)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if device.type == "cpu":
                # Optimal thread count: total cores - 1 (leave one for system)
                num_cores = os.cpu_count()
                optimal_threads = max(1, num_cores - 1)
                torch.set_num_threads(optimal_threads)
                
                # Disable inter-op parallelism for better CPU cache utilization
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass  # Already set, ignore
                
                # Check if MKL is available
                mkl_available = 'mkl' in torch.__config__.show().lower()
                
                # Set MKL environment variables if available
                if mkl_available:
                    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
                    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
                
                mkl_str = "with MKL" if mkl_available else "without MKL (slower)"
                self.log(f"Using CPU with {optimal_threads} threads {mkl_str}")
            else:
                self.log(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Use bfloat16 for speed (if CPU supports it)
            use_bfloat16 = device.type == "cpu" and hasattr(torch, 'bfloat16')
            dtype = torch.bfloat16 if use_bfloat16 else torch.float32
            dtype_str = "bfloat16" if use_bfloat16 else "float32"
            self.log(f"Using dtype: {dtype_str}")
            
            # ===== NOW SAFE TO DO DATA LOADING AND TOKENIZATION =====
            
            # 1. Load Data
            self.log(f"Loading text from {text_file_path}...")
            
            text = ""
            if text_file_path.endswith('.jsonl'):
                import json
                # Pre-load all JSONL entries at once (faster than line-by-line)
                entries = []
                with open(text_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            instr = data.get('instruction', '')
                            inp = data.get('input', '')
                            resp = data.get('response', '')
                            
                            entry_text = f"Instruction: {instr}\n"
                            if inp:
                                entry_text += f"Input: {inp}\n"
                            entry_text += f"Response: {resp}\n\n"
                            entries.append(entry_text)
                        except json.JSONDecodeError:
                            continue
                
                text = ''.join(entries)  # Join once instead of repeated concatenation
                self.log(f"Loaded {len(entries)} JSONL entries")
            else:
                with open(text_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Tokenization with SentencePiece or fallback to whitespace
            tokenizer_model_path = os.path.join(os.path.dirname(__file__), 'bullet_tokenizer.model')
            
            if SENTENCEPIECE_AVAILABLE and os.path.exists(tokenizer_model_path):
                self.log(f"Loading SentencePiece tokenizer from {tokenizer_model_path}...")
                sp = spm.SentencePieceProcessor()
                sp.load(tokenizer_model_path)
                
                # Tokenize the text
                self.log("Tokenizing dataset with SentencePiece BPE...")
                tokens = sp.encode(text, out_type=int)
                data = torch.tensor(tokens, dtype=torch.long)
                
                vocab_size = sp.get_piece_size()
                self.log(f"SentencePiece vocab size: {vocab_size}. Total tokens: {len(tokens)}")
                
                # Update config vocab_size to match tokenizer
                config['vocab_size'] = vocab_size
                
                # Save vocab for export
                vocab_path = os.path.join(output_dir, "vocab.txt")
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    for i in range(vocab_size):
                        piece = sp.id_to_piece(i)
                        f.write(f"{piece} 1.0\n")
            else:
                # Fallback to simple whitespace tokenizer
                if not SENTENCEPIECE_AVAILABLE:
                    self.log("SentencePiece not available, using simple tokenizer")
                else:
                    self.log(f"Tokenizer model not found at {tokenizer_model_path}, using simple tokenizer")
                
                self.log("Building vocabulary...")
                words = sorted(list(set(text.split())))
                if len(words) > config['vocab_size']:
                    words = words[:config['vocab_size']]
                
                vocab = {w: i for i, w in enumerate(words)}
                
                # Encode
                self.log("Tokenizing dataset...")
                tokens = [vocab.get(w, 0) for w in text.split()]
                data = torch.tensor(tokens, dtype=torch.long)
                
                self.log(f"Vocab size: {len(vocab)}. Total tokens: {len(tokens)}")
                
                # Save vocab for export
                vocab_path = os.path.join(output_dir, "vocab.txt")
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    for w in words:
                        f.write(f"{w} 1.0\n")
            
            self.log("Initializing model architecture...")
            # 2. Init Model (BULLET-spec compliant)
            model = BulletTransformer(
                vocab_size=config['vocab_size'],
                dim=config['dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                max_seq_len=config['max_seq_len']
            )
            
            # Move to device and dtype in one operation
            model = model.to(device=device, dtype=dtype)
            
            # Apply torch.compile for PyTorch 2.0+ (significant speedup)
            if hasattr(torch, 'compile') and device.type == "cpu":
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    self.log("Applied torch.compile() for faster execution")
                except Exception as e:
                    self.log(f"torch.compile() not available: {e}")
            
            self.log(f"Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
            # Advanced Training Setup
            # 1. Improved Optimizer with better hyperparameters
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                betas=(0.9, 0.95),  # Better for transformers
                weight_decay=config.get('weight_decay', 0.01),
                eps=1e-8
            )
            criterion = nn.CrossEntropyLoss()
            
            # 2. EMA (Exponential Moving Average) for better inference
            use_ema = config.get('use_ema', True)
            ema_decay = config.get('ema_decay', 0.999)
            
            if use_ema:
                from torch.optim.swa_utils import AveragedModel
                ema_model = AveragedModel(
                    model,
                    avg_fn=lambda avg_param, model_param, num_averaged: 
                        ema_decay * avg_param + (1 - ema_decay) * model_param
                )
                self.log(f"EMA enabled with decay={ema_decay}")
            else:
                ema_model = None
            
            # 3. Data Split (Train/Val)
            # Reserve 5% for validation
            val_size = int(len(data) * 0.05)
            train_data = data[:-val_size]
            val_data = data[-val_size:]
            
            self.log(f"Data split: {len(train_data)} train tokens, {len(val_data)} val tokens")
            
            # 4. Checkpointing setup
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            
            # 5. Training Loop Configuration
            batch_size = config['batch_size']
            seq_len = config['max_seq_len']
            gradient_accumulation_steps = config.get('gradient_accumulation_steps', 8)
            effective_batch_size = batch_size * gradient_accumulation_steps
            
            self.total_epochs = config['epochs']
            
            # Calculate steps dynamically based on data size
            mini_batch_size = batch_size
            total_train_batches = (len(train_data) - seq_len - 1) // (mini_batch_size * seq_len)
            steps_per_epoch = total_train_batches // gradient_accumulation_steps
            total_steps = steps_per_epoch * self.total_epochs
            
            # Warmup: 10% of total steps or 100, whichever is smaller
            warmup_steps = min(100, int(total_steps * 0.1))
            
            self.log(f"Training Config:")
            self.log(f"- Effective Batch Size: {effective_batch_size} (Accumulation: {gradient_accumulation_steps})")
            self.log(f"- Total Steps: {total_steps}")
            self.log(f"- Warmup Steps: {warmup_steps}")
            self.log(f"- LR Schedule: Cosine with Warmup")
            
            # Re-define scheduler with dynamic steps
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            # 3. Training Loop Configuration
            self.log(f"Starting training loop... (logs every 10 batches)")
            self.progress = 0.0
            
            global_step = 0
            best_val_loss = float('inf')
            patience_counter = 0
            patience_limit = config.get('patience_early_stop', 5)
            save_every_steps = config.get('save_every_steps', 2500)
            eval_every_steps = config.get('eval_every_steps', 1000)
            
            for epoch in range(self.total_epochs):
                if not self.is_training: break

                self.current_epoch = epoch + 1
                total_loss = 0
                num_batches = 0
                start_time = time.time()
                
                optimizer.zero_grad()
                
                # Training Loop
                for i in range(0, len(train_data) - seq_len - 1, mini_batch_size * seq_len):
                    if not self.is_training: break

                    # Debug log for first few batches
                    if num_batches < 5:
                        self.log(f"[DEBUG] Starting batch {num_batches+1}")

                    chunk_len = mini_batch_size * seq_len
                    if i + chunk_len + 1 > len(train_data): break
                    
                    chunk = train_data[i:i+chunk_len]
                    targets_chunk = train_data[i+1:i+chunk_len+1]
                    
                    inputs = chunk.view(mini_batch_size, seq_len)
                    targets = targets_chunk.view(mini_batch_size, seq_len)
                    
                    # Forward pass
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, config['vocab_size']), targets.view(-1))
                    
                    # Scale loss
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                    # Update weights
                    if (num_batches + 1) % gradient_accumulation_steps == 0:
                        # Gradient Clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('clip_grad_norm', 1.0))
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # Update EMA
                        if use_ema and ema_model is not None:
                            ema_model.update_parameters(model)
                        
                        # Validation and checkpointing
                        if global_step % eval_every_steps == 0:
                            model.eval()
                            with torch.no_grad():
                                # Quick val check on a few batches
                                val_loss = 0
                                val_batches = 0
                                for j in range(0, len(val_data) - seq_len - 1, mini_batch_size * seq_len):
                                    if val_batches >= 20: break # Limit val batches
                                    v_chunk = val_data[j:j+chunk_len]
                                    v_targets = val_data[j+1:j+chunk_len+1]
                                    
                                    if len(v_chunk) < chunk_len or len(v_targets) < chunk_len:
                                        break
                                    
                                    v_in = v_chunk.view(mini_batch_size, seq_len)
                                    v_tgt = v_targets.view(mini_batch_size, seq_len)
                                    v_logits = model(v_in)
                                    v_l = criterion(v_logits.view(-1, config['vocab_size']), v_tgt.view(-1))
                                    val_loss += v_l.item()
                                    val_batches += 1
                                
                                if val_batches > 0:
                                    avg_val_loss = val_loss / val_batches
                                    perplexity = math.exp(min(avg_val_loss, 20))  # Cap for numerical stability
                                    self.log(f"🔍 Validation Step {global_step} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f}")
                                    
                                    # Save best checkpoint
                                    if avg_val_loss < best_val_loss:
                                        best_val_loss = avg_val_loss
                                        patience_counter = 0
                                        
                                        # Save checkpoint
                                        checkpoint = {
                                            'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'epoch': epoch,
                                            'global_step': global_step,
                                            'val_loss': avg_val_loss,
                                            'config': config
                                        }
                                        
                                        if use_ema and ema_model is not None:
                                            checkpoint['ema_state_dict'] = ema_model.state_dict()
                                        
                                        torch.save(checkpoint, best_checkpoint_path)
                                        self.log(f"💾 Saved best checkpoint (val_loss={avg_val_loss:.4f})")
                                    else:
                                        patience_counter += 1
                                        if patience_counter >= patience_limit:
                                            self.log(f"⏹️ Early stopping triggered (patience={patience_limit})")
                                            self.is_training = False
                                            break
                            
                            model.train()

                    total_loss += loss.item() * gradient_accumulation_steps
                    num_batches += 1
                    
                    # Log every batch for debugging
                    if num_batches % 1 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        self.log(f"Epoch {epoch+1} | Batch {num_batches}/{total_train_batches} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | LR: {current_lr:.2e}")
                        
                        # Update progress
                        batch_progress = num_batches / total_train_batches
                        self.progress = ((epoch + batch_progress) / self.total_epochs) * 0.9

                
                if not self.is_training: break # Stop check

                epoch_time = time.time() - start_time
                avg_loss = total_loss / max(1, num_batches)
                self.log(f"Epoch {epoch+1}/{self.total_epochs} Completed in {epoch_time:.2f}s - Avg Loss: {avg_loss:.4f}")
                self.progress = (epoch + 1) / self.total_epochs * 0.8 # 80% for training

            
            if not self.is_training:
                self.log("Training stopped by user.")
                return

            # 4. Load best checkpoint for export
            self.log("Loading best checkpoint for export...")
            if os.path.exists(best_checkpoint_path):
                checkpoint = torch.load(best_checkpoint_path, map_location=device)
                
                # Use EMA model if available, otherwise use regular model
                if use_ema and ema_model is not None and 'ema_state_dict' in checkpoint:
                    ema_model.load_state_dict(checkpoint['ema_state_dict'])
                    export_model = ema_model.module  # Get the underlying model from AveragedModel
                    self.log(f"Using EMA model for export (val_loss={checkpoint['val_loss']:.4f})")
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    export_model = model
                    self.log(f"Using best checkpoint for export (val_loss={checkpoint['val_loss']:.4f})")
            else:
                self.log("No checkpoint found, using final training state")
                export_model = model

            # 5. Export
            self.log("Training complete. Exporting to .bullet...")
            model_name = f"model_{int(time.time())}.bullet"
            output_path = os.path.join(output_dir, model_name)
            
            # Create dummy vocab file for builder (already created during tokenization)
            # vocab_path is already set above
            
            builder = bullet_bindings.BulletBuilder(output_path)
            builder.load_vocab(vocab_path)
            builder.set_metadata(config['vocab_size'], config['num_layers'], config['num_heads'], config['max_seq_len'])
            
            state_dict = export_model.state_dict()
            for name, tensor in state_dict.items():
                # Map names to match bullet-core expectations
                # PyTorch: layers.0.norm_attn.weight -> Bullet: layers.0.norm.attn.weight
                # PyTorch: layers.0.norm_ffn.weight -> Bullet: layers.0.norm.ffn.weight
                
                if "norm_attn" in name:
                    name = name.replace("norm_attn", "norm.attn")
                elif "norm_ffn" in name:
                    name = name.replace("norm_ffn", "norm.ffn")
                
                # Skip biases if bullet-core doesn't support them (it uses RMSNorm usually, which has weight but no bias?)
                # Wait, bullet-core uses RMSNorm. PyTorch LayerNorm has bias by default.
                # BulletTransformer uses RMSNorm (spec-compliant).
                # RMSNorm only has weight parameter, no bias.
                # This matches bullet-core's RMSNorm implementation perfectly.
                # Skip any bias parameters for norm layers (shouldn't exist, but just in case)
                
                if "norm" in name and "bias" in name:
                    continue
                    
                data_np = tensor.detach().cpu().float().numpy().flatten()
                shape = list(tensor.shape)
                builder.add_tensor(name, shape, data_np)
                
            builder.build()
            self.log(f"Model exported to {model_name}")
            self.model_path = model_name
            self.progress = 1.0
            
        except Exception as e:
            self.log(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False

train_manager = TrainingManager()
