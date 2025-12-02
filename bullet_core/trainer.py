"""
Production Trainer with all stability features
"""

import numpy as np
from pathlib import Path
from typing import Optional
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from python.tensor import Tensor
from python.transformer import GPT
from python.optim import AdamW
from python import nn

from utils.config import Config
from utils.logger import BulletLogger
from utils.loss_monitor import LossMonitor
from utils.sanitize import load_and_clean_dataset, validate_dataset
from python.training_utils import CheckpointManager, set_seed

class Trainer:
    """Production-grade trainer with all improvements"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Set seed for deterministic training
        set_seed(config.seed)
        
        # Initialize components
        self.logger = BulletLogger()
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_last_n=config.keep_last_n
        )
        
        self.monitor = LossMonitor(
            factor=config.spike_threshold if hasattr(config, 'spike_threshold') else 2.0
        )
        
        # Load data
        self.load_data()
        
        # Build model
        self.build_model()
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Try to recover from checkpoint
        if config.auto_recover:
            self.try_recover()
        else:
            self.start_step = 0
            
    def load_data(self):
        """Load and prepare dataset with multiprocessing"""
        from python.tokenizer import BPETokenizer
        
        print("📚 Loading tokenizer...")
        self.tokenizer = BPETokenizer()
        try:
            self.tokenizer.load(self.config.tokenizer_path)
        except FileNotFoundError:
            print(f"⚠️ Tokenizer not found at {self.config.tokenizer_path}")
            print("🔄 Rebuilding tokenizer automatically...")
            from python.tokenizer import train_marathi_tokenizer
            self.tokenizer = train_marathi_tokenizer(
                data_path=self.config.data_path,
                vocab_size=self.config.vocab_size,
                save_path=self.config.tokenizer_path
            )
        
        print("📂 Loading and cleaning dataset...")
        texts = load_and_clean_dataset(self.config.data_path)
        validate_dataset(texts)
        
        # FIX 3: Sort by length to stabilize gradients (heuristic for stream training)
        print("📊 Sorting dataset by length...")
        texts.sort(key=len)
        
        # Tokenize with multiprocessing
        print("🔤 Tokenizing (Multiprocessed)...")
        
        try:
            with ProcessPoolExecutor(max_workers=4) as executor:
                # Chunk texts to reduce overhead
                chunk_size = max(1, len(texts) // 16)
                chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
                
                results = list(executor.map(self._tokenize_chunk, chunks))
                
            all_tokens = []
            for res in results:
                all_tokens.extend(res)
        except Exception as e:
            print(f"⚠️ Multiprocessing failed ({e}), falling back to single process...")
            all_tokens = []
            for text in texts:
                tokens = self.tokenizer.encode(text)
                # FIX 2: Limit max sequence length
                if len(tokens) > self.config.max_seq_len:
                    tokens = tokens[:self.config.max_seq_len]
                all_tokens.extend(tokens)
        
        self.data = np.array(all_tokens, dtype=np.int32)
        
        # Split train/val
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        
        print(f"✅ Train tokens: {len(self.train_data):,}")
        print(f"✅ Val tokens: {len(self.val_data):,}")

    def _tokenize_chunk(self, texts):
        """Helper for multiprocessing"""
        tokens = []
        for text in texts:
            t = self.tokenizer.encode(text)
            # FIX 2: Limit max sequence length
            if len(t) > self.config.max_seq_len:
                t = t[:self.config.max_seq_len]
            tokens.extend(t)
        return tokens
    
    def build_model(self):
        """Build transformer model"""
        print("🏗️  Building model...")
        self.model = GPT(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_head=self.config.n_heads,
            n_layer=self.config.n_layers,
            max_len=self.config.max_seq_len
        )
        
        total_params = sum(p.data.size for p in self.model.parameters())
        print(f"✅ Parameters: {total_params:,}")
        print(f"✅ Size: ~{total_params * 4 / 1024:.1f} KB")
    
    def try_recover(self):
        """Try to recover from last checkpoint"""
        checkpoint = self.checkpoint_manager.load_latest()
        if checkpoint:
            # Restore model parameters
            params = list(self.model.parameters())
            for i, param in enumerate(params):
                key = f'param_{i}'
                if key in checkpoint['model_params']:
                    param.data = checkpoint['model_params'][key]
            
            # Restore optimizer if available
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None and hasattr(self.optimizer, 'load_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                
            print(f"✅ Recovered from step {checkpoint['step']}")
            self.start_step = checkpoint['step']
        else:
            self.start_step = 0
    
    def get_batch(self, split='train'):
        """Get random batch"""
        data = self.train_data if split == 'train' else self.val_data
        ix = random.randint(0, len(data) - self.config.max_seq_len - 1)
        
        x = data[ix:ix + self.config.max_seq_len]
        y = data[ix + 1:ix + self.config.max_seq_len + 1]
        
        x = x.reshape(1, self.config.max_seq_len)
        y = y.reshape(1, self.config.max_seq_len)
        
        return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)
    
    def train(self):
        """Main training loop"""
        print("=" * 60)
        print(f"🚀 Starting training for {self.config.max_steps} steps")
        print("=" * 60)
        
        tokens_processed = 0
        grad_accum_steps = getattr(self.config, 'grad_accum_steps', 1)
        warmup_steps = int(self.config.max_steps * 0.05)
        base_lr = self.config.learning_rate
        
        for step in range(self.start_step, self.config.max_steps):
            # FIX 1: Realistic Learning Rate & Warmup
            if step < warmup_steps:
                lr = base_lr * (step / warmup_steps)
            else:
                lr = base_lr
            
            self.optimizer.lr = lr
            
            # Gradient Accumulation Loop
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            
            for _ in range(grad_accum_steps):
                # Get batch
                x, y = self.get_batch('train')
                tokens_processed += x.data.size
                
                # Forward
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                # Scale loss for accumulation
                loss_val = loss.data if hasattr(loss, 'data') else loss
                epoch_loss += float(loss_val) / grad_accum_steps
                
                # Backward
                (loss / grad_accum_steps).backward()
            
            # Update Loss Monitor
            self.monitor.update(epoch_loss)
            
            # Check for spikes
            if self.monitor.spike(epoch_loss):
                print(self.monitor.recover_message(epoch_loss, sum(self.monitor.losses)/len(self.monitor.losses)))
                
                # Save spiking batch for inspection
                try:
                    import pickle
                    with open(f"spike_batch_step_{step}.pkl", "wb") as f:
                        pickle.dump({'x': x.data, 'y': y.data, 'loss': epoch_loss}, f)
                    print(f"⚠️ Saved spiking batch to spike_batch_step_{step}.pkl")
                except Exception as e:
                    print(f"⚠️ Failed to save spiking batch: {e}")

                if self.config.auto_recover:
                    self.try_recover()
                    # Reduce LR temporarily
                    self.optimizer.lr *= 0.5
                    continue
            
            # Gradient clipping
            self.clip_gradients()
            
            # Update
            self.optimizer.step()
            
            # Logging
            if step % 10 == 0:
                self.logger.log(
                    step=step,
                    loss=epoch_loss,
                    lr=self.optimizer.lr,
                    total_steps=self.config.max_steps,
                    tokens_processed=tokens_processed
                )
            
            # Checkpointing
            if step % self.config.save_every == 0 and step > 0:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=step,
                    loss=epoch_loss,
                    config=self.config.to_dict()
                )

    def clip_gradients(self):
        """Clip gradients by global norm"""
        total_norm = 0.0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = np.sum(param.grad ** 2)
                total_norm += param_norm
        
        total_norm = np.sqrt(total_norm)
        
        if total_norm > self.config.grad_clip:
            clip_coef = self.config.grad_clip / (total_norm + 1e-6)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad * clip_coef
