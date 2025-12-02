"""
Training utilities and Trainer class for Bullet-Core
Production-ready training loop with all bells and whistles
"""

import os
import pickle
import time
import numpy as np
from typing import Optional, Callable
from .tensor import Tensor

class Trainer:
    """
    Production-ready training loop
    
    Features:
    - Progress tracking and logging
    - Gradient clipping
    - Validation
    - Early stopping
    - Checkpointing (save/resume training)
    - Learning rate scheduling
    
    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=mse_loss,
        ...     train_data=train_loader,
        ...     val_data=val_loader,
        ...     max_epochs=100
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn: Callable,
        train_data,
        val_data=None,
        scheduler=None,
        max_epochs=100,
        gradient_clip=1.0,
        checkpoint_dir='./checkpoints',
        early_stopping_patience=10,
        log_interval=10
    ):
        """
        Args:
            model: neural network model
            optimizer: optimizer instance
            loss_fn: loss function (takes pred, target)
            train_data: training data iterator
            val_data: validation data iterator (optional)
            scheduler: learning rate scheduler (optional)
            max_epochs: maximum number of epochs
            gradient_clip: gradient clipping threshold (0 = no clipping)
            checkpoint_dir: directory to save checkpoints
            early_stopping_patience: epochs to wait before early stopping
            log_interval: log every N batches
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.val_data = val_data
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.log_interval = log_interval
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(self.train_data):
            # Convert to Tensors if needed
            if not isinstance(x, Tensor):
                x = Tensor(x, requires_grad=False)
            if not isinstance(y, Tensor):
                y = Tensor(y, requires_grad=False)
            
            # Forward pass
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                self.clip_gradients()
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step (if per-batch scheduling)
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Accumulate loss
            epoch_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Batch {batch_idx + 1} | Loss: {avg_loss:.4f} | LR: {self.optimizer.lr:.6f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        elapsed = time.time() - start_time
        
        return avg_loss, elapsed
    
    def validate(self):
        """Validate on validation set"""
        if self.val_data is None:
            return None
        
        val_loss = 0.0
        num_batches = 0
        
        for x, y in self.val_data:
            # Convert to Tensors
            if not isinstance(x, Tensor):
                x = Tensor(x, requires_grad=False)
            if not isinstance(y, Tensor):
                y = Tensor(y, requires_grad=False)
            
            # Forward pass (no gradients needed)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            
            val_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
            num_batches += 1
        
        return val_loss / num_batches if num_batches > 0 else 0.0
    
    def clip_gradients(self):
        """Clip gradients by global norm"""
        total_norm = 0.0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = np.sum(param.grad ** 2)
                total_norm += param_norm
        
        total_norm = np.sqrt(total_norm)
        
        if total_norm > self.gradient_clip:
            clip_coef = self.gradient_clip / (total_norm + 1e-6)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad * clip_coef
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model and optimizer state"""
        checkpoint = {
            'epoch': epoch,
            'model_state': [p.data for p in self.model.parameters()],
            'optimizer_state': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pkl')
            with open(best_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, path):
        """Load checkpoint and resume training"""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore model parameters
        for param, saved_data in zip(self.model.parameters(), checkpoint['model_state']):
            param.data = saved_data
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print("=" * 60)
        print(f"Starting training for {self.max_epochs} epochs")
        print(f"Model parameters: {sum(p.data.size for p in self.model.parameters())}")
        print("=" * 60)
        
        for epoch in range(self.current_epoch, self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_time = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                self.val_losses.append(val_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} ({train_time:.2f}s)")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self.optimizer.lr:.6f}")
            
            # Check for improvement
            is_best = False
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                print(f"   No improvement for {self.early_stopping_patience} epochs")
                break
        
        print("\n" + "=" * 60)
        print("🎉 Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
