"""
Model Checkpoint Utilities for Bullet-Core
Save and load model weights, optimizer state, and training metadata
"""

import pickle
import json
import os
import numpy as np
from pathlib import Path

def save_checkpoint(model, optimizer, step, train_losses, val_losses, 
                   best_val_loss, save_path, metadata=None):
    """
    Save complete training checkpoint
    
    Args:
        model: The model to save
        optimizer: Optimizer state
        step: Current training step
        train_losses: List of training losses
        val_losses: List of validation losses
        best_val_loss: Best validation loss so far
        save_path: Path to save checkpoint
        metadata: Optional dict with additional info
    """
    checkpoint = {
        'step': step,
        'model_state': {},
        'optimizer_state': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'metadata': metadata or {}
    }
    
    # Save model parameters
    for i, param in enumerate(model.parameters()):
        checkpoint['model_state'][f'param_{i}'] = param.data.copy()
    
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"✅ Checkpoint saved to: {save_path}")
    print(f"   Step: {step}, Val Loss: {best_val_loss:.4f}")
    
    return save_path

def load_checkpoint(model, optimizer, load_path):
    """
    Load training checkpoint
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to restore state
        load_path: Path to checkpoint file
    
    Returns:
        dict with training state (step, losses, etc.)
    """
    with open(load_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Restore model parameters
    for i, param in enumerate(model.parameters()):
        if f'param_{i}' in checkpoint['model_state']:
            param.data = checkpoint['model_state'][f'param_{i}'].copy()
    
    # Restore optimizer state
    if checkpoint['optimizer_state'] and hasattr(optimizer, 'load_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    print(f"✅ Checkpoint loaded from: {load_path}")
    print(f"   Resuming from step: {checkpoint['step']}")
    print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return {
        'step': checkpoint['step'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'best_val_loss': checkpoint['best_val_loss'],
        'metadata': checkpoint.get('metadata', {})
    }

def save_model_only(model, save_path, metadata=None):
    """
    Save just the model weights (for inference)
    
    Args:
        model: Model to save
        save_path: Path to save model
        metadata: Optional metadata dict
    """
    model_data = {
        'parameters': {},
        'metadata': metadata or {}
    }
    
    for i, param in enumerate(model.parameters()):
        model_data['parameters'][f'param_{i}'] = param.data.copy()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to: {save_path}")
    
    return save_path

def load_model_only(model, load_path):
    """
    Load just model weights (for inference)
    
    Args:
        model: Model to load weights into
        load_path: Path to model file
    
    Returns:
        metadata dict
    """
    with open(load_path, 'rb') as f:
        model_data = pickle.load(f)
    
    for i, param in enumerate(model.parameters()):
        if f'param_{i}' in model_data['parameters']:
            param.data = model_data['parameters'][f'param_{i}'].copy()
    
    print(f"✅ Model loaded from: {load_path}")
    
    return model_data.get('metadata', {})

def save_training_config(config, save_path):
    """Save training configuration as JSON"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved to: {save_path}")

def get_checkpoint_info(checkpoint_dir):
    """
    Get information about all checkpoints in a directory
    
    Returns:
        List of dicts with checkpoint info
    """
    checkpoints = []
    
    if not os.path.exists(checkpoint_dir):
        return checkpoints
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    ckpt = pickle.load(f)
                
                checkpoints.append({
                    'path': filepath,
                    'filename': filename,
                    'step': ckpt.get('step', 0),
                    'val_loss': ckpt.get('best_val_loss', float('inf')),
                    'size_mb': os.path.getsize(filepath) / (1024 * 1024)
                })
            except:
                pass
    
    # Sort by step
    checkpoints.sort(key=lambda x: x['step'])
    
    return checkpoints

def get_best_checkpoint(checkpoint_dir):
    """Get path to checkpoint with best validation loss"""
    checkpoints = get_checkpoint_info(checkpoint_dir)
    
    if not checkpoints:
        return None
    
    best = min(checkpoints, key=lambda x: x['val_loss'])
    return best['path']

def cleanup_old_checkpoints(checkpoint_dir, keep_best=3, keep_latest=2):
    """
    Remove old checkpoints, keeping only the best and latest
    
    Args:
        checkpoint_dir: Directory with checkpoints
        keep_best: Number of best checkpoints to keep
        keep_latest: Number of latest checkpoints to keep
    """
    checkpoints = get_checkpoint_info(checkpoint_dir)
    
    if len(checkpoints) <= keep_best + keep_latest:
        return  # Not enough to clean up
    
    # Sort by validation loss (best first)
    by_loss = sorted(checkpoints, key=lambda x: x['val_loss'])
    best_paths = set(c['path'] for c in by_loss[:keep_best])
    
    # Sort by step (latest first)
    by_step = sorted(checkpoints, key=lambda x: x['step'], reverse=True)
    latest_paths = set(c['path'] for c in by_step[:keep_latest])
    
    # Keep union of best and latest
    keep_paths = best_paths | latest_paths
    
    # Remove others
    removed = 0
    for ckpt in checkpoints:
        if ckpt['path'] not in keep_paths:
            try:
                os.remove(ckpt['path'])
                removed += 1
                print(f"  Removed old checkpoint: {ckpt['filename']}")
            except:
                pass
    
    if removed > 0:
        print(f"✅ Cleaned up {removed} old checkpoints")

# Example usage
if __name__ == "__main__":
    print("Checkpoint utilities loaded successfully!")
    print("\nUsage:")
    print("  save_checkpoint(model, optimizer, step, ...)")
    print("  load_checkpoint(model, optimizer, path)")
    print("  save_model_only(model, path)")
    print("  load_model_only(model, path)")
