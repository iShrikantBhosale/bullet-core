"""
Training utilities for stable, production-ready training
"""

import random
import numpy as np
import os
import pickle
from pathlib import Path

def set_seed(seed=42):
    """
    Set all random seeds for deterministic training
    Same config → same results
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

class CheckpointManager:
    """
    Automatic checkpoint saving and recovery
    Never lose training progress
    """
    
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.keep_last_n = keep_last_n
    
    def save(self, model, optimizer, step, loss, config):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pkl"
        
        # Extract model parameters
        model_params = {}
        for i, param in enumerate(model.parameters()):
            model_params[f'param_{i}'] = param.data.copy()
        
        checkpoint = {
            'step': step,
            'model_params': model_params,
            'optimizer_state': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
            'loss': loss,
            'config': config
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_latest(self):
        """Load the most recent checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pkl"))
        
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        print(f"📥 Restoring from {latest}")
        
        with open(latest, 'rb') as f:
            return pickle.load(f)
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pkl"))
        
        if len(checkpoints) > self.keep_last_n:
            for old_ckpt in checkpoints[:-self.keep_last_n]:
                old_ckpt.unlink()

def sanitize_text(text):
    """
    Clean text data to prevent crashes
    
    - Normalize unicode
    - Remove bad characters
    - Remove blank lines
    """
    import unicodedata
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove control characters except newline/tab
    text = ''.join(char for char in text if char == '\n' or char == '\t' or not unicodedata.category(char).startswith('C'))
    
    # Remove multiple blank lines
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    text = '\n'.join(lines)
    
    return text
