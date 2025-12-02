"""
Configuration loader and validator
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Training configuration with validation"""
    
    def __init__(self, config_path: str = None, **kwargs):
        if config_path:
            self.load_from_file(config_path)
        else:
            self.__dict__.update(kwargs)
        
        self.validate()
    
    def load_from_file(self, path: str):
        """Load config from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.__dict__.update(config_dict)
    
    def validate(self):
        """Validate configuration"""
        errors = []
        
        # Required fields
        required = ['vocab_size', 'd_model', 'n_layers', 'n_heads', 
                   'learning_rate', 'max_seq_len']
        for field in required:
            if not hasattr(self, field):
                errors.append(f"Missing required field: {field}")
        
        # Validation rules
        if hasattr(self, 'max_seq_len') and self.max_seq_len < 16:
            errors.append(f"max_seq_len too small: {self.max_seq_len} < 16")
        
        if hasattr(self, 'vocab_size') and self.vocab_size <= 100:
            errors.append(f"vocab_size too small: {self.vocab_size} <= 100")
        
        if hasattr(self, 'd_model') and hasattr(self, 'n_heads'):
            if self.d_model % self.n_heads != 0:
                errors.append(f"d_model ({self.d_model}) not divisible by n_heads ({self.n_heads})")
        
        if hasattr(self, 'learning_rate'):
            if self.learning_rate <= 0 or self.learning_rate > 0.01:
                errors.append(f"Unsafe learning_rate: {self.learning_rate}")
        
        if hasattr(self, 'batch_size'):
            if self.batch_size <= 0 or self.batch_size > 128:
                errors.append(f"Unsafe batch_size: {self.batch_size}")
        
        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(f"  ❌ {e}" for e in errors))
        
        print("✅ Config validated successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        return f"Config({self.to_dict()})"
