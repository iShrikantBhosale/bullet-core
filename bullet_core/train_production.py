"""
Main training script - Production Grade
Uses modular architecture with all improvements
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from trainer import Trainer

def main():
    # Load config
    config_path = str(Path(__file__).parent / "configs/marathi_small.yaml")
    print(f"📋 Loading config from {config_path}...")
    
    config = Config(config_path)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
