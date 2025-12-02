import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Set seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # These ensure deterministic behavior on CUDA but might slow down training
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"🌱 Random seed set to {seed}")
