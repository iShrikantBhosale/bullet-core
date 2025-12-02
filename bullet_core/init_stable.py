"""
Stable Weight Initialization for V3.0
Conservative initialization to prevent gradient explosion
"""

import numpy as np
import math

def init_weights_conservative(model, n_layer):
    """
    Initialize model weights with conservative scaling
    
    Args:
        model: Transformer model
        n_layer: Number of layers (for depth-aware scaling)
    """
    print("\n🔧 Applying conservative weight initialization...")
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'embedding' in name:
                # Embedding layers: small std
                std = 0.02
                param.data = np.random.normal(0.0, std, param.data.shape).astype(np.float32)
                print(f"  ✓ {name}: std={std:.4f}")
                
            elif 'linear' in name or 'proj' in name:
                # Linear layers: Xavier with layer scaling
                fan_in = param.data.shape[1] if len(param.data.shape) > 1 else param.data.shape[0]
                std = 0.02 / math.sqrt(2 * n_layer)  # Scale by depth
                param.data = np.random.normal(0.0, std, param.data.shape).astype(np.float32)
                print(f"  ✓ {name}: std={std:.4f} (depth-scaled)")
                
        elif 'bias' in name:
            # All biases: zero
            param.data = np.zeros_like(param.data).astype(np.float32)
    
    print("✅ Conservative initialization complete\n")

# Usage in train_marathi_upgraded.py:
# from init_stable import init_weights_conservative
# init_weights_conservative(model, N_LAYER)
