"""
Test Script for Upgraded Marathi Philosophy Model
Loads the trained model and generates text for specific prompts.
"""

import sys
import os
import time
import numpy as np
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.tensor import Tensor
from python.transformer import GPT
from python.tokenizer import BPETokenizer

# Configuration
CHECKPOINT_DIR = "./marathi_checkpoints_upgraded"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final_model.pkl")
TOKENIZER_PATH = "./marathi_tokenizer.json"

def load_model(path):
    print(f"Loading model from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    metadata = data.get('metadata', {})
    
    # Reconstruct model
    model = GPT(
        vocab_size=metadata['vocab_size'],
        d_model=metadata['d_model'],
        n_head=metadata['n_head'],
        n_layer=metadata['n_layer'],
        max_len=metadata['block_size']
    )
    
    # Load weights
    for i, param in enumerate(model.parameters()):
        if f'param_{i}' in data['parameters']:
            param.data = data['parameters'][f'param_{i}']
            
    return model, metadata

def sample_with_strategy(logits, temperature=0.7, top_k=40, top_p=0.9):
    """Sample from logits with temperature, top-k, and top-p filtering"""
    # Apply temperature
    logits = logits / temperature
    
    # Softmax
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    
    # Top-k filtering
    if top_k > 0:
        top_k_indices = np.argsort(probs)[-top_k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        # Sample from top-k
        sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
        return top_k_indices[sampled_idx]
    
    return np.random.choice(len(probs), p=probs)

def generate(model, tokenizer, prompt="", max_new_tokens=100, block_size=128):
    """Generate text"""
    context = tokenizer.encode(prompt) if prompt else [0]
    generated = context.copy()
    
    start_time = time.time()
    
    for _ in range(max_new_tokens):
        # Get context
        ctx = generated[-block_size:] if len(generated) >= block_size else generated
        ctx_padded = ctx + [0] * (block_size - len(ctx))
        
        # Forward
        ctx_tensor = Tensor(np.array([ctx_padded], dtype=np.int32), requires_grad=False)
        logits = model(ctx_tensor)
        
        # Get logits for last position
        logits_last = logits.data[0, len(generated) if len(generated) < block_size else -1, :]
        
        # Sample
        next_token = sample_with_strategy(logits_last)
        generated.append(int(next_token))
    
    elapsed = time.time() - start_time
    tps = max_new_tokens / elapsed
    
    return tokenizer.decode(generated), tps

def main():
    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)
    
    # Load Model
    model, metadata = load_model(MODEL_PATH)
    
    print("\nModel Specs:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
        
    # Test Prompts
    prompts = [
        "जीवन",
        "आत्मा",
        "ध्यान",
        "कर्म",
        "सुख"
    ]
    
    print("\n" + "="*60)
    print("GENERATION TESTS")
    print("="*60)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        
        output, tps = generate(
            model, 
            tokenizer, 
            prompt, 
            max_new_tokens=100, 
            block_size=metadata['block_size']
        )
        
        print(output)
        print("-" * 60)
        print(f"Speed: {tps:.1f} tokens/sec")

if __name__ == "__main__":
    main()
