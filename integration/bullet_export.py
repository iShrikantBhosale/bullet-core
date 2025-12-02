import torch
import os
import sys
import struct
import numpy as np
import subprocess

def fnv1a64(string):
    hash_val = 0xcbf29ce484222325
    prime = 0x1099511628211
    for char in string:
        hash_val ^= ord(char)
        hash_val *= prime
        hash_val &= 0xFFFFFFFFFFFFFFFF
    return hash_val

def export_to_bullet(model, vocab_path, out_path, builder_path="../bullet-builder"):
    """
    Export a PyTorch model to .bullet format using the C++ builder.
    """
    print(f"Exporting model to {out_path}...")
    
    # 1. Prepare Metadata
    # Assume model has config attributes
    hidden_size = getattr(model, 'hidden_size', 256)
    num_layers = getattr(model, 'num_layers', 2)
    num_heads = getattr(model, 'num_heads', 4)
    max_context = getattr(model, 'max_context', 128)
    
    # 2. Create Builder Command
    # We can't easily pass tensors via CLI args.
    # Strategy: Write tensors to temp files or use a pipe?
    # Or better: Use the builder binary to JUST create the structure, 
    # but wait, the builder binary takes vocab and outpath and generates DUMMY data.
    # The prompt says "calls bullet-builder internally".
    # But `bullet-builder.cpp` as written generates dummy data.
    # 
    # FIX: We need to modify `bullet-builder.cpp` to accept tensor data?
    # Or we write a Python script that mimics `bullet-builder.cpp` logic but in Python?
    # The prompt says "calls bullet-builder internally".
    # This implies `bullet-builder` should be able to read weights.
    # 
    # However, `bullet-builder.cpp` currently generates random weights.
    # To make this work "for real", we would need to modify `bullet-builder.cpp` to read from a file.
    # 
    # ALTERNATIVE: Since I cannot modify `bullet-builder.cpp` heavily now (it's "final"),
    # I will implement the BQ4 quantization and file writing IN PYTHON here.
    # This is often cleaner for exporters anyway.
    # 
    # WAIT. The prompt says "calls bullet-builder internally".
    # Maybe it means "uses the builder logic"?
    # 
    # Let's look at `bullet-builder.cpp`. It has `BulletBuilder` class.
    # If I exposed `BulletBuilder` via pybind11, I could use it!
    # 
    # BUT `bullet_python_bindings.cpp` only exposed `BulletModel` (runtime).
    # I should expose `BulletBuilder` too!
    # 
    # Let's update `bullet_python_bindings.cpp` to include `BulletBuilder`.
    # Then this script can import it and use it directly.
    # 
    # I will update `bullet_python_bindings.cpp` in the next step.
    # For now, I will write this script assuming `bullet_bindings.BulletBuilder` exists.
    
    try:
        import bullet_bindings
    except ImportError:
        print("Error: bullet_bindings not found. Build it first.")
        return

    builder = bullet_bindings.BulletBuilder(out_path)
    builder.load_vocab(vocab_path)
    builder.set_metadata(hidden_size, num_layers, num_heads, max_context)
    
    # 3. Iterate State Dict
    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
        # Convert to float32 numpy
        data = tensor.cpu().float().numpy().flatten()
        
        # Shape
        shape = list(tensor.shape)
        
        # Add to builder
        # We need to pass shape as list of ints
        builder.add_tensor(name, shape, data)
        
    builder.build()
    print("Export complete.")

if __name__ == "__main__":
    print("This is a library. Use export_to_bullet(model, vocab, out).")
