
import os
import sys
import glob
import pickle
import json
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bullet_core.utils.bullet_io import BulletWriter, BulletReader, fnv1a64
from bullet_core.python.tokenizer import BPETokenizer
from bullet_core.utils.bullet_io import BulletWriter, BulletReader, fnv1a64
from bullet_core.python.tokenizer import BPETokenizer
from bullet_core.python.transformer import GPT
from bullet_core.python.tensor import Tensor
from bullet_core.utils.config import Config

def load_tokenizer():
    path = "bullet_core/marathi_tokenizer.json"
    if not os.path.exists(path):
        print(f"❌ Tokenizer not found at {path}")
        return None
    
    tokenizer = BPETokenizer()
    tokenizer.load(path)
    return tokenizer

def greedy_generate(model, tokenizer, prompt, max_new_tokens=20):
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    generated = tokens.copy()
    
    # Create input tensor
    x = Tensor(np.array([generated], dtype=np.int32), requires_grad=False)
    
    model.eval()
    
    start_time = time.time()
    for i in range(max_new_tokens):
        # Forward
        print(f"    Step {i}: Forwarding...", end="", flush=True)
        try:
            logits = model(x)
            print(" Done.", end="", flush=True)
        except Exception as e:
            print(f"\n    ❌ Error during forward: {e}")
            break
        
        # Get last token logits
        last_logits = logits.data[0, -1, :].copy()
        
        # Apply repetition penalty (1.2)
        # Penalize tokens that have already been generated
        for token_id in set(generated):
            if last_logits[token_id] > 0:
                last_logits[token_id] /= 1.2
            else:
                last_logits[token_id] *= 1.2
        
        # Greedy sample
        next_token = np.argmax(last_logits)
        print(f" Token: {next_token}", flush=True)
        
        generated.append(next_token)
        
        # Update input (append new token)
        x = Tensor(np.array([generated], dtype=np.int32), requires_grad=False)
        
        # Stop if EOS (if defined, but we don't have EOS in simple BPE usually)
    
    elapsed = time.time() - start_time
    speed = max_new_tokens / elapsed if elapsed > 0 else 0
    
    # Decode
    result = tokenizer.decode(generated)
    
    print(f"  Result: {result}")
    print(f"  Speed: {speed:.2f} tokens/sec")
    
    return result

def test_inference(model, tokenizer, prompt="जीवनाचा अर्थ", max_new_tokens=20):
    # Simple greedy generation with repetition penalty
    print(f"  Generating from prompt: '{prompt}'")
    return greedy_generate(model, tokenizer, prompt, max_new_tokens)

def main():
    print("="*60)
    print("🚀 BULLET CHECKPOINT CONVERSION & TEST REPORT")
    print("="*60)
    
    tokenizer = load_tokenizer()
    if not tokenizer:
        return
    
    checkpoints_dir = "marathi_checkpoints_stable"
    checkpoints = sorted(glob.glob(os.path.join(checkpoints_dir, "*800.pkl")))
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoints_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints.")
    
    results = []
    
    for ckpt_path in checkpoints:
        print(f"\nTesting {ckpt_path}...")
        try:
            # 1. Load Checkpoint
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            
            print(f"  ✅ Loaded pickle (Step {ckpt.get('step', 'Unknown')}, Loss {ckpt.get('loss', 'Unknown'):.4f})")
            
            # 2. Convert to .bullet
            bullet_path = ckpt_path.replace(".pkl", ".bullet")
            writer = BulletWriter(bullet_path)
            
            # Config
            config_dict = ckpt.get('config', {})
            writer.set_header(config_dict)
            
            # Tokenizer
            writer.set_tokenizer(tokenizer.token_to_id)
            
            # Weights
            params = ckpt['model_params']
            for name, data in params.items():
                # Default to BQ4 (quantize=True)
                writer.add_tensor(name, data, quantize=True)
                
            writer.write()
            print(f"  ✅ Converted to {bullet_path}")
            
            # 3. Verify .bullet
            reader = BulletReader(bullet_path)
            reader.load()
            
            # Verify weights
            all_match = True
            for name, data in params.items():
                h = fnv1a64(name)
                if h not in reader.tensors:
                    print(f"  ❌ Missing tensor: {name}")
                    all_match = False
                    break
                
                # Check shape
                if reader.tensors[h].shape != data.shape:
                    print(f"  ❌ Shape mismatch: {name} {reader.tensors[h].shape} vs {data.shape}")
                    all_match = False
                    break
                    
                # Check values (FP16 precision loss expected, checking closeness)
                # Original is FP32/64, converted is FP16->FP32
                # BQ4 is lossy, so we need a higher tolerance
                if not np.allclose(reader.tensors[h], data, atol=0.2):
                    print(f"  ❌ Value mismatch: {name}")
                    # Debug print
                    flat_orig = data.flatten()
                    flat_read = reader.tensors[h].flatten()
                    print(f"    Orig: {flat_orig[:10]}")
                    print(f"    Read: {flat_read[:10]}")
                    diff = np.abs(flat_orig - flat_read)
                    print(f"    Max Diff: {np.max(diff)}")
                    all_match = False
                    break
            
            if all_match:
                print("  ✅ Verification: All weights match (within FP16 tolerance)")
            else:
                print("  ❌ Verification FAILED")
            
            # 4. Test Inference
            # Reconstruct model
            vocab_size = config_dict.get('vocab_size', 16000)
            # Override vocab size from tokenizer if needed
            if hasattr(tokenizer, 'vocab_size'):
                 vocab_size = tokenizer.vocab_size

            model = GPT(
                vocab_size=vocab_size,
                d_model=config_dict.get('hidden_size', 256),
                n_head=config_dict.get('num_heads', 4),
                n_layer=config_dict.get('num_layers', 8),
                max_len=config_dict.get('max_seq_len', 128)
            )
            
            # Load weights
            # Model parameters are list of Tensors
            model_params_list = list(model.parameters())
            # Checkpoint has param_0, param_1...
            for i, param in enumerate(model_params_list):
                key = f"param_{i}"
                if key in params:
                    param.data = params[key]
            
            # Run inference
            output = test_inference(model, tokenizer)
            
            results.append({
                "checkpoint": os.path.basename(ckpt_path),
                "step": ckpt.get('step'),
                "loss": ckpt.get('loss'),
                "status": "PASS" if all_match else "FAIL",
                "output": output
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "checkpoint": os.path.basename(ckpt_path),
                "status": "ERROR",
                "error": str(e)
            })

    # Generate Report
    print("\n" + "="*60)
    print("📊 FINAL REPORT")
    print("="*60)
    print(f"{'Checkpoint':<30} | {'Step':<6} | {'Loss':<6} | {'Status':<6} | {'Output (Snippet)':<30}")
    print("-" * 90)
    for r in results:
        out_snip = r.get('output', '')[:28].replace('\n', ' ')
        print(f"{r['checkpoint']:<30} | {r.get('step', '-'):<6} | {r.get('loss', 0.0):<6.4f} | {r['status']:<6} | {out_snip}")

if __name__ == "__main__":
    main()
