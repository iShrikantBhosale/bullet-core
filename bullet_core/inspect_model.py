import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from python.checkpoint import load_model_only, get_checkpoint_info
from python.transformer import GPT
from python.tokenizer import BPETokenizer

checkpoint_dir = "./marathi_checkpoints_upgraded"
final_model_path = os.path.join(checkpoint_dir, "final_model.pkl")

print(f"Inspecting {final_model_path}...")
try:
    # We need a dummy model to load weights into, but load_model_only returns metadata
    # We can just use pickle directly to peek at metadata if we don't want to instantiate the model
    import pickle
    with open(final_model_path, 'rb') as f:
        data = pickle.load(f)
    
    metadata = data.get('metadata', {})
    print("\nModel Metadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

except Exception as e:
    print(f"Error loading final model: {e}")

# Check checkpoints
print("\nCheckpoints:")
checkpoints = get_checkpoint_info(checkpoint_dir)
for ckpt in checkpoints:
    print(f"  {ckpt['filename']}: Step {ckpt['step']}, Val Loss {ckpt['val_loss']:.4f}")
