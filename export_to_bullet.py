import torch
import struct
import os
from train_factbullet import Config, BulletTransformer

def export_model():
    config = Config()
    device = "cpu" # Export on CPU
    
    # Load Tokenizer to get vocab size
    from train_factbullet import SimpleTokenizer
    tokenizer = SimpleTokenizer(config.tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {config.vocab_size}")

    # Load Model
    print("Loading model...")
    model = BulletTransformer(config).to(device)
    state_dict = torch.load(f"{config.checkpoint_dir}/final_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    output_path = "docs/model_weights.bin"
    print(f"Exporting to {output_path}...")
    
    with open(output_path, "wb") as f:
        # 1. Header
        # Magic: "BULL" = 0x4C4C5542 (Little Endian) -> 0x42554C4C (Big Endian representation usually, but let's stick to simple int)
        # Let's use a distinct magic number. 0x42554C4C is 'BULL' in ASCII.
        magic = 0x42554C4C 
        version = 1
        weights = dict(model.state_dict())
        num_weights = len(weights)
        
        f.write(struct.pack("<I", magic))
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<I", num_weights))
        
        print(f"Header: Magic={hex(magic)}, Ver={version}, Num={num_weights}")
        
        # 2. Weights
        for name, tensor in weights.items():
            tensor = tensor.cpu().float().numpy()
            
            # Name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            
            # Shape
            shape = tensor.shape
            ndim = len(shape)
            f.write(struct.pack("<I", ndim))
            for dim in shape:
                f.write(struct.pack("<I", dim))
                
            # Data
            f.write(tensor.tobytes())
            
            print(f"Wrote {name} {shape}")

    print("Export complete.")

if __name__ == "__main__":
    export_model()
