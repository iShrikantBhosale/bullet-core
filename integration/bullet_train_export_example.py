import torch
import torch.nn as nn
import os
from bullet_export import export_to_bullet
from bullet_py_api import Bullet

# 1. Define Tiny Model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 32
        self.num_layers = 1
        self.num_heads = 2
        self.max_context = 64
        
        self.tok_embeddings = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.ModuleDict({'attn': nn.LayerNorm(32), 'ffn': nn.LayerNorm(32)}),
                'attention': nn.ModuleDict({
                    'query': nn.Linear(32, 32),
                    'key': nn.Linear(32, 32),
                    'value': nn.Linear(32, 32),
                    'output': nn.Linear(32, 32)
                }),
                'ffn': nn.ModuleDict({
                    'w1': nn.Linear(32, 128), # Gate
                    'w3': nn.Linear(32, 128), # Up
                    'w2': nn.Linear(128, 32)  # Down
                })
            })
        ])
        self.final_layernorm = nn.LayerNorm(32)
        self.output_head = nn.Linear(32, 100)

    def forward(self, x):
        return x

# 2. Main Flow
def main():
    print("1. Creating Model...")
    model = TinyModel()
    
    # 3. Create Dummy Vocab
    print("2. Creating Vocab...")
    with open("vocab.txt", "w") as f:
        for i in range(100):
            f.write(f"token_{i} 1.0\n")
            
    # 4. Export
    print("3. Exporting...")
    export_to_bullet(model, "vocab.txt", "tiny.bullet")
    
    # 5. Load and Run
    print("4. Loading in Bullet...")
    try:
        b = Bullet("tiny.bullet")
        res = b.chat("token_1")
        print(f"Result: {res}")
    except Exception as e:
        print(f"Runtime Error (Expected if bindings not built): {e}")

if __name__ == "__main__":
    main()
