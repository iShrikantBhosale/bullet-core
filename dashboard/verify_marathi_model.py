import bullet_bindings
import sys
import os

def verify_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return

    print(f"Loading model: {model_path}")
    model = bullet_bindings.BulletModel(model_path)
    
    prompts = [
        "जीवनाचा उद्देश काय आहे?",
        "How to find inner peace?",
        "कर्म म्हणजे काय?"
    ]
    
    for p in prompts:
        print(f"\nPrompt: {p}")
        output = model.generate(p, 100)
        print(f"Response: {output}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 verify_marathi_model.py <model_path>")
    else:
        verify_model(sys.argv[1])
