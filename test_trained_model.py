#!/usr/bin/env python3
"""
Test the trained Bullet model with interactive inference
"""
import sys
import os

# Add integration to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integration'))

try:
    import bullet_bindings
except ImportError:
    print("❌ Error: bullet_bindings not found!")
    print("Please install: cd integration && pip install --no-cache-dir .")
    sys.exit(1)

def test_model(model_path):
    """Test the trained model with sample prompts"""
    
    print("=" * 60)
    print("🚀 Bullet Model Inference Test")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📦 Loading model: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / 1024:.1f} KB")
    
    try:
        model = bullet_bindings.BulletModel(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test prompts (mix of English and Marathi)
    test_prompts = [
        "जीवनाचा उद्देश काय आहे?",  # What is the purpose of life?
        "What is dharma?",
        "कर्म म्हणजे काय?",  # What is karma?
        "How to find peace?",
        "सुख कसे मिळवावे?",  # How to attain happiness?
    ]
    
    print("\n" + "=" * 60)
    print("🧪 Running Test Prompts")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/5]")
        print(f"Prompt: {prompt}")
        print("-" * 60)
        
        try:
            response = model.generate(prompt, 50)  # Generate 50 tokens
            print(f"Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("💬 Interactive Mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\n🔹 Your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("Generating...")
            response = model.generate(prompt, 50)
            print(f"📝 Response: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Default model path
    model_path = "dashboard/backend/outputs/model_1764597609.bullet"
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    test_model(model_path)
