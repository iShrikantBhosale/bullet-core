
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bullet_core.python.tensor import Tensor
from bullet_core.python import nn

def test_tanh():
    print("Testing Tanh...")
    x_data = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    x = Tensor(x_data, requires_grad=True)
    y = x.tanh()
    
    # Forward check
    expected = np.tanh(x_data)
    assert np.allclose(y.data, expected), f"Forward failed: {y.data} vs {expected}"
    
    # Backward check
    y.sum().backward()
    # d/dx tanh(x) = 1 - tanh(x)^2
    expected_grad = 1 - np.tanh(x_data)**2
    assert np.allclose(x.grad, expected_grad), f"Backward failed: {x.grad} vs {expected_grad}"
    print("✅ Tanh Passed")

def test_gelu():
    print("Testing GELU...")
    x_data = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    x = Tensor(x_data, requires_grad=True)
    gelu = nn.GELU()
    y = gelu(x)
    
    # Forward check (approximate)
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    s = np.sqrt(2 / np.pi)
    expected = 0.5 * x_data * (1 + np.tanh(s * (x_data + 0.044715 * x_data**3)))
    
    assert np.allclose(y.data, expected, atol=1e-6), f"Forward failed: {y.data} vs {expected}"
    
    y.sum().backward()
    print(f"GELU Grads: {x.grad}")
    # Just check it runs and produces gradients
    assert x.grad is not None
    print("✅ GELU Passed")

if __name__ == "__main__":
    try:
        test_tanh()
        test_gelu()
        print("🎉 All Phase 2 tests passed!")
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        sys.exit(1)
