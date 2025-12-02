"""
Numerical gradient tests for autograd
Validates backward passes against finite differences
"""

import numpy as np
from bullet_core.tensor import Tensor
from bullet_core import autograd

def numerical_gradient(func, x, eps=1e-4):
    """
    Compute numerical gradient using finite differences
    
    Args:
        func: function that takes x and returns scalar
        x: input array
        eps: finite difference step size
    
    Returns:
        Numerical gradient
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # f(x + eps)
        x[idx] = old_value + eps
        pos = func(x)
        
        # f(x - eps)
        x[idx] = old_value - eps
        neg = func(x)
        
        # Restore
        x[idx] = old_value
        
        # Gradient
        grad[idx] = (pos - neg) / (2 * eps)
        it.iternext()
    
    return grad

def test_matmul_backward():
    """Test matrix multiplication gradients"""
    print("\n🧪 Testing matmul backward...")
    
    A = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    B = Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
    
    # Forward + backward
    C = A @ B
    loss = C.sum()
    loss.backward()
    
    # Numerical gradient for A
    def f_A(a):
        return (Tensor(a) @ B).sum().data.item()
    
    num_grad_A = numerical_gradient(f_A, A.data.copy())
    
    # Check
    error = np.abs(A.grad - num_grad_A).max()
    assert error < 1e-3, f"Matmul gradient error: {error}"
    print(f"  ✅ Matmul A gradient: max error = {error:.6f}")
    
    # Numerical gradient for B
    def f_B(b):
        return (A @ Tensor(b)).sum().data.item()
    
    num_grad_B = numerical_gradient(f_B, B.data.copy())
    error = np.abs(B.grad - num_grad_B).max()
    assert error < 1e-3, f"Matmul gradient error: {error}"
    print(f"  ✅ Matmul B gradient: max error = {error:.6f}")

def test_softmax_backward():
    """Test softmax gradients"""
    print("\n🧪 Testing softmax backward...")
    
    x = Tensor(np.random.randn(2, 5).astype(np.float32), requires_grad=True)
    
    # Forward + backward
    y = x.softmax()
    loss = y.sum()
    loss.backward()
    
    # Numerical gradient
    def f(x_val):
        return Tensor(x_val).softmax().sum().data.item()
    
    num_grad = numerical_gradient(f, x.data.copy())
    
    # Check
    error = np.abs(x.grad - num_grad).max()
    assert error < 1e-3, f"Softmax gradient error: {error}"
    print(f"  ✅ Softmax gradient: max error = {error:.6f}")

def test_rmsnorm_backward():
    """Test RMSNorm gradients"""
    print("\n🧪 Testing RMSNorm backward...")
    
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    weight = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
    
    # Forward + backward
    y = x.rmsnorm(weight)
    loss = y.sum()
    loss.backward()
    
    # Numerical gradient for x
    def f_x(x_val):
        return Tensor(x_val).rmsnorm(weight).sum().data.item()
    
    num_grad_x = numerical_gradient(f_x, x.data.copy())
    error = np.abs(x.grad - num_grad_x).max()
    assert error < 1e-3, f"RMSNorm x gradient error: {error}"
    print(f"  ✅ RMSNorm x gradient: max error = {error:.6f}")
    
    # Numerical gradient for weight
    def f_w(w_val):
        return x.rmsnorm(Tensor(w_val)).sum().data.item()
    
    num_grad_w = numerical_gradient(f_w, weight.data.copy())
    error = np.abs(weight.grad - num_grad_w).max()
    assert error < 1e-3, f"RMSNorm weight gradient error: {error}"
    print(f"  ✅ RMSNorm weight gradient: max error = {error:.6f}")

def test_add_mul_backward():
    """Test element-wise operation gradients"""
    print("\n🧪 Testing add/mul backward...")
    
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    
    # Test addition
    c = a + b
    loss = c.sum()
    loss.backward()
    
    assert np.allclose(a.grad, np.ones_like(a.data)), "Add gradient incorrect"
    assert np.allclose(b.grad, np.ones_like(b.data)), "Add gradient incorrect"
    print("  ✅ Add gradient correct")
    
    # Reset gradients
    a.zero_grad()
    b.zero_grad()
    
    # Test multiplication
    c = a * b
    loss = c.sum()
    loss.backward()
    
    assert np.allclose(a.grad, b.data), "Mul gradient incorrect"
    assert np.allclose(b.grad, a.data), "Mul gradient incorrect"
    print("  ✅ Mul gradient correct")

def test_power_backward():
    """Test power operation gradients"""
    print("\n🧪 Testing power backward...")
    
    x = Tensor(np.random.randn(3, 4).astype(np.float32) + 2, requires_grad=True)  # Positive values
    
    # Forward + backward
    y = x ** 2
    loss = y.sum()
    loss.backward()
    
    # Numerical gradient
    def f(x_val):
        return (Tensor(x_val) ** 2).sum().data.item()
    
    num_grad = numerical_gradient(f, x.data.copy())
    error = np.abs(x.grad - num_grad).max()
    assert error < 1e-3, f"Power gradient error: {error}"
    print(f"  ✅ Power gradient: max error = {error:.6f}")

def test_sum_mean_backward():
    """Test reduction operation gradients"""
    print("\n🧪 Testing sum/mean backward...")
    
    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    
    # Test sum
    loss = x.sum()
    loss.backward()
    
    assert np.allclose(x.grad, np.ones_like(x.data)), "Sum gradient incorrect"
    print("  ✅ Sum gradient correct")
    
    # Reset
    x.zero_grad()
    
    # Test mean
    loss = x.mean()
    loss.backward()
    
    expected_grad = np.ones_like(x.data) / x.data.size
    assert np.allclose(x.grad, expected_grad), "Mean gradient incorrect"
    print("  ✅ Mean gradient correct")

def test_linear_layer():
    """Test Linear layer gradients"""
    print("\n🧪 Testing Linear layer...")
    
    from bullet_core.nn import Linear
    
    layer = Linear(4, 3)
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    
    # Forward + backward
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "Input gradient missing"
    assert layer.weight.grad is not None, "Weight gradient missing"
    assert layer.bias.grad is not None, "Bias gradient missing"
    
    # Numerical check for weight
    def f_w(w):
        layer.weight.data = w
        return layer(x).sum().data.item()
    
    num_grad_w = numerical_gradient(f_w, layer.weight.data.copy())
    error = np.abs(layer.weight.grad - num_grad_w).max()
    assert error < 1e-3, f"Linear weight gradient error: {error}"
    print(f"  ✅ Linear layer gradients: max error = {error:.6f}")

def test_simple_mlp():
    """Test simple MLP training"""
    print("\n🧪 Testing simple MLP training...")
    
    from bullet_core.nn import Linear, Sequential
    
    # Create simple 2-layer MLP
    model = Sequential(
        Linear(4, 8),
        Linear(8, 2)
    )
    
    # Dummy data
    x = Tensor(np.random.randn(10, 4).astype(np.float32), requires_grad=True)
    target = Tensor(np.random.randn(10, 2).astype(np.float32))
    
    # Forward pass
    pred = model(x)
    
    # MSE loss
    loss = ((pred - target) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Check all parameters have gradients
    for i, param in enumerate(model.parameters()):
        assert param.grad is not None, f"Parameter {i} has no gradient"
    
    print(f"  ✅ MLP training: loss = {loss.data.item():.4f}")
    print(f"  ✅ All {len(model.parameters())} parameters have gradients")

if __name__ == "__main__":
    print("=" * 60)
    print("Bullet-Core Autograd Tests")
    print("=" * 60)
    
    test_matmul_backward()
    test_softmax_backward()
    test_rmsnorm_backward()
    test_add_mul_backward()
    test_power_backward()
    test_sum_mean_backward()
    test_linear_layer()
    test_simple_mlp()
    
    print("\n" + "=" * 60)
    print("🎉 All autograd tests passed!")
    print("=" * 60)
