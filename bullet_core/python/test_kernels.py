"""
Test suite for Bullet-Core CPU kernels
Validates correctness against NumPy reference implementations
"""

import numpy as np
import pytest
from bullet_core import ops

def test_matmul_basic():
    """Test basic matrix multiplication"""
    A = np.random.randn(128, 256).astype(np.float32)
    B = np.random.randn(256, 512).astype(np.float32)
    
    # Our implementation
    C_ours = ops.matmul(A, B)
    
    # NumPy reference
    C_numpy = A @ B
    
    # Check correctness
    np.testing.assert_allclose(C_ours, C_numpy, rtol=1e-5, atol=1e-6)
    print("✅ GEMM basic test passed")

def test_matmul_transpose():
    """Test matrix multiplication with transposes"""
    A = np.random.randn(128, 256).astype(np.float32)
    B = np.random.randn(512, 256).astype(np.float32)
    
    # Test A.T @ B
    C_ours = ops.matmul(A, B, transpose_A=True, transpose_B=False)
    C_numpy = A.T @ B
    np.testing.assert_allclose(C_ours, C_numpy, rtol=1e-5, atol=1e-6)
    
    # Test A @ B.T
    C_ours = ops.matmul(A, B, transpose_A=False, transpose_B=True)
    C_numpy = A @ B.T
    np.testing.assert_allclose(C_ours, C_numpy, rtol=1e-5, atol=1e-6)
    
    print("✅ GEMM transpose test passed")

def test_softmax():
    """Test softmax activation"""
    x = np.random.randn(32, 512).astype(np.float32)
    
    # Our implementation
    y_ours = ops.softmax(x)
    
    # NumPy reference
    y_numpy = np.exp(x - x.max(axis=-1, keepdims=True))
    y_numpy = y_numpy / y_numpy.sum(axis=-1, keepdims=True)
    
    np.testing.assert_allclose(y_ours, y_numpy, rtol=1e-5, atol=1e-6)
    
    # Check probabilities sum to 1
    assert np.allclose(y_ours.sum(axis=-1), 1.0)
    
    print("✅ Softmax test passed")

def test_rmsnorm():
    """Test RMS normalization"""
    x = np.random.randn(32, 256).astype(np.float32)
    weight = np.random.randn(256).astype(np.float32)
    
    # Our implementation
    y_ours = ops.rmsnorm(x, weight)
    
    # NumPy reference
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    y_numpy = (x / rms) * weight
    
    np.testing.assert_allclose(y_ours, y_numpy, rtol=1e-4, atol=1e-5)
    print("✅ RMSNorm test passed")

def test_layernorm():
    """Test layer normalization"""
    x = np.random.randn(32, 256).astype(np.float32)
    weight = np.random.randn(256).astype(np.float32)
    bias = np.random.randn(256).astype(np.float32)
    
    # Our implementation
    y_ours = ops.layernorm(x, weight, bias)
    
    # NumPy reference
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y_numpy = ((x - mean) / np.sqrt(var + 1e-6)) * weight + bias
    
    np.testing.assert_allclose(y_ours, y_numpy, rtol=1e-4, atol=1e-5)
    print("✅ LayerNorm test passed")

def test_embedding():
    """Test embedding lookup"""
    vocab_size, embedding_dim = 5000, 256
    table = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    indices = np.array([0, 10, 100, 1000, 4999], dtype=np.int32)
    
    # Our implementation
    emb_ours = ops.embedding(table, indices)
    
    # NumPy reference
    emb_numpy = table[indices]
    
    np.testing.assert_allclose(emb_ours, emb_numpy, rtol=1e-6, atol=1e-7)
    print("✅ Embedding test passed")

def test_vector_ops():
    """Test vector operations"""
    a = np.random.randn(1024).astype(np.float32)
    b = np.random.randn(1024).astype(np.float32)
    
    # Test add
    c_ours = ops.add(a, b)
    c_numpy = a + b
    np.testing.assert_allclose(c_ours, c_numpy, rtol=1e-6, atol=1e-7)
    
    # Test mul
    c_ours = ops.mul(a, b)
    c_numpy = a * b
    np.testing.assert_allclose(c_ours, c_numpy, rtol=1e-6, atol=1e-7)
    
    # Test scale
    c_ours = ops.scale(a, 2.5)
    c_numpy = a * 2.5
    np.testing.assert_allclose(c_ours, c_numpy, rtol=1e-6, atol=1e-7)
    
    print("✅ Vector ops test passed")

if __name__ == "__main__":
    print("Running Bullet-Core CPU kernel tests...\n")
    
    test_matmul_basic()
    test_matmul_transpose()
    test_softmax()
    test_rmsnorm()
    test_layernorm()
    test_embedding()
    test_vector_ops()
    
    print("\n🎉 All tests passed!")
