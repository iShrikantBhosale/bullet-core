"""
High-level Python API for Bullet-Core operations

Provides NumPy-compatible wrappers.
Replaced C++ kernels with pure NumPy for stability.
"""

import numpy as np

def matmul(A, B, transpose_A=False, transpose_B=False):
    """
    Matrix multiplication (NumPy backed)
    """
    if transpose_A:
        A = A.T
    if transpose_B:
        B = B.T
        
    return np.matmul(A, B)

def softmax(x, axis=-1):
    """
    Numerically stable softmax (NumPy backed)
    """
    # Shift for stability
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp

def rmsnorm(x, weight, eps=1e-6):
    """
    RMS Normalization (NumPy backed)
    """
    # RMS = x / sqrt(mean(x^2) + eps)
    # x: (..., dim)
    # weight: (dim,)
    
    sq_mean = np.mean(x**2, axis=-1, keepdims=True)
    norm_x = x * (sq_mean + eps)**(-0.5)
    return norm_x * weight

def layernorm(x, weight, bias, eps=1e-6):
    """
    Layer Normalization (NumPy backed)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm_x = (x - mean) / np.sqrt(var + eps)
    return norm_x * weight + bias

def embedding(table, indices):
    """
    Embedding lookup (NumPy backed)
    """
    return table[indices]

# Vector operations
def add(a, b):
    return np.add(a, b)

def mul(a, b):
    return np.multiply(a, b)

def sub(a, b):
    return np.subtract(a, b)

def scale(a, scalar):
    return np.multiply(a, scalar)
