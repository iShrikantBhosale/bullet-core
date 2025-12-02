"""
High-level Python API for Bullet-Core operations

Provides NumPy-compatible wrappers around C++ SIMD kernels.
"""

import numpy as np
try:
    from . import bullet_core_cpp
except ImportError:
    import bullet_core_cpp

def matmul(A, B, transpose_A=False, transpose_B=False):
    """
    Matrix multiplication with SIMD optimization
    
    Args:
        A: numpy array of shape (M, K) or (K, M) if transpose_A
        B: numpy array of shape (K, N) or (N, K) if transpose_B
        transpose_A: whether to transpose A
        transpose_B: whether to transpose B
    
    Returns:
        C: numpy array of shape (M, N)
    
    Example:
        >>> A = np.random.randn(128, 256).astype(np.float32)
        >>> B = np.random.randn(256, 512).astype(np.float32)
        >>> C = matmul(A, B)
        >>> assert C.shape == (128, 512)
    """
    assert A.dtype == np.float32, "Only fp32 supported in Phase 1"
    assert B.dtype == np.float32, "Only fp32 supported in Phase 1"
    
    if transpose_A:
        M, K1 = A.shape[1], A.shape[0]
    else:
        M, K1 = A.shape[0], A.shape[1]
    
    if transpose_B:
        K2, N = B.shape[1], B.shape[0]
    else:
        K2, N = B.shape[0], B.shape[1]
    
    assert K1 == K2, f"Dimension mismatch: {K1} != {K2}"
    
    C = np.zeros((M, N), dtype=np.float32)
    A_contig = np.ascontiguousarray(A)
    B_contig = np.ascontiguousarray(B)
    
    bullet_core_cpp.gemm_f32(A_contig, B_contig, C, M, N, K1, transpose_A, transpose_B)
    return C

def softmax(x, axis=-1):
    """
    Numerically stable softmax
    
    Args:
        x: numpy array
        axis: axis along which to apply softmax (only -1 supported in Phase 1)
    
    Returns:
        Softmax of x along specified axis
    
    Example:
        >>> x = np.random.randn(32, 512).astype(np.float32)
        >>> y = softmax(x)
        >>> assert np.allclose(y.sum(axis=-1), 1.0)
    """
    assert x.dtype == np.float32, "Only fp32 supported"
    assert axis == -1, "Only axis=-1 supported in Phase 1"
    
    batch = int(np.prod(x.shape[:-1]))
    dim = x.shape[-1]
    
    x_flat = np.ascontiguousarray(x.reshape(batch, dim))
    out = np.zeros_like(x_flat)
    
    bullet_core_cpp.softmax_f32(x_flat, out, batch, dim)
    return out.reshape(x.shape)

def rmsnorm(x, weight, eps=1e-6):
    """
    RMS Normalization (used in LLaMA/BULLET models)
    
    Args:
        x: input tensor of shape (..., dim)
        weight: weight tensor of shape (dim,)
        eps: epsilon for numerical stability
    
    Returns:
        Normalized tensor of same shape as x
    
    Example:
        >>> x = np.random.randn(32, 256).astype(np.float32)
        >>> weight = np.ones(256, dtype=np.float32)
        >>> y = rmsnorm(x, weight)
    """
    assert x.dtype == np.float32, "Only fp32 supported"
    assert weight.dtype == np.float32, "Only fp32 supported"
    
    batch = int(np.prod(x.shape[:-1]))
    dim = x.shape[-1]
    assert weight.shape == (dim,), f"Weight shape mismatch: {weight.shape} != ({dim},)"
    
    x_flat = np.ascontiguousarray(x.reshape(batch, dim))
    weight_contig = np.ascontiguousarray(weight)
    out = np.zeros_like(x_flat)
    
    bullet_core_cpp.rmsnorm_f32(x_flat, weight_contig, out, batch, dim, eps)
    return out.reshape(x.shape)

def layernorm(x, weight, bias, eps=1e-6):
    """
    Standard Layer Normalization
    
    Args:
        x: input tensor of shape (..., dim)
        weight: weight tensor of shape (dim,)
        bias: bias tensor of shape (dim,)
        eps: epsilon for numerical stability
    
    Returns:
        Normalized tensor of same shape as x
    """
    assert x.dtype == np.float32, "Only fp32 supported"
    assert weight.dtype == np.float32, "Only fp32 supported"
    assert bias.dtype == np.float32, "Only fp32 supported"
    
    batch = int(np.prod(x.shape[:-1]))
    dim = x.shape[-1]
    
    x_flat = np.ascontiguousarray(x.reshape(batch, dim))
    weight_contig = np.ascontiguousarray(weight)
    bias_contig = np.ascontiguousarray(bias)
    out = np.zeros_like(x_flat)
    
    bullet_core_cpp.layernorm_f32(x_flat, weight_contig, bias_contig, out, batch, dim, eps)
    return out.reshape(x.shape)

def embedding(table, indices):
    """
    Embedding table lookup with prefetching
    
    Args:
        table: embedding table of shape (vocab_size, embedding_dim)
        indices: indices to lookup, shape (num_indices,)
    
    Returns:
        Embeddings of shape (num_indices, embedding_dim)
    
    Example:
        >>> table = np.random.randn(5000, 256).astype(np.float32)
        >>> indices = np.array([0, 10, 100, 1000], dtype=np.int32)
        >>> emb = embedding(table, indices)
        >>> assert emb.shape == (4, 256)
    """
    assert table.dtype == np.float32, "Only fp32 supported"
    assert indices.dtype == np.int32, "Indices must be int32"
    
    vocab_size, embedding_dim = table.shape
    num_indices = indices.shape[0]
    
    table_contig = np.ascontiguousarray(table)
    indices_contig = np.ascontiguousarray(indices)
    out = np.zeros((num_indices, embedding_dim), dtype=np.float32)
    
    bullet_core_cpp.embedding_lookup_f32(table_contig, indices_contig, out, num_indices, embedding_dim)
    return out

# Vector operations
def add(a, b):
    """Element-wise addition (SIMD-optimized)"""
    # Handle broadcasting
    if a.shape != b.shape:
        try:
            # Broadcast to common shape
            shape = np.broadcast_shapes(a.shape, b.shape)
            a = np.broadcast_to(a, shape).copy()  # MUST copy to make writable
            b = np.broadcast_to(b, shape).copy()
        except ValueError:
            raise ValueError(f"Shapes mismatch: {a.shape} vs {b.shape}")

    assert a.shape == b.shape and a.dtype == np.float32
    out = np.zeros_like(a)
    bullet_core_cpp.vector_add_f32(
        np.ascontiguousarray(a.flatten()),
        np.ascontiguousarray(b.flatten()),
        out.ravel(), a.size  # Use ravel() to get a view, not a copy
    )
    return out

def mul(a, b):
    """Element-wise multiplication (SIMD-optimized)"""
    # Handle broadcasting
    if a.shape != b.shape:
        try:
            # Broadcast to common shape
            shape = np.broadcast_shapes(a.shape, b.shape)
            a = np.broadcast_to(a, shape).copy()  # MUST copy to make writable
            b = np.broadcast_to(b, shape).copy()
        except ValueError:
            raise ValueError(f"Shapes mismatch: {a.shape} vs {b.shape}")

    assert a.shape == b.shape and a.dtype == np.float32
    out = np.zeros_like(a)
    bullet_core_cpp.vector_mul_f32(
        np.ascontiguousarray(a.flatten()),
        np.ascontiguousarray(b.flatten()),
        out.ravel(), a.size  # Use ravel() to get a view, not a copy
    )
    return out

def sub(a, b):
    """Element-wise subtraction (SIMD-optimized)"""
    # Handle broadcasting
    if a.shape != b.shape:
        try:
            # Broadcast to common shape
            shape = np.broadcast_shapes(a.shape, b.shape)
            a = np.broadcast_to(a, shape).copy()  # MUST copy to make writable
            b = np.broadcast_to(b, shape).copy()
        except ValueError:
            raise ValueError(f"Shapes mismatch: {a.shape} vs {b.shape}")

    assert a.shape == b.shape and a.dtype == np.float32
    out = np.zeros_like(a)
    bullet_core_cpp.vector_sub_f32(
        np.ascontiguousarray(a.flatten()),
        np.ascontiguousarray(b.flatten()),
        out.ravel(), a.size  # Use ravel() to get a view, not a copy
    )
    return out

def scale(a, scalar):
    """Scalar multiplication (SIMD-optimized)"""
    assert a.dtype == np.float32
    out = np.zeros_like(a)
    bullet_core_cpp.vector_scale_f32(
        np.ascontiguousarray(a.flatten()),
        float(scalar),
        out.flatten(), a.size
    )
    return out
