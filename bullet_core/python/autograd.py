"""
Autograd operations with forward and backward passes
All operations use Phase 1 CPU kernels for forward pass
"""

import numpy as np
from .tensor import Tensor
from . import ops

# ===== MATRIX OPERATIONS =====

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication with autograd
    
    Forward: C = A @ B
    Backward: dL/dA = dL/dC @ B.T, dL/dB = A.T @ dL/dC
    """
    # Forward pass using Phase 1 kernel
    out_data = ops.matmul(a.data, b.data)
    out = Tensor(out_data, requires_grad=(a.requires_grad or b.requires_grad),
                 _children=(a, b), _op='matmul')
    
    # Backward pass
    def _backward():
        if a.requires_grad:
            # dL/dA = dL/dOut @ B.T
            grad_a = ops.matmul(out.grad, b.data, transpose_B=True)
            a.grad = grad_a if a.grad is None else a.grad + grad_a
        
        if b.requires_grad:
            # dL/dB = A.T @ dL/dOut
            grad_b = ops.matmul(a.data, out.grad, transpose_A=True)
            b.grad = grad_b if b.grad is None else b.grad + grad_b
    
    out._backward = _backward
    return out

# ===== ELEMENT-WISE OPERATIONS =====

def unbroadcast(grad, shape):
    """Sum gradient over broadcasted dimensions"""
    if grad.shape == shape:
        return grad
    
    ndim_grad = len(grad.shape)
    ndim_shape = len(shape)
    
    if ndim_grad > ndim_shape:
        # Sum over extra leading dimensions
        grad = grad.sum(axis=tuple(range(ndim_grad - ndim_shape)))
    
    # Sum over dimensions where shape is 1
    axes = []
    for i, dim in enumerate(shape):
        if dim == 1:
            axes.append(i)
    if axes:
        grad = grad.sum(axis=tuple(axes), keepdims=True)
        
    return grad

def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition with autograd"""
    out_data = ops.add(a.data, b.data)
    out = Tensor(out_data, requires_grad=(a.requires_grad or b.requires_grad),
                 _children=(a, b), _op='add')
    
    def _backward():
        if a.requires_grad:
            grad_a = unbroadcast(out.grad, a.shape)
            a.grad = grad_a if a.grad is None else a.grad + grad_a
        if b.requires_grad:
            grad_b = unbroadcast(out.grad, b.shape)
            b.grad = grad_b if b.grad is None else b.grad + grad_b
    
    out._backward = _backward
    return out

def mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication with autograd"""
    out_data = ops.mul(a.data, b.data)
    out = Tensor(out_data, requires_grad=(a.requires_grad or b.requires_grad),
                 _children=(a, b), _op='mul')
    
    def _backward():
        if a.requires_grad:
            grad_a = unbroadcast(out.grad * b.data, a.shape)
            a.grad = grad_a if a.grad is None else a.grad + grad_a
        if b.requires_grad:
            grad_b = unbroadcast(out.grad * a.data, b.shape)
            b.grad = grad_b if b.grad is None else b.grad + grad_b
    
    out._backward = _backward
    return out

def sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction with autograd"""
    out_data = a.data - b.data
    out = Tensor(out_data, requires_grad=(a.requires_grad or b.requires_grad),
                 _children=(a, b), _op='sub')
    
    def _backward():
        if a.requires_grad:
            grad_a = unbroadcast(out.grad, a.shape)
            a.grad = grad_a if a.grad is None else a.grad + grad_a
        if b.requires_grad:
            grad_b = unbroadcast(-out.grad, b.shape)
            b.grad = grad_b if b.grad is None else b.grad + grad_b
    
    out._backward = _backward
    return out

def pow_op(a: Tensor, power: float) -> Tensor:
    """Power operation with autograd"""
    out_data = a.data ** power
    out = Tensor(out_data, requires_grad=a.requires_grad,
                 _children=(a,), _op=f'pow({power})')
    
    def _backward():
        if a.requires_grad:
            grad_a = power * (a.data ** (power - 1)) * out.grad
            a.grad = grad_a if a.grad is None else a.grad + grad_a
    
    out._backward = _backward
    return out

# ===== REDUCTION OPERATIONS =====

def sum_op(a: Tensor, axis=None, keepdims=False) -> Tensor:
    """Sum reduction with autograd"""
    out_data = a.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad,
                 _children=(a,), _op='sum')
    
    def _backward():
        if a.requires_grad:
            # Broadcast gradient back to original shape
            grad = out.grad
            if axis is not None:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, a.shape)
            else:
                grad = np.full(a.shape, grad, dtype=np.float32)
            
            a.grad = grad if a.grad is None else a.grad + grad
    
    out._backward = _backward
    return out

def mean_op(a: Tensor, axis=None, keepdims=False) -> Tensor:
    """Mean reduction with autograd"""
    out_data = a.data.mean(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad,
                 _children=(a,), _op='mean')
    
    # Calculate number of elements being averaged
    if axis is None:
        n = a.data.size
    else:
        n = a.data.shape[axis] if isinstance(axis, int) else np.prod([a.data.shape[ax] for ax in axis])
    
    def _backward():
        if a.requires_grad:
            grad = out.grad / n
            if axis is not None:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, a.shape)
            else:
                grad = np.full(a.shape, grad, dtype=np.float32)
            
            a.grad = grad if a.grad is None else a.grad + grad
    
    out._backward = _backward
    return out

# ===== ACTIVATION FUNCTIONS =====

def relu(x: Tensor) -> Tensor:
    """ReLU activation with autograd"""
    out_data = np.maximum(0, x.data)
    out = Tensor(out_data, requires_grad=x.requires_grad,
                 _children=(x,), _op='relu')
    
    def _backward():
        if x.requires_grad:
            grad_x = out.grad * (x.data > 0).astype(np.float32)
            x.grad = grad_x if x.grad is None else x.grad + grad_x
    
    out._backward = _backward
    return out

def softmax(x: Tensor, axis=-1) -> Tensor:
    """
    Softmax activation with autograd
    
    Forward: softmax(x) = exp(x) / sum(exp(x))
    Backward: Jacobian-vector product
    """
    # Forward pass using Phase 1 kernel
    out_data = ops.softmax(x.data, axis=axis)
    out = Tensor(out_data, requires_grad=x.requires_grad,
                 _children=(x,), _op='softmax')
    
    # Backward pass
    def _backward():
        if x.requires_grad:
            # Softmax Jacobian-vector product
            # dL/dx = softmax * (dL/dy - sum(dL/dy * softmax))
            s = out.data
            grad_out = out.grad
            
            sum_term = (grad_out * s).sum(axis=axis, keepdims=True)
            grad_x = s * (grad_out - sum_term)
            
            x.grad = grad_x if x.grad is None else x.grad + grad_x
    
    out._backward = _backward
    return out

# ===== NORMALIZATION =====

def rmsnorm(x: Tensor, weight: Tensor, eps=1e-6) -> Tensor:
    """
    RMS Normalization with autograd
    
    Forward: y = (x / rms(x)) * weight
    where rms(x) = sqrt(mean(x^2) + eps)
    """
    # Forward pass using Phase 1 kernel
    out_data = ops.rmsnorm(x.data, weight.data, eps)
    out = Tensor(out_data, requires_grad=(x.requires_grad or weight.requires_grad),
                 _children=(x, weight), _op='rmsnorm')
    
    # Cache values for backward
    rms = np.sqrt(np.mean(x.data**2, axis=-1, keepdims=True) + eps)
    normalized = x.data / rms
    
    # Backward pass
    def _backward():
        if x.requires_grad:
            grad_out = out.grad
            dim = x.data.shape[-1]
            
            # Gradient w.r.t normalized input
            grad_norm = grad_out * weight.data
            
            # Gradient w.r.t input (chain rule through normalization)
            grad_x = (grad_norm - normalized * (grad_norm * normalized).sum(axis=-1, keepdims=True)) / rms
            
            x.grad = grad_x if x.grad is None else x.grad + grad_x
        
        if weight.requires_grad:
            # dL/dweight = sum(dL/dout * normalized)
            grad_w = (out.grad * normalized).sum(axis=tuple(range(len(x.data.shape)-1)))
            weight.grad = grad_w if weight.grad is None else weight.grad + grad_w
    
    out._backward = _backward
    return out

def layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps=1e-6) -> Tensor:
    """
    Layer Normalization with autograd
    
    Forward: y = ((x - mean) / sqrt(var + eps)) * weight + bias
    """
    # Forward pass using Phase 1 kernel
    out_data = ops.layernorm(x.data, weight.data, bias.data, eps)
    out = Tensor(out_data, requires_grad=(x.requires_grad or weight.requires_grad or bias.requires_grad),
                 _children=(x, weight, bias), _op='layernorm')
    
    # Cache values for backward
    mean = x.data.mean(axis=-1, keepdims=True)
    var = x.data.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    normalized = (x.data - mean) / std
    
    # Backward pass
    def _backward():
        if x.requires_grad:
            grad_out = out.grad
            dim = x.data.shape[-1]
            
            # Complex gradient through normalization
            grad_norm = grad_out * weight.data
            
            # Gradient w.r.t input
            grad_var = (grad_norm * (x.data - mean) * -0.5 * (var + eps) ** -1.5).sum(axis=-1, keepdims=True)
            grad_mean = (grad_norm * -1 / std).sum(axis=-1, keepdims=True) + grad_var * (-2 * (x.data - mean)).sum(axis=-1, keepdims=True) / dim
            
            grad_x = grad_norm / std + grad_var * 2 * (x.data - mean) / dim + grad_mean / dim
            
            x.grad = grad_x if x.grad is None else x.grad + grad_x
        
        if weight.requires_grad:
            grad_w = (out.grad * normalized).sum(axis=tuple(range(len(x.data.shape)-1)))
            weight.grad = grad_w if weight.grad is None else weight.grad + grad_w
        
        if bias.requires_grad:
            grad_b = out.grad.sum(axis=tuple(range(len(x.data.shape)-1)))
            bias.grad = grad_b if bias.grad is None else bias.grad + grad_b
    
    out._backward = _backward
    return out

# ===== SHAPE OPERATIONS =====

def reshape(x: Tensor, shape) -> Tensor:
    """Reshape with autograd"""
    out_data = x.data.reshape(shape)
    out = Tensor(out_data, requires_grad=x.requires_grad,
                 _children=(x,), _op='reshape')
    
    def _backward():
        if x.requires_grad:
            grad_x = out.grad.reshape(x.shape)
            x.grad = grad_x if x.grad is None else x.grad + grad_x
    
    out._backward = _backward
    return out

def transpose_op(x: Tensor) -> Tensor:
    """Transpose 2D tensor"""
    out_data = x.data.T
    out = Tensor(out_data, requires_grad=x.requires_grad,
                 _children=(x,), _op='transpose')
    
    def _backward():
        if x.requires_grad:
            grad_x = out.grad.T
            x.grad = grad_x if x.grad is None else x.grad + grad_x
            
    out._backward = _backward
    return out

def log_op(x: Tensor) -> Tensor:
    """Natural logarithm"""
    out_data = np.log(x.data)
    out = Tensor(out_data, requires_grad=x.requires_grad,
                 _children=(x,), _op='log')
    
    def _backward():
        if x.requires_grad:
            grad_x = out.grad / x.data
            x.grad = grad_x if x.grad is None else x.grad + grad_x
            
    out._backward = _backward
    return out
