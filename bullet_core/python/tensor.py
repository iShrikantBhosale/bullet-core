"""
Tensor class with automatic differentiation support
Implements tinygrad-style autograd with tape-based backpropagation
"""

import numpy as np
from typing import Optional, Tuple, Set, List

class Tensor:
    """
    Tensor with automatic gradient tracking
    
    Example:
        >>> x = Tensor([1, 2, 3], requires_grad=True)
        >>> y = x * 2
        >>> loss = y.sum()
        >>> loss.backward()
        >>> print(x.grad)  # [2, 2, 2]
    """
    
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        """
        Args:
            data: numpy array or list
            requires_grad: whether to track gradients
            _children: parent tensors in computation graph
            _op: operation that created this tensor
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        
        # Computation graph (for backprop)
        self._prev: Set['Tensor'] = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def backward(self, gradient=None):
        """
        Compute gradients via backpropagation
        
        Args:
            gradient: initial gradient (defaults to ones)
        """
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data, dtype=np.float32)
            else:
                raise RuntimeError("gradient must be specified for non-scalar tensors")
        
        # Build topological order
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()
        
        def build_topo(v: 'Tensor'):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient
        self.grad = gradient.astype(np.float32)
        
        # Backpropagate in reverse topological order
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Reset gradients to None"""
        self.grad = None
    
    def detach(self):
        """Return a new tensor with no gradient tracking"""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def numpy(self):
        """Return data as numpy array"""
        return self.data
    
    # Operator overloading
    def __matmul__(self, other):
        from .autograd import matmul
        return matmul(self, other)
    
    def __add__(self, other):
        from .autograd import add
        other = other if isinstance(other, Tensor) else Tensor(other)
        return add(self, other)
    
    def __mul__(self, other):
        from .autograd import mul
        other = other if isinstance(other, Tensor) else Tensor(other)
        return mul(self, other)
    
    def __sub__(self, other):
        from .autograd import sub
        other = other if isinstance(other, Tensor) else Tensor(other)
        return sub(self, other)
    
    def __pow__(self, power):
        from .autograd import pow_op
        return pow_op(self, power)
    
    def __neg__(self):
        return self * -1
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    # Reverse operators
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return Tensor(other) - self
    
    # Reduction operations
    def sum(self, axis=None, keepdims=False):
        """Sum reduction"""
        from .autograd import sum_op
        return sum_op(self, axis, keepdims)
    
    def mean(self, axis=None, keepdims=False):
        """Mean reduction"""
        from .autograd import mean_op
        return mean_op(self, axis, keepdims)
    
    # Activation functions
    def relu(self):
        """ReLU activation"""
        from .autograd import relu
        return relu(self)
    
    def softmax(self, axis=-1):
        """Softmax activation"""
        from .autograd import softmax
        return softmax(self, axis)
    
    def tanh(self):
        """Tanh activation"""
        from .autograd import tanh_op
        return tanh_op(self)
    
    # Normalization
    def rmsnorm(self, weight, eps=1e-6):
        """RMS Normalization"""
        from .autograd import rmsnorm
        return rmsnorm(self, weight, eps)
    
    def layernorm(self, weight, bias, eps=1e-6):
        """Layer Normalization"""
        from .autograd import layernorm
        return layernorm(self, weight, bias, eps)
    
    # Reshaping
    def reshape(self, *shape):
        """Reshape tensor"""
        from .autograd import reshape
        return reshape(self, shape)

    def transpose(self, axis1=-2, axis2=-1):
        """Transpose tensor"""
        # For 2D, simple transpose
        if self.data.ndim == 2:
            from .autograd import transpose_op
            return transpose_op(self)
        else:
            raise NotImplementedError("Only 2D transpose supported currently")
    


    def __repr__(self):
        grad_str = f", grad_fn=<{self._op}>" if self._op else ""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{grad_str})"
    
    def __str__(self):
        return f"Tensor({self.data})"
