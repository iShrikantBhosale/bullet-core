"""
Neural network layers using autograd
PyTorch-style API for building models
"""

import numpy as np
from .tensor import Tensor
from typing import List

class Module:
    """Base class for all neural network modules"""
    
    def parameters(self) -> List[Tensor]:
        """Return list of trainable parameters"""
        return []
    
    def zero_grad(self):
        """Reset gradients for all parameters"""
        for p in self.parameters():
            p.zero_grad()
    
    def train(self):
        """Set module to training mode"""
        self.training = True
        return self
    
    def eval(self):
        """Set module to evaluation mode"""
        self.training = False
        return self
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    """
    Fully connected layer: y = x @ weight + bias
    
    Args:
        in_features: input dimension
        out_features: output dimension
        bias: whether to include bias term
    
    Example:
        >>> layer = Linear(256, 512)
        >>> x = Tensor(np.random.randn(32, 256), requires_grad=True)
        >>> y = layer(x)
        >>> y.shape  # (32, 512)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(
            (np.random.randn(in_features, out_features) * scale).astype(np.float32),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        if len(original_shape) > 2:
            # Flatten to (N, in_features)
            x_flat = x.reshape(-1, self.in_features)
            out = x_flat @ self.weight
            # Reshape back to (*original_shape[:-1], out_features)
            out = out.reshape(*original_shape[:-1], self.out_features)
        else:
            out = x @ self.weight
            
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def parameters(self) -> List[Tensor]:
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

class RMSNorm(Module):
    """
    RMS Normalization layer (used in LLaMA/BULLET models)
    
    y = (x / rms(x)) * weight
    where rms(x) = sqrt(mean(x^2) + eps)
    
    Args:
        dim: normalization dimension
        eps: epsilon for numerical stability
    
    Example:
        >>> norm = RMSNorm(256)
        >>> x = Tensor(np.random.randn(32, 256), requires_grad=True)
        >>> y = norm(x)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        return x.rmsnorm(self.weight, self.eps)
    
    def parameters(self) -> List[Tensor]:
        return [self.weight]
    
    def __repr__(self):
        return f"RMSNorm(dim={self.dim}, eps={self.eps})"

class LayerNorm(Module):
    """
    Standard Layer Normalization
    
    y = ((x - mean) / sqrt(var + eps)) * weight + bias
    
    Args:
        dim: normalization dimension
        eps: epsilon for numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        return x.layernorm(self.weight, self.bias, self.eps)
    
    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]
    
    def __repr__(self):
        return f"LayerNorm(dim={self.dim}, eps={self.eps})"

class Embedding(Module):
    """
    Embedding layer: lookup table for discrete tokens
    
    Args:
        num_embeddings: vocabulary size
        embedding_dim: embedding dimension
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Random initialization
        self.weight = Tensor(
            (np.random.randn(num_embeddings, embedding_dim) * 0.01).astype(np.float32),
            requires_grad=True
        )
    
    def forward(self, indices) -> Tensor:
        """
        Args:
            indices: integer indices (numpy array or Tensor)
        
        Returns:
            Embeddings for the given indices
        """
        from . import ops
        if isinstance(indices, Tensor):
            indices = indices.data
            
        original_shape = indices.shape
        indices_flat = indices.reshape(-1)
        
        emb_data = ops.embedding(self.weight.data, indices_flat.astype(np.int32))
        emb_data = emb_data.reshape(*original_shape, self.embedding_dim)
        
        out = Tensor(emb_data, requires_grad=self.weight.requires_grad,
                     _children=(self.weight,), _op='embedding')
        
        # Backward pass
        def _backward():
            if self.weight.requires_grad:
                # Gradient accumulation for embedding table
                grad_w = np.zeros_like(self.weight.data)
                indices_flat = indices.reshape(-1)
                grad_flat = out.grad.reshape(-1, self.embedding_dim)
                
                for i, idx in enumerate(indices_flat):
                    grad_w[int(idx)] += grad_flat[i]
                
                self.weight.grad = grad_w if self.weight.grad is None else self.weight.grad + grad_w
        
        out._backward = _backward
        return out
    
    def parameters(self) -> List[Tensor]:
        return [self.weight]
    
    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"

class Sequential(Module):
    """
    Sequential container for stacking layers
    
    Example:
        >>> model = Sequential(
        ...     Linear(256, 512),
        ...     RMSNorm(512),
        ...     Linear(512, 256)
        ... )
        >>> x = Tensor(np.random.randn(32, 256), requires_grad=True)
        >>> y = model(x)
    """
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        layer_str = '\n  '.join([f"({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"Sequential(\n  {layer_str}\n)"
class ReLU(Module):
    """
    ReLU activation: y = max(0, x)
    """
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
    def __repr__(self):
        return "ReLU()"

class CrossEntropyLoss(Module):
    """
    Cross Entropy Loss
    Combines LogSoftmax and NLLLoss
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            logits: (B, T, V) or (B, V)
            target: (B, T) or (B,) - indices
        """
        # Flatten
        V = logits.shape[-1]
        logits_flat = logits.reshape(-1, V)
        target_flat = target.reshape(-1)
        
        # Softmax (maintains gradients)
        probs = logits_flat.softmax(axis=-1)
        
        # NLL: Select probabilities for target indices
        # One-hot encoding
        N = logits_flat.shape[0]
        y_onehot = np.zeros((N, V), dtype=np.float32)
        y_onehot[np.arange(N), target_flat.data.astype(int)] = 1.0
        y_onehot_tensor = Tensor(y_onehot, requires_grad=False)
        
        # Cross Entropy = -mean(sum(y_true * log(y_pred)))
        # Element-wise: y_onehot * probs gives us the selected probabilities
        selected_probs = (y_onehot_tensor * probs).sum(axis=1)  # (N,)
        
        # Log of selected probabilities
        log_probs = selected_probs.log()
        
        # Mean negative log likelihood
        loss = -log_probs.mean()
        
        return loss

# Add log method to Tensor
def log(self):
    from .autograd import log_op
    return log_op(self)
Tensor.log = log

class CrossEntropyLoss(Module):
    """
    Cross Entropy Loss
    Combines LogSoftmax and NLLLoss
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            logits: (B, T, V) or (B, V)
            target: (B, T) or (B,) - indices
        """
        # Flatten
        V = logits.shape[-1]
        logits_flat = logits.reshape(-1, V)
        target_flat = target.reshape(-1)
        
        # Log Softmax
        # log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
        # Max trick for stability
        # max_logits = logits_flat.data.max(axis=1, keepdims=True)
        # We need to do this with Tensors to track gradients
        # But max() reduction might not be implemented in autograd yet.
        # Let's implement a simplified version using the provided softmax
        
        # softmax = exp(x) / sum(exp(x))
        probs = logits_flat.softmax(axis=-1)
        
        # NLL: -log(probs[target])
        # We need to select the probabilities corresponding to targets
        # This requires advanced indexing which might not be in Tensor
        
        # Workaround: One-hot encoding targets
        N = logits_flat.shape[0]
        
        # Create one-hot targets (numpy)
        y_onehot = np.zeros((N, V), dtype=np.float32)
        y_onehot[np.arange(N), target_flat.data.astype(int)] = 1.0
        y_onehot = Tensor(y_onehot, requires_grad=False)
        
        # Cross Entropy = -sum(y_true * log(y_pred))
        # Add epsilon to avoid log(0)
        loss = -(y_onehot * (probs + 1e-9).log()).sum() / N
        
        return loss

# Add log method to Tensor
def log(self):
    from .autograd import log_op
    return log_op(self)
Tensor.log = log
