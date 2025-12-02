"""
Optimizers for Bullet-Core
Implements SGD, Adam, and AdamW optimizers
"""

import numpy as np
import pickle
from typing import List
from .tensor import Tensor

class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, params, lr=0.01):
        """
        Args:
            params: iterable of Tensors (model parameters)
            lr: learning rate
        """
        self.params = list(params)
        self.lr = lr
        self.state = {}
    
    def zero_grad(self):
        """Reset gradients for all parameters"""
        for param in self.params:
            param.zero_grad()
    
    def step(self):
        """Update parameters (implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement step()")
    
    def state_dict(self):
        """Get optimizer state for checkpointing"""
        return {
            'state': self.state,
            'param_groups': [{'lr': self.lr}]
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint"""
        self.state = state_dict['state']
        self.lr = state_dict['param_groups'][0]['lr']


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum
    
    Args:
        params: model parameters
        lr: learning rate
        momentum: momentum factor (0-1)
        weight_decay: L2 regularization coefficient
    
    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        >>> loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """
    
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize momentum buffers
        if momentum > 0:
            for i, param in enumerate(self.params):
                self.state[i] = {'momentum_buffer': np.zeros_like(param.data)}
    
    def step(self):
        """Perform a single optimization step"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # Momentum
            if self.momentum > 0:
                if i not in self.state:
                    self.state[i] = {'momentum_buffer': np.zeros_like(param.data)}
                
                buf = self.state[i]['momentum_buffer']
                buf = self.momentum * buf + grad
                self.state[i]['momentum_buffer'] = buf
                grad = buf
            
            # Update parameters
            param.data = (param.data - self.lr * grad).astype(np.float32)


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation)
    
    Args:
        params: model parameters
        lr: learning rate
        betas: (beta1, beta2) coefficients for running averages
        eps: epsilon for numerical stability
        weight_decay: L2 regularization coefficient
    
    Example:
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize state for each parameter
        for i, param in enumerate(self.params):
            self.state[i] = {
                'step': 0,
                'm': np.zeros_like(param.data),  # First moment estimate
                'v': np.zeros_like(param.data)   # Second moment estimate
            }
    
    def step(self):
        """Perform a single optimization step"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            state = self.state[i]
            
            # Increment step count
            state['step'] += 1
            
            # Weight decay (L2 regularization in gradient)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # Update biased first moment estimate
            state['m'] = self.betas[0] * state['m'] + (1 - self.betas[0]) * grad
            
            # Update biased second raw moment estimate
            state['v'] = self.betas[1] * state['v'] + (1 - self.betas[1]) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = state['m'] / (1 - self.betas[0] ** state['step'])
            
            # Compute bias-corrected second raw moment estimate
            v_hat = state['v'] / (1 - self.betas[1] ** state['step'])
            
            # Update parameters
            param.data = (param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(np.float32)


class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay)
    
    Decouples weight decay from gradient-based update,
    which is more effective than standard Adam.
    
    Args:
        params: model parameters
        lr: learning rate
        betas: (beta1, beta2) coefficients for running averages
        eps: epsilon for numerical stability
        weight_decay: weight decay coefficient
    
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize state
        for i, param in enumerate(self.params):
            self.state[i] = {
                'step': 0,
                'm': np.zeros_like(param.data),
                'v': np.zeros_like(param.data)
            }
    
    def step(self):
        """Perform a single optimization step"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            state = self.state[i]
            
            state['step'] += 1
            
            # Gradient clipping to prevent overflow
            grad = np.clip(grad, -1.0, 1.0)
            
            # Update biased first moment estimate (without weight decay)
            state['m'] = self.betas[0] * state['m'] + (1 - self.betas[0]) * grad
            
            # Update biased second raw moment estimate
            state['v'] = self.betas[1] * state['v'] + (1 - self.betas[1]) * (grad ** 2)
            
            # Bias correction
            m_hat = state['m'] / (1 - self.betas[0] ** state['step'])
            v_hat = state['v'] / (1 - self.betas[1] ** state['step'])
            
            # Update parameters with decoupled weight decay
            param.data = (param.data - self.lr * (
                m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param.data
            )).astype(np.float32)
