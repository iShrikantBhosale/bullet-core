"""
Learning Rate Schedulers for Bullet-Core
Implements various LR scheduling strategies
"""

import numpy as np

class LRScheduler:
    """Base class for learning rate schedulers"""
    
    def __init__(self, optimizer):
        """
        Args:
            optimizer: Optimizer instance
        """
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        new_lr = self.get_lr()
        self.optimizer.lr = new_lr
        return new_lr
    
    def get_lr(self):
        """Calculate new learning rate (implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement get_lr()")


class ConstantLR(LRScheduler):
    """
    Constant learning rate (no scheduling)
    
    Useful as a baseline or when no scheduling is needed.
    """
    
    def get_lr(self):
        return self.base_lr


class LinearWarmup(LRScheduler):
    """
    Linear warmup from 0 to base_lr over warmup_steps
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: number of warmup steps
        min_lr: minimum learning rate after warmup
    
    Example:
        >>> scheduler = LinearWarmup(optimizer, warmup_steps=1000)
        >>> for epoch in range(epochs):
        ...     train_epoch()
        ...     scheduler.step()
    """
    
    def __init__(self, optimizer, warmup_steps, min_lr=0.0):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
    
    def get_lr(self):
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.min_lr + (self.base_lr - self.min_lr) * self.step_count / self.warmup_steps
        return self.base_lr


class CosineAnnealing(LRScheduler):
    """
    Cosine annealing learning rate schedule
    
    Smoothly decreases learning rate following a cosine curve.
    
    Args:
        optimizer: Optimizer instance
        T_max: maximum number of iterations
        eta_min: minimum learning rate
    
    Example:
        >>> scheduler = CosineAnnealing(optimizer, T_max=10000, eta_min=1e-6)
    """
    
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * \
               (1 + np.cos(np.pi * self.step_count / self.T_max)) / 2


class StepLR(LRScheduler):
    """
    Step decay learning rate schedule
    
    Decays learning rate by gamma every step_size epochs.
    
    Args:
        optimizer: Optimizer instance
        step_size: period of learning rate decay
        gamma: multiplicative factor of learning rate decay
    
    Example:
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    """
    
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.step_count // self.step_size))


class ExponentialLR(LRScheduler):
    """
    Exponential decay learning rate schedule
    
    Args:
        optimizer: Optimizer instance
        gamma: multiplicative factor of learning rate decay
    """
    
    def __init__(self, optimizer, gamma=0.95):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def get_lr(self):
        return self.base_lr * (self.gamma ** self.step_count)


class WarmupCosineAnnealing(LRScheduler):
    """
    Combines linear warmup with cosine annealing
    
    Common schedule for transformer training.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: number of warmup steps
        T_max: maximum number of iterations (after warmup)
        eta_min: minimum learning rate
    """
    
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self):
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / self.T_max
            return self.eta_min + (self.base_lr - self.eta_min) * \
                   (1 + np.cos(np.pi * progress)) / 2
