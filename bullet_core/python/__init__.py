"""
Bullet-Core: SIMD-Optimized CPU Kernels for Deep Learning

A lightweight, high-performance compute engine for training micro-transformers
on CPU with SIMD optimizations (AVX, SSE4.2) and OpenMP parallelization.

Designed for systems with limited resources (16GB RAM, GT 730 GPU).

Phase 1: CPU kernels (GEMM, softmax, RMSNorm, etc.)
Phase 2: Autograd engine (Tensor, gradients, backprop)
Phase 3: CUDA GPU acceleration (GT 730 optimized)
Phase 4: Optimizers & training pipeline
"""

try:
    from . import bullet_core_cpp
except ImportError:
    try:
        import bullet_core_cpp
    except ImportError:
        import warnings
        warnings.warn("bullet_core_cpp not built yet. Run: pip install -e .")
        bullet_core_cpp = None

from . import ops
from .tensor import Tensor
from . import autograd
from . import nn
from . import optim
from . import scheduler
from . import trainer

__version__ = "0.4.0"
__all__ = ['ops', 'Tensor', 'autograd', 'nn', 'optim', 'scheduler', 'trainer', 'bullet_core_cpp']
