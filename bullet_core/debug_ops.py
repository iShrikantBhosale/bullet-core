"""
Debug ops.add
"""

import numpy as np
from bullet_core import ops

# Test simple add
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

result = ops.add(a, b)
print(f"Simple add:")
print(f"  a: {a}")
print(f"  b: {b}")
print(f"  result: {result}")

# Test broadcasting
a = np.random.randn(2, 3, 4).astype(np.float32)
b = np.random.randn(2, 3, 4).astype(np.float32)

result = ops.add(a, b)
print(f"\n3D add (same shape):")
print(f"  a shape: {a.shape}, range: [{a.min():.4f}, {a.max():.4f}]")
print(f"  b shape: {b.shape}, range: [{b.min():.4f}, {b.max():.4f}]")
print(f"  result shape: {result.shape}, range: [{result.min():.4f}, {result.max():.4f}]")
print(f"  Expected: {(a + b).min():.4f} to {(a + b).max():.4f}")
