"""
Test C++ kernel directly
"""

import numpy as np
try:
    from bullet_core import bullet_core_cpp
except:
    import bullet_core_cpp

# Test simple add
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
out = np.zeros(4, dtype=np.float32)

print(f"Before C++ call:")
print(f"  a: {a}")
print(f"  b: {b}")
print(f"  out: {out}")

bullet_core_cpp.vector_add_f32(a, b, out, 4)

print(f"\nAfter C++ call:")
print(f"  out: {out}")
print(f"  Expected: {a + b}")

# Test with flatten
a2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b2d = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
out2d = np.zeros_like(a2d)

print(f"\n2D test:")
print(f"  a2d: {a2d}")
print(f"  b2d: {b2d}")

# Flatten
a_flat = a2d.flatten()
b_flat = b2d.flatten()
out_flat = out2d.flatten()

print(f"\nFlattened:")
print(f"  a_flat: {a_flat}")
print(f"  b_flat: {b_flat}")
print(f"  out_flat (before): {out_flat}")

bullet_core_cpp.vector_add_f32(a_flat, b_flat, out_flat, a_flat.size)

print(f"  out_flat (after): {out_flat}")
print(f"  out2d: {out2d}")
print(f"  Expected: {a2d + b2d}")
