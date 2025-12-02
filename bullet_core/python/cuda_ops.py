"""
CUDA operations wrapper for Python
Provides high-level API for GPU operations
"""

import numpy as np

try:
    from . import bullet_core_cuda as cuda_backend
    CUDA_AVAILABLE = cuda_backend.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    cuda_backend = None

class CudaTensor:
    """Wrapper for GPU tensors"""
    
    def __init__(self, data):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        
        self.shape = data.shape
        self.dtype = data.dtype
        
        # Allocate GPU memory
        self.nbytes = data.nbytes
        self.ptr = cuda_backend.malloc(self.nbytes)
        
        # Copy data to GPU
        if isinstance(data, np.ndarray):
            cuda_backend.to_cuda(np.ascontiguousarray(data.astype(np.float32)))
    
    def cpu(self):
        """Copy tensor back to CPU"""
        return cuda_backend.from_cuda(self.ptr, self.shape)
    
    def __del__(self):
        """Free GPU memory"""
        if hasattr(self, 'ptr') and self.ptr is not None:
            cuda_backend.free(self.ptr)

# High-level API functions
def is_available():
    """Check if CUDA is available"""
    return CUDA_AVAILABLE

def get_device_name():
    """Get GPU device name"""
    if not CUDA_AVAILABLE:
        return "No CUDA device"
    return cuda_backend.get_device_name()

def get_device_info():
    """Get detailed device information"""
    if not CUDA_AVAILABLE:
        return {"available": False}
    
    return {
        "available": True,
        "name": cuda_backend.get_device_name(),
        "device_id": cuda_backend.get_device_id(),
        "compute_capability": cuda_backend.get_compute_capability(),
        "total_memory": cuda_backend.get_total_memory(),
        "used_vram": cuda_backend.get_used_vram(),
        "free_vram": cuda_backend.get_free_vram(),
        "usage_percent": cuda_backend.get_usage_percent()
    }

def matmul(A, B, transpose_A=False, transpose_B=False):
    """
    Matrix multiplication on GPU
    
    Args:
        A: numpy array or CudaTensor
        B: numpy array or CudaTensor
        transpose_A: transpose A
        transpose_B: transpose B
    
    Returns:
        Result as numpy array
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
    
    # Convert to CudaTensor if needed
    if isinstance(A, np.ndarray):
        A = CudaTensor(A)
    if isinstance(B, np.ndarray):
        B = CudaTensor(B)
    
    # Determine dimensions
    if transpose_A:
        M, K1 = A.shape[1], A.shape[0]
    else:
        M, K1 = A.shape[0], A.shape[1]
    
    if transpose_B:
        K2, N = B.shape[1], B.shape[0]
    else:
        K2, N = B.shape[0], B.shape[1]
    
    assert K1 == K2, f"Dimension mismatch: {K1} != {K2}"
    
    # Allocate output
    C_data = np.zeros((M, N), dtype=np.float32)
    C = CudaTensor(C_data)
    
    # Launch kernel
    cuda_backend.gemm(A.ptr, B.ptr, C.ptr, M, N, K1, transpose_A, transpose_B)
    
    # Copy result back
    return C.cpu()

def softmax(x):
    """Softmax on GPU"""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
    
    if isinstance(x, np.ndarray):
        x = CudaTensor(x)
    
    batch = int(np.prod(x.shape[:-1]))
    dim = x.shape[-1]
    
    out_data = np.zeros_like(x.cpu())
    out = CudaTensor(out_data)
    
    cuda_backend.softmax(x.ptr, out.ptr, batch, dim)
    
    return out.cpu()

def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm on GPU"""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
    
    if isinstance(x, np.ndarray):
        x = CudaTensor(x)
    if isinstance(weight, np.ndarray):
        weight = CudaTensor(weight)
    
    batch = int(np.prod(x.shape[:-1]))
    dim = x.shape[-1]
    
    out_data = np.zeros_like(x.cpu())
    out = CudaTensor(out_data)
    
    cuda_backend.rmsnorm(x.ptr, weight.ptr, out.ptr, batch, dim, eps)
    
    return out.cpu()

# Print device info on import
if CUDA_AVAILABLE:
    info = get_device_info()
    print(f"✅ CUDA available: {info['name']}")
    print(f"   Compute Capability: {info['compute_capability'] / 10:.1f}")
    print(f"   Total Memory: {info['total_memory'] / 1024**3:.2f} GB")
else:
    print("⚠️  CUDA not available - using CPU only")
