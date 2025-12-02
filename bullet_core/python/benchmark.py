"""
Performance benchmarks for Bullet-Core CPU kernels
Measures GFLOPS and speedup vs pure NumPy
"""

import time
import numpy as np
from bullet_core import ops

def benchmark_gemm():
    """Benchmark matrix multiplication"""
    print("\n📊 GEMM Benchmark")
    print("-" * 60)
    
    sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    
    for M, N, K in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Warmup
        _ = ops.matmul(A, B)
        
        # Benchmark our implementation
        start = time.time()
        iterations = 100 if M < 512 else 10
        for _ in range(iterations):
            _ = ops.matmul(A, B)
        elapsed_ours = time.time() - start
        
        # Benchmark NumPy
        start = time.time()
        for _ in range(iterations):
            _ = A @ B
        elapsed_numpy = time.time() - start
        
        # Calculate GFLOPS
        flops = 2 * M * N * K * iterations
        gflops_ours = flops / (elapsed_ours * 1e9)
        gflops_numpy = flops / (elapsed_numpy * 1e9)
        speedup = elapsed_numpy / elapsed_ours
        
        print(f"Size ({M}x{N}x{K}):")
        print(f"  Ours:  {gflops_ours:6.2f} GFLOPS ({elapsed_ours/iterations*1000:6.2f} ms/iter)")
        print(f"  NumPy: {gflops_numpy:6.2f} GFLOPS ({elapsed_numpy/iterations*1000:6.2f} ms/iter)")
        print(f"  Speedup: {speedup:.2f}x")

def benchmark_softmax():
    """Benchmark softmax"""
    print("\n📊 Softmax Benchmark")
    print("-" * 60)
    
    sizes = [(32, 512), (64, 1024), (128, 2048)]
    
    for batch, dim in sizes:
        x = np.random.randn(batch, dim).astype(np.float32)
        
        # Warmup
        _ = ops.softmax(x)
        
        # Benchmark
        start = time.time()
        iterations = 1000
        for _ in range(iterations):
            _ = ops.softmax(x)
        elapsed_ours = time.time() - start
        
        # NumPy reference
        start = time.time()
        for _ in range(iterations):
            y = np.exp(x - x.max(axis=-1, keepdims=True))
            _ = y / y.sum(axis=-1, keepdims=True)
        elapsed_numpy = time.time() - start
        
        speedup = elapsed_numpy / elapsed_ours
        print(f"Size ({batch}x{dim}): {elapsed_ours/iterations*1000:.3f} ms/iter, Speedup: {speedup:.2f}x")

def benchmark_rmsnorm():
    """Benchmark RMSNorm"""
    print("\n📊 RMSNorm Benchmark")
    print("-" * 60)
    
    sizes = [(32, 256), (64, 512), (128, 1024)]
    
    for batch, dim in sizes:
        x = np.random.randn(batch, dim).astype(np.float32)
        weight = np.random.randn(dim).astype(np.float32)
        
        # Warmup
        _ = ops.rmsnorm(x, weight)
        
        # Benchmark
        start = time.time()
        iterations = 1000
        for _ in range(iterations):
            _ = ops.rmsnorm(x, weight)
        elapsed_ours = time.time() - start
        
        # NumPy reference
        start = time.time()
        for _ in range(iterations):
            rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
            _ = (x / rms) * weight
        elapsed_numpy = time.time() - start
        
        speedup = elapsed_numpy / elapsed_ours
        print(f"Size ({batch}x{dim}): {elapsed_ours/iterations*1000:.3f} ms/iter, Speedup: {speedup:.2f}x")

def benchmark_embedding():
    """Benchmark embedding lookup"""
    print("\n📊 Embedding Lookup Benchmark")
    print("-" * 60)
    
    vocab_size, embedding_dim = 5000, 256
    table = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    
    seq_lengths = [64, 128, 256, 512]
    
    for seq_len in seq_lengths:
        indices = np.random.randint(0, vocab_size, seq_len, dtype=np.int32)
        
        # Warmup
        _ = ops.embedding(table, indices)
        
        # Benchmark
        start = time.time()
        iterations = 1000
        for _ in range(iterations):
            _ = ops.embedding(table, indices)
        elapsed_ours = time.time() - start
        
        # NumPy reference
        start = time.time()
        for _ in range(iterations):
            _ = table[indices]
        elapsed_numpy = time.time() - start
        
        speedup = elapsed_numpy / elapsed_ours
        print(f"Seq len {seq_len}: {elapsed_ours/iterations*1000:.3f} ms/iter, Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    print("=" * 60)
    print("Bullet-Core CPU Kernel Benchmarks")
    print("=" * 60)
    
    benchmark_gemm()
    benchmark_softmax()
    benchmark_rmsnorm()
    benchmark_embedding()
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
