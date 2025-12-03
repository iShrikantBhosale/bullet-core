#!/usr/bin/env python3
"""
OpenCL Detection and Configuration Helper for Bullet OS
Detects OpenCL platforms and devices, and helps configure environment variables.
"""

import os
import sys

def check_pyopencl():
    """Check if PyOpenCL is installed"""
    try:
        import pyopencl as cl
        return True, cl
    except ImportError:
        return False, None

def detect_opencl_platforms():
    """Detect all OpenCL platforms and devices"""
    has_pyopencl, cl = check_pyopencl()
    
    if not has_pyopencl:
        print("❌ PyOpenCL is not installed!")
        print("\nInstall it with:")
        print("  pip install pyopencl")
        print("\nOr with conda:")
        print("  conda install -c conda-forge pyopencl")
        return False
    
    try:
        platforms = cl.get_platforms()
        
        if not platforms:
            print("❌ No OpenCL platforms found!")
            print("\nPlease install OpenCL drivers for your GPU.")
            print("See OPENCL_INSTALLATION_GUIDE.md for instructions.")
            return False
        
        print("=" * 70)
        print("🎮 OpenCL Detection Results")
        print("=" * 70)
        print(f"\n✅ Found {len(platforms)} OpenCL platform(s)\n")
        
        total_devices = 0
        gpu_devices = []
        
        for i, platform in enumerate(platforms):
            print(f"Platform {i}: {platform.name}")
            print(f"  Vendor: {platform.vendor}")
            print(f"  Version: {platform.version}")
            
            try:
                devices = platform.get_devices()
                print(f"  Devices: {len(devices)}")
                
                for j, device in enumerate(devices):
                    device_type = cl.device_type.to_string(device.type)
                    print(f"\n  Device {j}: {device.name}")
                    print(f"    Type: {device_type}")
                    print(f"    Max Compute Units: {device.max_compute_units}")
                    print(f"    Global Memory: {device.global_mem_size // (1024**2)} MB")
                    print(f"    Local Memory: {device.local_mem_size // 1024} KB")
                    print(f"    Max Work Group Size: {device.max_work_group_size}")
                    print(f"    Max Clock Frequency: {device.max_clock_frequency} MHz")
                    
                    total_devices += 1
                    
                    if 'GPU' in device_type.upper():
                        gpu_devices.append((i, j, platform.name, device.name))
                
            except Exception as e:
                print(f"  ⚠️  Error getting devices: {e}")
            
            print()
        
        print("=" * 70)
        print(f"Summary: {len(platforms)} platform(s), {total_devices} device(s)")
        
        if gpu_devices:
            print(f"✅ Found {len(gpu_devices)} GPU device(s)")
            print("\n📝 Recommended configuration:")
            platform_id, device_id, platform_name, device_name = gpu_devices[0]
            print(f"\n  Platform: {platform_name}")
            print(f"  Device: {device_name}")
            print(f"\n  Environment variables:")
            print(f"    export BULLET_USE_GPU=1")
            print(f"    export BULLET_GPU_BACKEND=opencl")
            print(f"    export BULLET_OPENCL_PLATFORM={platform_id}")
            print(f"    export BULLET_OPENCL_DEVICE={device_id}")
            
            # Generate shell script
            with open("setup_opencl.sh", "w") as f:
                f.write("#!/bin/bash\n")
                f.write("# Auto-generated OpenCL configuration for Bullet OS\n\n")
                f.write("export BULLET_USE_GPU=1\n")
                f.write("export BULLET_GPU_BACKEND=opencl\n")
                f.write(f"export BULLET_OPENCL_PLATFORM={platform_id}\n")
                f.write(f"export BULLET_OPENCL_DEVICE={device_id}\n")
                f.write("\necho '✅ OpenCL environment configured!'\n")
                f.write(f"echo 'Platform: {platform_name}'\n")
                f.write(f"echo 'Device: {device_name}'\n")
            
            os.chmod("setup_opencl.sh", 0o755)
            
            print(f"\n✅ Created setup_opencl.sh")
            print(f"\nTo activate, run:")
            print(f"  source setup_opencl.sh")
        else:
            print("⚠️  No GPU devices found. OpenCL will use CPU.")
        
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"❌ Error detecting OpenCL: {e}")
        print("\nPlease ensure OpenCL drivers are properly installed.")
        print("See OPENCL_INSTALLATION_GUIDE.md for instructions.")
        return False

def check_current_config():
    """Check current Bullet OS GPU configuration"""
    print("\n" + "=" * 70)
    print("🔧 Current Bullet OS Configuration")
    print("=" * 70)
    
    use_gpu = os.environ.get('BULLET_USE_GPU', '0')
    gpu_backend = os.environ.get('BULLET_GPU_BACKEND', 'none')
    opencl_platform = os.environ.get('BULLET_OPENCL_PLATFORM', 'auto')
    opencl_device = os.environ.get('BULLET_OPENCL_DEVICE', 'auto')
    
    print(f"\nBULLET_USE_GPU: {use_gpu}")
    print(f"BULLET_GPU_BACKEND: {gpu_backend}")
    print(f"BULLET_OPENCL_PLATFORM: {opencl_platform}")
    print(f"BULLET_OPENCL_DEVICE: {opencl_device}")
    
    if use_gpu == '1' and gpu_backend == 'opencl':
        print("\n✅ GPU acceleration is ENABLED with OpenCL")
    elif use_gpu == '1':
        print(f"\n⚠️  GPU acceleration is enabled but backend is '{gpu_backend}' (not OpenCL)")
    else:
        print("\n❌ GPU acceleration is DISABLED")
        print("\nTo enable OpenCL, run:")
        print("  source setup_opencl.sh")
    
    print("=" * 70)

def run_benchmark():
    """Run a simple OpenCL benchmark"""
    has_pyopencl, cl = check_pyopencl()
    
    if not has_pyopencl:
        print("❌ PyOpenCL not installed. Cannot run benchmark.")
        return
    
    try:
        import numpy as np
        import time
        
        print("\n" + "=" * 70)
        print("⚡ Running OpenCL Benchmark")
        print("=" * 70)
        
        # Get first GPU device
        platforms = cl.get_platforms()
        devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
        
        if not devices:
            devices = platforms[0].get_devices()
        
        ctx = cl.Context([devices[0]])
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        print(f"\nDevice: {devices[0].name}")
        
        # Matrix multiplication benchmark
        sizes = [256, 512, 1024, 2048]
        
        kernel_code = """
        __kernel void matmul(
            __global const float* A,
            __global const float* B,
            __global float* C,
            const int N)
        {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
        """
        
        prg = cl.Program(ctx, kernel_code).build()
        
        print("\nMatrix Multiplication Benchmark:")
        print(f"{'Size':<10} {'Time (ms)':<15} {'GFLOPS':<15}")
        print("-" * 40)
        
        for N in sizes:
            # Create random matrices
            A = np.random.rand(N, N).astype(np.float32)
            B = np.random.rand(N, N).astype(np.float32)
            C = np.zeros((N, N), dtype=np.float32)
            
            # Create buffers
            mf = cl.mem_flags
            A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
            
            # Execute kernel
            event = prg.matmul(queue, (N, N), None, A_buf, B_buf, C_buf, np.int32(N))
            event.wait()
            
            # Get timing
            elapsed = 1e-6 * (event.profile.end - event.profile.start)  # Convert to ms
            
            # Calculate GFLOPS
            flops = 2 * N**3  # Matrix multiplication FLOPs
            gflops = (flops / elapsed) / 1e6
            
            print(f"{N}x{N:<6} {elapsed:<15.2f} {gflops:<15.2f}")
        
        print("\n✅ Benchmark complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def main():
    """Main function"""
    print("\n🚀 Bullet OS - OpenCL Configuration Helper\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--benchmark':
            run_benchmark()
            return
        elif sys.argv[1] == '--config':
            check_current_config()
            return
        elif sys.argv[1] == '--help':
            print("Usage: python setup_opencl.py [--benchmark|--config|--help]")
            print("\nOptions:")
            print("  (no args)    Detect OpenCL platforms and devices")
            print("  --benchmark  Run OpenCL performance benchmark")
            print("  --config     Show current Bullet OS GPU configuration")
            print("  --help       Show this help message")
            return
    
    # Default: detect OpenCL
    success = detect_opencl_platforms()
    
    if success:
        check_current_config()
        
        print("\n📚 Next steps:")
        print("  1. Run: source setup_opencl.sh")
        print("  2. Test: python setup_opencl.py --benchmark")
        print("  3. Train: python bullet_core/train_marathi_gpu.py")
        print("\nFor detailed instructions, see OPENCL_INSTALLATION_GUIDE.md")

if __name__ == "__main__":
    main()
