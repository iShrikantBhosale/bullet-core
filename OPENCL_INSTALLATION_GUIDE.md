# 🎮 OpenCL Installation Guide for Bullet OS

**Enable GPU acceleration for Bullet OS using OpenCL**

This guide will help you install and configure OpenCL to accelerate Bullet OS training and inference on your GPU.

---

## 📋 Table of Contents

1. [What is OpenCL?](#what-is-opencl)
2. [Prerequisites](#prerequisites)
3. [Installation by Platform](#installation-by-platform)
4. [Verification](#verification)
5. [Configuring Bullet OS](#configuring-bullet-os)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tips](#performance-tips)

---

## 🔍 What is OpenCL?

**OpenCL (Open Computing Language)** is an open standard for parallel programming across CPUs, GPUs, and other processors. Unlike CUDA (NVIDIA-only), OpenCL works with:

- ✅ **NVIDIA GPUs** (GeForce, Quadro, Tesla)
- ✅ **AMD GPUs** (Radeon, FirePro)
- ✅ **Intel GPUs** (Integrated & Arc)
- ✅ **CPUs** (Fallback option)

For Bullet OS, OpenCL provides GPU acceleration for:
- Matrix operations (GEMM)
- Attention mechanisms
- Layer normalization
- Softmax operations

---

## ⚙️ Prerequisites

Before installing OpenCL, ensure you have:

1. **A compatible GPU** (NVIDIA, AMD, or Intel)
2. **Updated GPU drivers**
3. **Build tools** (gcc, g++, cmake)
4. **Python 3.8+** (for Bullet OS)

Check your GPU:
```bash
# For NVIDIA
lspci | grep -i nvidia

# For AMD
lspci | grep -i amd

# For Intel
lspci | grep -i vga
```

---

## 🚀 Installation by Platform

### 🐧 Linux (Ubuntu/Debian)

#### Option 1: NVIDIA GPU (Recommended for GeForce/Quadro)

```bash
# Update package list
sudo apt update

# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535  # Or latest version

# Install OpenCL headers and ICD loader
sudo apt install opencl-headers ocl-icd-opencl-dev

# Install NVIDIA OpenCL implementation
sudo apt install nvidia-opencl-dev nvidia-opencl-icd-535

# Install clinfo for verification
sudo apt install clinfo
```

#### Option 2: AMD GPU

```bash
# Update package list
sudo apt update

# Install OpenCL headers and ICD loader
sudo apt install opencl-headers ocl-icd-opencl-dev

# Install AMD ROCm OpenCL (for modern AMD GPUs)
# Add ROCm repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocm-opencl-dev

# Install clinfo for verification
sudo apt install clinfo
```

#### Option 3: Intel GPU

```bash
# Update package list
sudo apt update

# Install OpenCL headers and ICD loader
sudo apt install opencl-headers ocl-icd-opencl-dev

# Install Intel Compute Runtime
sudo apt install intel-opencl-icd

# Install clinfo for verification
sudo apt install clinfo
```

#### Option 4: CPU-only (Fallback)

```bash
# Install OpenCL headers and ICD loader
sudo apt install opencl-headers ocl-icd-opencl-dev

# Install PoCL (Portable Computing Language) for CPU
sudo apt install pocl-opencl-icd

# Install clinfo for verification
sudo apt install clinfo
```

---

### 🪟 Windows

#### NVIDIA GPU

1. **Install NVIDIA GPU Drivers**
   - Download from: https://www.nvidia.com/Download/index.aspx
   - Install the latest driver for your GPU

2. **Install CUDA Toolkit** (includes OpenCL)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Install CUDA Toolkit 12.x or later
   - OpenCL is included automatically

3. **Verify Installation**
   - Download GPU Caps Viewer: https://www.geeks3d.com/dlz/
   - Run and check OpenCL tab

#### AMD GPU

1. **Install AMD GPU Drivers**
   - Download from: https://www.amd.com/en/support
   - Install Adrenalin drivers

2. **OpenCL is included** with AMD drivers
   - No additional installation needed

3. **Verify Installation**
   - Download GPU Caps Viewer or GPU-Z

#### Intel GPU

1. **Install Intel Graphics Drivers**
   - Download from: https://www.intel.com/content/www/us/en/download-center/home.html
   - Install latest graphics drivers

2. **Install Intel OpenCL Runtime**
   - Download from: https://github.com/intel/compute-runtime/releases
   - Install the appropriate version for your GPU

---

### 🍎 macOS

> **Note**: macOS deprecated OpenCL in favor of Metal. For best performance on macOS, consider using Metal backend instead.

```bash
# OpenCL is pre-installed on macOS
# Verify with:
/System/Library/Frameworks/OpenCL.framework/Versions/Current/Libraries/openclinfo

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install clinfo for verification
brew install clinfo
```

---

## ✅ Verification

After installation, verify OpenCL is working:

### Check OpenCL Platforms and Devices

```bash
clinfo
```

**Expected output:**
```
Number of platforms: 1
  Platform Name: NVIDIA CUDA
  Platform Vendor: NVIDIA Corporation
  Platform Version: OpenCL 3.0 CUDA 12.2.0
  
  Number of devices: 1
    Device Name: GeForce GT 730
    Device Type: GPU
    Device Vendor: NVIDIA Corporation
    Max Compute Units: 2
    Max Work Group Size: 1024
    Global Memory: 2048 MB
```

### Quick Test

Create a test file `test_opencl.py`:

```python
import pyopencl as cl

# List all OpenCL platforms and devices
platforms = cl.get_platforms()
print(f"Found {len(platforms)} OpenCL platform(s):")

for i, platform in enumerate(platforms):
    print(f"\nPlatform {i}: {platform.name}")
    print(f"  Vendor: {platform.vendor}")
    print(f"  Version: {platform.version}")
    
    devices = platform.get_devices()
    print(f"  Devices: {len(devices)}")
    
    for j, device in enumerate(devices):
        print(f"\n  Device {j}: {device.name}")
        print(f"    Type: {cl.device_type.to_string(device.type)}")
        print(f"    Max Compute Units: {device.max_compute_units}")
        print(f"    Global Memory: {device.global_mem_size // (1024**2)} MB")
        print(f"    Local Memory: {device.local_mem_size // 1024} KB")
```

Install PyOpenCL and run:
```bash
pip install pyopencl
python test_opencl.py
```

---

## 🔧 Configuring Bullet OS

Once OpenCL is installed, configure Bullet OS to use GPU acceleration:

### 1. Set Environment Variables

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# Enable GPU acceleration
export BULLET_USE_GPU=1

# Set GPU backend to OpenCL
export BULLET_GPU_BACKEND=opencl

# Optional: Specify OpenCL platform (if you have multiple)
export BULLET_OPENCL_PLATFORM=0

# Optional: Specify OpenCL device (if you have multiple GPUs)
export BULLET_OPENCL_DEVICE=0
```

Apply changes:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### 2. Verify Bullet OS Detects GPU

```bash
cd /home/shri/Desktop/bulletOs/bullet_core
python -c "from python.gpu_utils import detect_opencl; detect_opencl()"
```

### 3. Build Bullet OS with OpenCL Support

```bash
cd /home/shri/Desktop/bulletOs/bullet_core

# Install PyOpenCL
pip install pyopencl

# Build C++ extensions with OpenCL
mkdir -p build && cd build
cmake .. -DUSE_OPENCL=ON
make -j$(nproc)
cd ..
```

### 4. Run Training with GPU

```bash
# Train with OpenCL acceleration
python train_marathi_gpu.py
```

**Expected output:**
```
======================================================================
MARATHI PHILOSOPHY TRANSFORMER - GPU-ENABLED TRAINING
======================================================================

🎮 GPU Status: AVAILABLE (OpenCL)
   Platform: NVIDIA CUDA
   Device: GeForce GT 730
   Global Memory: 2048 MB
   Compute Units: 2

📋 Configuration:
  Device: OPENCL
  Model: 4 layers, 256 dim, 4 heads
  Context: 128 tokens
  Training: 10000 steps, LR=0.0005
```

---

## 🐛 Troubleshooting

### Issue 1: `clinfo` shows no platforms

**Solution:**
```bash
# Check if ICD loader is installed
dpkg -l | grep ocl-icd

# Reinstall ICD loader
sudo apt install --reinstall ocl-icd-libopencl1

# Check ICD files
ls /etc/OpenCL/vendors/
```

### Issue 2: Permission denied errors

**Solution:**
```bash
# Add user to video group (for GPU access)
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Logout and login again
```

### Issue 3: PyOpenCL installation fails

**Solution:**
```bash
# Install build dependencies
sudo apt install python3-dev libffi-dev

# Install PyOpenCL with verbose output
pip install pyopencl --verbose

# If still fails, try conda
conda install -c conda-forge pyopencl
```

### Issue 4: "No OpenCL devices found" in Bullet OS

**Solution:**
```bash
# Check environment variables
echo $BULLET_USE_GPU
echo $BULLET_GPU_BACKEND

# Verify OpenCL works outside Bullet OS
python test_opencl.py

# Check if OpenCL libraries are in path
ldconfig -p | grep OpenCL
```

### Issue 5: GT 730 not detected

**Solution:**
```bash
# GT 730 requires older NVIDIA drivers
# Install compatible driver
sudo apt install nvidia-driver-470

# Verify driver
nvidia-smi

# Check OpenCL support
clinfo | grep "GeForce GT 730"
```

---

## ⚡ Performance Tips

### 1. Optimize for GT 730 (2GB VRAM)

The GT 730 has limited VRAM. Optimize settings:

```python
# In train_marathi_gpu.py
BATCH_SIZE = 1          # Keep at 1 for 2GB VRAM
BLOCK_SIZE = 128        # Reduce if OOM errors
D_MODEL = 256           # Reduce if OOM errors
N_LAYER = 4             # Reduce if OOM errors
```

### 2. Monitor GPU Usage

```bash
# For NVIDIA
watch -n 1 nvidia-smi

# For AMD
watch -n 1 radeontop

# For Intel
intel_gpu_top
```

### 3. Enable Profiling

```bash
# Set OpenCL profiling
export BULLET_OPENCL_PROFILE=1

# Run training
python train_marathi_gpu.py
```

### 4. Benchmark OpenCL vs CPU

```bash
# CPU baseline
export BULLET_USE_GPU=0
time python train_marathi.py --steps 100

# OpenCL GPU
export BULLET_USE_GPU=1
export BULLET_GPU_BACKEND=opencl
time python train_marathi_gpu.py --steps 100
```

### 5. Expected Speedup

| Hardware | Steps/sec | Speedup |
|----------|-----------|---------|
| CPU (4 cores) | ~2 steps/s | 1x |
| GT 730 (OpenCL) | ~5-8 steps/s | 2.5-4x |
| GTX 1060 (OpenCL) | ~15-20 steps/s | 7.5-10x |
| RTX 3060 (OpenCL) | ~40-50 steps/s | 20-25x |

---

## 📚 Additional Resources

### Official Documentation
- [OpenCL Official Site](https://www.khronos.org/opencl/)
- [PyOpenCL Documentation](https://documen.tician.de/pyopencl/)
- [NVIDIA OpenCL](https://developer.nvidia.com/opencl)
- [AMD ROCm OpenCL](https://rocmdocs.amd.com/en/latest/Programming_Guides/Opencl-programming-guide.html)

### Bullet OS Documentation
- [BULLET_CORE_ARCHITECTURE.md](BULLET_CORE_ARCHITECTURE.md) - System architecture
- [BULLET_SPEC_v1.0.md](BULLET_SPEC_v1.0.md) - File format specification
- [README.md](README.md) - Quick start guide

### Community
- [Bullet OS GitHub](https://github.com/iShrikantBhosale/bullet-core)
- [Issues & Support](https://github.com/iShrikantBhosale/bullet-core/issues)

---

## 🎯 Quick Reference

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `BULLET_USE_GPU` | `0` or `1` | Enable/disable GPU acceleration |
| `BULLET_GPU_BACKEND` | `opencl`, `cuda`, `metal` | GPU backend to use |
| `BULLET_OPENCL_PLATFORM` | `0`, `1`, ... | OpenCL platform index |
| `BULLET_OPENCL_DEVICE` | `0`, `1`, ... | OpenCL device index |
| `BULLET_OPENCL_PROFILE` | `0` or `1` | Enable profiling |

### Common Commands

```bash
# Check OpenCL platforms
clinfo

# List OpenCL devices
clinfo -l

# Test PyOpenCL
python -c "import pyopencl as cl; print(cl.get_platforms())"

# Train with OpenCL
export BULLET_USE_GPU=1
export BULLET_GPU_BACKEND=opencl
python bullet_core/train_marathi_gpu.py

# Benchmark
python bullet_core/benchmark_gpu.py --backend opencl
```

---

## ✅ Checklist

Before training with OpenCL, ensure:

- [ ] OpenCL drivers installed (`clinfo` works)
- [ ] PyOpenCL installed (`pip install pyopencl`)
- [ ] Environment variables set (`BULLET_USE_GPU=1`, `BULLET_GPU_BACKEND=opencl`)
- [ ] GPU detected by Bullet OS (check startup logs)
- [ ] Sufficient VRAM for model (2GB minimum for GT 730)
- [ ] Permissions configured (user in `video` and `render` groups)

---

## 🚀 Next Steps

Once OpenCL is configured:

1. **Train your first model**: `python bullet_core/train_marathi_gpu.py`
2. **Monitor performance**: Use `nvidia-smi` or `radeontop`
3. **Optimize settings**: Adjust batch size and model size for your GPU
4. **Export to .bullet**: `python test_checkpoints.py`
5. **Deploy**: Use the trained model for inference

---

**Happy Training! 🎉**

For issues or questions, open an issue on [GitHub](https://github.com/iShrikantBhosale/bullet-core/issues).
