# Bullet OS Integration Guide

This directory contains tools to integrate Bullet OS with PyTorch and Python.

## Components

1.  **Python Bindings** (`bullet_bindings`): Native C++ extension to run `.bullet` models in Python.
2.  **Exporter** (`bullet_export.py`): Converts PyTorch models to `.bullet` format.
3.  **CLI** (`bullet_cli`): Command-line tool for inference.

## Installation

### Prerequisites
- Python 3.8+
- C++17 Compiler (GCC/Clang)
- `pybind11` (`pip install pybind11`)

### Install Python Package
```bash
cd integration
pip install .
```

### Build CLI
```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### 1. Exporting a PyTorch Model
```python
from bullet_export import export_to_bullet
# ... define model ...
export_to_bullet(model, "vocab.txt", "model.bullet")
```

### 2. Running in Python
```python
from bullet_py_api import Bullet

model = Bullet("model.bullet")
print(model.chat("Hello world"))
```

### 3. Running via CLI
```bash
./bullet_cli run model.bullet "Hello world"
```

## File Structure
- `bullet_python_bindings.cpp`: Pybind11 glue code.
- `bullet_export.py`: PyTorch -> Bullet converter.
- `bullet_cli.cpp`: Standalone C++ runner.
