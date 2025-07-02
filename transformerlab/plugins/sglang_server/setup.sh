#!/bin/bash

set -e

# Step 0: Install common dependencies
uv pip install zmq orjson uvloop sgl-kernel compressed_tensors msgspec partial_json_parser torchao xgrammar ninja 'flashinfer-python>=0.2.6.post1'

# Step 1: Detect backend
if command -v rocminfo &>/dev/null; then
    echo "[setup] ROCm detected."
    BACKEND="rocm"
else
    echo "[setup] ROCm not detected. Assuming CUDA."
    BACKEND="cuda"
fi

# Step 2: Install correct torch + torchvision
if [ "$BACKEND" = "rocm" ]; then
    echo "[setup] Installing ROCm-compatible PyTorch (assumes ROCm 5.7)"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
else
    echo "[setup] Installing CUDA 12.8-compatible PyTorch"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# Step 3: Install sglang without letting it override torch
uv pip install sglang[all]==0.4.8.post1

# Step 4: Conditionally install bitsandbytes
if [ "$BACKEND" != "rocm" ]; then
    uv pip install bitsandbytes
else
    echo "[setup] Skipping bitsandbytes for ROCm."
fi
