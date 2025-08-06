#!/usr/bin/env bash
# Everything should be installed by default
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Compute Capability: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)"
    
    # Check if all GPUs have compute capability > 9.0
    min_compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | sort -n | head -1)
    # Use awk to compare floating point numbers
    if awk "BEGIN {exit !($min_compute_cap > 9.0)}"; then
        echo "All GPUs have compute capability > 9.0"
        uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
        uv pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
        uv pip install "triton==3.4.0"
        uv pip install "kernels>=0.9.0" "peft>=0.17.0" "trl>=0.21.0" "trackio" "transformers>=4.55.0"
    fi
fi

if ! command -v rocminfo &> /dev/null; then
    uv pip install bitsandbytes
fi