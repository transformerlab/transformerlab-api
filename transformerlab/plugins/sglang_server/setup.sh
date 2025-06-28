#!/bin/bash

sudo apt-get install -y ninja-build
# Install transformerlab-inference from local path
uv pip install "sglang[all]==0.4.8.post1"

# Only install bitsandbytes if ROCm is not available
if ! command -v rocminfo &> /dev/null; then
    uv pip install bitsandbytes
fi
