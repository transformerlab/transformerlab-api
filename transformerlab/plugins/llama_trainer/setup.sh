#!/usr/bin/env bash
#pip install "datasets==2.9.0" "accelerate==0.21.0" "evaluate==0.4.0" loralib
uv pip install trl
# if we're NOT on AMD/ROCm, install bitsandbytes for quantization support
if ! command -v rocminfo &> /dev/null; then
    uv pip install bitsandbytes
fi
