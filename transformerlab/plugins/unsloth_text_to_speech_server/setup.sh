#!/usr/bin/env bash
uv pip install unsloth
uv pip install snac

if command -v rocminfo &> /dev/null; then
    # Install Unsloth from source
    git clone https://github.com/billishyahao/unsloth.git && cd unsloth && git checkout billhe/rocm && pip install .
    uv pip install unsloth_zoo
    # Install ROCm Bitsandbytes from source 
    git clone --recurse https://github.com/ROCm/bitsandbytes && cd bitsandbytes && git checkout rocm_enabled_multi_backend && pip install -r requirements-dev.txt && cmake -DCOMPUTE_BACKEND=hip -S . && make -j  && pip install -e .
fi