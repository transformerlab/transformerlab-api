#!/bin/bash

sudo apt-get update && sudo apt-get install -y ninja-build
# Install transformerlab-inference from local path
uv pip install sglang zmq orjson uvloop sgl-kernel compressed_tensors msgspec partial_json_parser torchao xgrammar ninja 'flashinfer-python>=0.2.6.post1'
# Only install bitsandbytes if ROCm is not available
if ! command -v rocminfo &> /dev/null; then
    uv pip install bitsandbytes
fi
