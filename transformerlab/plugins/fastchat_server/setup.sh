#!/usr/bin/env bash
# Everything should be installed by default
uv pip install "transformerlab-inference>=0.2.38"

if ! command -v rocminfo &> /dev/null; then
    uv pip install bitsandbytes
fi