#!/usr/bin/env bash
uv pip install mlx==0.25.1 --upgrade
uv pip install "mlx-lm==0.24.0" --upgrade
uv pip install "mlx_embedding_models==0.0.11"

# Uncomment and replace with your local installation for testing
# uv pip install -e /Users/deep.gandhi/transformerlab-repos/FastChat/
