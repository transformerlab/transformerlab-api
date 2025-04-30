#!/usr/bin/env bash
uv pip install trl
uv pip install mlx==0.25.1 --upgrade
uv pip install "mlx-lm==0.24.0" --upgrade
# requires:
# mlx>=0.0.7
# transformers
# numpya