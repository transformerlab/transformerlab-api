#!/usr/bin/env bash
#uv pip install mlx==0.24.0 # Using this version to work around bug: https://github.com/Blaizzy/mlx-audio/issues/207
uv pip install "mlx-audio"
python -m ensurepip --upgrade
