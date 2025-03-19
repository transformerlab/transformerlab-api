#!/usr/bin/env bash
uv pip install trl
uv pip install mlx==0.23.2 --upgrade
uv pip install "mlx-lm==0.22.1" --upgrade
uv pip install tensorboardX # for tensorboard
# requires:
# mlx>=0.0.7
# transformers
# numpya