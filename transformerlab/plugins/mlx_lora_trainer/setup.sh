#!/usr/bin/env bash
uv pip install trl
uv pip install mlx==0.22.0 --upgrade
uv pip install "mlx-lm==0.21.1" --upgrade
uv pip install tensorboardX # for tensorboard
uv pip install wandb
# requires:
# mlx>=0.0.7
# transformers
# numpya