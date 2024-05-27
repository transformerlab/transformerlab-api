#!/usr/bin/env bash
pip install trl
pip install mlx==0.14.0 --upgrade
pip install "mlx-lm==0.14.0" --upgrade
git clone https://github.com/ml-explore/mlx-examples.git
pip install tensorboardX # for tensorboard
# requires:
# mlx>=0.0.7
# transformers
# numpya