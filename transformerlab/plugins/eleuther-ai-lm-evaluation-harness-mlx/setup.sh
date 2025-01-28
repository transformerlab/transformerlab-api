#!/usr/bin/env bash
# git clone https://github.com/EleutherAI/lm-evaluation-harness
# cd lm-evaluation-harness
# pip install -e .
# pip install lm-eval==0.4.7
# pip install "lm-eval[api]"
git clone https://github.com/EleutherAI/lm-evaluation-harness-mlx
cd lm-evaluation-harness-mlx
git checkout mlx
pip install -e .
