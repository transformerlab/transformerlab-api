#!/usr/bin/env bash

# Check if the 'lm-evaluation-harness-mlx' directory exists and remove it if it does
if [ -d "./lm-evaluation-harness-mlx" ]; then
  rm -rf ./lm-evaluation-harness-mlx
fi

# Clone the repository
git clone https://github.com/chimezie/lm-evaluation-harness-mlx || { echo "Git clone failed or repository already exists"; exit 1; }

# Navigate to the directory
cd lm-evaluation-harness-mlx

git checkout mlx

# Install dependencies
pip install -e .

uv pip install tensorboardX