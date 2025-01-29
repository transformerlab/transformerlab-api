#!/usr/bin/env bash
# Clone the repository
git clone https://github.com/chimezie/lm-evaluation-harness-mlx || { echo "Git clone failed or repository already exists"; exit 1; }

# Navigate to the directory
cd lm-evaluation-harness-mlx

git checkout mlx

# Install dependencies
pip install -e .