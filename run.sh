#!/bin/bash
MINICONDA_DIRNAME="miniconda3"

# Check if conda activate file exists:
if [ ! -f "$HOME/$MINICONDA_DIRNAME/bin/activate" ]; then
    echo "Conda is installed but it's not stored in $HOME/$MINICONDA_DIRNAME/"
    # Check if conda is installed in the macbrew location:
    if [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        echo "Conda is installed in the macbrew location -- running activate script"
        . "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    fi
else
    # activate the conda base env
    source "$HOME/$MINICONDA_DIRNAME/bin/activate"
fi

# The following conda command "run" is equivalent to
# conda activate transformerlab; unicorn api:app --port 8000 --host
conda run -n transformerlab --live-stream uvicorn api:app --port 8000 --host 0.0.0.0 