#!/bin/bash
# -e: Exit immediately on error. -u: treat unset variables as an error and exit immediately.
set -eu

OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" == "Darwin" ]; then
    OS="MacOSX"
fi

MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-$OS-$ARCH.sh"
MINICONDA_DIRNAME="miniconda3"

if ! command -v conda &> /dev/null; then
    echo Conda is not installed.

    # download and install conda (headless/silent)
    echo Downloading "$MINICONDA_URL"

    curl -o miniconda_installer.sh "$MINICONDA_URL" && bash miniconda_installer.sh -b -p "$HOME/$MINICONDA_DIRNAME" && rm miniconda_installer.sh
    # Install conda to bash and zsh
    ~/miniconda3/bin/conda init bash
    if [ -n "$(command -v zsh)" ]; then
        ~/miniconda3/bin/conda init zsh
fi

# Check if conda activate file exists:
if [ ! -f "$HOME/$MINICONDA_DIRNAME/bin/activate" ]; then
    echo "Conda is installed but it's not stored in $HOME/$MINICONDA_DIRNAME/"
    echo "If the following script doesn't work, run conda init and follow the"
    echo "instructions there"
    # Check if conda is installed in the macbrew location:
    if [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        echo "Conda is installed in the macbrew location -- running activate script"
        . "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    fi
else
    # activate the conda base env
    source "$HOME/$MINICONDA_DIRNAME/bin/activate"
fi


# Create the conda environment for Tranformer Lab
ENV_NAME="transformerlab"

if { conda env list | grep "$ENV_NAME"; } >/dev/null 2>&1; then
    echo "Conda environment $ENV_NAME already exists."
else
    echo conda create -y -n "$ENV_NAME" python=3.11
    conda create -y -n "$ENV_NAME" python=3.11
fi

# Activate the newly created environment
echo conda activate "$ENV_NAME"
conda activate "$ENV_NAME"

# Verify that the environment is activated by displaying the Python version
which python

# store if the box has an nvidia graphics card
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    # Check if nvidia-smi is available
    echo "nvidia-smi is available"
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits) || echo "Issue with NVIDIA SMI"
    echo $GPU_INFO
    if [ -n "$GPU_INFO" ]; then
        echo "NVIDIA GPU detected: $GPU_INFO"
        HAS_GPU=true 
    else
        echo "Nvidia SMI exists, No NVIDIA GPU detected. Perhaps you need to re-install NVIDIA drivers."
    fi
fi

echo "HAS_GPU=$HAS_GPU"

if [ "$HAS_GPU" = true ] ; then
    echo "Your computer has a GPU; installing cuda:"
    conda install -y cuda -c nvidia/label/cuda-12.1.1

    echo "Installing requirements:"
    # Install the python requirements
    pip install -r requirements.txt
else
    echo "No NVIDIA GPU detected drivers detected. Install NVIDIA drivers to enable GPU support."
    echo "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions"
    echo "Installing Tranformer Lab requirements without GPU support"

    pip install -r requirements-no-gpu.txt
fi

# Deactivate the environment when done 
conda deactivate
