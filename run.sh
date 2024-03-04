#!/bin/bash
set -eu

MINICONDA_DIRNAME=${MINICONDA_DIRNAME:-miniconda3}
ENV_NAME="transformerlab"

err_report() {
  echo "Error in run.sh on line $1"
}

trap 'err_report $LINENO' ERR

if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Conda and try again."
else
    echo "✅ Conda is installed."
fi


# Check if the conda environment is activated:
if { conda env list | grep "$ENV_NAME"; } >/dev/null 2>&1; then
    echo "✅ Conda environment $ENV_NAME exists."
else
    echo "❌ Conda environment $ENV_NAME does not exist. Please run ./install.sh and try again."
    exit 1
fi

echo "👏 Enabling conda in shell"
eval "$(conda shell.bash hook)"

echo "👏 Activating transformerlab conda environment"
conda activate transformerlab

# Check if the uvicorn command works:
if ! command -v uvicorn &> /dev/null; then
    echo "❌ Uvicorn is not installed. This usually means that the installation of dependencies failed. Run ./install.sh to install the dependencies."
    exit 1
else
    echo "✅ Uvicorn is installed."
fi

echo "👏 Starting the API server"
uvicorn api:app --port 8000 --host 0.0.0.0 