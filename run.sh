#!/bin/bash
set -eu

MINICONDA_DIRNAME=${MINICONDA_DIRNAME:-miniconda3}
CONDA_BIN=${HOME}/${MINICONDA_DIRNAME}/bin/conda
ENV_NAME="transformerlab"

echo "Your shell is $SHELL"
echo "Conda's binary is at ${CONDA_BIN}"

err_report() {
  echo "Error in run.sh on line $1"
}

trap 'err_report $LINENO' ERR

if ! command -v ${CONDA_BIN} &> /dev/null; then
    echo "❌ Conda is not installed at ${HOME}/${MINICONDA_DIRNAME}. Please install Conda there (and only there) and try again."
else
    echo "✅ Conda is installed."
fi

echo "👏 Enabling conda in shell"

eval "$(${CONDA_BIN} shell.bash hook)"

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