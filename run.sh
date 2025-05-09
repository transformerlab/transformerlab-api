#!/bin/bash
set -e

ENV_NAME="transformerlab"
TLAB_DIR="$HOME/.transformerlab"
TLAB_CODE_DIR="${TLAB_DIR}/src"

MINIFORGE_ROOT=${TLAB_DIR}/miniforge3
CONDA_BIN=${MINIFORGE_ROOT}/bin/conda
ENV_DIR=${TLAB_DIR}/envs/${ENV_NAME}
CUSTOM_ENV=false

HOST="::"
PORT="8338"

RELOAD=false

# echo "Your shell is $SHELL"
# echo "Conda's binary is at ${CONDA_BIN}"
# echo "Your current directory is $(pwd)"

err_report() {
  echo "Error in run.sh on line $1"
}

# trap 'err_report $LINENO' ERR

if ! command -v ${CONDA_BIN} &> /dev/null; then
    echo "❌ Conda is not installed at ${MINIFORGE_ROOT}. Please run ./install.sh and try again."
else
    echo "✅ Conda is installed."
fi

while getopts crp:h: flag
do
    case "${flag}" in
        c) CUSTOM_ENV=true;;
        r) RELOAD=true;;
        p) PORT=${OPTARG};;
        h) HOST=${OPTARG};;
    esac
done

# Print out everything that was discovered above
# echo "👏 Using host: ${HOST}
# 👏 Using port: ${PORT}
# 👏 Using reload: ${RELOAD}
# 👏 Using custom environment: ${CUSTOM_ENV}"

if [ "$CUSTOM_ENV" = true ]; then
    echo "🔧 Using current conda environment, I won't activate for you"
else
    # echo "👏 Using default conda environment: ${ENV_DIR}"
    echo "👏 Enabling conda in shell"

    eval "$(${CONDA_BIN} shell.bash hook)"

    echo "👏 Activating transformerlab conda environment"
    conda activate "${ENV_DIR}"
fi

# Check if the uvicorn command works:
if ! command -v uvicorn &> /dev/null; then
    echo "❌ Uvicorn is not installed. This usually means that the installation of dependencies failed. Run ./install.sh to install the dependencies."
    exit 1
else
    echo -n ""
    # echo "✅ Uvicorn is installed."
fi

# Check if NVIDIA GPU is available and add necessary paths
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected, adding CUDA libraries to path"
    # Add common NVIDIA library paths
    export LD_LIBRARY_PATH=${ENV_DIR}/lib:$LD_LIBRARY_PATH
fi

echo "▶️ Starting the API server:"
if [ "$RELOAD" = true ]; then
    echo "🔁 Reload the server on file changes"
    uv run -v uvicorn api:app --reload --port ${PORT} --host ${HOST}
else
    uv run -v uvicorn api:app --port ${PORT} --host ${HOST} --no-access-log
fi
