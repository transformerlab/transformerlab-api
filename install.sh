#!/bin/bash
set -e

ENV_NAME="transformerlab"
TLAB_DIR="$HOME/.transformerlab"
TLAB_CODE_DIR="${TLAB_DIR}/src"
TLAB_STATIC_WEB_DIR="${TLAB_DIR}/webapp"

OLD_MINICONDA_ROOT=${TLAB_DIR}/miniconda3 # old place -- used to detect if an old install exists
MINIFORGE_ROOT=${TLAB_DIR}/miniforge3
CONDA_BIN=${MINIFORGE_ROOT}/bin/conda
MAMBA_BIN=${MINIFORGE_ROOT}/bin/mamba
ENV_DIR=${TLAB_DIR}/envs/${ENV_NAME}
RUN_DIR=$(pwd)

##############################
# Helper Functions
##############################

err_report() {
  echo "Error on line $1"
}

trap 'err_report $LINENO' ERR

abort() {
  printf "%s\n" "$@" >&2
  exit 1
}

if [[ -t 1 ]]
then
  tty_escape() { printf "\033[%sm" "$1"; }
else
  tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_underline="$(tty_escape "4;39")"
tty_blue="$(tty_mkbold 34)"
tty_red="$(tty_mkbold 31)"
tty_bold="$(tty_mkbold 39)"
tty_reset="$(tty_escape 0)"

shell_join() {
  local arg
  printf "%s" "$1"
  shift
  for arg in "$@"
  do
    printf " "
    printf "%s" "${arg// /\ }"
  done
}

chomp() {
  printf "%s" "${1/"$'\n'"/}"
}

ohai() {
  printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$(shell_join "$@")"
}

warn() {
  printf "${tty_red}Warning${tty_reset}: %s\n" "$(chomp "$1")" >&2
}

title() {
  echo ""
  printf "%s#########################################################################%s\n" "${tty_blue}" "${tty_reset}"
  printf "${tty_blue}#### ${tty_bold} %s${tty_reset}\n" "$(shell_join "$@")"
  printf "%s#########################################################################%s\n" "${tty_blue}" "${tty_reset}"
}

check_conda() {
  if ! command -v "${CONDA_BIN}" &> /dev/null; then
    abort "❌ Conda is not installed at ${MINIFORGE_ROOT}. Please install Conda using '${TLAB_DIR}/src/install.sh install_conda' and try again."
  else
    ohai "✅ Conda is installed at ${MINIFORGE_ROOT}."
  fi
}

check_mamba() {
  if ! command -v "${MAMBA_BIN}" &> /dev/null; then
    abort "❌ Mamba is not installed at ${MINIFORGE_ROOT}. Please install Mamba using '${TLAB_DIR}/src/install.sh install_conda' and try again."
  else
    ohai "✅ Mamba is installed at ${MINIFORGE_ROOT}."
  fi
}

check_python() {
  if ! command -v python &> /dev/null; then
    abort "❌ Python is not installed as 'python'. Please install Python and try again or it could be installed as 'python3'"
  else
    # store python version in variable:
    PYTHON_VERSION=$(python --version)
    ohai "✅ Python is installed: $PYTHON_VERSION"
  fi
}

unset_conda_for_sure() {
  { conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null
  { mamba deactivate && mamba deactivate && mamba deactivate; } 2> /dev/null
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH
  unset PYTHONHOME
}

# We've seen users who installed conda using root have problems if their
# ~/.conda directory is not writable. This checks for that.
check_if_conda_environments_dot_text_is_writable() {
    # Check if a file called ~/.conda/environments.txt exists:
  if [ -f "$HOME/.conda/environments.txt" ]; then
    # Now check if it is writable:
    if [ -w "$HOME/.conda/environments.txt" ]; then
      echo -n
      # echo "✅ The file ~/.conda/environments.txt is writable."
    else
      abort "❌ The file $HOME/.conda/environments.txt exists but is not writable. Please run [sudo chown -R \$USER ~/.conda] in the terminal to fix conda permissions."
    fi
  else
    echo -n
    # echo "The file $HOME/.conda/environments.txt does not exist. No problem we will create it below"
  fi
}

# First check OS.
# WSL will return "Linux" which is OK. We will check KERNEL to detect WSL.
OS="$(uname)"
KERNEL=$(uname -r)
if [[ "${OS}" == "Linux" ]]
then
  TLAB_ON_LINUX=1
elif [[ "${OS}" == "Darwin" ]]
then
  TLAB_ON_MACOS=1
else
  abort "Transformer Lab is only supported on macOS and Linux, you are running ${OS}."
fi

# Stack Overflow says the best way to check for WSL is looking for Microsoft in the uname kernel
if [[ "${KERNEL}" =~ ([Mm]icrosoft|[Ww][Ss][Ll]2) ]]; then
  TLAB_ON_WSL=1
else
  TLAB_ON_WSL=0
fi

##############################
## Step 1: Download Transformer Lab
## and  place it in the ~/.transformerlab/src directory.
##############################

download_transformer_lab() {
  title "Step 1: Download the latest release of Transformer Lab"
  echo "🌘 Step 1: START"

  # First check that curl is installed:
  if ! command -v curl &> /dev/null; then
    abort "❌ curl is not installed on the remote host. Please install curl and try again."
  else
    ohai "✅ curl is installed."
  fi
  
  # Figure out the path to the lastest release of Transformer Lab
  LATEST_RELEASE_VERSION=$(curl -Ls -o /dev/null -w %{url_effective} https://github.com/transformerlab/transformerlab-api/releases/latest)
  LATEST_RELEASE_VERSION=$(basename "$LATEST_RELEASE_VERSION")
  LATEST_RELEASE_VERSION_WITHOUT_V=$(echo "$LATEST_RELEASE_VERSION" | sed 's/v//g')
  echo "Latest Release of the API on Github: $LATEST_RELEASE_VERSION"
  TLAB_URL="https://github.com/transformerlab/transformerlab-api/archive/refs/tags/${LATEST_RELEASE_VERSION}.tar.gz"
  echo "Download Location: $TLAB_URL"

  # If the user has not installed Transformer Lab, then we should install it.
  ohai "Installing Transformer Lab ${LATEST_RELEASE_VERSION}..."
  # Fetch the latest version of Transformer Lab from GitHub:
  mkdir -p "${TLAB_DIR}"
  curl -L "${TLAB_URL}" -o "${TLAB_DIR}/transformerlab.tar.gz"
  NEW_DIRECTORY_NAME="transformerlab-api-${LATEST_RELEASE_VERSION_WITHOUT_V}"
  rm -rf "${TLAB_DIR:?}/${NEW_DIRECTORY_NAME:?}"
  rm -rf "${TLAB_CODE_DIR}"
  tar -xf "${TLAB_DIR}/transformerlab.tar.gz" -C "${TLAB_DIR}"
  mv "${TLAB_DIR}/${NEW_DIRECTORY_NAME}" "${TLAB_CODE_DIR}"
  rm "${TLAB_DIR}/transformerlab.tar.gz"
  # Create a file called LATEST_VERSION that contains the latest version of Transformer Lab.
  echo "${LATEST_RELEASE_VERSION}" > "${TLAB_CODE_DIR}/LATEST_VERSION"

  # Now do the same thing for the web app which is in a different repo called https://github.com/transformerlab/transformerlab-app
  # Step 1: First get the latest release version:
  TLAB_APP_URL="https://github.com/transformerlab/transformerlab-app/releases/latest/download/transformerlab_web.tar.gz"
  echo "APP Download Location: $TLAB_APP_URL"

  # Delete and recreate the target static files directory
  echo "Creating clean directory at ${TLAB_STATIC_WEB_DIR}"
  rm -rf "${TLAB_STATIC_WEB_DIR:?}" 2>/dev/null || true
  mkdir -p "${TLAB_STATIC_WEB_DIR}"

  # Download and extract, handling possible failure
  if curl -L --fail "${TLAB_APP_URL}" -o /tmp/transformerlab_web.tar.gz; then
    # Extraction succeeded, proceed with unpacking
    tar -xzf /tmp/transformerlab_web.tar.gz -C "${TLAB_STATIC_WEB_DIR}"
    
    # Move contents up one level and clean up
    mv "${TLAB_STATIC_WEB_DIR}/transformerlab_web/"* "${TLAB_STATIC_WEB_DIR}/" 2>/dev/null || true
    rmdir "${TLAB_STATIC_WEB_DIR}/transformerlab_web" 2>/dev/null || true
    
    # Remove the temporary file
    rm /tmp/transformerlab_web.tar.gz
    
    echo "Web app successfully installed."
  else
    echo "Warning: Could not download web app from ${TLAB_APP_URL}. Continuing without web app installation."
  fi

  echo "🌕 Step 1: COMPLETE"
}

##############################
## Step 2: Install Conda
##############################

install_conda() {
  title "Step 2: Install Conda"
  echo "🌘 Step 2: START"

  unset_conda_for_sure

  # first check if the old miniconda folder exists, and if so, delete it
  # This is because we have switched to using Miniforge instead of Miniconda.
  if [ -n "${OLD_MINICONDA_ROOT:-}" ] && [ -d "$OLD_MINICONDA_ROOT" ]; then
    echo "[INFO] Deleting deprecated Miniconda installation at $OLD_MINICONDA_ROOT"
    rm -rf "$OLD_MINICONDA_ROOT"
  else
    echo "[INFO] No deprecated Miniconda installation found"
  fi

  # check if conda already exists:
  if ! command -v "${CONDA_BIN}" &> /dev/null; then
    echo "Conda is not installed at ${MINIFORGE_ROOT}."
    OS=$(uname -s)
    ARCH=$(uname -m)

    if [ "$OS" == "Darwin" ]; then
        OS="MacOSX"
    fi

    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$OS-$ARCH.sh"
    echo Downloading "$MINIFORGE_URL"

    # Change the directory to the Transformer Lab directory
    mkdir -p "$TLAB_DIR"
    cd "$TLAB_DIR"
    # first check if the MINIFORGEROOT exists, and if so, delete it:
    if [ -d "$MINIFORGE_ROOT" ]; then
      echo "Deleting existing Miniforge installation at $MINIFORGE_ROOT"
      rm -rf "$MINIFORGE_ROOT"
    fi
    curl -L -o miniforge_installer.sh "$MINIFORGE_URL" && bash miniforge_installer.sh -b -p "$MINIFORGE_ROOT" && rm miniforge_installer.sh
    # Install conda to bash and zsh. We keep these commented out
    # to avoid adding our conda to the user's shell as the default.
    # $MINIFORGE_ROOT/bin/conda init bash
    # if [ -n "$(command -v zsh)" ]; then
    #     $MINIFORGE_ROOT/bin/conda init zsh
    # fi
  else
      ohai "Conda is installed at ${MINIFORGE_ROOT}, we do not need to install it"
  fi

  # Enable conda in shell
  eval "$(${CONDA_BIN} shell.bash hook)"

  check_conda
  check_mamba

  conda info
  
  echo "🌕 Step 2: COMPLETE"
}

##############################
## Step 3: Create the Conda Environment
##############################

create_conda_environment() {
  title "Step 3: Create the Conda Environment"
  echo "🌘 Step 3: START"

  check_if_conda_environments_dot_text_is_writable

  check_conda

  unset_conda_for_sure

  eval "$(${CONDA_BIN} shell.bash hook)"

  conda info --envs

  # Create the conda environment for Transformer Lab
  if { conda env list | grep "$ENV_DIR"; } >/dev/null 2>&1; then
      echo "✅ Conda environment $ENV_DIR already exists."
  else
      echo mamba create -y -n "$ENV_DIR" python=3.11
      conda create -y -k --prefix "$ENV_DIR" python=3.11
  fi

  # Activate the newly created environment
  echo conda activate "$ENV_DIR"
  conda activate "$ENV_DIR"
  echo "🌕 Step 3: COMPLETE"
}

##############################
## Step 4: Install Dependencies
##############################

install_dependencies() {
  title "Step 4: Install Dependencies"
  echo "Warning: this step may take a while to complete the first time."
  echo "In this step, all Python dependencies for a full ML workspace"
  echo "will be installed in the conda environment."
  echo "🌘 Step 4: START"

  unset_conda_for_sure
  eval "$(${CONDA_BIN} shell.bash hook)"
  conda activate "$ENV_DIR"

  check_python

  # Detect GPU type: NVIDIA vs AMD (ROCm)
  HAS_NVIDIA=false
  HAS_AMD=false

  if command -v nvidia-smi &> /dev/null; then
      echo "nvidia-smi is available"
      GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits) || echo "Issue with NVIDIA SMI"
      if [ -n "$GPU_INFO" ]; then
          echo "NVIDIA GPU detected: $GPU_INFO"
          HAS_NVIDIA=true
      else
          echo "nvidia-smi exists but no NVIDIA GPU found"
      fi
  elif command -v rocminfo &> /dev/null; then
      echo "rocminfo is available"
      HAS_AMD=true
      # AMD_INFO=$(rocm-smi --showproductname | awk '/^GPU/ {print $2}') || echo "Issue with ROCm SMI" 
      # if [ -n "$AMD_INFO" ]; then
      #     echo "AMD GPU detected: $AMD_INFO"
          # HAS_AMD=true
      # else
      #     echo "rocm-smi exists but no AMD GPU found"
      # fi
  fi

  # install uv
  pip install uv
  
  echo "HAS_NVIDIA=$HAS_NVIDIA, HAS_AMD=$HAS_AMD"
  PIP_WHEEL_FLAGS="--upgrade"

  if [ "$HAS_NVIDIA" = true ]; then
      echo "Your computer has a GPU; installing cuda:"
      conda install -y cuda==12.8.1 --force-reinstall -c nvidia/label/cuda-12.8.1

      echo "Installing requirements:"
      # Install the python requirements
      if [ -e "$RUN_DIR/requirements-uv.txt" ]; then
        REQS_PATH="$RUN_DIR/requirements-uv.txt"
      elif [ -e "$TLAB_CODE_DIR/requirements-uv.txt" ]; then
        REQS_PATH="$TLAB_CODE_DIR/requirements-uv.txt"
      else
        echo "Error: requirements-uv.txt not found in run directory or src location."
        exit 1
      fi

      uv pip install ${PIP_WHEEL_FLAGS} -r ${REQS_PATH}

  elif [ "$HAS_AMD" = true ]; then
      echo "Installing requirements for ROCm:"
      if [ -e "$RUN_DIR/requirements-rocm-uv.txt" ]; then
        REQS_PATH="$RUN_DIR/requirements-rocm-uv.txt"
      elif [ -e "$TLAB_CODE_DIR/requirements-rocm-uv.txt" ]; then
        REQS_PATH="$TLAB_CODE_DIR/requirements-rocm-uv.txt"
      else
        echo "Error: requirements-rocm-uv.txt not found in run directory or src location."
        exit 1
      fi

      PIP_WHEEL_FLAGS+=" --index https://download.pytorch.org/whl/rocm6.4"
      uv pip install ${PIP_WHEEL_FLAGS} -r ${REQS_PATH}

      if [ "$TLAB_ON_WSL" = 1 ]; then
        location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
        cd "${location}/torch/lib/" || exit 1
        rm -f libhsa-runtime64.so*
        # cp /opt/rocm/lib/libhsa-runtime64.so.1.14.0 .
        # ln -sf libhsa-runtime64.so.1.14.0 libhsa-runtime64.so.1
        # ln -sf libhsa-runtime64.so.1 libhsa-runtime64.so
      fi

  else
      echo "No NVIDIA GPU detected drivers detected. Install NVIDIA drivers to enable GPU support."
      echo "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions"
      echo "Installing Tranformer Lab requirements without GPU support"

      if [ -e "$RUN_DIR/requirements-no-gpu-uv.txt" ]; then
        REQS_PATH="$RUN_DIR/requirements-no-gpu-uv.txt"
      elif [ -e "$TLAB_CODE_DIR/requirements-no-gpu-uv.txt" ]; then
        REQS_PATH="$TLAB_CODE_DIR/requirements-no-gpu-uv.txt"
      else
        echo "Error: requirements-no-gpu-uv.txt not found in run directory or src location."
        exit 1
      fi

      if [[ -z "${TLAB_ON_MACOS}" ]]; then
          # Add the CPU-specific PyTorch index for non-macOS systems
          PIP_WHEEL_FLAGS+=" --index https://download.pytorch.org/whl/cpu"
      fi

      echo "Using requirements from ${REQS_PATH}"
      # Run the installation with dynamic flags
      uv pip install ${PIP_WHEEL_FLAGS} -r ${REQS_PATH}
  fi

  # Check if the uvicorn command works:
  if ! command -v uvicorn &> /dev/null; then
    abort "❌ Uvicorn is not installed. This usually means that the installation of dependencies failed."
  else
    ohai "✅ Uvicorn is installed."
  fi

  # Record the status after this install for debugging and to check if an install has been attmeped
  PIP_LIST=$(pip list --format json)
  echo "${PIP_LIST}" > "${TLAB_CODE_DIR}/INSTALLED_DEPENDENCIES"
  echo "🌕 Step 4: COMPLETE"
}

list_installed_packages() {
  unset_conda_for_sure
  eval "$(${CONDA_BIN} shell.bash hook)"
  conda activate "${ENV_DIR}"
  pip list --format json
}

list_environments() {
  check_if_conda_environments_dot_text_is_writable
  unset_conda_for_sure
  eval "$(${CONDA_BIN} shell.bash hook)"
  conda env list
}

doctor() {
  title "Doctor"
  ohai "Checking if everything is installed correctly."
  echo "Your machine is: $OS"
  echo "Your shell is: $SHELL"

  
  if command -v "${CONDA_BIN}" &> /dev/null; then
    echo "Your conda version is: $(${CONDA_BIN} --version)" || echo "Issue with conda"
    echo "Conda is seen in path at at: $(which conda)" || echo "Conda is not in your path"
  else
    echo "Conda is not installed at ${MINIFORGE_ROOT}. Please install Conda using '${TLAB_DIR}/src/install.sh install_conda' and try again."
  fi
  if command -v nvidia-smi &> /dev/null; then
    echo "Your nvidia-smi version is: $(nvidia-smi --version)"
  else
    echo "nvidia-smi is not installed."
  fi
  check_conda
  check_python

}

print_success_message() {
  title "Installation Complete"
  echo "------------------------------------------"
  echo "Transformer Lab is installed to:"
  echo "  ${TLAB_DIR}"
  echo "Your workspace is located at:"
  echo "  ${TLAB_DIR}/workspace"
  echo "Your conda environment is at:"
  echo "  ${ENV_DIR}"
  echo "You can run Transformer Lab with:"
  echo "  conda activate ${ENV_DIR}"
  echo "  cd ${TLAB_CODE_DIR}"
  echo "  ./run.sh"
  echo "------------------------------------------"
  echo
}

# Check if there are arguments to this script, and if so, run the appropriate function.
if [[ "$#" -eq 0 ]]; then
  title "Performing a full installation of Transformer Lab."
  download_transformer_lab
  install_conda
  create_conda_environment
  install_dependencies
  print_success_message
else
  for arg in "$@"
  do
    case $arg in
      download_transformer_lab)
        download_transformer_lab
        ;;
      install_conda)
        install_conda
        ;;
      create_conda_environment)
        create_conda_environment
        ;;
      install_dependencies)
        install_dependencies
        ;;
      doctor)
        doctor
        ;;
      list_installed_packages)
        list_installed_packages
        ;;
      list_environments)
        list_environments
        ;;
      *)
        # Print allowed arguments
        echo "Allowed arguments: [download_transformer_lab, install_conda, create_conda_environment, install_dependencies] or leave blank to perform a full installation."
        abort "❌ Unknown argument: $arg"
        ;;
    esac
  done
fi
