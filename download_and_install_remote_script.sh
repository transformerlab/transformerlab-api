#!/bin/bash
set -e

VERSION="0.1.4"
ENV_NAME="transformerlab"

TFL_DIR="$HOME/.transformerlab"
TFL_CODE_DIR="${TFL_DIR}/src"
# TFL_URL="https://github.com/transformerlab/transformerlab-api/archive/refs/heads/main.zip"

LATEST_RELEASE_VERSION=$(curl -Ls -o /dev/null -w %{url_effective} https://github.com/transformerlab/transformerlab-api/releases/latest)
LATEST_RELEASE_VERSION=$(basename $LATEST_RELEASE_VERSION)
LATEST_RELEASE_VERSION_WITHOUT_V=$(echo $LATEST_RELEASE_VERSION | sed 's/v//g')
echo "Latest Release on Github: $LATEST_RELEASE_VERSION"
echo "Latest Release without V: $LATEST_RELEASE_VERSION_WITHOUT_V"
TFL_URL="https://github.com/transformerlab/transformerlab-api/archive/refs/tags/${LATEST_RELEASE_VERSION}.zip"
echo "Download Location: $TFL_URL"

# This script is meant to be run  on a new computer. 
# It will pull down the API and install
# it at ~/.transfomerlab/src

abort() {
  printf "%s\n" "$@" >&2
  exit 1
}

# string formatters
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

ohai() {
  printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$(shell_join "$@")"
}

warn() {
  printf "${tty_red}Warning${tty_reset}: %s\n" "$(chomp "$1")" >&2
}

# First check OS.
OS="$(uname)"
if [[ "${OS}" == "Linux" ]]
then
  TFL_ON_LINUX=1
elif [[ "${OS}" == "Darwin" ]]
then
  TFL_ON_MACOS=1
else
  abort "Transformer Lab is only supported on macOS and Linux."
fi


# If the user has not installed Transformer Lab, then we should install it.
ohai "Installing Transformer Lab ${LATEST_RELEASE_VERSION}..."
# Fetch the latest version of Transformer Lab from GitHub:
mkdir -p "${TFL_DIR}"
curl -L "${TFL_URL}" -o "${TFL_DIR}/transformerlab.zip"
NEW_DIRECTORY_NAME="transformerlab-api-${LATEST_RELEASE_VERSION_WITHOUT_V}"
rm -rf "${TFL_DIR}/${NEW_DIRECTORY_NAME}"
rm -rf "${TFL_CODE_DIR}"
unzip -o "${TFL_DIR}/transformerlab.zip" -d "${TFL_DIR}"
mv "${TFL_DIR}/${NEW_DIRECTORY_NAME}" "${TFL_CODE_DIR}"
rm "${TFL_DIR}/transformerlab.zip"
# Create a file called LATEST_VERSION that contains the latest version of Transformer Lab.
echo "${LATEST_RELEASE_VERSION}" > "${TFL_CODE_DIR}/LATEST_VERSION"

# Now time to install dependencies and requirements.txt by running
# the init.sh script.
INIT_SCRIPT="${TFL_CODE_DIR}/init.sh"

# check if conda environment already exists:
if ! command -v conda &> /dev/null; then
  echo "Conda is not installed."
  source "$INIT_SCRIPT"
else
  if { conda env list | grep "$ENV_NAME"; } >/dev/null 2>&1; then
    if [[ "$1" == "--force" ]]
    then
      ohai "Forcing conda dependencies to be reinstalled."
      source "$INIT_SCRIPT"
    else
      ohai "Conda dependencies look like they've been installed already."
    fi
  else
    ohai "Installing conda and conda dependencies..."
    source "$INIT_SCRIPT"
  fi
fi


ohai "Installation successful!"
echo "------------------------------------------"
echo "Transformer Lab is installed to:"
echo "  ${TFL_DIR}"
echo "Your workspace is located at:"
echo "  ${TFL_DIR}/workspace"
echo "You can run Transformer Lab with:"
echo "  conda activate ${ENV_NAME}"
echo "  ${TFL_DIR}/src/run.sh"
echo "------------------------------------------"
echo