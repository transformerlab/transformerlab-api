#!/bin/bash
# -e: Exit immediately on error. -u: treat unset variables as an error and exit immediately.
set -eu

# We are deprecating this file. For now it just calles download_and_install_remote_script.sh

err_report() {
  echo "Error on line $1"
}

trap 'err_report $LINENO' ERR

# Need the directory of this script to reference other files
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

$SCRIPT_DIR/download_and_install_remote_script.sh install_dependencies