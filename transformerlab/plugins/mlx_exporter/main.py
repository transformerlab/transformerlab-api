# This plugin exports a model to MLX format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess
import argparse
import sys


def get_python_executable(plugin_dir):
    """Check if a virtual environment exists and return the appropriate Python executable"""
    # Check for virtual environment in the plugin directory
    venv_path = os.path.join(plugin_dir, "venv")

    if os.path.isdir(venv_path):
        print("Virtual environment found, using it for evaluation...")
        # Determine the correct path to the Python executable based on the platform
        python_executable = os.path.join(venv_path, "bin", "python")

        if os.path.exists(python_executable):
            return python_executable

    # Fall back to system Python if venv not found or executable doesn't exist
    print("No virtual environment found, using system Python...")
    return sys.executable


# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description="Convert a model to MLX format.")
parser.add_argument("--output_dir", type=str, help="Directory to save the model in.")
parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Name of model to export.")
parser.add_argument("--q_bits", default="4", type=str, help="Bits per weight for quantization.")
args, unknown = parser.parse_known_args()

output_path = args.output_dir

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))
python_executable = get_python_executable(plugin_dir)

# Call MLX Convert function
export_process = subprocess.run(
    [
        python_executable,
        "-u",
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        args.model_name,
        "--mlx-path",
        output_path,
        "-q",
        "--q-bits",
        str(args.q_bits),
    ],
    cwd=plugin_dir,
    capture_output=True,
)
