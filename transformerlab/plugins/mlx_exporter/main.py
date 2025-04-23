# This plugin exports a model to MLX format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess
import argparse


try:
    from transformerlab.plugin import get_python_executable
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable


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

env = os.environ.copy()
env["PATH"] = python_executable.replace("/python", ":") + env["PATH"]

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
    env=env,
)
