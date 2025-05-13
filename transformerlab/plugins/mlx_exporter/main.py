# This plugin exports a model to MLX format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess
<<<<<<< Updated upstream
import argparse
=======
import sys
import time
import threading
import select
import fcntl
import argparse
import json

try:
    from transformerlab.sdk.v1.export import tlab_exporter
    from transformerlab.plugin import get_python_executable, get_db_connection, Job
except ImportError or ModuleNotFoundError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable, get_db_connection, Job
    from transformerlab.plugin_sdk.transformerlab.sdk.v1.export import tlab_exporter
>>>>>>> Stashed changes


try:
    from transformerlab.plugin import get_python_executable
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable


<<<<<<< Updated upstream
# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description="Convert a model to MLX format.")
parser.add_argument("--output_dir", type=str, help="Directory to save the model in.")
parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Name of model to export.")
parser.add_argument("--q_bits", default="4", type=str, help="Bits per weight for quantization.")
args, unknown = parser.parse_known_args()

output_path = args.output_dir
=======
@tlab_exporter.exporter_job_wrapper(progress_start=0, progress_end=100)
def mlx_export():
    plugin_dir = os.path.realpath(os.path.dirname(__file__))
    python_executable = get_python_executable(plugin_dir)
>>>>>>> Stashed changes

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))
python_executable = get_python_executable(plugin_dir)

<<<<<<< Updated upstream
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
=======
    command = [
        python_executable, "-u", "-m", "mlx_lm", "convert",
        "--hf-path", tlab_exporter.params.get("model_name"),
        "--mlx-path", tlab_exporter.params.get("output_dir"),
        "-q", "--q-bits", str(tlab_exporter.params.get("q_bits")),
    ]

    print("Starting MLX conversion...")
    print(f"Running command: {' '.join(command)}")
    tlab_exporter.add_job_data("command", " ".join(command))
    
    tlab_exporter.progress_update(5)
    tlab_exporter.add_job_data("status", "Starting MLX conversion")

    try:
        with subprocess.Popen(
            command,
            cwd=plugin_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            universal_newlines=True,
            bufsize=1
        ) as process:

            all_output_lines = []
            progress_value = 5

            for line in process.stdout:
                line = line.strip()
                all_output_lines.append(line)
                print(line, flush=True)
                
                if "Loading" in line:
                    progress_value = 15
                    tlab_exporter.add_job_data("status", "Loading model")
                elif "Fetching" in line:
                    progress_value = 35
                    tlab_exporter.add_job_data("status", "Fetching model files")
                elif "Using dtype" in line:
                    progress_value = 50
                    tlab_exporter.add_job_data("status", "Preparing quantization")
                elif "Quantizing" in line:
                    progress_value = 65
                    tlab_exporter.add_job_data("status", "Quantizing model")
                elif "Quantized model" in line:
                    progress_value = 80
                    tlab_exporter.add_job_data("status", "Finalizing model")

                tlab_exporter.progress_update(progress_value)

            return_code = process.wait()
            tlab_exporter.add_job_data("stdout", "\n".join(all_output_lines))

            if return_code != 0:
                error_msg = f"MLX conversion failed with return code {return_code}"
                print(error_msg)
                tlab_exporter.add_job_data("status", error_msg)
                raise RuntimeError(error_msg)

    except Exception as e:
        error_msg = f"MLX conversion failed with exception: {str(e)}"
        print(error_msg)
        tlab_exporter.add_job_data("status", error_msg)
        raise

    print("MLX conversion completed successfully!")
    tlab_exporter.add_job_data("status", "MLX conversion complete")
    tlab_exporter.progress_update(100)

    return "Successful export to MLX format"

mlx_export()
>>>>>>> Stashed changes
