# This plugin exports a model to MLX format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess

from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable
from transformerlab.plugin_sdk.transformerlab.sdk.v1.export import tlab_exporter   


tlab_exporter.add_argument("--q_bits", default="4", type=str, help="Bits per weight for quantization.")


@tlab_exporter.exporter_job_wrapper(progress_start=0, progress_end=100)
def mlx_export():
    """Export a model to MLX format"""

    # Directory to run conversion subprocess
    plugin_dir = os.path.realpath(os.path.dirname(__file__))
    python_executable = get_python_executable(plugin_dir)

    env = os.environ.copy()
    env["PATH"] = python_executable.replace("/python", ":") + env["PATH"]

    tlab_exporter.progress_update(10)
    tlab_exporter.add_job_data("status", "Starting MLX conversion")

    # Call MLX Convert function
    export_process = subprocess.run(
        [
            python_executable,
            "-u",
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            tlab_exporter.params.get("model_name"),
            "--mlx-path",
            tlab_exporter.params.get("output_dir"),
            "-q",
            "--q-bits",
            str(tlab_exporter.params.get("q_bits")),
        ],
        cwd=plugin_dir,
        capture_output=True,
        env=env,
    )

    # Add output to job data
    stdout = export_process.stdout.decode("utf-8")
    stderr = export_process.stderr.decode("utf-8")
    tlab_exporter.add_job_data("stdout", stdout)
    tlab_exporter.add_job_data("stderr", stderr)

    # Final progress update
    tlab_exporter.progress_update(100)
    tlab_exporter.add_job_data("status", "MLX conversion complete")

    return "Successful export to MLX format"


mlx_export()
