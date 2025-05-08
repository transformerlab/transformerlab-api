# This plugin exports a model to GGUF format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess

from huggingface_hub import snapshot_download

try:
    from transformerlab.sdk.v1.export import tlab_exporter
    from transformerlab.plugin import get_python_executable
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable
    from transformerlab.plugin_sdk.transformerlab.sdk.v1.export import tlab_exporter


tlab_exporter.add_argument("--output_model_id", type=str, help="Directory to save the model in.")
tlab_exporter.add_argument(
    "--outtype", default="q8_0", type=str, help="GGUF output format. q8_0 quantizes the model to 8 bits."
)


@tlab_exporter.exporter_job_wrapper(progress_start=0, progress_end=100)
def gguf_export():
    """Export a model to GGUF format"""

    input_model = tlab_exporter.params.get("model_name")
    outtype = tlab_exporter.params.get("outtype")

    # For internals to work we need the directory and output filename to be the same
    output_filename = tlab_exporter.params.get("output_model_id")
    output_path = os.path.join(tlab_exporter.params.get("output_dir"), output_filename)

    # Directory to run conversion subprocess
    plugin_dir = os.path.realpath(os.path.dirname(__file__))
    python_executable = get_python_executable(plugin_dir)

    tlab_exporter.progress_update(10)
    tlab_exporter.add_job_data("status", "Starting GGUF conversion")

    # The model _should_ be available locally
    # but call hugging_face anyways so we get the proper path to it
    model_path = input_model
    if not os.path.exists(model_path):
        tlab_exporter.add_job_data("status", "Downloading model from Hugging Face")
        tlab_exporter.progress_update(20)
        model_path = snapshot_download(
            repo_id=input_model,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.py",
                "tokenizer.model",
                "*.tiktoken",
            ],
        )

    tlab_exporter.progress_update(30)
    tlab_exporter.add_job_data("status", "Quantizing model to 8-bit format")

    subprocess_cmd = [
        python_executable,
        os.path.join(plugin_dir, "llama.cpp", "convert_hf_to_gguf.py"),
        "--outfile",
        output_path,
        "--outtype",
        outtype,
        model_path,
    ]

    tlab_exporter.progress_update(60)
    tlab_exporter.add_job_data("status", "Converting model to GGUF format")

    # Call GGUF Convert function
    export_process = subprocess.run(
        subprocess_cmd, cwd=plugin_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Add output to job data
    stdout = export_process.stdout
    stderr = export_process.stderr
    tlab_exporter.add_job_data("stdout", stdout)
    tlab_exporter.add_job_data("stderr", stderr)

    # Final progress update
    tlab_exporter.progress_update(100)
    tlab_exporter.add_job_data("status", "GGUF conversion complete")

    return "Successful export to GGUF format"


gguf_export()
