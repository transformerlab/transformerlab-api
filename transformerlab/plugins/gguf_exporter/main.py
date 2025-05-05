# This plugin exports a model to GGUF format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess
import argparse

from huggingface_hub import snapshot_download

try:
    from transformerlab.plugin import get_python_executable
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable


# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description="Convert a model to GGUF format.")
parser.add_argument("--output_dir", type=str, help="Directory to save the model in.")
parser.add_argument("--output_model_id", type=str, help="Directory to save the model in.")
parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Name of model to export.")
parser.add_argument("--exported_model_name", default="my-exported-model", help="Name of the exported model folder.")
parser.add_argument(
    "--outtype", default="q8_0", type=str, help="GGUF output format. q8_0 quantizes the model to 8 bits."
)
args, unknown = parser.parse_known_args()

# input arguments
input_model = args.model_name
outtype = args.outtype

output_path = os.path.expanduser(f"~/.transformerlab/workspace/models/{args.exported_model_name}")

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))
python_executable = get_python_executable(plugin_dir)

# The model _should_ be available locally
# but call hugging_face anyways so we get the proper path to it
model_path = input_model
if not os.path.exists(model_path):
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

# TODO: This default quantizes to 8-bit. Need to read that in as a parameter.
subprocess_cmd = [
    python_executable,
    os.path.join(plugin_dir, "llama.cpp", "convert_hf_to_gguf.py"),
    "--outfile",
    output_path,
    "--outtype",
    outtype,
    model_path,
]

# Call GGUF Convert function
export_process = subprocess.run(
    subprocess_cmd, cwd=plugin_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
print(export_process.stdout)
