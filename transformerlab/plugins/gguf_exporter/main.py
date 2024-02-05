# This plugin exports a model to GGUF format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess
import sys
import argparse

from huggingface_hub import snapshot_download

# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description='Convert a model to GGUF format.')
parser.add_argument('--output_dir', type=str, help='Directory to save the model in.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str, help='Name of model to export.')
args, unknown = parser.parse_known_args()

# input arguments
input_model = args.model_name
output_filename = f"{input_model.split('/')[-1]}.gguf"
output_path = os.path.join(args.output_dir, output_filename) 

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))

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
    sys.executable,
    os.path.join(plugin_dir, 'llama.cpp', 'convert.py'),
    '--outfile', output_path,
    '--outtype', 'q8_0',
    model_path
]

# Call GGUF Convert function
export_process = subprocess.run(
    subprocess_cmd,
    cwd=plugin_dir,
    capture_output=True
)