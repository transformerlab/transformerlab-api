# This plugin exports a model to GGUF format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess
import sys
import argparse

# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description='Convert a model to GGUF format.')
parser.add_argument('--output_dir', type=str, help='Directory to save the model in.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str, help='Name of model to export.')
args, unknown = parser.parse_known_args()

# input arguments
input_model_id = args.model_name
output_filename = f"{input_model_id.split('/')[-1]}.gguf"
output_path = os.path.join(args.output_dir, output_filename) 

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))

subprocess_cmd = [
    sys.executable,
    os.path.join(plugin_dir, 'llama.cpp', 'convert.py'),
    '--outfile', output_path,
    '--outtype', 'q4_0',
    input_model_id
]

# Call GGUF Convert function
# TODO: This default quantizes to 4-bit. Need to read that in as a parameter.
export_process = subprocess.run(
    subprocess_cmd,
    cwd=plugin_dir,
    capture_output=True
)