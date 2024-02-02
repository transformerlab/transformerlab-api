# This plugin exports a model to GGUF format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess
import sqlite3
import argparse
import sys
import json
import time

# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description='Convert a model to GGUF format.')
parser.add_argument('--output_dir', type=str, help='Directory to save the model in.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str, help='Name of model to export.')
parser.add_argument('--model_architecture', default='hf-causal', type=str, help='Type of model to export.')
parser.add_argument('--output_model_id', default='New Model', type=str, help='Exported ID of the model is also the name of the output directory.')
parser.add_argument('--quant_bits', default='4', type=str, help='Bits per weight for quantization.')
args, unknown = parser.parse_known_args()

output_model_id = args.output_model_id
output_path = args.output_dir

# TODO: Verify that the model uses a supported format (see main.json for list)
model_architecture = args.model_architecture

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))

# Full directory to output quantized model
root_dir = os.environ.get("LLM_LAB_ROOT_PATH")

# Call GGUF Convert function
# TODO: This is a placeholder for now! Sleep 10 seconds to test app functionality
print("Exporting", args.model_name, "to GGUF format in", output_path)
export_process = subprocess.run(
#    ["python", '-u', '-m',  _TODO_conversion_class, '--hf-path', args.model_name, '--output-path', output_path, '-q', '--q-bits', str(args.quant_bits)],
    ["sleep", '5'],
    cwd=plugin_dir,
    capture_output=True
)

if (export_process.returncode == 0):

    print("Export to GGUF completed successfully")

else:
    print("Export to GGUF failed. Return code:", export_process.returncode)
