# This plugin exports a model to GGUF format so you can interact and train on a MBP with Apple Silicon
import os
import subprocess
import argparse

# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description='Convert a model to GGUF format.')
parser.add_argument('--output_dir', type=str, help='Directory to save the model in.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str, help='Name of model to export.')
parser.add_argument('--quant_bits', default='4', type=str, help='Bits per weight for quantization.')
args, unknown = parser.parse_known_args()

output_path = args.output_dir

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))

# Call GGUF Convert function
# TODO: This is a placeholder for now! Sleep 10 seconds to test app functionality
export_process = subprocess.run(
#    ["python", '-u', '-m',  _TODO_conversion_class, '--hf-path', args.model_name, '--output-path', output_path, '-q', '--q-bits', str(args.quant_bits)],
    ["sleep", '5'],
    cwd=plugin_dir,
    capture_output=True
)