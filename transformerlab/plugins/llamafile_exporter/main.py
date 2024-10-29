import os
import argparse
import subprocess
import shutil

from huggingface_hub import snapshot_download


# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description='Create a fully contained self-executing llamafile.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str, help='Name of model to create llamafile from.')
parser.add_argument('--model_path', default='gpt-j-6b', type=str, help='Path to directory or file containing the model.')
parser.add_argument('--output_dir', type=str, help='Directory to save the model in.')

# TODO: The app generates an ID but it's not consistent with how llamafiles are normally named
#parser.add_argument('--output_model_id', type=str, help='ID of outputted llamafile.')
args, unknown = parser.parse_known_args()

# We need to pass the model ID in the .args file
# Make sure we remove the author part (everything before and including "/")
input_model = args.model_name
input_model_id_without_author = input_model.split("/")[-1]

# But we need the actual model path to get the GGUF file
input_model_path = args.model_path

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))

# Output details - ignoring the output_model_id passed by app
outfile = f"{input_model_id_without_author}.llamafile"
output_dir = args.output_dir

# Setup arguments for executing this model
argsfile = os.path.join(plugin_dir, ".args")
argsoutput = f"""-m
{input_model_id_without_author}
--host
0.0.0.0
-ngl
9999
...
"""

# Create a .args file to include in the llamafile
with open(argsfile, 'w') as f:
    f.write(argsoutput)

# Create a copy of pre-built llamafile to use as a base 
shutil.copy(os.path.join(plugin_dir, "llamafile"), os.path.join(plugin_dir, outfile))

# Merge files together in single executable using zipalign
subprocess_cmd = [
    "sh", "./zipalign", "-j0",
    outfile,
    input_model_path,
    ".args"
]
export_process = subprocess.run(
    subprocess_cmd,
    cwd=plugin_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)
print(export_process.stdout)

# Move file to output_dir
shutil.move(os.path.join(plugin_dir, outfile), os.path.join(output_dir, outfile))
