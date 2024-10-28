import os
import argparse
import subprocess
import shutil

from huggingface_hub import snapshot_download


# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description='Create a fully contained self-executing llamafile.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str, help='Name of model to create llamafile from.')
parser.add_argument('--output_dir', type=str, help='Directory to save the model in.')

# TODO: The app generates an ID but it's not consistent with how llamafiles are normally named
#parser.add_argument('--output_model_id', type=str, help='ID of outputted llamafile.')
args, unknown = parser.parse_known_args()

# Figure out where the actual input model file is
# The model _should_ be available locally
# And we download GGUF files to the transformerlab models dir
# TODO: This is not working!
input_model = args.model_name
input_model_path = input_model
print("Input model path:", input_model_path)

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))

# Output details
outfile = f"{input_model}.llamafile"
output_dir = args.output_dir

# Setup arguments for executing this model
argsfile = os.path.join(plugin_dir, ".args")
argsoutput = f"""-m
{input_model}
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
    capture_output=True
)

# Move file to output_dir
shutil.move(os.path.join(plugin_dir, outfile), os.path.join(output_dir, outfile))
