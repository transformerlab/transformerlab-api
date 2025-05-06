import os
import argparse
import subprocess
import shutil


# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser(description="Create a fully contained self-executing llamafile.")
parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Name of model to create llamafile from.")
parser.add_argument(
    "--model_path", default="gpt-j-6b", type=str, help="Path to directory or file containing the model."
)
parser.add_argument("--output_dir", type=str, help="Directory to save the model in.")
parser.add_argument("--output_model_id", type=str, help="ID of outputted llamafile.")
args, unknown = parser.parse_known_args()

# Get model details from arguments
input_model = args.model_name
input_model_path = args.model_path
output_model_id = args.output_model_id
output_dir = args.output_dir

# Directory to run conversion subprocess
plugin_dir = os.path.realpath(os.path.dirname(__file__))

# From export.py we know that output_model_id is the base name without extension
# Create the filename with .llamafile extension
model_name = output_model_id
output_filename = f"{model_name}.llamafile"

# Setup arguments for executing this model
argsfile = os.path.join(plugin_dir, ".args")
argsoutput = f"""-m
{model_name}
--host
0.0.0.0
-ngl
9999
...
"""

# Create a .args file to include in the llamafile
with open(argsfile, "w") as f:
    f.write(argsoutput)

# Create a copy of pre-built llamafile to use as a base
shutil.copy(os.path.join(plugin_dir, "llamafile"), os.path.join(plugin_dir, output_filename))

# Merge files together in single executable using zipalign
subprocess_cmd = ["sh", "./zipalign", "-j0", output_filename, input_model_path, ".args"]
export_process = subprocess.run(
    subprocess_cmd, cwd=plugin_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
print(export_process.stdout)

# Move file to output_dir - the directory already exists (created by export.py)
# The info.json file will be created by export.py after this script runs
shutil.move(os.path.join(plugin_dir, output_filename), os.path.join(output_dir, output_filename))