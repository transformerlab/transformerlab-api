# This plugin exports a model to MLX format so you can interact and train on a MBP with Apple Silicon
import os
import sqlite3
import argparse
import sys

# Connect to the LLM Lab database (to update job status)
# llmlab_root_dir = os.getenv('LLM_LAB_ROOT_PATH')
# db = sqlite3.connect(llmlab_root_dir + "/workspace/llmlab.sqlite3")

# Get all arguments provided to this script using argparse
# NOTE: This takes an adaptor name but doesn't do anything with the adaptor at this time
parser = argparse.ArgumentParser(description='Convert a model to MLX format.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str, help='Name of model to export.')
parser.add_argument('--model_architecture', default='hf-causal', type=str, help='Type of model to export.')
parser.add_argument('--model_adaptor', default='', type=str, help='Name of model adaptor.')
args, unknown = parser.parse_known_args()

# TODO: Verify that the model uses a supported format
model_architecture = args.model_architecture
print("Model architecture: ", model_architecture)

# TODO: Call the MLX convert function
print("Exporting model", args.model_name, "and adaptor", args.model_adaptor, "to MLX")

