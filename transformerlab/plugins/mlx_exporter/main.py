# This plugin exports a model to MLX format so you can interact and train on a MBP with Apple Silicon
import os
import sqlite3
import argparse
import sys
import json

# Connect to the LLM Lab database (to update job status)
# llmlab_root_dir = os.getenv('LLM_LAB_ROOT_PATH')
# db = sqlite3.connect(llmlab_root_dir + "/workspace/llmlab.sqlite3")


if __name__ == "__main__":
    # Get all arguments provided to this script using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args, unknown = parser.parse_known_args()

    with open(args.input_file) as json_file:
        input = json.load(json_file)

    # Testing: Verify input
    print("Input to Script:")
    print(input)
