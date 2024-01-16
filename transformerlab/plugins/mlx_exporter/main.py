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

    # Testing: Verify input
    model_name = "model"
    model_architecture = "architecture"
    adaptor_name = "adaptor"

    print("Exporting model", model_name, "and adaptor", adaptor_name, "to MLX")
