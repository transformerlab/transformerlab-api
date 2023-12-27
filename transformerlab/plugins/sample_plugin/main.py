# This is a sample plugin that just displays the parameters that are provided to it
# It is used to test the plugin system
import os
import sqlite3
import argparse
import sys
import json

# Connect to the LLM Lab database (you can use this to update job status in the jobs table)
# llmlab_root_dir = os.getenv('LLM_LAB_ROOT_PATH')
# db = sqlite3.connect(llmlab_root_dir + "/workspace/llmlab.sqlite3")


if __name__ == "__main__":
    # Get all arguments provided to this script using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--param1', type=str, help='First parameter')
    parser.add_argument('--param2', type=str, help='Second parameter')
    # allow other arguments:
    parser.add_argument('--other_args', nargs=argparse.REMAINDER)
    args, unknown = parser.parse_known_args()

    print("Sample plugin called with parameters:")
    print("param1: " + args.param1)
    print("param2: " + args.param2)

    # treat stdin as one json string:
    experiment_info = json.loads(sys.stdin.read())
    print(experiment_info)
