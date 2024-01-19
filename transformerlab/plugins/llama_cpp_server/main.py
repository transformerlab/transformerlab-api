import os
import sqlite3
import argparse
import sys
import json
from llama_cpp import server
import uvicorn
from llama_cpp.server.app import create_app, Settings


# Connect to the LLM Lab database (you can use this to update job status in the jobs table)
# llmlab_root_dir = os.getenv('LLM_LAB_ROOT_PATH')
# db = sqlite3.connect(llmlab_root_dir + "/workspace/llmlab.sqlite3")


if __name__ == "__main__":
    # Get all arguments provided to this script using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    args, unknown = parser.parse_known_args()

    print("Running llama-cpp-server", file=sys.stderr)

    model = args.model_path

    # Below we set n_gpu_layers=1 which works for Mac MPS but we should dynamically set
    # the options based on the GPU available
    # users can edit the following line in the plugin editor themselves for now

    app = create_app(Settings(
        model=f"workspace/models/{model}/{model}", n_gpu_layers=1), )
    uvicorn.run(app, host="0.0.0.0", port=8001)
