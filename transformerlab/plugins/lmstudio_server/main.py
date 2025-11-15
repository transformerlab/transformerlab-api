"""
LM Studio Server

This plugin integrates LM Studio models into TransformerLab, allowing users to utilize models stored in the LM Studio format.
"""

import argparse
import os
import subprocess
import json
import uuid
from hashlib import sha256
from pathlib import Path
import sys
import lmstudio
import time
import requests

worker_id = str(uuid.uuid4())[:8]

LMSTUDIO_STARTUP_TIMEOUT = 180  # seconds

try:
    from transformerlab.plugin import register_process
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import register_process

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
parser.add_argument("--parameters", type=str, default="{}")

args, unknown = parser.parse_known_args()
# model_path can be a hugging face ID or a local file in Transformer Lab
# But LM Studio models are always stored as a local path because
# we are using a specific LM Studio model file
if os.path.exists(args.model_path):
    model_path = args.model_path
else:
    raise FileNotFoundError(
        f"The specified LM Studio model '{args.model_path}' was not found. Please select a valid LM Studio model file to proceed."
    )

llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")

parameters = args.parameters
parameters = json.loads(parameters)
# Now go through the parameters object and remove the key that is equal to "inferenceEngine":
if "inferenceEngine" in parameters:
    del parameters["inferenceEngine"]

if "inferenceEngineFriendlyName" in parameters:
    del parameters["inferenceEngineFriendlyName"]

# Get plugin directory
real_plugin_dir = os.path.realpath(os.path.dirname(__file__))

port = int(parameters.get("port", 1234))
env = os.environ.copy()
env["LMSTUDIO_HOST"] = f"127.0.0.1:{port}"
print("Starting LM Studio server...", file=sys.stderr)

process = subprocess.Popen(["lms", "server", "start", f"--port {port}"], env=env)

lmstudio_models_url = f"http://127.0.0.1:{port}/v1/models"
start_time = time.time()
while True:
    try:
        response = requests.get(lmstudio_models_url)
        if response.status_code == 200:
            print("LM Studio server is up and running.", file=sys.stderr)
            break
    except requests.ConnectionError:
        pass
    if time.time() - start_time > LMSTUDIO_STARTUP_TIMEOUT:
        print("Timeout waiting for LM Studio server to start.", file=sys.stderr)
        process.terminate()
        sys.exit(1)
    time.sleep(1)

register_process(process.pid)
