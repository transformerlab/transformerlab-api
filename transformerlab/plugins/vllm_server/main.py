import argparse
import json
import os
import subprocess
import sys
import time
import requests
import gc
import torch
from pathlib import Path



try:
    from transformerlab.plugin import get_python_executable, register_process
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable, register_process


# Clear CUDA memory (if CUDA is available)
def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        print(">>> [main] Emptying CUDA memory cache and collecting garbage...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
clear_vram()
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
parser.add_argument("--parameters", type=str, default="{}")
args, unknown = parser.parse_known_args()

model = args.model_path

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

# Get Python executable (from venv if available)
python_executable = get_python_executable(real_plugin_dir)

port = int(parameters.get("port", 8000))
# host = "127.0.0.1"
print("Starting vLLM server...", file=sys.stderr)

workspace = os.environ["_TFL_WORKSPACE_DIR"]
VLLM_TEMP_IMG_DIR=Path(f"{workspace}/plugins/vllm_server/tmp_img")

vllm_args = [
    python_executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model", model,
    "--dtype", "float16",
    "--port", str(port),
    "--gpu-memory-utilization", "0.9",
    "--trust-remote-code",
    "--enforce-eager",
    "--allowed-local-media-path", str(VLLM_TEMP_IMG_DIR)
]

# Add tensor parallel size if multiple GPUs are available
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    vllm_args.extend(["--tensor-parallel-size", str(num_gpus)])

vllm_proc = subprocess.Popen(vllm_args, stdout=None, stderr=subprocess.PIPE)

# Wait for vLLM server to be ready
vllm_url = f"http://localhost:{port}/v1/models"
timeout = 300  # seconds
start_time = time.time()
while True:
    try:
        resp = requests.get(vllm_url, timeout=3)
        if resp.status_code == 200:
            print("vLLM server is ready", file=sys.stderr)
            break
    except Exception:
        pass
    if time.time() - start_time > timeout:
        print("ERROR: Timeout waiting for vLLM server to be ready", file=sys.stderr)
        sys.exit(1)
    time.sleep(1)

proxy_args = [
    python_executable, 
    "-m", 
    "fastchat.serve.openai_api_proxy_worker",
    "--model-path", model,
    "--proxy-url", f"http://localhost:{port}/v1",
   "--model", model,
    ]

# print("Starting FastChat OpenAI API Proxy worker...", file=sys.stderr)
proxy_proc = subprocess.Popen(proxy_args, stdout=None, stderr=subprocess.PIPE)

# save both worker process id and vllm process id to file
# this will allow transformer lab to kill both later
register_process([proxy_proc.pid, vllm_proc.pid])

# read output:
for line in iter(proxy_proc.stderr.readline, b""):
    print(line, file=sys.stderr)

print("Vllm worker exited", file=sys.stderr)
clear_vram()
sys.exit(1)  # 99 is our code for CUDA OOM