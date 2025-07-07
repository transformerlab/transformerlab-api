import argparse
import json
import os
import subprocess
import sys
import time
import requests


try:
    from transformerlab.plugin import get_python_executable
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
parser.add_argument("--parameters", type=str, default="{}")
args, unknown = parser.parse_known_args()

model = args.model_path

llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")

parameters = args.parameters
parameters = json.loads(parameters)

print("Starting OpenAI API Proxy Server", file=sys.stderr)

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

vllm_args = [
    python_executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model", model,
    "--dtype", "float16",
    "--port", str(port),
    "--max-model-len", str(parameters.get("max_model_len", 2048)),
    "--gpu-memory-utilization", "0.9",
    "--enforce-eager",
    "--trust-remote-code",
]
print("Starting vLLM OpenAI API server...", file=sys.stderr)
vllm_proc = subprocess.Popen(vllm_args, stdout=None, stderr=subprocess.PIPE)

# Wait for vLLM server to be ready
vllm_url = f"http://localhost:{port}/v1/models"
timeout = 180  # seconds
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
        print("Timeout waiting for vLLM server to be ready", file=sys.stderr)
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
with open(f"{llmlab_root_dir}/worker.pid", "w") as f:
    f.write(f"{proxy_proc.pid}\n")
    f.write(f"{vllm_proc.pid}\n")

# read output:
for line in iter(proxy_proc.stderr.readline, b""):
    print(line, file=sys.stderr)

print("Vllm proxy worker exited", file=sys.stderr)
sys.exit(1)  # 99 is our code for CUDA OOM