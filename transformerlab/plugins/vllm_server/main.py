import argparse
import json
import os
import subprocess
import sys
import time
import torch
import threading

from fastchat.constants import TEMP_IMAGE_DIR


try:
    from transformerlab.plugin import get_python_executable, register_process
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable, register_process


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
print("Starting vLLM server...", file=sys.stderr)

os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

vllm_args = [
    python_executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    model,
    "--dtype",
    "float16",
    "--port",
    str(port),
    "--gpu-memory-utilization",
    "0.9",
    "--trust-remote-code",
    "--enforce-eager",
    "--allowed-local-media-path",
    str(TEMP_IMAGE_DIR),
]

# Add tensor parallel size if multiple GPUs are available
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    vllm_args.extend(["--tensor-parallel-size", str(num_gpus)])

# We need to read both STDOUT (to determine when the server is up)
# and STDOUT (to report on errors)
vllm_proc = subprocess.Popen(vllm_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# Wait for vLLM server to be ready
# This is a magic string we watch for to know the vllm server has started
watch_string = "init engine"
start_success = False

# Read over stderr and print out any error output
# Break as soon as we detect the server is up (based on watch string)
for line in iter(vllm_proc.stdout.readline, b""):
    decoded = line.decode()
    if watch_string in decoded:
        print("vLLM server started successfully")
        start_success = True
        break

    error_msg = decoded.strip()
    print("[vLLM]", error_msg, file=sys.stderr)

# If we didn't detect the startup string then report the error and exit
if not start_success:
    vllm_proc.wait()
    print("vLLM Startup Failed with exit code", vllm_proc.returncode)
    print(error_msg)
    sys.exit(1)

proxy_args = [
    python_executable,
    "-m",
    "fastchat.serve.openai_api_proxy_worker",
    "--model-path",
    model,
    "--proxy-url",
    f"http://localhost:{port}/v1",
    "--model",
    model,
]

proxy_proc = subprocess.Popen(proxy_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# save both worker process id and vllm process id to file
# this will allow transformer lab to kill both later
register_process([proxy_proc.pid, vllm_proc.pid])


# Read output from both processes (vLLM and proxy) simultaneously
def read_stream(proc, prefix):
    """Read from a process stdout (which includes stderr) and print with prefix"""
    for line in iter(proc.stdout.readline, b""):
        if line:
            print(f"[{prefix}]", line.decode().strip(), file=sys.stderr)


# Create threads to read from both processes
vllm_thread = threading.Thread(target=read_stream, args=(vllm_proc, "vLLM"), daemon=True)
proxy_thread = threading.Thread(target=read_stream, args=(proxy_proc, "Proxy"), daemon=True)

vllm_thread.start()
proxy_thread.start()

# Wait for either process to exit
while vllm_proc.poll() is None and proxy_proc.poll() is None:
    time.sleep(1)

# If one exits, report which one
if vllm_proc.poll() is not None:
    print(f"vLLM process exited with code {vllm_proc.poll()}", file=sys.stderr)
if proxy_proc.poll() is not None:
    print(f"Proxy process exited with code {proxy_proc.poll()}", file=sys.stderr)

print("Vllm worker exited", file=sys.stderr)
