import argparse
import json
import os
import asyncio
import sys
import time
import requests
import gc
import torch
from fastchat.constants import TEMP_IMAGE_DIR


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


async def launch_server():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--parameters", type=str, default="{}")
    args, unknown = parser.parse_known_args()

    model = args.model_path

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

    vllm_proc = await asyncio.create_subprocess_exec(
        *vllm_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    async def print_stderr(proc, prefix):
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            print(f"[{prefix}]", line.decode().strip(), file=sys.stderr)

    # Start printing vLLM stderr asynchronously
    vllm_stderr_task = asyncio.create_task(print_stderr(vllm_proc, "vLLM"))

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
        "--model-path",
        model,
        "--proxy-url",
        f"http://localhost:{port}/v1",
        "--model",
        model,
    ]

    proxy_proc = await asyncio.create_subprocess_exec(
        *proxy_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Start printing proxy stderr asynchronously
    proxy_stderr_task = asyncio.create_task(print_stderr(proxy_proc, "Proxy"))

    # Wait for vLLM server to be ready
    vllm_url = f"http://localhost:{port}/v1/models"
    timeout = 180
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
        await asyncio.sleep(1)

    # Optionally, wait for proxy process to finish (or handle as needed)
    await proxy_proc.wait()
    await vllm_proc.wait()
    await vllm_stderr_task
    await proxy_stderr_task

    print("Vllm worker exited", file=sys.stderr)
    clear_vram()


asyncio.run(launch_server())
