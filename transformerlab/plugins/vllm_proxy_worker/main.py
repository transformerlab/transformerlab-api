import argparse
import json
import os
import subprocess
import sys
import threading
import time
import socket

try:
    from transformerlab.plugin import get_python_executable
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable


def stream_output(pipe, label):
    """Continuously read from pipe and print lines with label."""
    for line in iter(pipe.readline, b''):
        print(f"[{label}]", line.decode(errors='replace').rstrip(), file=sys.stderr)
    pipe.close()


def wait_for_port(host, port, timeout=30):
    """Wait until a TCP port is open on host or timeout."""
    start = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            if time.time() - start > timeout:
                return False
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--parameters", type=str, default="{}")
    args = parser.parse_args()

    model = args.model_path
    parameters = json.loads(args.parameters)

    # Clean parameters
    parameters.pop("inferenceEngine", None)
    parameters.pop("inferenceEngineFriendlyName", None)

    llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH", ".")

    python_executable = get_python_executable(os.path.realpath(os.path.dirname(__file__)))

    port = int(parameters.get("port", 8000))
    host = "127.0.0.1"

    # Start vLLM server subprocess
    vllm_args = [
        python_executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--dtype", str(parameters.get("model_dtype", "auto")),
        "--port", str(port),
        "--max-model-len", str(parameters.get("max_model_len", 2048)),
        "--gpu-memory-utilization", str(parameters.get("gpu_memory_utilization", 0.9)),
    ]

    print("Starting vLLM OpenAI API server...", file=sys.stderr)
    vllm_proc = subprocess.Popen(vllm_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Start threads to read vLLM server output
    threading.Thread(target=stream_output, args=(vllm_proc.stdout, "vLLM-stdout"), daemon=True).start()
    threading.Thread(target=stream_output, args=(vllm_proc.stderr, "vLLM-stderr"), daemon=True).start()

    # Wait for vLLM server to be ready (port open)
    if not wait_for_port(host, port, timeout=30):
        print(f"Error: vLLM server did not start listening on {host}:{port} within timeout.", file=sys.stderr)
        vllm_proc.terminate()
        vllm_proc.wait()
        sys.exit(1)

    print(f"vLLM server is up and running on {host}:{port}", file=sys.stderr)

    # Start FastChat proxy worker subprocess
    proxy_args = [
        python_executable,
        "-m", "fastchat.serve.openai_api_proxy_worker",
        "--model-path", model,
        "--proxy-url", f"http://{host}:{port}/v1",
        "--model", os.path.basename(model),
        "--model-names", os.path.basename(model),
    ]

    print("Starting FastChat OpenAI API Proxy worker...", file=sys.stderr)
    proxy_proc = subprocess.Popen(proxy_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Start threads to read proxy worker output
    threading.Thread(target=stream_output, args=(proxy_proc.stdout, "ProxyWorker-stdout"), daemon=True).start()
    threading.Thread(target=stream_output, args=(proxy_proc.stderr, "ProxyWorker-stderr"), daemon=True).start()

    # Save proxy worker PID for external management
    try:
        with open(os.path.join(llmlab_root_dir, "worker.pid"), "w") as f:
            f.write(str(proxy_proc.pid))
    except Exception as e:
        print(f"Warning: Could not write worker PID file: {e}", file=sys.stderr)

    # Wait for proxy worker to finish (it usually runs indefinitely until killed)
    proxy_return_code = proxy_proc.wait()
    print(f"Proxy worker exited with code {proxy_return_code}", file=sys.stderr)

    # If proxy worker exits, also terminate vLLM server
    if vllm_proc.poll() is None:
        print("Terminating vLLM server...", file=sys.stderr)
        vllm_proc.terminate()
        vllm_proc.wait()

    # Exit with proxy worker's exit code
    sys.exit(proxy_return_code)


if __name__ == "__main__":
    main()
