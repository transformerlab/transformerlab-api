import argparse
import json
import os
import subprocess
import sys

try:
    from transformerlab.plugin import get_python_executable
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable


# Get all arguments provided to this script using argparse
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

popen_args = [
    python_executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model", model,
    "--dtype", args.model_dtype,
    "--port", args.port,
    "--max-model-len", args.max_model_len,
    "--gpu-memory-utilization", args.gpu_memory_utilization,
]

process = subprocess.Popen(popen_args)

popen_args = [
    python_executable, 
    "-m", 
    "fastchat.serve.openai_api_proxy_worker",
    "--model-path", model,
    "--proxy-url", f"http://localhost:{args.port}/v1",
    "--model", model.split("/")[-1],
    "--model-names", [args.model],
    ]


# Add all parameters to the command
# for key, value in parameters.items():
#     popen_args.extend([f"--{key}", str(value)])

print(popen_args)
proc = subprocess.Popen(popen_args, stderr=subprocess.PIPE, stdout=None)

# save worker process id to file
# this will allow transformer lab to kill it later
with open(f"{llmlab_root_dir}/worker.pid", "w") as f:
    f.write(str(proc.pid))

# read output:
for line in iter(proc.stderr.readline, b""):
    print(line, file=sys.stderr)

print("OpenAI API Proxy Server exited", file=sys.stderr)
sys.exit(1)  # 99 is our code for CUDA OOM