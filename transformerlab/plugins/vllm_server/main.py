import argparse
import json
import os
import subprocess
import sys


def get_python_executable(plugin_dir):
    """Check if a virtual environment exists and return the appropriate Python executable"""
    # Check for virtual environment in the plugin directory
    venv_path = os.path.join(plugin_dir, "venv")

    if os.path.isdir(venv_path):
        print("Virtual environment found, using it for evaluation...")
        # Determine the correct path to the Python executable based on the platform
        python_executable = os.path.join(venv_path, "bin", "python")

        if os.path.exists(python_executable):
            return python_executable

    # Fall back to system Python if venv not found or executable doesn't exist
    print("No virtual environment found, using system Python...")
    return sys.executable


# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
parser.add_argument("--parameters", type=str, default="{}")
args, unknown = parser.parse_known_args()

model = args.model_path

llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")

parameters = args.parameters
parameters = json.loads(parameters)

print("Starting VLLM Server", file=sys.stderr)

# Quantization is not yet supported but once it is, we need to add the following to index:
# "quantization": {
#     "title": "Quantization",
#     "type": "string",
#     "enum": [
#             "aqlm",
#             "awq",
#             "deepspeedfp",
#             "fp8",
#             "fbgemm_fp8",
#             "marlin",
#             "gptq_marlin_24",
#             "gptq_marlin",
#             "awq_marlin",
#             "gptq",
#             "squeezellm",
#             "compressed-tensors",
#             "bitsandbytes",
#             "None"
#     ]
# }

# Now go through the parameters object and remove the key that is equal to "inferenceEngine":
if "inferenceEngine" in parameters:
    del parameters["inferenceEngine"]

if "max-model-len" in parameters:
    if parameters["max-model-len"] == "":
        del parameters["max-model-len"]

if "inferenceEngineFriendlyName" in parameters:
    del parameters["inferenceEngineFriendlyName"]

if "num_gpus" in parameters:
    del parameters["num_gpus"]

# The command to run a VLLM server is:
# python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
# but we can also run it through FastChat's VLLM integration:
# https://github.com/lm-sys/FastChat/blob/main/docs/vllm_integration.md

# Get plugin directory
real_plugin_dir = os.path.realpath(os.path.dirname(__file__))

# Get Python executable (from venv if available)
python_executable = get_python_executable(real_plugin_dir)

popen_args = [python_executable, "-m", "fastchat.serve.vllm_worker", "--model-path", model]

# Add all parameters to the command
for key, value in parameters.items():
    popen_args.extend([f"--{key}", str(value)])

print(popen_args)
proc = subprocess.Popen(popen_args, stderr=subprocess.PIPE, stdout=None)

# save worker process id to file
# this will allow transformer lab to kill it later
with open(f"{llmlab_root_dir}/worker.pid", "w") as f:
    f.write(str(proc.pid))

# read output:
for line in iter(proc.stderr.readline, b""):
    print(line, file=sys.stderr)

print("VLLM Server exited", file=sys.stderr)
sys.exit(1)  # 99 is our code for CUDA OOM
