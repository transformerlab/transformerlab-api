import argparse
import json
import os
import subprocess
import sys
import transformerlab.plugin

# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
parser.add_argument("--parameters", type=str, default="{}")
args, unknown = parser.parse_known_args()

model = args.model_path

llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")

parameters = args.parameters
parameters = json.loads(parameters)

print("Starting VLLM Server with OpenAI API compatibility", file=sys.stderr)

# Clean up parameters
if "inferenceEngine" in parameters:
    del parameters["inferenceEngine"]

if "max-model-len" in parameters:
    if parameters["max-model-len"] == "":
        del parameters["max-model-len"]
    # VLLM OpenAI server uses "max_model_len" instead of "max-model-len"
    else:
        parameters["max_model_len"] = parameters.pop("max-model-len")

if "inferenceEngineFriendlyName" in parameters:
    del parameters["inferenceEngineFriendlyName"]

# num_gpus is supported by VLLM's OpenAI API server
# No need to remove it

# Use VLLM's OpenAI API server directly
popen_args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", "21009"]


# Add all parameters to the command
# Convert kebab-case to snake_case for VLLM parameters
for key, value in parameters.items():
    # Convert kebab-case to snake_case if needed
    key_snake_case = key.replace("-", "_")
    popen_args.extend([f"--{key_snake_case}", str(value)])

print(popen_args)
proc = subprocess.Popen(popen_args, stderr=subprocess.PIPE, stdout=None)

transformerlab.plugin.set_db_config_value("INFERENCE_SERVER_URL", "http://localhost:21009")


# save worker process id to file
# this will allow transformer lab to kill it later
with open(f"{llmlab_root_dir}/worker.pid", "w") as f:
    f.write(str(proc.pid))

# read output:
for line in iter(proc.stderr.readline, b""):
    print(line, file=sys.stderr)

print("VLLM Server exited", file=sys.stderr)
sys.exit(1)  # 99 is our code for CUDA OOM