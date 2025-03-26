import argparse
import json
import os
import subprocess
import sys

import torch


def isnum(s):
    return s.strip().isdigit()

# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
parser.add_argument("--adaptor-path", type=str)
parser.add_argument("--parameters", type=str, default="{}")
parser.add_argument("--plugin_dir", type=str)

args, unknown = parser.parse_known_args()

print("Starting Standard FastChat Worker", file=sys.stderr)

model = args.model_path
adaptor = args.adaptor_path

if adaptor != "":
    model = adaptor

parameters = args.parameters
parameters = json.loads(parameters)

if parameters.get("eight_bit") == "on":
    eight_bit = True
else:
    eight_bit = False

gpu_ids = parameters.get("gpu_ids", "")
if gpu_ids is not None and gpu_ids != "":
    gpu_ids_formatted = gpu_ids.split(",")
    if len(gpu_ids_formatted) > 1:
        num_gpus = len(gpu_ids_formatted)
        # To remove any spacing issues which may arise
        gpu_ids = ",".join([gpu_id.strip() for gpu_id in gpu_ids_formatted])
        # If gpu_ids is not formatted correctly then use all GPUs by default
        if num_gpus == 0 or not isnum(gpu_ids_formatted[0]):
            num_gpus = torch.cuda.device_count()
            gpu_ids = ""
    else:
        num_gpus = 1
        gpu_ids = gpu_ids_formatted[0]
else:
    num_gpus = torch.cuda.device_count()

if gpu_ids is None:
    gpu_ids = ""

# Auto detect backend if device not specified
device = parameters.get("device", None)
if device is None or device == "":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        num_gpus = 0


llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")
PLUGIN_DIR = args.plugin_dir

popen_args = [sys.executable, f"{PLUGIN_DIR}/model_worker.py", "--model-path", model, "--device", device]
if num_gpus:
    popen_args.extend(["--gpus", gpu_ids])
    popen_args.extend(["--num-gpus", str(num_gpus)])
if eight_bit:
    popen_args.append("--load-8bit")


print(popen_args)
proc = subprocess.Popen(popen_args, stderr=subprocess.PIPE, stdout=None)

# save worker process id to file
# this will allow transformer lab to kill it later
with open(f"{llmlab_root_dir}/worker.pid", "w") as f:
    f.write(str(proc.pid))

# read output:
for line in proc.stderr:
    # if line contains "Ready to serve" then we can break
    if "torch.cuda.OutOfMemoryError" in line.decode("utf-8"):
        print("CUDA Out of memory error", file=sys.stderr)
        sys.exit(99)  # 99 is our code for CUDA OOM
    print(line.decode("utf-8"), file=sys.stderr)

print("FastChat Worker exited", file=sys.stderr)
sys.exit(1)
