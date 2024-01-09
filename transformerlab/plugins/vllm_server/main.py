import argparse
import os
import subprocess
import sys

# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str)
args, unknown = parser.parse_known_args()

print("Starting VLLM Server", file=sys.stderr)

model = args.model_path

llmlab_root_dir = os.getenv('LLM_LAB_ROOT_PATH')

# The command to run a VLLM server is:
# python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
# but we can also run it through FastChat's VLLM integration:
# https://github.com/lm-sys/FastChat/blob/main/docs/vllm_integration.md

popen_args = [sys.executable, '-m',
              'fastchat.serve.vllm_worker', '--model-path', model]

print(popen_args)
proc = subprocess.Popen(popen_args, stderr=subprocess.PIPE, stdout=None)

# save worker process id to file
# this will allow transformer lab to kill it later
with open(f"{llmlab_root_dir}/worker.pid", "w") as f:
    f.write(str(proc.pid))

# read output:
for line in iter(proc.stderr.readline, b''):
    print(line, file=sys.stderr)

print("VLLM Server exited", file=sys.stderr)
