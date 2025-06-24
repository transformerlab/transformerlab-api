import argparse
import json
import os
import subprocess
import sys
import logging
import signal
import psutil
import torch
import select
import gc

try:
    from transformerlab.plugin import get_python_executable
except ImportError:
    from transformerlab.plugin_sdk.transformerlab.plugin import get_python_executable

def kill_sglang_subprocesses():
    print(">>> [main] Checking for lingering sglang scheduler subprocesses...")
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            cmdline_list = proc.info.get("cmdline")
            if not cmdline_list:  # Handles None or empty list
                continue

            cmdline = " ".join(cmdline_list)
            if "sglang" in cmdline or "sglang::scheduler" in cmdline:
                print(f">>> [main] Killing lingering sglang process: PID {proc.pid}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


def kill_existing_workers():
    print(">>> [main] Checking for existing model_worker.py processes...")
    for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"])
            if "model_worker.py" in cmdline:
                print(f">>> [main] Terminating model_worker PID={proc.info['pid']}")
                os.kill(proc.info["pid"], signal.SIGKILL)
        except Exception as e:
            print(f">>> [main] Failed to inspect or kill process: {e}")

# Clear CUDA memory (if CUDA is available)
def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        print(">>> [main] Emptying CUDA memory cache and collecting garbage...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def isnum(s):
    return s.strip().isdigit()


# Get all arguments provided to this script using argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
parser.add_argument("--adaptor-path", type=str)
parser.add_argument("--parameters", type=str, default="{}")
parser.add_argument("--plugin_dir", type=str)

args, unknown = parser.parse_known_args()

print("Starting SGLang Worker", file=sys.stderr)

model = args.model_path
adaptor = args.adaptor_path

if adaptor != "":
    model = adaptor

parameters = args.parameters
parameters = json.loads(parameters)

eight_bit = False
four_bit = False
if parameters.get("load_compressed", "None") != "None":
    if parameters.get("load_compressed", "None") == "8-bit":
        eight_bit = True
        four_bit = False
    elif parameters.get("load_compressed", "None") == "4-bit":
        eight_bit = False
        four_bit = True


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

# Get plugin directory
real_plugin_dir = os.path.realpath(os.path.dirname(__file__))

# Get Python executable (from venv if available)
python_executable = get_python_executable(real_plugin_dir)

popen_args = [python_executable, f"{PLUGIN_DIR}/model_worker.py", "--model-path", model, "--device", device]

model_dtype = parameters.get("model_dtype")
# Set model dtype if provided
if model_dtype is not None and model_dtype != "" and model_dtype != "auto":
    popen_args.extend(["--dtype", model_dtype])
if num_gpus:
    popen_args.extend(["--gpus", gpu_ids])
    popen_args.extend(["--num-gpus", str(num_gpus)])
if eight_bit:
    popen_args.append("--load-8bit")
if four_bit:
    popen_args.append("--load-4bit")

free_mem = torch.cuda.mem_get_info()[0] / (1024**2)  # in MiB
print(f">>> [main] Free GPU memory: {free_mem:.2f} MiB")
if free_mem < 1000:
    print("⚠️ Warning: Less than 1 GB GPU memory free before starting model. Might fail with OOM.")

proc = subprocess.Popen(popen_args, stderr=subprocess.PIPE, stdout=None)

# save worker process id to file
# this will allow transformer lab to kill it later
with open(f"{llmlab_root_dir}/worker.pid", "w") as f:
    f.write(str(proc.pid))

# read output:
try:
    stderr_fd = proc.stderr.fileno()
    poller = select.poll()
    poller.register(stderr_fd, select.POLLIN)

    while True:
        if poller.poll(1000):  # Wait max 1 second
            line = proc.stderr.readline()
            if not line:
                break  # EOF
            decoded = line.decode("utf-8")
            print(decoded, file=sys.stderr)
            if "torch.cuda.OutOfMemoryError" in decoded:
                print("CUDA Out of memory error", file=sys.stderr)
                proc.kill()
                kill_existing_workers()
                clear_vram()
                kill_sglang_subprocesses()
                sys.exit(99)
        elif proc.poll() is not None:
            # Process has exited
            break
finally:
    print(">>> [main] Waiting for model_worker to terminate completely...")
    proc.wait()
    print(">>> [main] Model worker terminated. Proceeding with cleanup.")
    kill_existing_workers()
    kill_sglang_subprocesses()
    time.sleep(1)
    clear_vram()
    print(">>> [main] Cleanup completed.")


print("SGLang Worker exited", file=sys.stderr)
sys.exit(1)

