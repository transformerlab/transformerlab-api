from watchfiles import awatch
import atexit
import json
import os
import platform
import sys
import subprocess
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

# Could also use https://github.com/gpuopenanalytics/pynvml but this is simpler
import psutil
import torch
from fastapi import APIRouter
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)

from transformerlab.shared import dirs

pyTorch_version = torch.__version__
print(f"🔥 PyTorch version: {pyTorch_version}")

# Check for version of flash_attn:
flash_attn_version = ""
try:
    from flash_attn import __version__ as flash_attn_version
    print(f"⚡️ Flash Attention is installed, version {flash_attn_version}")
except ImportError:
    flash_attn_version = "n/a"
    print("🟡 Flash Attention is not installed. If you are running on GPU, install to accelerate inference and training. https://github.com/Dao-AILab/flash-attention")


# Read in static system info
system_info = {
    "cpu": platform.machine(),
    "name": platform.node(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "os": platform.system(),
    "os_alias": platform.system_alias(
        platform.system(), platform.release(), platform.version()
    ),
    "gpu": [],
    "gpu_memory": "",
    "device": "cpu",
    "cuda_version": "n/a",
    "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "n/a"),
    "conda_prefix": os.environ.get("CONDA_PREFIX", "n/a"),
    "flash_attn_version": flash_attn_version,
    "pytorch_version": torch.__version__,
}

# Determine which device to use (cuda/mps/cpu)
if torch.cuda.is_available():
    system_info["device"] = "cuda"

    # we have a GPU so initialize the nvidia python bindings
    nvmlInit()

    # get CUDA version:
    system_info["cuda_version"] = torch.version.cuda

    print(f"🏄 PyTorch is using CUDA, version {torch.version.cuda}")

elif torch.backends.mps.is_available():
    system_info["device"] = "mps"
    print("🏄 PyTorch is using MPS for Apple Metal acceleration")

router = APIRouter(prefix="/server", tags=["serverinfo"])


@router.get("/info")
async def get_computer_information():
    # start with our static system information and add current performance details
    r = system_info
    r.update(
        {
            "cpu_percent": psutil.cpu_percent(),
            "cpu_count": psutil.cpu_count(),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage("/")._asdict(),
            "gpu_memory": "",
        }
    )

    g = []

    try:
        deviceCount = nvmlDeviceGetCount()
        # print('device count: ', deviceCount)
        for i in range(deviceCount):
            info = {}

            handle = nvmlDeviceGetHandleByIndex(i)

            # Certain versions of the NVML library on WSL return a byte string,
            # and this creates a utf error. This is a workaround:
            device_name = nvmlDeviceGetName(handle)
            # print('device name: ', device_name)

            # check if device_name is a byte string, if so convert to string:
            if (isinstance(device_name, bytes)):
                device_name = device_name.decode()

            info["name"] = device_name

            memory = nvmlDeviceGetMemoryInfo(handle)
            info["total_memory"] = memory.total
            info["free_memory"] = memory.free
            info["used_memory"] = memory.used

            u = nvmlDeviceGetUtilizationRates(handle)
            info["utilization"] = u.gpu

            # info["temp"] = nvmlDeviceGetTemperature(handle)
            g.append(info)
    except Exception:  # Catch all exceptions and print them
        # print(f"Error retrieving GPU information: {e}")

        g.append(
            {
                "name": "cpu",
                "total_memory": "n/a",
                "free_memory": "n/a",
                "used_memory": "n/a",
                "utilization": "n/a",
            }
        )

    r["gpu"] = g

    return r


@router.get("/python_libraries")
async def get_python_library_versions():
    # get the list of installed python packages
    packages = subprocess.check_output(
        sys.executable + " -m pip list --format=json", shell=True)

    packages = packages.decode("utf-8")
    packages = json.loads(packages)
    return packages


@router.get("/pytorch_collect_env")
async def get_pytorch_collect_env():
    # run python -m torch.utils.collect_env and return the output
    output = subprocess.check_output(
        sys.executable + " -m torch.utils.collect_env", shell=True
    )
    return output.decode("utf-8")


def cleanup_at_exit():
    if torch.cuda.is_available():
        nvmlShutdown()


atexit.register(cleanup_at_exit)


GLOBAL_LOG_PATH = dirs.GLOBAL_LOG_PATH


async def watch_file(filename: str, start_from_beginning=False, force_polling=False) -> AsyncGenerator[str, None]:
    print(f"👀 Watching file: {filename}")

    # create the file if it doesn't already exist:
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("")

    last_position = 0
    if start_from_beginning:
        last_position = 0
        with open(filename, "r") as f:
            f.seek(last_position)
            new_lines = f.readlines()
            yield (f"data: {json.dumps(new_lines)}\n\n")
            last_position = f.tell()
    else:
        try:
            with open(filename, "r") as f:
                f.seek(0, os.SEEK_END)
                last_position = f.tell()
        except Exception as e:
            print(f"Error seeking to end of file: {e}")

    async for changes in awatch(filename, force_polling=force_polling):
        print(f"📝 File changed: {filename}")
        with open(filename, "r") as f:
            f.seek(last_position)
            new_lines = f.readlines()
            yield (f"data: {json.dumps(new_lines)}\n\n")
            last_position = f.tell()


@router.get("/stream_log")
async def watch_log():
    return StreamingResponse(
        watch_file(GLOBAL_LOG_PATH),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )
