import atexit
import json
import os
import platform
import sys
import subprocess


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

# Check for version of flash_attn:
try:
    from flash_attn import __version__ as flash_attn_version
    print("⚡️ Flash Attention is installed.")
    print(f"flash_attn version: {flash_attn_version}")
except ImportError:
    flash_attention_version = "n/a"
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
    "flash_attn_version": flash_attn_version
}

# Determine which device to use (cuda/mps/cpu)
if torch.cuda.is_available():
    system_info["device"] = "cuda"

    # we have a GPU so initialize the nvidia python bindings
    nvmlInit()

    # get CUDA version:
    system_info["cuda_version"] = torch.version.cuda

elif torch.backends.mps.is_available():
    system_info["device"] = "mps"

print("Using", system_info["device"])


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
        for i in range(deviceCount):
            info = {}

            handle = nvmlDeviceGetHandleByIndex(i)

            info["name"] = nvmlDeviceGetName(handle)

            memory = nvmlDeviceGetMemoryInfo(handle)
            info["total_memory"] = memory.total
            info["free_memory"] = memory.free
            info["used_memory"] = memory.used

            u = nvmlDeviceGetUtilizationRates(handle)
            info["utilization"] = u.gpu

            # info["temp"] = nvmlDeviceGetTemperature(handle)
            g.append(info)
    except:  # noqa: E722 (TODO: what are the exceptions to chat here?)
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


def cleanup_at_exit():
    if torch.cuda.is_available():
        nvmlShutdown()


atexit.register(cleanup_at_exit)
