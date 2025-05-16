from watchfiles import awatch
import json
import os
import platform
import asyncio
import sys
import subprocess
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

# Could also use https://github.com/gpuopenanalytics/pynvml but this is simpler
import psutil
import torch
from fastapi import APIRouter

try:
    from pynvml import (
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
    )
    HAS_AMD = False
except Exception:
    from pyrsmi import rocml
    HAS_AMD = True


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
    print(
        "🟡 Flash Attention is not installed. If you are running on GPU, install to accelerate inference and training. https://github.com/Dao-AILab/flash-attention"
    )


# Read in static system info
system_info = {
    "cpu": platform.machine(),
    "name": platform.node(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "os": platform.system(),
    "os_alias": platform.system_alias(platform.system(), platform.release(), platform.version()),
    "gpu": [],
    "gpu_memory": "",
    "device": "cpu",
    "device_type": "cpu",
    "cuda_version": "n/a",
    "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "n/a"),
    "conda_prefix": os.environ.get("CONDA_PREFIX", "n/a"),
    "flash_attn_version": flash_attn_version,
    "pytorch_version": torch.__version__,
}

# Determine which device to use (cuda/mps/cpu)
if torch.cuda.is_available():
    system_info["device"] = "cuda"
    if not HAS_AMD:
        nvmlInit()
        system_info["cuda_version"] = torch.version.cuda
        system_info["device_type"] = "nvidia"
        pytorch_device = "CUDA"
    else:
        rocml.smi_initialize()
        system_info["device_type"] = "amd"
        system_info["cuda_version"] = torch.version.hip
        pytorch_device = "ROCm"
        

    print(f"🏄 PyTorch is using {pytorch_device}, version {system_info['cuda_version']}")

elif torch.backends.mps.is_available():
    system_info["device"] = "mps"
    system_info["device_type"] = "apple_silicon"
    print("🏄 PyTorch is using MPS for Apple Metal acceleration")

router = APIRouter(prefix="/server", tags=["serverinfo"])


async def get_mac_disk_usage():
    if sys.platform != "darwin":
        return None  # Ensure it only runs on macOS

    try:
        # Run the subprocess asynchronously
        process = await asyncio.create_subprocess_shell(
            "diskutil apfs list | awk '/Capacity In Use By Volumes/'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if stderr:
            print(f"Error retrieving disk usage: {stderr.decode().strip()}")
            return None

        mac_disk_usage = stdout.decode("utf-8").strip()

        # Extract the numeric value before "B" (Bytes) and convert to int
        if "Capacity In Use By Volumes:" in mac_disk_usage:
            mac_disk_usage_cleaned = int(
                mac_disk_usage.split("Capacity In Use By Volumes:")[1].strip().split("B")[0].strip()
            )
            return mac_disk_usage_cleaned

    except Exception as e:
        print(f"Error retrieving disk usage: {e}")

    return None


async def get_macmon_data():
    if sys.platform != "darwin":
        return None  # Ensure it only runs on macOS

    try:
        from macmon import MacMon

        macmon = MacMon()
        data = await macmon.get_metrics_async()
        json_data = json.loads(data)
        return json_data

    except Exception as e:
        print(f"Error retrieving macmon data: {e}")

    return None


@router.get("/info")
async def get_computer_information():
    # start with our static system information and add current performance details
    r = system_info

    # Get the current disk usage if its a mac
    mac_disk_usage = await get_mac_disk_usage()

    # Get data from macmon if its a mac
    macmon_data = await get_macmon_data()

    disk_usage = psutil.disk_usage("/")._asdict()
    if mac_disk_usage:
        disk_usage["used"] = mac_disk_usage
        disk_usage["free"] = disk_usage["total"] - mac_disk_usage
        disk_usage["percent"] = round((mac_disk_usage / disk_usage["total"]) * 100, 2)

    r.update(
        {
            "cpu_percent": psutil.cpu_percent(),
            "cpu_count": psutil.cpu_count(),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": disk_usage,
            "gpu_memory": "",
        }
    )

    g = []

    if macmon_data:
        r["mac_metrics"] = macmon_data

    try:
        if HAS_AMD:
            deviceCount = rocml.smi_get_device_count()
        else:
            deviceCount = nvmlDeviceGetCount()
        # print('device count: ', deviceCount)
        for i in range(deviceCount):
            info = {}
            if HAS_AMD:
                handle = rocml.smi_get_device_id(i)
            else:
                handle = nvmlDeviceGetHandleByIndex(i)

            # Certain versions of the NVML library on WSL return a byte string,
            # and this creates a utf error. This is a workaround:
            if not HAS_AMD:
                device_name = nvmlDeviceGetName(handle)
            else:
                device_name = rocml.smi_get_device_name(i)
            # print('device name: ', device_name)

            # check if device_name is a byte string, if so convert to string:
            if isinstance(device_name, bytes):
                device_name = device_name.decode(errors="ignore")

            info["name"] = device_name
            if not HAS_AMD:
                memory = nvmlDeviceGetMemoryInfo(handle)
                info["total_memory"] = memory.total
                info["free_memory"] = memory.free
                info["used_memory"] = memory.used

                u = nvmlDeviceGetUtilizationRates(handle)
                info["utilization"] = u.gpu
            else:
                info["total_memory"] = rocml.smi_get_device_memory_total(i)
                info["used_memory"] = rocml.smi_get_device_memory_used(i)
                info["free_memory"] = rocml.smi_get_device_memory_total(i) - rocml.smi_get_device_memory_used(i)
                info["utilization"] = rocml.smi_get_device_utilization(i)

            # info["temp"] = nvmlDeviceGetTemperature(handle)
            g.append(info)
    except Exception:  # Catch all exceptions and print them

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
    packages = subprocess.check_output(sys.executable + " -m pip list --format=json", shell=True)

    packages = packages.decode("utf-8")
    packages = json.loads(packages)
    return packages


@router.get("/pytorch_collect_env")
async def get_pytorch_collect_env():
    # run python -m torch.utils.collect_env and return the output
    output = subprocess.check_output(sys.executable + " -m torch.utils.collect_env", shell=True)
    return output.decode("utf-8")


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
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )
