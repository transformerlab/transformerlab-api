import functools
import gc

import torch


try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False

try:
    HAS_XPU = torch.xpu.is_available()
except Exception:
    HAS_XPU = False


def clean_memory():
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    if HAS_XPU:
        torch.xpu.empty_cache()
    if HAS_MPS:
        torch.mps.empty_cache()


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    r"""
    Do not call this function from training scripts. Use accelerator.device instead.
    """
    if HAS_CUDA:
        device = torch.device("cuda")
    elif HAS_XPU:
        device = torch.device("xpu")
    elif HAS_MPS:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"get_preferred_device() -> {device}")
    return device
