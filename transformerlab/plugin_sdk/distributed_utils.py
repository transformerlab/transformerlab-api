"""
Distributed Training Utilities for TransformerLab Plugins

This module provides utilities for plugins to easily work with distributed training.
Plugins can use these utilities to detect if they're running in distributed mode
and access the accelerator instance.
"""

import os
import sys
import logging
from typing import Optional, Any


def is_distributed_training() -> bool:
    """
    Check if the current execution is in distributed training mode.

    Returns:
        bool: True if running in distributed mode, False otherwise
    """
    return os.environ.get("DISTRIBUTED_TRAINING", "").lower() == "true"


def get_accelerator() -> Optional[Any]:
    """
    Get the accelerator instance if available.

    Returns:
        Optional[Accelerator]: The accelerator instance if available, None otherwise
    """
    if hasattr(sys.modules["__main__"], "accelerator"):
        return sys.modules["__main__"].accelerator
    return None


def get_world_size() -> int:
    """
    Get the world size (total number of processes).

    Returns:
        int: World size, defaults to 1 if not in distributed mode
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    """
    Get the global rank of the current process.

    Returns:
        int: Global rank, defaults to 0 if not in distributed mode
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank of the current process.

    Returns:
        int: Local rank, defaults to 0 if not in distributed mode
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).

    Returns:
        bool: True if main process, False otherwise
    """
    return os.environ.get("IS_MAIN_PROCESS", "true").lower() == "true"


def should_save_outputs() -> bool:
    """
    Check if the current process should save outputs (typically only rank 0).

    Returns:
        bool: True if should save outputs, False otherwise
    """
    return is_main_process()


def print_distributed_info():
    """
    Print information about the distributed training setup.
    Only prints from the main process to avoid cluttered output.
    """
    if is_main_process():
        print("Distributed Training Information:")
        print(f"  - World size: {get_world_size()}")
        print(f"  - Current rank: {get_rank()}")
        print(f"  - Local rank: {get_local_rank()}")
        print(f"  - Is main process: {is_main_process()}")


def setup_distributed_logging(logger):
    """
    Configure logging for distributed training.
    Only the main process will log to avoid duplicate messages.

    Args:
        logger: The logger instance to configure
    """
    if not is_main_process():
        logger.setLevel(logging.WARNING)  # Suppress info/debug from non-main processes


# Example usage in a plugin:
"""
# In your plugin's main.py:

from plugin_sdk.distributed_utils import (
    is_distributed_training,
    get_accelerator,
    should_save_outputs,
    print_distributed_info
)

def main():
    if is_distributed_training():
        print_distributed_info()
        accelerator = get_accelerator()
        
        # Use accelerator for distributed training
        model = accelerator.prepare(model)
        optimizer = accelerator.prepare(optimizer)
        dataloader = accelerator.prepare(dataloader)
        
        # Train your model...
        
        # Only save on main process
        if should_save_outputs():
            model.save_pretrained("output_dir")
    else:
        # Regular single-device training
        pass
"""
