#!/usr/bin/env python3
"""
Universal Distributed Training Wrapper for Transformer Lab Plugins

This wrapper enables ANY existing Transformer Lab training plugin to run
in distributed mode across multiple machines using HuggingFace Accelerate.

Usage:
    # Local multi-GPU (existing behavior)
    python plugin_main.py --config_file config.json
    
    # Distributed across machines (new behavior)  
    accelerate launch --multi_node --num_machines=3 --machine_rank=0 \
        --main_process_ip=192.168.1.100 --main_process_port=29500 \
        distributed_training_wrapper.py --original_plugin=llama_trainer \
        --config_file=config.json

How it works:
1. Wraps any existing plugin's main.py
2. Sets up Accelerate distributed environment
3. Modifies plugin configuration for distributed training
4. Handles model saving coordination (only rank 0 saves)
5. Provides distributed logging and monitoring
"""

import os
import sys
import json
import importlib.util
import argparse
from pathlib import Path


def setup_distributed_environment():
    """Set up environment variables for distributed training"""

    # Import accelerate here to ensure it's available
    try:
        from accelerate import Accelerator, DistributedDataParallelKwargs
    except ImportError:
        print("ERROR: accelerate is not installed. Please install it with: pip install accelerate")
        sys.exit(1)

    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    return accelerator


def modify_config_for_distributed(config_path, accelerator, original_plugin):
    """
    Modify the training configuration for distributed training
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # Adjust batch size for distributed training
    if "batch_size" in config:
        # Keep the same effective batch size across all processes
        original_batch_size = int(config["batch_size"])
        per_device_batch_size = max(1, original_batch_size // accelerator.num_processes)
        config["batch_size"] = per_device_batch_size
        print(f"Adjusted batch size from {original_batch_size} to {per_device_batch_size} per device")

    # Adjust learning rate for distributed training (linear scaling)
    if "learning_rate" in config:
        original_lr = float(config["learning_rate"])
        # Scale learning rate by number of processes (common practice)
        scaled_lr = original_lr * accelerator.num_processes
        config["learning_rate"] = scaled_lr
        print(f"Scaled learning rate from {original_lr} to {scaled_lr} for {accelerator.num_processes} processes")

    # Adjust output directories for distributed training
    rank = accelerator.process_index

    if "output_dir" in config:
        original_output = config["output_dir"]
        # Only rank 0 saves to the main output directory
        if rank == 0:
            config["output_dir"] = original_output
        else:
            # Other ranks save to temporary directories
            config["output_dir"] = f"{original_output}_temp_rank_{rank}"

    if "adaptor_output_dir" in config:
        original_adaptor = config["adaptor_output_dir"]
        # Only rank 0 saves the final adaptor
        if rank == 0:
            config["adaptor_output_dir"] = original_adaptor
        else:
            # Other ranks save to temporary directories
            config["adaptor_output_dir"] = f"{original_adaptor}_temp_rank_{rank}"

    # Add distributed training flags
    config["distributed_training"] = True
    config["world_size"] = accelerator.num_processes
    config["rank"] = rank
    config["local_rank"] = accelerator.local_process_index
    config["is_main_process"] = accelerator.is_main_process

    # Write modified config to a temporary file
    temp_config_path = f"{config_path}_rank_{rank}.json"
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=2)

    return temp_config_path


def load_and_run_plugin(plugin_name, config_path, accelerator):
    """
    Dynamically load and run the specified plugin with distributed support
    """
    # Find the plugin directory
    plugin_dir = Path(__file__).parent / plugin_name
    if not plugin_dir.exists():
        print(f"ERROR: Plugin directory not found: {plugin_dir}")
        sys.exit(1)

    plugin_main = plugin_dir / "main.py"
    if not plugin_main.exists():
        print(f"ERROR: Plugin main.py not found: {plugin_main}")
        sys.exit(1)

    print(f"Loading plugin: {plugin_name} from {plugin_dir}")

    # Modify the plugin's environment for distributed training
    original_cwd = os.getcwd()
    os.chdir(str(plugin_dir))

    # Add plugin directory to path
    sys.path.insert(0, str(plugin_dir))

    try:
        # Modify config for distributed training
        modified_config_path = modify_config_for_distributed(config_path, accelerator, plugin_name)

        # Set up command line arguments that the plugin expects
        original_argv = sys.argv.copy()
        sys.argv = [str(plugin_main), "--config_file", modified_config_path]

        # Import and run the plugin
        spec = importlib.util.spec_from_file_location("plugin_main", plugin_main)
        plugin_module = importlib.util.module_from_spec(spec)

        # Execute the plugin
        print(f"Starting distributed training on rank {accelerator.process_index}/{accelerator.num_processes}")
        spec.loader.exec_module(plugin_module)

        # Cleanup
        if os.path.exists(modified_config_path):
            os.remove(modified_config_path)

        sys.argv = original_argv

    except Exception as e:
        print(f"ERROR running plugin {plugin_name}: {e}")
        raise
    finally:
        os.chdir(original_cwd)
        if str(plugin_dir) in sys.path:
            sys.path.remove(str(plugin_dir))


def create_accelerate_config_for_multi_node(num_machines, machine_rank, main_process_ip, main_process_port):
    """
    Create an accelerate config file for multi-node training
    """
    try:
        import torch

        num_gpus_per_machine = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except ImportError:
        num_gpus_per_machine = 1

    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "downcast_bf16": "no",
        "gpu_ids": "all",
        "machine_rank": machine_rank,
        "main_process_ip": main_process_ip,
        "main_process_port": main_process_port,
        "main_training_function": "main",
        "mixed_precision": "bf16",
        "num_machines": num_machines,
        "num_processes": num_machines * num_gpus_per_machine,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
    }

    # Write config to a temporary file
    config_path = "/tmp/accelerate_config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def main():
    parser = argparse.ArgumentParser(description="Universal Distributed Training Wrapper")
    parser.add_argument("--original_plugin", required=True, help="Name of the original plugin to run")
    parser.add_argument("--config_file", required=True, help="Path to the training configuration file")
    parser.add_argument("--num_machines", type=int, help="Number of machines in distributed setup")
    parser.add_argument("--machine_rank", type=int, help="Rank of this machine (0 for main)")
    parser.add_argument("--main_process_ip", help="IP address of the main process")
    parser.add_argument("--main_process_port", type=int, default=29500, help="Port for main process")

    args = parser.parse_args()

    print("=" * 60)
    print("Universal Distributed Training Wrapper")
    print("=" * 60)
    print(f"Plugin: {args.original_plugin}")
    print(f"Config: {args.config_file}")

    if args.num_machines and args.num_machines > 1:
        print(f"Distributed setup: {args.num_machines} machines")
        print(f"This machine rank: {args.machine_rank}")
        print(f"Main process: {args.main_process_ip}:{args.main_process_port}")
    else:
        print("Single machine multi-GPU setup")

    print("=" * 60)

    # Set up distributed environment
    accelerator = setup_distributed_environment()

    print("Accelerator initialized:")
    print(f"  - Process index: {accelerator.process_index}")
    print(f"  - Local process index: {accelerator.local_process_index}")
    print(f"  - Num processes: {accelerator.num_processes}")
    print(f"  - Device: {accelerator.device}")
    print(f"  - Is main process: {accelerator.is_main_process}")

    # Run the original plugin with distributed support
    try:
        load_and_run_plugin(args.original_plugin, args.config_file, accelerator)

        if accelerator.is_main_process:
            print("\n" + "=" * 60)
            print("Distributed training completed successfully!")
            print("=" * 60)

    except Exception as e:
        print(f"ERROR: Distributed training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
