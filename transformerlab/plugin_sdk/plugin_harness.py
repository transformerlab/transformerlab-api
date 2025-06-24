"""
This is the main file that gets called by popen when a plugin is run.
It must get passed:

    --plugin_dir            full path to the directory containing the plugin

All other parameters can be passed as if you are calling the plugin directly.

For distributed training, additional parameters:
    --distributed_mode      Enable distributed training mode
    --accelerate_config     JSON string with accelerate configuration

"""

import sys
import os
import json
import argparse
import traceback


parser = argparse.ArgumentParser()
parser.add_argument("--plugin_dir", type=str, required=True)
parser.add_argument("--distributed_mode", action="store_true", help="Enable distributed training mode")
parser.add_argument("--accelerate_config", type=str, help="JSON string with accelerate configuration")
args, unknown = parser.parse_known_args()

# Add the plugin directory to the path
# Note that this will allow the plugin to import files in this file's directory
# So the plugin is able to import the SDK
sys.path.append(args.plugin_dir)

# Handle distributed training mode
if args.distributed_mode:
    print("Setting up distributed training environment...")
    
    # Parse accelerate configuration
    accelerate_config = {}
    if args.accelerate_config:
        try:
            accelerate_config = json.loads(args.accelerate_config)
        except json.JSONDecodeError as e:
            print(f"Error parsing accelerate config: {e}")
            sys.exit(1)
    
    # Set up distributed training environment
    try:
        from accelerate import Accelerator, DistributedDataParallelKwargs
        
        # Initialize accelerator with configuration
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        
        # Set distributed training environment variables
        os.environ["DISTRIBUTED_TRAINING"] = "true"
        os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
        os.environ["RANK"] = str(accelerator.process_index)
        os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
        os.environ["IS_MAIN_PROCESS"] = str(accelerator.is_main_process)
        
        print("Distributed training setup complete:")
        print(f"  - World size: {accelerator.num_processes}")
        print(f"  - Rank: {accelerator.process_index}")
        print(f"  - Local rank: {accelerator.local_process_index}")
        print(f"  - Is main process: {accelerator.is_main_process}")
        
        # Store accelerator globally for plugin access
        sys.modules['__main__'].accelerator = accelerator
        
    except ImportError:
        print("ERROR: accelerate is not installed. Please install it with: pip install accelerate")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to set up distributed training: {e}")
        traceback.print_exc()
        sys.exit(1)

try:
    import main
except ImportError as e:
    print(f"Error executing plugin: {e}")
    traceback.print_exc()

    # if e is a ModuleNotFoundError, the plugin is missing a required package
    if isinstance(e, ModuleNotFoundError):
        print("ModuleNotFoundError means a Python package is missing. This is usually fixed by reinstalling the plugin")

    sys.exit(1)

# Also execute the function main.main(), if it exists
if "main" in dir(main) and callable(getattr(main, "main")):
    main.main()
