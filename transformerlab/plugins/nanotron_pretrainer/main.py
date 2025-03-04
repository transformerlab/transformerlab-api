import os
import yaml
import argparse
import subprocess
import json
import requests
import sys

parser = argparse.ArgumentParser(description="Nanotron Pre-training")

# Only accept input file parameter
parser.add_argument("--input_file", type=str, help="Path to JSON configuration file")

args, other = parser.parse_known_args()

args, unknown = parser.parse_known_args()

# Load configuration from JSON file
input_config = None
try:
    with open(args.input_file) as json_file:
        input_config = json.load(json_file)
except Exception as e:
    print(f"Error loading configuration file: {e}")
    sys.exit(1)

# Extract configuration
config = input_config["config"]

print("Arguments:")
print(args)


def get_gpu_count_from_url(url=None):
    """
    Get the number of available GPUs from a URL or fallback to local detection

    Args:
        url: URL to query for GPU count (if None, will try to detect locally)

    Returns:
        int: Number of available GPUs, defaults to 1 if detection fails
    """
    # If URL is provided, try to fetch GPU count from there
    if url:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Assuming the response has a field for GPU count
                return data.get("gpu_count", 1)
        except Exception as e:
            print(f"Failed to get GPU count from URL: {e}")

    # Fallback: Try to detect locally using PyTorch
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception as e:
        print(f"Failed to get GPU count using PyTorch: {e}")

    # Ultimate fallback: return 1
    print("Could not determine GPU count, defaulting to 1")
    return 1


def generate_nanotron_config(config):
    """
    Generate YAML configuration for Nanotron from input parameters

    Args:
        config: Dictionary containing configuration parameters

    Returns:
        dict: Complete Nanotron configuration
    """

    # Format the run name with date and job ID
    job_id = config.get("job_id", str(0))
    # run_name = config.get("run_name", "nanotron_run_%date_%jobid")
    # run_name = run_name.replace("%date", datetime.now().strftime("%Y-%m-%d"))
    # run_name = run_name.replace("%jobid", job_id)
    run_name = config.get("template_name", f"nanotron_run_{job_id}")
    checkpoint_path = os.path.join(os.environ.get("_TFL_WORKSPACE_DIR", "."), "models", "run_name", "checkpoints")

    # Create the config dictionary
    nanotron_config = {
        "checkpoints": {
            "checkpoint_interval": int(config.get("checkpoint_interval", 1000)),
            "checkpoints_path": checkpoint_path,
            "checkpoints_path_is_shared_file_system": False,
            "resume_checkpoint_path": None,
            "save_initial_state": False,
        },
        "data_stages": [
            {
                "data": {
                    "dataset": {
                        "dataset_overwrite_cache": False,
                        "dataset_processing_num_proc_per_process": 1,
                        "hf_dataset_config_name": None,
                        "hf_dataset_or_datasets": config.get("dataset_name", "stas/openwebtext-10k"),
                        "hf_dataset_splits": config.get("dataset_split", "train"),
                        "text_column_name": config.get("text_column_name", "text"),
                    },
                    "num_loading_workers": 1,
                    "seed": int(config.get("seed", 42)),
                },
                "name": "Stable Training Stage",
                "start_training_step": 1,
            },
            {
                "data": {
                    "dataset": {
                        "dataset_overwrite_cache": False,
                        "dataset_processing_num_proc_per_process": 1,
                        "hf_dataset_config_name": None,
                        "hf_dataset_or_datasets": config.get("dataset_name", "stas/openwebtext-10k"),
                        "hf_dataset_splits": config.get("dataset_split", "train"),
                        "text_column_name": config.get("text_column_name", "text"),
                    },
                    "num_loading_workers": 1,
                    "seed": int(config.get("seed", 42)),
                },
                "name": "Annealing Phase",
                "start_training_step": int(config.get("annealing_start_step", 10)),
            },
        ],
        "general": {
            "benchmark_csv_path": None,
            "consumed_train_samples": None,
            "ignore_sanity_checks": True,
            "project": config.get("project_name", "nanotron_project"),
            "run": run_name,
            "seed": int(config.get("seed", 42)),
            "step": None,
        },
        "lighteval": None,
        "logging": {"iteration_step_info_interval": 1, "log_level": "info", "log_level_replica": "info"},
        "model": {
            "ddp_bucket_cap_mb": 25,
            "dtype": config.get("mixed_precision", "bfloat16"),
            "init_method": {"std": 0.025},
            "make_vocab_size_divisible_by": 1,
            "model_config": {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_act": "silu",
                "hidden_size": int(config.get("model_hidden_size", 16)),
                "initializer_range": 0.02,
                "intermediate_size": int(config.get("model_intermediate_size", 64)),
                "is_llama_config": True,
                "max_position_embeddings": int(config.get("maximum_sequence_length", 256)),
                "num_attention_heads": int(config.get("model_num_attention_heads", 4)),
                "num_hidden_layers": int(config.get("model_num_layers", 2)),
                "num_key_value_heads": int(config.get("model_num_key_value_heads", 4)),
                "pad_token_id": None,
                "pretraining_tp": 1,
                "rms_norm_eps": 1.0e-05,
                "rope_scaling": None,
                "tie_word_embeddings": True,
                "use_cache": True,
                "vocab_size": 256,  # Will be determined by the tokenizer in practice
            },
        },
        "optimizer": {
            "accumulate_grad_in_fp32": True,
            "clip_grad": 1.0,
            "learning_rate_scheduler": {
                "learning_rate": float(config.get("learning_rate", 5e-4)),
                "lr_decay_starting_step": None,
                "lr_decay_steps": int(config.get("train_steps", 10000)) - int(config.get("warmup_steps", 2)),
                "lr_decay_style": "cosine",
                "lr_warmup_steps": int(config.get("warmup_steps", 2)),
                "lr_warmup_style": "linear",
                "min_decay_lr": 1.0e-05,
            },
            "optimizer_factory": {
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_eps": 1.0e-08,
                "name": "adamW",
                "torch_adam_is_fused": True,
            },
            "weight_decay": float(config.get("weight_decay", 0.01)),
            "zero_stage": 0,
        },
        "parallelism": {
            "dp": int(config.get("data_parallel_size", 2)),
            "expert_parallel_size": 1,
            "pp": int(config.get("pipeline_parallel_size", 1)),
            "pp_engine": "1f1b",
            "tp": int(config.get("tensor_parallel_size", 1)),
            "tp_linear_async_communication": True,
            "tp_mode": "REDUCE_SCATTER",
        },
        "profiler": None,
        "tokenizer": {
            "tokenizer_max_length": None,
            "tokenizer_name_or_path": config.get("tokenizer_name", "robot-test/dummy-tokenizer-wordlevel"),
            "tokenizer_revision": None,
        },
        "tokens": {
            "batch_accumulation_per_replica": 1,
            "limit_test_batches": 0,
            "limit_val_batches": 0,
            "micro_batch_size": int(config.get("micro_batch_size", 2)),
            "sequence_length": int(config.get("maximum_sequence_length", 256)),
            "train_steps": int(config.get("train_steps", 10000)),
            "val_check_interval": -1,
        },
    }

    return nanotron_config


def run_nanotron():
    # Create the Nanotron configuration
    nanotron_config = generate_nanotron_config(config)

    # Save the configuration to a YAML file
    output_dir = os.path.join(os.getcwd(), "nanotron_output")
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "nanotron_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(nanotron_config, f, default_flow_style=False)

    print(f"Generated Nanotron configuration at: {config_path}")

    # Get the number of GPUs to use
    if config.get("gpu_ids") and config["gpu_ids"].lower() != "auto":
        # Use specified GPU IDs
        gpu_ids = config["gpu_ids"].split(",")
        num_gpus = len(gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_ids"]
    else:
        # Get GPU count from URL or local detection
        num_gpus = get_gpu_count_from_url(config.get("gpu_info_url"))

    # Create run_train.py script
    run_train_path = os.path.join(
        os.environ["_TFL_WORKSPACE_DIR"], "plugins", "nanotron_pretrainer", "nanotron", "run_train.py"
    )
    #     with open(run_train_path, "w") as f:
    #         f.write("""
    # import os
    # import sys
    # import subprocess

    # def main():
    #     # Get the config file path from command line arguments
    #     config_file = sys.argv[sys.argv.index('--config-file')+1] if '--config-file' in sys.argv else None
    #     if not config_file:
    #         print("Error: Config file not specified")
    #         sys.exit(1)

    #     # Run Nanotron training
    #     subprocess.run(["python", "-m", "nanotron.scripts.train", "--config", config_file])

    # if __name__ == "__main__":
    #     main()
    # """)

    #     # Set executable permission
    #     os.chmod(run_train_path, 0o755)

    # Run training with torchrun
    env = os.environ.copy()
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        run_train_path,
        "--config-file",
        config_path,
    ]

    print(f"Running Nanotron with command: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)


run_nanotron()
