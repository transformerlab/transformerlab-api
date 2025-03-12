import os
import subprocess
import sys
import requests
import yaml
import re
import time
from tensorboardX import SummaryWriter
import torch

from transformerlab.tfl_decorators import tfl_trainer


def get_gpu_count():
    """Get the number of available GPUs using PyTorch"""
    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception as e:
        print(f"Failed to get GPU count using PyTorch: {e}")
    
    print("Could not determine GPU count, defaulting to 1")
    return 1

def generate_nanotron_config():
    """
    Generate YAML configuration for Nanotron from input parameters
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        dict: Complete Nanotron configuration
    """
    # Format the run name with job ID
    run_name = getattr(tfl_trainer, "template_name", "nanotron_run") + "_" + str(tfl_trainer.job_id)
    checkpoint_path = os.path.join(
        os.environ.get("_TFL_WORKSPACE_DIR", "."), "models", "pretrained", run_name, "checkpoints"
    )

    # Create the config dictionary
    nanotron_config = {
        "checkpoints": {
            "checkpoint_interval": int(getattr(tfl_trainer, "checkpoint_interval", 1000)),
            "checkpoints_path": checkpoint_path,
            "checkpoints_path_is_shared_file_system": False,
            "resume_checkpoint_path": None,
            "save_initial_state": False,
            "save_final_state": True,
        },
        "data_stages": [
            {
                "data": {
                    "dataset": {
                        "dataset_overwrite_cache": False,
                        "dataset_processing_num_proc_per_process": 1,
                        "hf_dataset_config_name": None,
                        "hf_dataset_or_datasets": getattr(tfl_trainer, "dataset_name", "stas/openwebtext-10k"),
                        "hf_dataset_splits": getattr(tfl_trainer, "dataset_split", "train"),
                        "text_column_name": getattr(tfl_trainer, "text_column_name", "text"),
                    },
                    "num_loading_workers": 1,
                    "seed": int(getattr(tfl_trainer, "seed", 42)),
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
                        "hf_dataset_or_datasets": getattr(tfl_trainer, "dataset_name", "stas/openwebtext-10k"),
                        "hf_dataset_splits": getattr(tfl_trainer, "dataset_split", "train"),
                        "text_column_name": getattr(tfl_trainer, "text_column_name", "text"),
                    },
                    "num_loading_workers": 1,
                    "seed": int(getattr(tfl_trainer, "seed", 42)),
                },
                "name": "Annealing Phase",
                "start_training_step": int(getattr(tfl_trainer, "annealing_start_step", 10)),
            },
        ],
        "general": {
            "benchmark_csv_path": None,
            "consumed_train_samples": None,
            "ignore_sanity_checks": True,
            "project": "TFL_Pretraining",
            "run": run_name,
            "seed": int(getattr(tfl_trainer, "seed", 42)),
            "step": None,
        },
        "lighteval": None,
        "logging": {"iteration_step_info_interval": 1, "log_level": "info", "log_level_replica": "info"},
        "model": {
            "ddp_bucket_cap_mb": 25,
            "dtype": getattr(tfl_trainer, "mixed_precision", "bfloat16"),
            "init_method": {"std": 0.025},
            "make_vocab_size_divisible_by": 1,
            "model_config": {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_act": "silu",
                "hidden_size": int(getattr(tfl_trainer, "model_hidden_size", 16)),
                "initializer_range": 0.02,
                "intermediate_size": int(getattr(tfl_trainer, "model_intermediate_size", 64)),
                "is_llama_config": True,
                "max_position_embeddings": int(getattr(tfl_trainer, "maximum_sequence_length", 256)),
                "num_attention_heads": int(getattr(tfl_trainer, "model_num_attention_heads", 4)),
                "num_hidden_layers": int(getattr(tfl_trainer, "model_num_layers", 2)),
                "num_key_value_heads": int(getattr(tfl_trainer, "model_num_key_value_heads", 4)),
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
                "learning_rate": float(getattr(tfl_trainer, "learning_rate", 5e-4)),
                "lr_decay_starting_step": None,
                "lr_decay_steps": int(getattr(tfl_trainer, "train_steps", 10000)) - int(getattr(tfl_trainer, "warmup_steps", 2)),
                "lr_decay_style": "cosine",
                "lr_warmup_steps": int(getattr(tfl_trainer, "warmup_steps", 2)),
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
            "weight_decay": float(getattr(tfl_trainer, "weight_decay", 0.01)),
            "zero_stage": 0,
        },
        "parallelism": {
            "dp": int(getattr(tfl_trainer, "data_parallel_size", 2)),
            "expert_parallel_size": 1,
            "pp": int(getattr(tfl_trainer, "pipeline_parallel_size", 1)),
            "pp_engine": "1f1b",
            "tp": int(getattr(tfl_trainer, "tensor_parallel_size", 1)),
            "tp_linear_async_communication": True,
            "tp_mode": "REDUCE_SCATTER",
        },
        "profiler": None,
        "tokenizer": {
            "tokenizer_max_length": None,
            "tokenizer_name_or_path": getattr(tfl_trainer, "tokenizer_name", "robot-test/dummy-tokenizer-wordlevel"),
            "tokenizer_revision": None,
        },
        "tokens": {
            "batch_accumulation_per_replica": 1,
            "limit_test_batches": 0,
            "limit_val_batches": 0,
            "micro_batch_size": int(getattr(tfl_trainer, "micro_batch_size", 2)),
            "sequence_length": int(getattr(tfl_trainer, "maximum_sequence_length", 256)),
            "train_steps": int(getattr(tfl_trainer, "train_steps", 10000)),
            "val_check_interval": -1,
        },
    }

    return nanotron_config

@tfl_trainer.job_wrapper(progress_start=0, progress_end=100)
def train_model():
    """Main training function using TrainerTFLPlugin"""
    
    # Create the Nanotron configuration
    nanotron_config = generate_nanotron_config()

    # Set up output paths
    run_name = getattr(tfl_trainer, "template_name", "nanotron_run") + "_" + str(tfl_trainer.job_id)
    
    # Create output directories
    output_path = os.path.join(
        os.environ.get("_TFL_WORKSPACE_DIR", "."), "models", "pretrained", run_name, "nanotron_config_files"
    )
    os.makedirs(output_path, exist_ok=True)
    
    # Save the configuration to a YAML file
    config_path = os.path.join(output_path, f"{run_name}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(nanotron_config, f, default_flow_style=False)
    
    # Setting up tensorboard
    output_dir = os.path.join(tfl_trainer.output_dir, f"job_{tfl_trainer.job_id}_{run_name}")
    writer = SummaryWriter(output_dir)
    print("Writing logs to:", output_dir)
    
    # Store the tensorboard output dir in the job
    tfl_trainer.add_job_data("tensorboard_output_dir", output_dir)
    
    print(f"Generated Nanotron configuration at: {config_path}")
    
    # Get the number of GPUs to use
    if getattr(tfl_trainer, "gpu_ids") and tfl_trainer.gpu_ids.lower() != "auto":
        # Use specified GPU IDs
        gpu_ids = tfl_trainer.gpu_ids.split(",")
        num_gpus = len(gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = tfl_trainer.gpu_ids
    else:
        # Get GPU count
        num_gpus = get_gpu_count()
    
    # Create run_train.py script
    run_train_path = os.path.join(
        os.environ["_TFL_WORKSPACE_DIR"], "plugins", "nanotron_pretrainer", "nanotron", "run_train.py"
    )
    
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
    
    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1
    )
    
    # Process output line by line
    for line in iter(process.stdout.readline, ""):
        print(line.rstrip())  # Echo the output
        
        # Look for iteration information
        if "[INFO|" in line and "iteration:" in line:
            try:
                # Extract iteration information using regex
                iteration_match = re.search(r"iteration: (\d+) / (\d+)", line)
                if iteration_match:
                    current_iter = int(iteration_match.group(1))
                    total_iter = int(iteration_match.group(2))
                    
                    # Calculate progress as percentage
                    progress_percentage = min(100, (current_iter / total_iter) * 100)
                    
                    # Update job progress
                    tfl_trainer.progress_update(progress_percentage)
                    
                    # Extract other metrics for TensorBoard
                    # Loss
                    loss_match = re.search(r"lm_loss: ([\d.]+)", line)
                    if loss_match:
                        loss_value = float(loss_match.group(1))
                        writer.add_scalar("train/loss", loss_value, current_iter)
                    
                    # Learning rate
                    lr_match = re.search(r"lr: ([\d.e\-]+)", line)
                    if lr_match:
                        lr_value = float(lr_match.group(1))
                        writer.add_scalar("train/learning_rate", lr_value, current_iter)
                    
                    # Tokens per second
                    tps_match = re.search(r"tokens_per_sec: ([\d.]+)K", line)
                    if tps_match:
                        tps_value = float(tps_match.group(1)) * 1000  # Convert K to actual value
                        writer.add_scalar("system/tokens_per_sec", tps_value, current_iter)
                    
                    # Gradient norm
                    grad_norm_match = re.search(r"grad_norm: ([\d.]+)", line)
                    if grad_norm_match:
                        grad_norm_value = float(grad_norm_match.group(1))
                        writer.add_scalar("train/gradient_norm", grad_norm_value, current_iter)
                    
                    # Hardware TFLOPS per GPU
                    tflops_match = re.search(r"hardware_tflops_per_gpu: ([\d.]+)", line)
                    if tflops_match:
                        tflops_value = float(tflops_match.group(1))
                        writer.add_scalar("system/tflops_per_gpu", tflops_value, current_iter)
                    
            except Exception as e:
                print(f"Error parsing progress: {e}")
    
    # Wait for process to complete
    process.wait()
    
    # Ensure we mark the job as 100% complete when done
    tfl_trainer.job.update_progress(100)
    
    # Convert Nanotron checkpoint to HF format
    checkpoint_path = os.path.join(
        os.environ.get("_TFL_WORKSPACE_DIR", "."), "models", "pretrained", run_name, "checkpoints"
    )
    try:
        with open(os.path.join(checkpoint_path, "latest.txt"), "r") as f:
            latest_checkpoint = f.read().strip()
        
        save_path = os.path.join(
            os.environ.get("_TFL_WORKSPACE_DIR", "."), "models", run_name,
        )
        latest_checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
        print("Latest checkpoint path:", latest_checkpoint_path)
        print("Save path:", save_path)
        
        convert_script_path = os.path.join(
            os.environ["_TFL_WORKSPACE_DIR"], "plugins", "nanotron_pretrainer", "convert_nanotron_to_hf.py"
        )
        
        cmd_convert = [
            "torchrun",
            "--nproc_per_node=1",
            convert_script_path,
            "--checkpoint_path",
            latest_checkpoint_path,
            "--save_path",
            save_path,
            "--tokenizer_name",
            getattr(tfl_trainer, "tokenizer_name", "robot-test/dummy-tokenizer-wordlevel"),
        ]
        
        process_convert = subprocess.Popen(
            cmd_convert, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1
        )
        
        for line in iter(process_convert.stdout.readline, ""):
            print(line.rstrip())  # Echo the output
        
        process_convert.wait()
        
        # Import Model in Transformer Lab
        response = requests.get(
            "http://localhost:8338/model/import_from_local_path", 
            params={"model_path": save_path}
        )
        
        if response.status_code != 200:
            print(f"Failed to import model to Transformer Lab: {response.text}")
            return f"Training succeeded but model import failed: {response.text}"
            
    except Exception as e:
        print(f"Error during model conversion: {e}")
        return f"Training succeeded but model conversion failed: {str(e)}"
    
    return "Nanotron training completed successfully"



train_model()