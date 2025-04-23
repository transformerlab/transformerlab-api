"""
Fine-Tuning with LoRA or QLoRA using MLX

https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md
"""

import json
import yaml
import re
import subprocess
import os
import copy
from jinja2 import Environment

# Import tlab_trainer from the SDK
from transformerlab.sdk.v1.train import tlab_trainer
from transformerlab.plugin import WORKSPACE_DIR

from transformerlab.plugin import get_python_executable


def get_python_executable(plugin_dir):
    """Check if a virtual environment exists and return the appropriate Python executable"""
    # Check for virtual environment in the plugin directory
    venv_path = os.path.join(plugin_dir, "venv")

    if os.path.isdir(venv_path):
        print("Virtual environment found, using it for evaluation...")
        # Determine the correct path to the Python executable based on the platform
        python_executable = os.path.join(venv_path, "bin", "python")

        if os.path.exists(python_executable):
            return python_executable

    # Fall back to system Python if venv not found or executable doesn't exist
    print("No virtual environment found, using system Python...")
    return sys.executable


def train_function(**params):
    """Train a model with given parameters and return metrics"""
    jinja_environment = Environment()
    plugin_dir = os.path.dirname(os.path.realpath(__file__))
    python_executable = get_python_executable(plugin_dir)

    # Extract configuration parameters
    lora_layers = params.get("lora_layers", "16")
    learning_rate = params.get("learning_rate", "5e-5")
    batch_size = str(params.get("batch_size", "4"))
    steps_per_eval = str(params.get("steps_per_eval", "200"))
    iters = params.get("iters", "1000")
    adaptor_name = params.get("adaptor_name", "default")
    fuse_model = params.get("fuse_model", True)

    # Check if LoRA parameters are set
    lora_rank = params.get("lora_rank", None)
    lora_alpha = params.get("lora_alpha", None)

    # LoRA parameters have to be passed in a config file
    config_file = None
    if lora_rank or lora_alpha:
        config_file = os.path.join(plugin_dir, "config.yaml")
        with open(config_file, "w") as file:
            # It looks like the MLX code doesn't actually read the alpha parameter!
            # Instead it uses another parameter called scale to imply alpha
            # scale = alpha / rank
            lora_scale = int(lora_alpha) / int(lora_rank) if lora_alpha and lora_rank else 1

            lora_config = {}
            lora_config["lora_parameters"] = {}
            lora_config["lora_parameters"]["alpha"] = lora_alpha
            lora_config["lora_parameters"]["rank"] = lora_rank
            lora_config["lora_parameters"]["scale"] = lora_scale
            lora_config["lora_parameters"]["dropout"] = 0
            yaml.dump(lora_config, file)
            print("LoRA config:")
            print(lora_config)

    # Load the dataset
    datasets = params.get("datasets", None)
    if not datasets:
        datasets = tlab_trainer.load_dataset(["train", "valid"])

    # Directory for storing temporary working files
    data_directory = f"{WORKSPACE_DIR}/plugins/mlx_lora_trainer/data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Go over each dataset split and render a new file based on the template
    formatting_template = jinja_environment.from_string(params.get("formatting_template"))

    for split_name in datasets:
        # Get the dataset from our loaded datasets
        dataset_split = datasets[split_name]

        print(f"Processing {split_name} dataset with {len(dataset_split)} examples.")

        # Output training files in templated format into data directory
        with open(f"{data_directory}/{split_name}.jsonl", "w") as f:
            for i in range(len(dataset_split)):
                data_line = dict(dataset_split[i])
                line = formatting_template.render(data_line)
                # Convert line breaks to "\n" so that the jsonl file is valid
                line = line.replace("\n", "\\n")
                line = line.replace("\r", "\\r")
                o = {"text": line}
                f.write(json.dumps(o) + "\n")

    print("Example formatted training example:")
    example = formatting_template.render(dict(datasets["train"][0]))
    print(example)

    # Set output directory for the adaptor
    adaptor_output_dir = params.get("adaptor_output_dir", "")
    if adaptor_output_dir == "" or adaptor_output_dir is None:
        adaptor_output_dir = os.path.join(WORKSPACE_DIR, "adaptors", params.get("model_name"), adaptor_name)
        print("Using default adaptor output directory:", adaptor_output_dir)
    if not os.path.exists(adaptor_output_dir):
        os.makedirs(adaptor_output_dir)

    # Get Python executable (from venv if available)
    python_executable = get_python_executable(plugin_dir)
    env = os.environ.copy()
    env["PATH"] = python_executable.replace("/python", ":") + env["PATH"]

    # Prepare the command for MLX LoRA training
    popen_command = [
        python_executable,
        "-um",
        "mlx_lm.lora",
        "--model",
        params.get("model_name"),
        "--iters",
        str(iters),
        "--train",
        "--adapter-path",
        adaptor_output_dir,
        "--num-layers",
        str(lora_layers),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--data",
        data_directory,
        "--steps-per-report",
        str(params.get("steps_per_report", "10")),
        "--steps-per-eval",
        str(steps_per_eval),
        "--save-every",
        str(params.get("save_every", "100")),
    ]

    # If a config file has been created then include it
    if config_file:
        popen_command.extend(["--config", config_file])

    print("Running command:")
    print(popen_command)

    print("Training beginning:")
    print("Adaptor will be saved in:", adaptor_output_dir)

    # Track metrics
    metrics = {}

    # Run the MLX LoRA training process
    with subprocess.Popen(
        popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, env=env
    ) as process:
        for line in process.stdout:
            # Parse progress from output
            pattern = r"Iter (\d+):"
            match = re.search(pattern, line)
            if match:
                iteration = int(match.group(1))
                percent_complete = float(iteration) / float(iters) * 100
                print("Progress: ", f"{percent_complete:.2f}%")
                tlab_trainer.progress_update(percent_complete)

                # Parse training metrics
                pattern = (
                    r"Train loss (\d+\.\d+), Learning Rate (\d+\.[e\-\d]+), It/sec (\d+\.\d+), Tokens/sec (\d+\.\d+)"
                )
                match = re.search(pattern, line)
                if match:
                    loss = float(match.group(1))
                    it_per_sec = float(match.group(3))
                    tokens_per_sec = float(match.group(4))
                    print("Training Loss: ", loss)

                    tlab_trainer.log_metric("train/loss", loss, iteration)
                    tlab_trainer.log_metric("train/it_per_sec", it_per_sec, iteration)
                    tlab_trainer.log_metric("train/tokens_per_sec", tokens_per_sec, iteration)

                    # Store the latest metrics to return
                    metrics["train/loss"] = loss
                    metrics["train/it_per_sec"] = it_per_sec
                    metrics["train/tokens_per_sec"] = tokens_per_sec

                # Parse validation metrics
                else:
                    pattern = r"Val loss (\d+\.\d+), Val took (\d+\.\d+)s"
                    match = re.search(pattern, line)
                    if match:
                        validation_loss = float(match.group(1))
                        print("Validation Loss: ", validation_loss)
                        tlab_trainer.log_metric("valid/loss", validation_loss, iteration)

                        # Store validation loss for sweep optimization
                        metrics["valid/loss"] = validation_loss

            print(line, end="", flush=True)

    # Check if the training process completed successfully
    if process.returncode and process.returncode != 0:
        print("An error occurred before training completed.")
        raise RuntimeError("Training failed.")

    print("Finished training.")

    # Check if this is a sweep run
    is_sweep_run = "run_id" in params

    # Fuse the model with the base model if requested (skip during sweep runs)
    if fuse_model and not is_sweep_run:
        print("Now fusing the adaptor with the model.")

        model_name = params.get("model_name")
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        fused_model_name = f"{model_name}_{adaptor_name}"
        fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)

        # Make the directory to save the fused model
        if not os.path.exists(fused_model_location):
            os.makedirs(fused_model_location)

        fuse_popen_command = [
            python_executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            params.get("model_name"),
            "--adapter-path",
            adaptor_output_dir,
            "--save-path",
            fused_model_location,
        ]

        with subprocess.Popen(
            fuse_popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, env=env
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)

            return_code = process.wait()

            if return_code == 0:
                json_data = {
                    "description": f"An MLX model trained and generated by Transformer Lab based on {params.get('model_name')}"
                }
                tlab_trainer.create_transformerlab_model(
                    fused_model_name=fused_model_name, model_architecture="MLX", json_data=json_data
                )
                print("Finished fusing the adaptor with the model.")
            else:
                print("Fusing model with adaptor failed: ", return_code)
                raise RuntimeError(f"Model fusion failed with return code {return_code}")

    # Return metrics for sweep optimization
    return metrics


@tlab_trainer.job_wrapper(wandb_project_name="TLab_Training", manual_logging=True)
def run_plugin():
    """Main entry point for the training plugin"""
    tlab_trainer._ensure_args_parsed()

    # Initialize parameters
    tlab_trainer.params.lora_layers = tlab_trainer.params.get("lora_layers", "16")
    tlab_trainer.params.learning_rate = tlab_trainer.params.get("learning_rate", "5e-5")
    tlab_trainer.params.batch_size = tlab_trainer.params.get("batch_size", "4")
    tlab_trainer.params.steps_per_eval = tlab_trainer.params.get("steps_per_eval", "200")
    tlab_trainer.params.iters = tlab_trainer.params.get("iters", "1000")
    tlab_trainer.params.adaptor_name = tlab_trainer.params.get("adaptor_name", "default")
    tlab_trainer.params.fuse_model = tlab_trainer.params.get("fuse_model", True)
    tlab_trainer.params.lora_rank = tlab_trainer.params.get("lora_rank", None)
    tlab_trainer.params.lora_alpha = tlab_trainer.params.get("lora_alpha", None)

    # Determine if we're doing a sweep
    run_sweep = tlab_trainer.params.get("run_sweeps")
    tlab_trainer.params.sweep_metric = "valid/loss"
    tlab_trainer.params.train_final_model = True
    tlab_trainer.params.lower_is_better = True

    if run_sweep:
        # Run hyperparameter sweep
        sweep_results = tlab_trainer.run_sweep(train_function)

        # If there's a best config, train a final model with it
        if tlab_trainer.params.get("train_final_model", True) and sweep_results["best_config"]:
            print("\n--- Training final model with best configuration ---")

            # Create parameters with the best configuration
            final_params = copy.deepcopy(tlab_trainer.params)
            for k, v in sweep_results["best_config"].items():
                final_params[k] = v

            # Run final training
            result = train_function(**final_params)
            return {**sweep_results, "final_model_metrics": result}

        return sweep_results
    else:
        # Run single training
        return train_function(**tlab_trainer.params)


# Execute the plugin when imported
run_plugin()
