import json
import re
import subprocess
import os
import sys
import copy
from jinja2 import Environment

# Import the TrainerTLabPlugin
from transformerlab.sdk.v1.train import tlab_trainer


# Setup Jinja environment
jinja_environment = Environment()


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
    """Train model with given parameters and return metrics"""
    # Get parameters for training
    learning_rate = params.get("learning_rate", "2e-4")
    batch_size = str(params.get("batch_size", 4))
    num_train_epochs = str(params.get("num_train_epochs", 4))
    adaptor_name = params.get("adaptor_name", "default")

    # Add optional LoRA parameters if provided
    lora_r = str(params.get("lora_r", 8))
    lora_alpha = str(params.get("lora_alpha", 16))
    lora_dropout = str(params.get("lora_dropout", 0.05))

    # Generate a model name using the original model and the passed adaptor
    model_name = params.get("model_name")
    input_model_no_author = model_name.split("/")[-1]
    project_name = f"{input_model_no_author}-{adaptor_name}".replace(".", "")

    # For sweep runs, add run_id to project name
    if "run_id" in params:
        project_name = f"{project_name}-{params['run_id']}"

    # Setup directories
    plugin_dir = os.path.dirname(os.path.realpath(__file__))
    python_executable = get_python_executable(plugin_dir)
    data_directory = os.path.join(plugin_dir, "data")

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Get template from parameters
    formatting_template = jinja_environment.from_string(params.get("formatting_template"))

    # Load datasets (train, test)
    dataset_types = ["train", "test"]

    # Use provided datasets if available, otherwise load them
    datasets = params.get("datasets", None)
    if not datasets:
        try:
            datasets = tlab_trainer.load_dataset(dataset_types=dataset_types)
            dataset_types = datasets.keys()
        except Exception as e:
            raise e
    else:
        dataset_types = datasets.keys()

    for dataset_type in dataset_types:
        print(f"Processing {dataset_type} dataset with {len(datasets[dataset_type])} examples.")

        # Output training files in templated format
        with open(f"{data_directory}/{dataset_type}.jsonl", "w") as f:
            for i in range(len(datasets[dataset_type])):
                data_line = dict(datasets[dataset_type][i])
                line = formatting_template.render(data_line)

                # Escape newlines for jsonl format
                line = line.replace("\n", "\\n")
                line = line.replace("\r", "\\r")
                o = {"text": line}
                f.write(json.dumps(o) + "\n")

    # Copy test.jsonl to valid.jsonl (validation = test)
    os.system(f"cp {data_directory}/test.jsonl {data_directory}/valid.jsonl")

    print("Example formatted training example:")
    example = formatting_template.render(datasets["train"][1])
    print(example)

    # Set output directory for the adaptor
    adaptor_output_dir = params.get("adaptor_output_dir", "")
    if not adaptor_output_dir:
        adaptor_output_dir = os.path.join(os.getcwd(), project_name)

    # Prepare autotrain command
    popen_command = [
        python_executable,
        "-m",
        "autotrain",
        "llm",
        "--train",
        "--model",
        model_name,
        "--data-path",
        data_directory,
        "--lr",
        str(learning_rate),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(num_train_epochs),
        "--trainer",
        "sft",
        "--peft",
        "--lora_r",
        lora_r,
        "--lora_alpha",
        lora_alpha,
        "--lora_dropout",
        lora_dropout,
        "--auto_find_batch_size",
        "--project-name",
        project_name,
    ]

    # Only merge adapter for non-sweep runs or final model
    if "run_id" not in params:
        popen_command.append("--merge-adapter")

    print("Running command:")
    print(popen_command)

    print("Training beginning:")

    # Store metrics for return
    metrics = {}

    # Run the subprocess with output monitoring
    with subprocess.Popen(
        popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True
    ) as process:
        iteration = 0
        it_per_sec = 0
        percent_complete = 0

        for line in process.stdout:
            # Parse progress from output lines
            # Progress complete pattern
            pattern = r"\s*(\d+)\%\|.+?(?=\d+/)(\d+)/.+?(?=\d+.\d+s/it)(\d+.\d+)s/it"
            match = re.search(pattern, line)
            if match:
                percent_complete = int(match.group(1))
                iteration = int(match.group(2))
                it_per_sec = float(match.group(3))
                # Update progress in TransformerLab
                tlab_trainer.progress_update(percent_complete)

            # Parse metrics for logging
            pattern = r"INFO.+?{'loss': (\d+\.\d+), 'grad_norm': (\d+\.\d+), 'learning_rate': (\d+\.\d+), 'epoch': (\d+\.\d+)}"
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                grad_norm = float(match.group(2))
                learning_rate = float(match.group(3))
                epoch = float(match.group(4))

                print("Progress: ", f"{percent_complete}%")
                print("Iteration: ", iteration)
                print("It/sec: ", it_per_sec)
                print("Loss: ", loss)
                print("Epoch:", epoch)

                # Store metrics for sweep optimization
                metrics["train/loss"] = loss
                metrics["train/grad_norm"] = grad_norm
                metrics["train/it_per_sec"] = it_per_sec

                # Log metrics to tensorboard and wandb
                tlab_trainer.log_metric("train/loss", loss, iteration)
                tlab_trainer.log_metric("train/grad_norm", grad_norm, iteration)
                tlab_trainer.log_metric("train/it_per_sec", it_per_sec, iteration)
                tlab_trainer.log_metric("train/learning_rate", learning_rate, iteration)
                tlab_trainer.log_metric("train/epoch", epoch, iteration)

            # Parse validation metrics
            pattern = r"INFO.+?{'eval_loss': (\d+\.\d+), 'eval_epoch': (\d+\.\d+)}"
            match = re.search(pattern, line)
            if match:
                eval_loss = float(match.group(1))
                eval_epoch = float(match.group(2))

                print("Validation Loss: ", eval_loss)
                print("Validation Epoch:", eval_epoch)

                # Store validation metrics for sweep optimization
                metrics["eval/loss"] = eval_loss

                # Log validation metrics
                tlab_trainer.log_metric("eval/loss", eval_loss, iteration)

            # Print the output line
            print(line, end="", flush=True)

    # Check if the process completed successfully
    if process.returncode and process.returncode != 0:
        print("An error occurred during training.")
        raise RuntimeError("Training failed.")

    # Clean up and move model
    is_sweep_run = "run_id" in params
    if not is_sweep_run:
        try:
            # Remove autotrain data as it's not needed anymore
            os.system(f"rm -rf {project_name}/autotrain_data")
        except Exception as e:
            print(f"Failed to delete unnecessary data: {e}")

        try:
            # Move the model to the specified output directory
            os.system(f"mv {project_name} {adaptor_output_dir}/")
        except Exception as e:
            raise e

    print("Finished training.")
    return metrics


@tlab_trainer.job_wrapper(manual_logging=True)
def train_model():
    """Main entry point for the training plugin"""
    tlab_trainer._ensure_args_parsed()

    # Initialize parameters
    if hasattr(tlab_trainer.params, "_config"):
        config = tlab_trainer.params._config
        # Transfer config items to params for easier access
        for key, value in config.items():
            tlab_trainer.params[key] = value

    # Set default parameters
    tlab_trainer.params.learning_rate = tlab_trainer.params.get("learning_rate", "2e-4")
    tlab_trainer.params.batch_size = tlab_trainer.params.get("batch_size", 4)
    tlab_trainer.params.num_train_epochs = tlab_trainer.params.get("num_train_epochs", 4)
    tlab_trainer.params.adaptor_name = tlab_trainer.params.get("adaptor_name", "default")
    tlab_trainer.params.lora_r = tlab_trainer.params.get("lora_r", 8)
    tlab_trainer.params.lora_alpha = tlab_trainer.params.get("lora_alpha", 16)
    tlab_trainer.params.lora_dropout = tlab_trainer.params.get("lora_dropout", 0.05)

    # Determine if we're doing a sweep
    run_sweep = tlab_trainer.params.get("run_sweeps")
    tlab_trainer.params.sweep_metric = "eval/loss"
    tlab_trainer.params.lower_is_better = True

    if run_sweep is not None:
        # Run hyperparameter sweep
        sweep_results = tlab_trainer.run_sweep(train_function)

        # If there's a best config, train a final model with it
        if tlab_trainer.params.get("train_final_model", True) and sweep_results["best_config"]:
            print("\n--- Training final model with best configuration ---")

            # Create parameters with the best configuration
            final_params = copy.deepcopy(tlab_trainer.params)
            for k, v in sweep_results["best_config"].items():
                final_params[k] = v

            final_params["train_final_model"] = True

            # Run final training
            result = train_function(**final_params)
            return {**sweep_results, "final_model_metrics": result}

        return sweep_results
    else:
        # Run single training
        return train_function(**tlab_trainer.params)


# Execute the plugin when imported
train_model()
