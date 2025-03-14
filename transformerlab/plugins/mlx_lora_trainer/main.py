"""
Fine-Tuning with LoRA or QLoRA using MLX

https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md
"""

import json
import yaml
import re
import subprocess
import sys
import os
from tensorboardX import SummaryWriter
from jinja2 import Environment

# Import tfl_trainer from the SDK
from transformerlab.tfl_decorators import tfl_trainer
from transformerlab.plugin import WORKSPACE_DIR, generate_model_json


@tfl_trainer.job_wrapper(progress_start=0, progress_end=100)
def train_mlx_lora():
    jinja_environment = Environment()
    plugin_dir = os.path.dirname(os.path.realpath(__file__))
    print("Plugin dir:", plugin_dir)

    # Extract configuration parameters
    lora_layers = getattr(tfl_trainer, "lora_layers", "16")
    learning_rate = getattr(tfl_trainer, "learning_rate", "5e-5")
    batch_size = str(getattr(tfl_trainer, "batch_size", "4"))
    steps_per_eval = str(getattr(tfl_trainer, "steps_per_eval", "200"))
    iters = getattr(tfl_trainer, "iters", "1000")
    adaptor_name = getattr(tfl_trainer, "adaptor_name", "default")
    fuse_model = getattr(tfl_trainer, "fuse_model", True)

    # Check if LoRA parameters are set
    lora_rank = getattr(tfl_trainer, "lora_rank", None)
    lora_alpha = getattr(tfl_trainer, "lora_alpha", None)

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

    # Setup wandb and tensorboard logging
    use_wandb = "wandb" in tfl_trainer.report_to
    if use_wandb:
        try:
            import wandb

            # Initialize wandb with proper configuration
            wandb_config = {
                "model": tfl_trainer.model_name,
                "lora_layers": lora_layers,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "steps_per_eval": steps_per_eval,
                "iters": iters,
                "adaptor_name": adaptor_name,
            }
            if lora_rank:
                wandb_config["lora_rank"] = lora_rank
            if lora_alpha:
                wandb_config["lora_alpha"] = lora_alpha

            wandb_run = wandb.init(project="MLX_LoRA_Training", config=wandb_config)
            print("Weights & Biases logging enabled.")
        except ImportError:
            print("wandb package not found. Continuing without W&B logging.")
            use_wandb = False
        except Exception as e:
            print(f"Error initializing wandb: {e}. Continuing without W&B logging.")
            use_wandb = False

    # Load the dataset using tfl_trainer
    datasets = tfl_trainer.load_dataset(["train", "valid"])

    # Directory for storing temporary working files
    data_directory = f"{WORKSPACE_DIR}/plugins/mlx_lora_trainer/data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Go over each dataset split and render a new file based on the template
    formatting_template = jinja_environment.from_string(tfl_trainer.formatting_template)

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
    adaptor_output_dir = getattr(tfl_trainer, "adaptor_output_dir", "")
    if adaptor_output_dir == "" or adaptor_output_dir is None:
        adaptor_output_dir = os.path.join(WORKSPACE_DIR, "adaptors", tfl_trainer.model_name, adaptor_name)
        print("Using default adaptor output directory:", adaptor_output_dir)
    if not os.path.exists(adaptor_output_dir):
        os.makedirs(adaptor_output_dir)

    # Prepare the command for MLX LoRA training
    popen_command = [
        sys.executable,
        "-um",
        "mlx_lm.lora",
        "--model",
        tfl_trainer.model_name,
        "--iters",
        iters,
        "--train",
        "--adapter-path",
        adaptor_output_dir,
        "--num-layers",
        lora_layers,
        "--batch-size",
        batch_size,
        "--learning-rate",
        learning_rate,
        "--data",
        data_directory,
        "--steps-per-report",
        getattr(tfl_trainer, "steps_per_report", "10"),
        "--steps-per-eval",
        steps_per_eval,
        "--save-every",
        getattr(tfl_trainer, "save_every", "100"),
    ]

    # If a config file has been created then include it
    if config_file:
        popen_command.extend(["--config", config_file])

    print("Running command:")
    print(popen_command)

    print("Training beginning:")
    print("Adaptor will be saved in:", adaptor_output_dir)

    # Setup tensorboard writer
    output_dir = tfl_trainer.tensorboard_output_dir
    writer = SummaryWriter(output_dir)
    print("Writing logs to:", output_dir)

    # Run the MLX LoRA training process
    with subprocess.Popen(
        popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True
    ) as process:
        for line in process.stdout:
            # Parse progress from output
            pattern = r"Iter (\d+):"
            match = re.search(pattern, line)
            if match:
                iteration = int(match.group(1))
                percent_complete = float(iteration) / float(iters) * 100
                print("Progress: ", f"{percent_complete:.2f}%")
                tfl_trainer.progress_update(percent_complete)

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

                    writer.add_scalar("train/loss", loss, iteration)
                    writer.add_scalar("train/it_per_sec", it_per_sec, iteration)
                    writer.add_scalar("train/tokens_per_sec", tokens_per_sec, iteration)
                    # Add wandb logging if enabled
                    if use_wandb:
                        wandb_run.log(
                            {
                                "train/loss": loss,
                                "train/it_per_sec": it_per_sec,
                                "train/tokens_per_sec": tokens_per_sec,
                                "iteration": iteration,
                            }
                        )

                # Parse validation metrics
                else:
                    pattern = r"Val loss (\d+\.\d+), Val took (\d+\.\d+)s"
                    match = re.search(pattern, line)
                    if match:
                        validation_loss = float(match.group(1))
                        print("Validation Loss: ", validation_loss)
                        writer.add_scalar("valid/loss", validation_loss, iteration)
                        wandb_run.log({"valid/loss": validation_loss, "iteration": iteration})

            print(line, end="", flush=True)

    # Check if the training process completed successfully
    if process.returncode and process.returncode != 0:
        print("An error occured before training completed.")
        raise RuntimeError("Training failed.")

    # At the end of the function before returning
    if use_wandb:
        wandb.finish()
    print("Finished training.")

    # Fuse the model with the base model if requested
    if not fuse_model:
        print(f"Adaptor training complete and saved at {adaptor_output_dir}.")
        return True
    else:
        print("Now fusing the adaptor with the model.")

        model_name = tfl_trainer.model_name
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        fused_model_name = f"{model_name}_{adaptor_name}"
        fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)

        # Make the directory to save the fused model
        if not os.path.exists(fused_model_location):
            os.makedirs(fused_model_location)

        fuse_popen_command = [
            sys.executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            tfl_trainer.model_name,
            "--adapter-path",
            adaptor_output_dir,
            "--save-path",
            fused_model_location,
        ]

        with subprocess.Popen(
            fuse_popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)

            return_code = process.wait()

            if return_code == 0:
                json_data = {
                    "description": f"An MLX model trained and generated by Transformer Lab based on {tfl_trainer.model_name}"
                }
                generate_model_json(fused_model_name, "MLX", json_data=json_data)
                print("Finished fusing the adaptor with the model.")
                return True
            else:
                print("Fusing model with adaptor failed: ", return_code)
                raise RuntimeError(f"Model fusion failed with return code {return_code}")


train_mlx_lora()
