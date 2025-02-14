"""
Fine-Tuning with LoRA or QLoRA using MLX

https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md

You must install MLX python:
pip install mlx-lm

LoRA or QLoRA finetuning.

usage: mlx_lm.lora [-h] [--model MODEL] [--train] [--data DATA]
                   [--lora-layers LORA_LAYERS] [--batch-size BATCH_SIZE]
                   [--iters ITERS] [--val-batches VAL_BATCHES]
                   [--learning-rate LEARNING_RATE]
                   [--steps-per-report STEPS_PER_REPORT]
                   [--steps-per-eval STEPS_PER_EVAL]
                   [--resume-adapter-file RESUME_ADAPTER_FILE]
                   [--adapter-path ADAPTER_PATH] [--save-every SAVE_EVERY]
                   [--test] [--test-batches TEST_BATCHES]
                   [--max-seq-length MAX_SEQ_LENGTH] [-c CONFIG]
                   [--grad-checkpoint] [--seed SEED] [--use-dora]

LoRA or QLoRA finetuning.

options:
  -h, --help            show this help message and exit
  --model MODEL         The path to the local model directory or Hugging Face
                        repo.
  --train               Do training
  --data DATA           Directory with {train, valid, test}.jsonl files
  --lora-layers LORA_LAYERS
                        Number of layers to fine-tune. Default is 16, use -1
                        for all.
  --batch-size BATCH_SIZE
                        Minibatch size.
  --iters ITERS         Iterations to train for.
  --val-batches VAL_BATCHES
                        Number of validation batches, -1 uses the entire
                        validation set.
  --learning-rate LEARNING_RATE
                        Adam learning rate.
  --steps-per-report STEPS_PER_REPORT
                        Number of training steps between loss reporting.
  --steps-per-eval STEPS_PER_EVAL
                        Number of training steps between validations.
  --resume-adapter-file RESUME_ADAPTER_FILE
                        Load path to resume training with the given adapters.
  --adapter-path ADAPTER_PATH
                        Save/load path for the adapters.
  --save-every SAVE_EVERY
                        Save the model every N iterations.
  --test                Evaluate on the test set after training
  --test-batches TEST_BATCHES
                        Number of test set batches, -1 uses the entire test
                        set.
  --max-seq-length MAX_SEQ_LENGTH
                        Maximum sequence length.
  -c CONFIG, --config CONFIG
                        A YAML configuration file with the training options
  --grad-checkpoint     Use gradient checkpointing to reduce memory use.
  --seed SEED           The PRNG seed
  --use-dora            Use DoRA to finetune.
"""

import json
import yaml
import re
import subprocess
import sys
import time
from datasets import load_dataset, get_dataset_split_names
import argparse
import os
from tensorboardX import SummaryWriter

import transformerlab.plugin
from jinja2 import Environment

jinja_environment = Environment()

plugin_dir = os.path.dirname(os.path.realpath(__file__))
print("Plugin dir:", plugin_dir)

# A few things we need from Plugin SDK
WORKSPACE_DIR = transformerlab.plugin.WORKSPACE_DIR

# Get all parameters provided to this script from Transformer Lab
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args, unknown = parser.parse_known_args()

print("Arguments:")
print(args)

input_config = None
# open the input file that provides configs
with open(args.input_file) as json_file:
    input_config = json.load(json_file)
config = input_config["config"]
if "fuse_model" not in config:
    config["fuse_model"] = True
print("Input:")
print(json.dumps(input_config, indent=4))

lora_layers = config["lora_layers"]
learning_rate = config["learning_rate"]
batch_size = config.get("batch_size", 4)
steps_per_eval = config.get("steps_per_eval", 200)
iters = config["iters"]

# Check if LoRA parameters are set
lora_rank = config.get("lora_rank", None)
lora_alpha = config.get("lora_alpha", None)

job = transformerlab.plugin.Job(config["job_id"])

# LoRA parameters have to be passed in a config file
config_file = None
if lora_rank or lora_alpha:
    config_file = os.path.join(plugin_dir, "config.yaml")
    with open(config_file, "w") as file:
        # It looks like the MLX code doesn't actually read the alpha parameter!
        # Instead it uses another parameter called scale to imply alpha
        # scale = alpha / rank
        lora_scale = int(lora_alpha) / int(lora_rank)

        lora_config = {}
        lora_config["lora_parameters"] = {}
        lora_config["lora_parameters"]["alpha"] = lora_alpha
        lora_config["lora_parameters"]["rank"] = lora_rank
        lora_config["lora_parameters"]["scale"] = lora_scale
        lora_config["lora_parameters"]["dropout"] = 0
        yaml.dump(lora_config, file)
        print("LoRA config:")
        print(lora_config)

# we need to adapter parameter so set a default
adaptor_name = config.get("adaptor_name", "default")
fuse_model = config.get("fuse_model", None)


# Get the dataset
try:
    dataset_target = transformerlab.plugin.get_dataset_path(config["dataset_name"])
except Exception as e:
    print(e)
    job.set_job_completion_status("failed", "Could not find dataset.")
    exit(1)

# RENDER EACH DATASET SPLIT THROUGH THE SUPPLIED TEMPLATE

# We need both a "train" and a "valid" split
# If only a "train" split exists then manually carve off
# 80% train, 10% test, 10% valid
# TODO: Make this something you can customize via parameters
available_splits = get_dataset_split_names(dataset_target)

# Verify that we have required "train" split
if "train" not in available_splits:
    print(f"Error: Missing required train slice in dataset {dataset_target}.")
    job.set_job_completion_status("failed", "This training algorithm requires a split called 'train' in the dataset.")
    exit(1)

# And then either use provided "valid" split or create one
# FUN: Some datasets call it "valid", others call it "validation"
if "validation" in available_splits:
    dataset_splits = {"train": "train", "valid": "validation"}

elif "valid" in available_splits:
    dataset_splits = {"train": "train", "valid": "valid"}

else:
    print(f"No validation slice found in dataset {dataset_target}:")
    print("Using a default 80/10/10 split for training, test and valid.")
    dataset_splits = {"train": "train[:80%]", "valid": "train[-10%:]"}

dataset = {}
formatting_template = jinja_environment.from_string(config["formatting_template"])

# Directory for storing temporary working files
# TODO: This should probably be stored per job.
data_directory = f"{WORKSPACE_DIR}/plugins/mlx_lora_trainer/data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Go over each dataset split and render a new file based on the template
for split_name in dataset_splits:
    # Load dataset
    dataset[split_name] = load_dataset(dataset_target, split=dataset_splits[split_name], trust_remote_code=True)

    print(f"Loaded {split_name} dataset with {len(dataset[split_name])} examples.")

    # output training files in templated format in to data directory
    with open(f"{data_directory}/{split_name}.jsonl", "w") as f:
        for i in range(len(dataset[split_name])):
            data_line = dataset[split_name][i]
            data_line = dict(data_line)
            line = formatting_template.render(data_line)
            # convert line breaks to "\n" so that the jsonl file is valid
            line = line.replace("\n", "\\n")
            line = line.replace("\r", "\\r")
            o = {"text": line}
            f.write(json.dumps(o) + "\n")


print("Example formatted training example:")
example = formatting_template.render(dataset["train"][1])
print(example)

# TODO: For now create adapter in the plugin directory but this should probably go somewhere else
# adaptor_output_dir = plugin_dir
adaptor_output_dir = config["adaptor_output_dir"]
if adaptor_output_dir == "" or adaptor_output_dir is None:
    print("No adaptor output directory specified.")
    adaptor_output_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "adaptors", args.model_name, args.adaptor_name)
    print("Using default adaptor output directory:", adaptor_output_dir)
if not os.path.exists(adaptor_output_dir):
    os.makedirs(adaptor_output_dir)

popen_command = [
    sys.executable,
    "-um",
    "mlx_lm.lora",
    "--model",
    config["model_name"],
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
    os.path.join(plugin_dir, "data"),
    "--steps-per-report",
    config["steps_per_report"],
    "--steps-per-eval",
    steps_per_eval,
    "--save-every",
    config["save_every"],
]

# If a config file has been created then include it
if config_file:
    popen_command.extend(["--config", config_file])

print("Running command:")
print(popen_command)

job.update_progress(0)

print("Training beginning:")
print("Adaptor will be saved in:", adaptor_output_dir)

# todays date with seconds:
today = time.strftime("%Y%m%d-%H%M%S")

output_dir = os.path.join(config["output_dir"], today)
# w = tf.summary.create_file_writer(os.path.join(config["output_dir"], "logs"))
writer = SummaryWriter(output_dir)
print("Writing logs to:", output_dir)

# Store the tensorboard output dir in the job
job.set_tensorboard_output_dir(output_dir)

with subprocess.Popen(
    popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True
) as process:
    for line in process.stdout:
        # Each output line from lora.py looks like
        # "Iter 190: ....."
        pattern = r"Iter (\d+):"
        match = re.search(pattern, line)
        if match:
            first_number = match.group(1)
            percent_complete = float(first_number) / float(iters) * 100
            print("Progress: ", f"{percent_complete:.2f}%")
            # print(percent_complete, ' ', config["job_id"])
            job.update_progress(percent_complete)

            if job.should_stop():
                print("Stopping job because of user interruption.")
                job.update_status("STOPPED")
                process.terminate()

            # Now parse the rest of the line and write to tensorboard
            # There are two types of output we are looking for:
            # 1. Training progress updates which look like:
            # "Iter 190: Train loss 1.997, It/sec 0.159, Tokens/sec 103.125"
            pattern = r"Train loss (\d+\.\d+), Learning Rate (\d+\.[e\-\d]+), It/sec (\d+\.\d+), Tokens/sec (\d+\.\d+)"
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                it_per_sec = float(match.group(3))
                tokens_per_sec = float(match.group(4))
                print("Training Loss: ", loss)
                print("It/sec: ", it_per_sec)
                print("Tokens/sec: ", tokens_per_sec)
                # The code snippet `with w.as_default(): tf.summary.scalar` is using TensorFlow's
                # `tf.summary.scalar` function to log scalar values to a TensorBoard summary writer
                # `w`.
                writer.add_scalar("loss", loss, int(first_number))
                writer.add_scalar("it_per_sec", it_per_sec, int(first_number))
                writer.add_scalar("tokens_per_sec", tokens_per_sec, int(first_number))

            # 2. Validation updates which look like:
            # "Iter 190: Val loss 1.009, Val took 1.696s"
            else:
                pattern = r"Val loss (\d+\.\d+), Val took (\d+\.\d+)s"
                match = re.search(pattern, line)
                if match:
                    validation_loss = float(match.group(1))
                    print("Validation Loss: ", validation_loss)
                    writer.add_scalar("validation-loss", validation_loss, int(first_number))

        print(line, end="", flush=True)

# Check if the training process completed successfully
# Terminate if not
if process.returncode and process.returncode != 0:
    print("An error occured before training completed.")
    job.set_job_completion_status("failed", "Failed during training.")
    exit(process.returncode)

print("Finished training.")

# TIME TO FUSE THE MODEL WITH THE BASE MODEL
if not fuse_model:
    print(f"Adaptor training complete and saved at {adaptor_output_dir}.")
    job.set_job_completion_status("success", "Adapter saved successfully.")

else:
    print("Now fusing the adaptor with the model.")

    model_name = config["model_name"]
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
        config["model_name"],
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

        # If model create was successful, create an info.json file so this can be read by the system
        print("Return code: ", return_code)
        if return_code == 0:
            json_data = {
                "description": f"An MLX model trained and generated by TransformerLab based on {config['model_name']}"
            }
            transformerlab.plugin.generate_model_json(fused_model_name, "MLX", json_data=json_data)
            print("Finished fusing the adaptor with the model.")
            job.set_job_completion_status("success", "Model fused successfully.")

        else:
            print("Fusing model with adaptor failed: ", return_code)
            job.set_job_completion_status("failed", "Model fusion failed.")
