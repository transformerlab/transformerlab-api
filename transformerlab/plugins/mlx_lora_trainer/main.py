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
import re
from string import Template
import subprocess
import sys
import time
from datasets import load_dataset
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
parser.add_argument('--input_file', type=str)
args, unknown = parser.parse_known_args()

print("Arguments:")
print(args)

input_config = None
# open the input file that provides configs
with open(args.input_file) as json_file:
    input_config = json.load(json_file)
config = input_config["config"]
print("Input:")
print(json.dumps(input_config, indent=4))

lora_layers = config["lora_layers"]
learning_rate = config["learning_rate"]
batch_size = config.get("batch_size", 4)
steps_per_eval = config.get("steps_per_eval", 200)
iters = config["iters"]

# we need to adapter parameter so set a default
adaptor_name = config.get('adaptor_name', "default")

# Get the dataset
try:
    dataset_target = transformerlab.plugin.get_dataset_path(
        config["dataset_name"])
except Exception as e:
    print(e)
    exit

dataset_types = ["train", "test"]
dataset = {}
formatting_template = jinja_environment.from_string(
    config["formatting_template"])

for dataset_type in dataset_types:

    # Load dataset
    try:
        dataset[dataset_type] = load_dataset(
            dataset_target, split=dataset_type, trust_remote_code=True)

    except ValueError as e:
        # This is to catch this error-> ValueError: Unknown split "test". Should be one of ['train']
        # Generally that means there is a single file in the dataset and we're trying to make a test dataset
        # So we're going to ignore that! (Unless we're trying to load the train dataset...check that)
        if (dataset_target == "train"):
            raise

        print(f"Continuing without any data for \"{dataset_type}\" slice.")
        # print(">", str(e))
        continue

    print(
        f"Loaded {dataset_type} dataset with {len(dataset[dataset_type])} examples.")

    # Directory for storing temporary working files
    data_directory = f"{WORKSPACE_DIR}/plugins/mlx_lora_trainer/data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # output training files in templated format in to data directory
    with open(f"{data_directory}/{dataset_type}.jsonl", "w") as f:
        for i in range(len(dataset[dataset_type])):
            data_line = dataset[dataset_type][i]
            data_line = dict(data_line)
            line = formatting_template.render(data_line)
            # convert line breaks to "\n" so that the jsonl file is valid
            line = line.replace("\n", "\\n")
            line = line.replace("\r", "\\r")
            o = {"text": line}
            f.write(json.dumps(o) + "\n")
            # trimming dataset as a hack, to reduce training time"
            # if (i > 40):
            #     break

# copy file test.jsonl to valid.jsonl. Our test set is the same as our validation set.
os.system(
    f"cp {WORKSPACE_DIR}/plugins/mlx_lora_trainer/data/test.jsonl {WORKSPACE_DIR}/plugins/mlx_lora_trainer/data/valid.jsonl")

print("Example formatted training example:")
example = formatting_template.render(dataset["train"][1])
print(example)

# TODO: For now create adapter in the plugin directory but this should probably go somewhere else
adaptor_output_dir = plugin_dir
# adaptor_output_dir = config["adaptor_output_dir"]
# if not os.path.exists(adaptor_output_dir):
#     os.makedirs(adaptor_output_dir)

popen_command = [sys.executable, "-m", "mlx_lm.lora",
                 "--model", config["model_name"],
                 "--iters", iters,
                 "--train",
                 "--adapter-path", adaptor_output_dir,
                 "--lora-layers", lora_layers,
                 "--batch-size", batch_size,
                 "--learning-rate", learning_rate,
                 "--data", os.path.join(plugin_dir, "data"),
                 "--steps-per-report", config['steps_per_report'],
                 "--steps-per-eval", steps_per_eval,
                 "--save-every", config["save_every"]]

print("Running command:")
print(popen_command)

job = transformerlab.plugin.Job(config["job_id"])
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
        popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:
    for line in process.stdout:
        # Each output line from lora.py looks like
        # "Iter 190: Train loss 1.997, It/sec 0.159, Tokens/sec 103.125"
        pattern = r"Iter (\d+):"
        match = re.search(pattern, line)
        if match:
            first_number = match.group(1)
            percent_complete = float(first_number) / float(iters) * 100
            print("Progress: ", f"{percent_complete:.2f}%")
            # print(percent_complete, ' ', config["job_id"])
            job.update_progress(percent_complete)

            if job.should_stop:
                print("Stopping job because of user interruption.")
                job.update_status("STOPPED")
                process.terminate()

            # Now parse the rest of the line and write to tensorboard
            pattern = r"Train loss (\d+\.\d+), It/sec (\d+\.\d+), Tokens/sec (\d+\.\d+)"
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                it_per_sec = float(match.group(2))
                tokens_per_sec = float(match.group(3))
                print("Loss: ", loss)
                print("It/sec: ", it_per_sec)
                print("Tokens/sec: ", tokens_per_sec)
                # The code snippet `with w.as_default(): tf.summary.scalar` is using TensorFlow's
                # `tf.summary.scalar` function to log scalar values to a TensorBoard summary writer
                # `w`.
                writer.add_scalar("loss", loss, int(first_number))
                writer.add_scalar("it_per_sec", it_per_sec, int(first_number))
                writer.add_scalar("tokens_per_sec",
                                  tokens_per_sec, int(first_number))

        print(line, end="", flush=True)

# Check if the training process completed successfully
# Terminate if not
if process.returncode and process.returncode != 0:
    print("An error occured before training completed.")
    exit(process.returncode)

print("Finished training.")

# TIME TO FUSE THE MODEL WITH THE BASE MODEL

print("Now fusing the adaptor with the model.")

model_name = config['model_name']
if "/" in model_name:
    model_name = model_name.split("/")[-1]
fused_model_name = f"{model_name}_{adaptor_name}"
fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)

# Make the directory to save the fused model
if not os.path.exists(fused_model_location):
    os.makedirs(fused_model_location)

fuse_popen_command = [
    sys.executable, "-m", "mlx_lm.fuse",
    "--model", config["model_name"],
    "--adapter-path", adaptor_output_dir,
    "--save-path", fused_model_location]

with subprocess.Popen(
        fuse_popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:
    for line in process.stdout:
        print(line, end="", flush=True)

    return_code = process.wait()

    # If model create was successful, create an info.json file so this can be read by the system
    print("Return code: ", return_code)
    if (return_code == 0):
        json_data = {
            "description": f"An MLX model trained and generated by TransformerLab based on {config['model_name']}"
        }
        transformerlab.plugin.generate_model_json(
            fused_model_name, "MLX", json_data=json_data)
        print("Finished fusing the adaptor with the model.")

    else:
        print("Fusing model with adaptor failed: ", return_code)
