"""
Fine-Tuning with LoRA or QLoRA using MLX

https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md

You must install MLX python:
pip install mlx-lm

LoRA or QLoRA finetuning.

options:
  -h, --help            show this help message and exit
  --model MODEL         The path to the local model directory or Hugging Face
                        repo.
  --max-tokens MAX_TOKENS, -m MAX_TOKENS
                        The maximum number of tokens to generate
  --temp TEMP           The sampling temperature
  --prompt PROMPT, -p PROMPT
                        The prompt for generation
  --train               Do training
  --data DATA           Directory with {train, valid, test}.jsonl files
  --lora-layers LORA_LAYERS
                        Number of layers to fine-tune
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
                        Load path to resume training with the given adapter
                        weights.
  --adapter-file ADAPTER_FILE
                        Save/load path for the trained adapter weights.
  --save-every SAVE_EVERY
                        Save the model every N iterations.
  --test                Evaluate on the test set after training
  --test-batches TEST_BATCHES
                        Number of test set batches, -1 uses the entire test
                        set.
  --seed SEED           The PRNG seed
"""

import json
import re
import sqlite3
from string import Template
import subprocess
import sys
import time
from datasets import load_dataset
import argparse
import os
from tensorboardX import SummaryWriter


root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
plugin_dir = os.path.dirname(os.path.realpath(__file__))
print("Plugin dir:", plugin_dir)

# Connect to the LLM Lab database
WORKSPACE_DIR = os.getenv("_TFL_WORKSPACE_DIR")
db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3")

# Get all parameters provided to this script from Transformer Lab
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--experiment_name', default='', type=str)
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
iters = config["iters"]

# we need to adapter parameter so set a default
adaptor_name = config.get('adaptor_name', "default")

# Get the dataset
# Datasets can be a huggingface ID or the name of a locally uploaded dataset
# Need to check the DB to figure out which because it changes how we load the dataset
# TODO: Refactor this to somehow simplify across training plugins
dataset_id = config["dataset_name"]
cursor = db.execute(
    "SELECT location FROM dataset WHERE dataset_id = ?", (dataset_id,))
row = cursor.fetchone()
cursor.close()

# if no rows exist then the dataset hasn't been installed!
if row is None:
    print(f"No dataset named {dataset_id} installed.")
    exit

# dataset_location will be either "local" or "huggingface"
# (and if it's something else we're going to treat "huggingface" as default)
dataset_location = row[0]

dataset_types = ["train", "test"]
dataset = {}
formatting_template = Template(config["formatting_template"])

for dataset_type in dataset_types:

    # Load dataset - if it's local then pass it the path to the dataset directory
    if (dataset_location == "local"):
        dataset_target = os.path.join(WORKSPACE_DIR, "datasets", dataset_id)

    # Otherwise assume it is a Huggingface ID
    else:
        dataset_target = dataset_id

    try:
        dataset[dataset_type] = load_dataset(
            dataset_target, split=dataset_type)

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
            line = formatting_template.substitute(dataset[dataset_type][i])
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
example = formatting_template.substitute(dataset["train"][1])
print(example)

# TODO: For now create adapter in the plugin directory but this should probably go somewhere else
adaptor_output_dir = plugin_dir
# adaptor_output_dir = config["adaptor_output_dir"]
# if not os.path.exists(adaptor_output_dir):
#     os.makedirs(adaptor_output_dir)
adaptor_file_name = os.path.join(adaptor_output_dir, "adaptor_name.npz")

popen_command = [sys.executable, "-u", f"{plugin_dir}/mlx-examples/lora/lora.py",
                 "--model", config["model_name"],
                 "--iters", iters,
                 "--train",
                 "--adapter-file", adaptor_file_name,
                 "--lora-layers", lora_layers,
                 "--learning-rate", learning_rate,
                 "--data", f"{plugin_dir}/data/",
                 "--steps-per-report", config['steps_per_report'],
                 #  "--steps_per_eval", config["steps_per_eval"],
                 "--save-every", config["save_every"]]

print("Running command:")
print(popen_command)


db.execute(
    "UPDATE job SET progress = ? WHERE id = ?",
    (0, config["job_id"]),
)
db.commit()

print("Training beginning:")
print("Adaptor will be saved as:", adaptor_file_name)

# w = tf.summary.create_file_writer(os.path.join(config["output_dir"], "logs"))
writer = SummaryWriter(os.path.join(config["output_dir"], "logs"))
print("Writing logs to:", os.path.join(config["output_dir"], "logs"))

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
            db.execute(
                "UPDATE job SET progress = ? WHERE id = ?",
                (percent_complete, config["job_id"]),
            )
            db.commit()

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
    sys.executable,
    f"{plugin_dir}/mlx-examples/lora/fuse.py",
    "--model", config["model_name"],
    "--adapter-file", adaptor_file_name,
    "--save-path", fused_model_location]

with subprocess.Popen(
        fuse_popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:
    for line in process.stdout:
        print(line, end="", flush=True)

    return_code = process.wait()

    # If model create was successful, create an info.json file so this can be read by the system
    print("Return code: ", return_code)
    if (return_code == 0):
        model_description = [{
            "model_id": f"TransformerLab-mlx/{fused_model_name}",
            "model_filename": "",
            "name": fused_model_name,
            "local_model": True,
            "json_data": {
                "uniqueID": f"TransformerLab-mlx/{fused_model_name}",
                "name": f"MLX",
                "description": f"An MLX modeled generated by TransformerLab based on {config['model_name']}",
                "architecture": "MLX",
                "huggingface_repo": ""
            }
        }]
        model_description_file = open(f"{fused_model_location}/info.json", "w")
        json.dump(model_description, model_description_file)
        model_description_file.close()

        print("Finished fusing the adaptor with the model.")

    else:
        print("Fusing model with adaptor failed: ", return_code)
