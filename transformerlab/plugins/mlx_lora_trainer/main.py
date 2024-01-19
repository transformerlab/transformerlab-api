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
from datasets import load_dataset
import argparse
import os

# Connect to the LLM Lab database
llmlab_root_dir = os.getenv('LLM_LAB_ROOT_PATH')
db = sqlite3.connect(llmlab_root_dir + "/workspace/llmlab.sqlite3")


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


# Get the dataset
dataset_id = config["dataset_name"]

dataset_types = ["train", "test"]
dataset = {}
formatting_template = Template(config["formatting_template"])

for dataset_type in dataset_types:
    dataset[dataset_type] = load_dataset(
        dataset_id, split=f"{dataset_type}[:100%]")
    print(
        f"Loaded {dataset_type} dataset with {len(dataset[dataset_type])} examples.")
    with open(f"{llmlab_root_dir}/workspace/experiments/{args.experiment_name}/plugins/mlx_lora_trainer/data/{dataset_type}.jsonl", "w") as f:
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
    f"cp {llmlab_root_dir}/workspace/experiments/{args.experiment_name}/plugins/mlx_lora_trainer/data/test.jsonl {llmlab_root_dir}/workspace/experiments/{args.experiment_name}/plugins/mlx_lora_trainer/data/valid.jsonl")

print("Example formatted training example:")
example = formatting_template.substitute(dataset["train"][1])
print(example)

# adaptor_output_dir = config["adaptor_output_dir"]
# if not os.path.exists(adaptor_output_dir):
#     os.makedirs(adaptor_output_dir)

# adaptor_file_name = f"{adaptor_output_dir}/{config['adaptor_name']}.npz"
adaptor_file_name = f"{llmlab_root_dir}/workspace/experiments/{args.experiment_name}/plugins/mlx_lora_trainer/{config['adaptor_name']}.npz"

root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
plugin_dir = f"{root_dir}/workspace/experiments/{args.experiment_name}/plugins/mlx_lora_trainer"

popen_command = [sys.executable, "-u", f"{plugin_dir}/mlx-examples/lora/lora.py",
                 "--model", config["model_name"], "--iters", iters, "--train", "--adapter-file",
                 adaptor_file_name, "--lora-layers", lora_layers, "--learning-rate",
                 learning_rate, "--data", f"{plugin_dir}/data/", "--steps-per-report", config['steps_per_report'],
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
print("Adaptor will be saved as:")
# print(f"{plugin_dir}/{config['adaptor_name']}.npz", flush=True)

# define a regext pattern to look for "Iter: 100" in the output
pattern = r"Iter (\d+):"

llmlab_root_dir = os.getenv('LLM_LAB_ROOT_PATH')
db = sqlite3.connect(llmlab_root_dir + "/workspace/llmlab.sqlite3")

with subprocess.Popen(
        popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:
    for line in process.stdout:
        # Use re.search to find the match
        match = re.search(pattern, line)

        # Extract the first number if a match is found
        # We do this because, @TODO later we can use this number to update progress
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
        print(line, end="", flush=True)

print("Finished training.")

# TIME TO FUSE THE MODEL WITH THE BASE MODEL

print("Now fusing the adaptor with the model.")

fused_model_name = f"{config['model_name']}_{config['adaptor_name']}"
fused_model_location = f"{llmlab_root_dir}/workspace/models/{fused_model_name}"

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

print("Finished fusing the adaptor with the model.")
