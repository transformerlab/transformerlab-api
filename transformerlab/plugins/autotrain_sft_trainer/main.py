import json
import re
from string import Template
import subprocess
import time
from datasets import load_dataset
import argparse
import os
from tensorboardX import SummaryWriter

import transformerlab.plugin
from jinja2 import Environment

jinja_environment = Environment()


# Setup some directories we'll use
plugin_dir = os.path.dirname(os.path.realpath(__file__))
WORKSPACE_DIR = transformerlab.plugin.WORKSPACE_DIR
TLAB_CODE_DIR = WORKSPACE_DIR

# Get all parameters provided to this script from Transformer Lab
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
args, unknown = parser.parse_known_args()

input_config = None
# open the input file that provides configs
with open(args.input_file) as json_file:
    input_config = json.load(json_file)
config = input_config["config"]
print("Input:")
print(json.dumps(input_config, indent=4))

# Parameters to pass to autotrain
learning_rate = config["learning_rate"]
batch_size = config.get("batch_size", 4)
num_train_epochs = config.get("num_train_epochs", 4)

# Generate a model name using the original model and the passed adaptor
adaptor_name = config.get('adaptor_name', "default")
input_model_no_author = config["model_name"].split("/")[-1]

project_name = f"{input_model_no_author}-{adaptor_name}".replace(".","")

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

# Directory for storing temporary working files
data_directory = os.path.join(plugin_dir, "data")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

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

    # output training files in templated format in to data directory
    with open(f"{data_directory}/{dataset_type}.jsonl", "w") as f:
        for i in range(len(dataset[dataset_type])):
            data_line = dataset[dataset_type][i]
            data_line = dict(data_line)
            line = formatting_template.render(data_line)

            # escape newline and return characters so this is valid jsonl
            line = line.replace("\n", "\\n")
            line = line.replace("\r", "\\r")
            o = {"text": line}
            f.write(json.dumps(o) + "\n")

# copy file test.jsonl to valid.jsonl. Our test set is the same as our validation set.
os.system(
    f"cp {data_directory}/test.jsonl {data_directory}/valid.jsonl")

print("Example formatted training example:")
example = formatting_template.render(dataset["train"][1])
print(example)

popen_command = ["autotrain", "llm",
                 "--train",
                 "--model", config["model_name"],
                 "--data-path", data_directory,
                 "--lr", learning_rate,
                 "--batch-size", batch_size,
                 "--epochs", num_train_epochs,
                 "--trainer", "sft",
                 "--peft",
                 "--merge-adapter",
                 "--auto_find_batch_size",  # automatically find optimal batch size
                 "--project-name", project_name 
                 ]

print("Running command:")
print(popen_command)

job = transformerlab.plugin.Job(config["job_id"])
job.update_progress(0)

print("Training beginning:")

# todays date with seconds:
today = time.strftime("%Y%m%d-%H%M%S")

output_dir = os.path.join(config["output_dir"], today)
writer = SummaryWriter(output_dir)
print("Writing logs to:", output_dir)

# Store the tensorboard output dir in the job
job.set_tensorboard_output_dir(output_dir)

with subprocess.Popen(
        popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:

    iteration = 0
    it_per_sec = 0
    percent_complete = 0
    for line in process.stdout:
        # Progress complete output lines looks like
        # "0%|          | 1/1710 [00:05<2:46:58,  5.86s/it]""
        # I'm sorry this regex is insane
        pattern = r"\s*(\d+)\%\|.+?(?=\d+/)(\d+)/.+?(?=\d+.\d+s/it)(\d+.\d+)s/it"
        match = re.search(pattern, line)
        if match:
            percent_complete = match.group(1)
            iteration = int(match.group(2))
            it_per_sec = float(match.group(3))
            # Don't update the job database here. Only when we get a major update.

        # Progress data for tensorboard comes from lines formatted like:
        # INFO     | 2024-06-25 18:01:04 | autotrain.trainers.common:on_log:226 -
        # {'loss': 1.7918, 'grad_norm': 0.55, 'learning_rate': 0.0007, 'epoch': 0.073}
        pattern = r"INFO.+?{'loss': (\d+\.\d+), 'grad_norm': (\d+\.\d+), 'learning_rate': (\d+\.\d+), 'epoch': (\d+\.\d+)}"
        match = re.search(pattern, line)
        if match:
            loss = float(match.group(1))
            grad_norm = float(match.group(2))
            learning_rate = float(match.group(3))
            epoch = float(match.group(4))
            token_per_sec = 0
            print("Progress: ", f"{percent_complete}%")
            print("Iteration: ", iteration)
            print("It/sec: ", it_per_sec)
            print("Loss: ", loss)
            print("Epoch:", epoch)
            job.update_progress(percent_complete)

            if job.should_stop:
                print("Stopping job because of user interruption.")
                job.update_status("STOPPED")
                process.terminate()

            # Output to tensorboard
            writer.add_scalar("loss", loss, iteration)
            writer.add_scalar("it_per_sec", it_per_sec, iteration)
            writer.add_scalar("learning_rate", learning_rate, iteration)
            writer.add_scalar("epoch", epoch, iteration)

        print(line, end="", flush=True)

# Clean up
# Autotrain outputs its data in a directory named <project_name>
# We don't need to keep the arrow-formatted data Autotrain uses, so we delete it
os.system(
    f"rm -rf {project_name}/autotrain_data")

# Move the model to the TransformerLab directory
os.system(
    f"mv {project_name} {config['adaptor_output_dir']}/")


print("Finished training.")
