"""
Fine-Tuning with Llama Factory

https://github.com/hiyouga/LLaMA-Factory/tree/main

Standard command:
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_sft.yaml

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
import yaml

########################################
# First set up arguments and parameters
########################################

root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
plugin_dir = os.path.dirname(os.path.realpath(__file__))
plugin_dir = os.path.join(plugin_dir)
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

model_name = config['model_name']
adaptor_output_dir = config['adaptor_output_dir']

lora_layers = config["lora_layers"]
learning_rate = config["learning_rate"]
batch_size = config.get("batch_size", 4)
steps_per_eval = config.get("steps_per_eval", 200)
iters = config["iters"]

# we need to adapter parameter so set a default
adaptor_name = config.get('adaptor_name', "default")

########################################
# Now process the Datatset
########################################

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

# Directory for storing temporary working files
data_directory = f"{WORKSPACE_DIR}/temp/llama_factory"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

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
    f"cp {data_directory}/test.jsonl {data_directory}/valid.jsonl")

print("Example formatted training example:")
example = formatting_template.substitute(dataset["train"][1])
print(example)

adaptor_output_dir = data_directory
adaptor_file_name = os.path.join(adaptor_output_dir, f"{adaptor_name}.npz")

########################################
# Generate a config YAML file that will be used by LLaMA-Factory
########################################

yaml_config_path = f"{data_directory}/llama3_lora_sft.yaml"


today = time.strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join(config["output_dir"], today)
print(f"Storing Tensorboard Output to: {output_dir}")

# In the json job_data column for this job, store the tensorboard output dir
db.execute(
    "UPDATE job SET job_data = json_insert(job_data, '$.tensorboard_output_dir', ?) WHERE id = ?",
    (output_dir, config["job_id"]),
)
db.commit()

# First copy a template file to the data directory
os.system(
    f"cp {plugin_dir}/LLaMA-Factory/examples/lora_single_gpu/llama3_lora_sft.yaml {yaml_config_path}")
# Now replace specific values in the file using the PyYAML library:
with open(yaml_config_path, 'r+') as file:
    yml = yaml.safe_load(file)
    print("Template configuration:")
    print(yml)
    # replace model_name with the model name
    yml["model_name_or_path"] = model_name
    yml["output_dir"] = adaptor_output_dir
    yml["logging_dir"] = output_dir
    print("--------")
    # Now write out the new file
    yaml.dump(yml, file)
    print("New configuration:")
    print(yml)


########################################
# Now train
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_sft.yaml
########################################

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
popen_command = ['llamafactory-cli', 'train', yaml_config_path]

print("Running command:")
print(popen_command)


db.execute(
    "UPDATE job SET progress = ? WHERE id = ?",
    (0, config["job_id"]),
)
db.commit()

print("Training beginning:")
print("Adaptor will be saved as:", adaptor_file_name)

with subprocess.Popen(
        popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, cwd=os.path.join(plugin_dir, 'LLaMA-Factory')) as process:
    for line in process.stdout:
        # Each output line from lora.py looks like
        # "  2%|▏         | 8/366 [00:15<11:28,  1.92s/it]"
        pattern = r'(\d+)%\|.*\| (\d+)\/(\d+) \[(\d+):(\d+)<(\d+):(\d+),  (\d+\.\d+)s\/it]'
        match = re.search(pattern, line)
        if match:
            percentage = match.group(1)
            current = match.group(2)
            total = match.group(3)
            minutes = match.group(4)
            seconds = match.group(5)
            it_s = match.group(8)

            print(
                f"Percentage: {percentage}, Current: {current}, Total: {total}, Minutes: {minutes}, Seconds: {seconds}, It/s: {it_s}")
            db.execute(
                "UPDATE job SET progress = ? WHERE id = ?",
                (percentage, config["job_id"]),
            )
            db.commit()

        # We will not log the training progress to tensorboard yet because the above output doesn't display it...
        # loss = float(match.group(1))
        # it_per_sec = float(match.group(2))
        # tokens_per_sec = float(match.group(3))
        # print("Loss: ", loss)
        # print("It/sec: ", it_per_sec)
        # print("Tokens/sec: ", tokens_per_sec)
        # # The code snippet `with w.as_default(): tf.summary.scalar` is using TensorFlow's
        # # `tf.summary.scalar` function to log scalar values to a TensorBoard summary writer
        # # `w`.
        # writer.add_scalar("loss", loss, int(first_number))
        # writer.add_scalar("it_per_sec", it_per_sec, int(first_number))
        # writer.add_scalar("tokens_per_sec",
        #                 tokens_per_sec, int(first_number))

        print(line, end="", flush=True)

print("Finished training.")

# TIME TO FUSE THE MODEL WITH THE BASE MODEL

# print("Now fusing the adaptor with the model.")

# model_name = config['model_name']
# if "/" in model_name:
#     model_name = model_name.split("/")[-1]
# fused_model_name = f"{model_name}_{adaptor_name}"
# fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)

# # Make the directory to save the fused model
# if not os.path.exists(fused_model_location):
#     os.makedirs(fused_model_location)

# fuse_popen_command = [
#     sys.executable,
#     f"{plugin_dir}/mlx-examples/lora/fuse.py",
#     "--model", config["model_name"],
#     "--adapter-file", adaptor_file_name,
#     "--save-path", fused_model_location]

# with subprocess.Popen(
#         fuse_popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:
#     for line in process.stdout:
#         print(line, end="", flush=True)

#     return_code = process.wait()

#     # If model create was successful, create an info.json file so this can be read by the system
#     print("Return code: ", return_code)
#     if (return_code == 0):
#         model_description = [{
#             "model_id": f"TransformerLab-mlx/{fused_model_name}",
#             "model_filename": "",
#             "name": fused_model_name,
#             "local_model": True,
#             "json_data": {
#                 "uniqueID": f"TransformerLab-mlx/{fused_model_name}",
#                 "name": f"MLX",
#                 "description": f"An MLX modeled generated by TransformerLab based on {config['model_name']}",
#                 "architecture": "MLX",
#                 "huggingface_repo": ""
#             }
#         }]
#         model_description_file = open(f"{fused_model_location}/info.json", "w")
#         json.dump(model_description, model_description_file)
#         model_description_file.close()

#         print("Finished fusing the adaptor with the model.")

#     else:
#         print("Fusing model with adaptor failed: ", return_code)
