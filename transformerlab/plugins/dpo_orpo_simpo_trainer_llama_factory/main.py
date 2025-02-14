"""
Fine-Tuning with Llama Factory

https://github.com/hiyouga/LLaMA-Factory/tree/main

Standard command:
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml

"""

import os
import subprocess
import time

import json
import yaml
import re
import argparse

import transformerlab.plugin

from datasets import load_dataset
from jinja2 import Environment

jinja_environment = Environment()


########################################
# First set up arguments and parameters
########################################

root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
plugin_dir = os.path.dirname(os.path.realpath(__file__))
print("Plugin dir:", plugin_dir)

# Connect to the LLM Lab database
WORKSPACE_DIR = transformerlab.plugin.WORKSPACE_DIR

# Get all parameters provided to this script from Transformer Lab
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--experiment_name", default="", type=str)
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

job = transformerlab.plugin.Job(config["job_id"])
job.update_progress(0)

model_name = config["model_name"]
adaptor_output_dir = config["adaptor_output_dir"]

lora_layers = config.get("lora_layers", 1)
learning_rate = config["learning_rate"]
batch_size = config.get("batch_size", 4)
steps_per_eval = config.get("steps_per_eval", 200)
iters = config.get("iters", -1)

# we need to adapter parameter so set a default
adaptor_name = config.get("adaptor_name", "default")

preference_strategy = config.get("pref_loss", "dpo")
if preference_strategy == "dpo":
    preference_strategy = "sigmoid"  # llama factory calls dpo "sigmoid"
if preference_strategy not in ["sigmoid", "orpo", "simpo"]:
    print("Invalid preference strategy")
    job.set_job_completion_status("failed", "Invalid preference strategy")
    exit()


# Directory for storing temporary working files
data_directory = f"{WORKSPACE_DIR}/temp/llama_factory_reward/data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)


def create_data_directory_in_llama_factory_format():
    """This function creates a directory in the data_directory location
    that contains the files in the format that LLaMA-Factory expects.
    The main file being a dataset_info.json file that acts as an index to the JSON training data
    """
    dataset_info = {
        "training_data": {
            "file_name": "train.json",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"},
        }
    }

    with open(f"{data_directory}/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)


########################################
# Now process the Datatset
########################################

try:
    dataset_target = transformerlab.plugin.get_dataset_path(config["dataset_name"])
except Exception as e:
    job.set_job_completion_status("failed", "failed to get dataset path")
    raise e

try:
    dataset = load_dataset(dataset_target, trust_remote_code=True)
except Exception as e:
    job.set_job_completion_status("failed", "failed to load dataset")
    raise e

# output dataset['train'] to a json file, row by row:
# This will exhaust memory if the data is large
try:
    with open(f"{data_directory}/train.json", "w") as f:
        all_data = []
        for row in dataset["train"]:
            all_data.append(row)
        json.dump(all_data, f, indent=2)
except Exception as e:
    job.set_job_completion_status("failed", "failed to process the dataset")
    raise e

########################################
# Generate a config YAML file that will be used by LLaMA-Factory
########################################
yaml_config_path = f"{data_directory}/llama3_lora_dpo.yaml"

today = time.strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join(config["output_dir"], today)
print(f"Storing Tensorboard Output to: {output_dir}")


# First copy a template file to the data directory
os.system(f"cp {plugin_dir}/LLaMA-Factory/examples/train_lora/llama3_lora_dpo.yaml {yaml_config_path}")
# Now replace specific values in the file using the PyYAML library:

yml = {}
with open(yaml_config_path, "r") as file:
    yml = yaml.safe_load(file)

try:
    create_data_directory_in_llama_factory_format()
except Exception as e:
    job.set_job_completion_status("failed", "failed to create llama factory data directory")
    raise e


print("Template configuration:")
print(yml)
yml["pref_loss"] = preference_strategy
yml["model_name_or_path"] = model_name
yml["output_dir"] = adaptor_output_dir
yml["logging_dir"] = output_dir
yml["learning_rate"] = float(config.get("learning_rate", 0.001))
yml["num_train_epochs"] = float(config.get("num_train_epochs", 1))
yml["max_steps"] = float(config.get("max_steps", -1))
yml["dataset_dir"] = data_directory
yml["dataset"] = "training_data"
yml["template"] = "llama3"
# Without resize_vocab the training fails for many models including Mistral
yml["resize_vocab"] = True
print("--------")

with open(yaml_config_path, "w") as file:
    # Now write out the new file
    yaml.dump(yml, file)
    print("New configuration:")
    print(yml)


########################################
# Now train
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_sft.yaml
########################################

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
popen_command = ["llamafactory-cli", "train", yaml_config_path]

print("Running command:")
print(popen_command)


# In the json job_data column for this job, store the tensorboard output dir
job.set_tensorboard_output_dir(output_dir)


print("Training beginning:")

error_output = ""

with subprocess.Popen(
    popen_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    universal_newlines=True,
    cwd=os.path.join(plugin_dir, "LLaMA-Factory"),
) as process:
    training_step_has_started = False

    for line in process.stdout:
        error_output += line

        if job.should_stop():
            print("Stopping job because of user interruption.")
            job.update_status("STOPPED")
            job.set_job_completion_status("failed", "user stopped the job")
            process.terminate()

        if "***** Running training *****" in line:
            training_step_has_started = True

        if not training_step_has_started:
            continue

        # Each output line from lora.py looks like
        # "  2%|▏         | 8/366 [00:15<11:28,  1.92s/it]"
        pattern = r"(\d+)%\|.*\| (\d+)\/(\d+) \[(\d+):(\d+)<(\d+):(\d+),(\s*)(\d+\.\d+)(.+)]"
        match = re.search(pattern, line)
        if match:
            percentage = match.group(1)
            current = match.group(2)
            total = match.group(3)
            minutes = match.group(4)
            seconds = match.group(5)
            it_s = match.group(8)

            print(
                f"Percentage: {percentage}, Current: {current}, Total: {total}, Minutes: {minutes}, Seconds: {seconds}, It/s: {it_s}"
            )
            job.update_progress(percentage)

        print(line, end="", flush=True)

    return_code = process.wait()

    if (
        return_code != 0
        and "TypeError: DPOTrainer.create_model_card() got an unexpected keyword argument 'license'" not in error_output
    ):
        job.set_job_completion_status("failed", "failed during training")
        raise RuntimeError(f"Training failed: {error_output}")

print("Finished training.")

# TIME TO FUSE THE MODEL WITH THE BASE MODEL

print("Now fusing the adaptor with the model.")

model_name = config["model_name"]
if "/" in model_name:
    model_name = model_name.split("/")[-1]
fused_model_name = f"{model_name}_{adaptor_name}"
fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)

# Make the directory to save the fused model
if not os.path.exists(fused_model_location):
    os.makedirs(fused_model_location)

yaml_config_path = f"{data_directory}/merge_llama3_lora_sft.yaml"
# First copy a template file to the data directory
os.system(f"cp {plugin_dir}/LLaMA-Factory/examples/merge_lora/llama3_lora_sft.yaml {yaml_config_path}")
yml = {}
with open(yaml_config_path, "r") as file:
    yml = yaml.safe_load(file)

yml["model_name_or_path"] = config["model_name"]
yml["adapter_name_or_path"] = adaptor_output_dir
yml["export_dir"] = fused_model_location
yml["resize_vocab"] = True

with open(yaml_config_path, "w") as file:
    yaml.dump(yml, file)
    print("New configuration:")
    print(yml)

fuse_popen_command = ["llamafactory-cli", "export", yaml_config_path]

with subprocess.Popen(
    fuse_popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)

    return_code = process.wait()

    # If model create was successful, create an info.json file so this can be read by the system
    print("Return code: ", return_code)
    if return_code == 0:
        model_description = [
            {
                "model_id": f"TransformerLab/{fused_model_name}",
                "model_filename": "",
                "name": fused_model_name,
                "local_model": True,
                "json_data": {
                    "uniqueID": f"TransformerLab/{fused_model_name}",
                    "name": "dpo_orpo_simpo_trainer_llama_factory",
                    "description": f"Model generated using Llama Factory in TransformerLab based on {config['model_name']}",
                    "architecture": config["model_architecture"],
                    "huggingface_repo": "",
                },
            }
        ]
        model_description_file = open(f"{fused_model_location}/info.json", "w")
        json.dump(model_description, model_description_file)
        model_description_file.close()

        print("Finished fusing the adaptor with the model.")
        job.set_job_completion_status("success", "succesfully trained model")
    else:
        print("Fusing model with adaptor failed: ", return_code)
        job.set_job_completion_status("failed", "failed to fuse model")
