"""
Fine-Tuning with Llama Factory

https://github.com/hiyouga/LLaMA-Factory/tree/main

Standard command:
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml
"""

import os
import asyncio
import time
import json
import yaml
import re

from transformerlab.sdk.v1.train import tlab_trainer
from transformerlab.plugin import WORKSPACE_DIR, get_python_executable


# Get environment variables
plugin_dir = os.path.dirname(os.path.realpath(__file__))
print("Plugin dir:", plugin_dir)

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


@tlab_trainer.async_job_wrapper()
async def run_reward_modeling():
    """Main function to run reward modeling with LlamaFactory"""
    # Access configuration through tlab_trainer
    config = tlab_trainer.params._config
    print("Input config:")
    print(json.dumps(config, indent=4))

    model_name = tlab_trainer.params.model_name
    adaptor_output_dir = tlab_trainer.params.adaptor_output_dir
    adaptor_name = tlab_trainer.params.get("adaptor_name", "default")

    # Process dataset
    try:
        datasets = tlab_trainer.load_dataset()
        dataset = datasets["train"]

        # Output dataset to a json file
        with open(f"{data_directory}/train.json", "w") as f:
            all_data = []
            for row in dataset:
                all_data.append(row)
            json.dump(all_data, f, indent=2)
    except Exception as e:
        print(f"Error processing dataset: {e}")
        raise

    # Generate config YAML file for LLaMA-Factory
    yaml_config_path = f"{data_directory}/llama3_lora_reward.yaml"

    today = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(config["output_dir"], today)

    # Copy template file and modify it
    os.system(f"cp {plugin_dir}/LLaMA-Factory/examples/train_lora/llama3_lora_reward.yaml {yaml_config_path}")

    with open(yaml_config_path, "r") as file:
        yml = yaml.safe_load(file)

    create_data_directory_in_llama_factory_format()

    print("Template configuration:")
    print(yml)

    # Update YAML configuration
    yml["model_name_or_path"] = model_name
    yml["output_dir"] = adaptor_output_dir
    yml["logging_dir"] = output_dir
    yml["learning_rate"] = float(config.get("learning_rate", 0.001))
    yml["num_train_epochs"] = float(config.get("num_train_epochs", 1))
    yml["max_steps"] = float(config.get("max_steps", -1))
    yml["dataset_dir"] = data_directory
    yml["dataset"] = "training_data"
    yml["template"] = "llama3"
    yml["resize_vocab"] = True
    print("--------")

    with open(yaml_config_path, "w") as file:
        yaml.dump(yml, file)
        print("New configuration:")
        print(yml)

    env = os.environ.copy()
    python_executable = get_python_executable(plugin_dir)
    env["PATH"] = python_executable.replace("/python", ":") + env["PATH"]

    if "venv" in python_executable:
        python_executable = python_executable.replace("venv/bin/python", "venv/bin/llamafactory-cli")

    # Set up environment and run training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    popen_command = [python_executable, "train", yaml_config_path]

    print("Running command:")
    print(popen_command)

    print("Training beginning:")

    process = await asyncio.create_subprocess_exec(
        *popen_command,
        cwd=os.path.join(plugin_dir, "LLaMA-Factory"),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert process.stdout is not None
    training_step_has_started = False

    async for line_bytes in process.stdout:
        line = line_bytes.decode("utf-8", errors="replace")

        if "***** Running training *****" in line:
            training_step_has_started = True

        if not training_step_has_started:
            continue

        # Parse progress from output lines
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
            tlab_trainer.progress_update(round(float(percentage), 1))

        print(line, end="", flush=True)

    return_code = await process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training subprocess failed with return code {return_code}")

    print("Finished training.")

    # Fuse the model with the base model
    print("Now fusing the adaptor with the model.")

    model_name_simple = model_name
    if "/" in model_name_simple:
        model_name_simple = model_name_simple.split("/")[-1]

    fused_model_name = f"{model_name_simple}_{adaptor_name}"
    fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)

    # Make directory for the fused model
    if not os.path.exists(fused_model_location):
        os.makedirs(fused_model_location)

    # Create config for model fusion
    yaml_config_path = f"{data_directory}/merge_llama3_lora_sft.yaml"
    os.system(f"cp {plugin_dir}/LLaMA-Factory/examples/merge_lora/llama3_lora_sft.yaml {yaml_config_path}")

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

    # Run fusion process
    fuse_popen_command = [python_executable, "export", yaml_config_path]

    process = await asyncio.create_subprocess_exec(
        *fuse_popen_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    assert process.stdout is not None
    async for line_bytes in process.stdout:
        print(line_bytes.decode(), end="", flush=True)

    return_code = await process.wait()

    # If model fusion was successful, create model info
    print("Return code: ", return_code)
    if return_code == 0:
        json_data = {
            "uniqueID": f"TransformerLab/{fused_model_name}",
            "name": fused_model_name,
            "description": f"Model generated using Llama Factory in Transformer Lab based on {config['model_name']}",
            "architecture": config["model_architecture"],
            "huggingface_repo": "",
        }

        tlab_trainer.create_transformerlab_model(
            fused_model_name=fused_model_name,
            model_architecture=config["model_architecture"],
            json_data=json_data,
        )

        print("Finished fusing the adaptor with the model.")
    else:
        print("Fusing model with adaptor failed: ", return_code)


run_reward_modeling()
