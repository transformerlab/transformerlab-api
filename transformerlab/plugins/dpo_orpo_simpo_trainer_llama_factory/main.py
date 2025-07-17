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


########################################
# First set up arguments and parameters
########################################


plugin_dir = os.path.dirname(os.path.realpath(__file__))
print("Plugin dir:", plugin_dir)


@tlab_trainer.async_job_wrapper(progress_start=0, progress_end=100, wandb_project_name="LlamaFactory_DPO")
async def run_train():
    # Directory for storing temporary working files
    data_directory = f"{WORKSPACE_DIR}/temp/llama_factory_reward/data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Typecasting the required arguments to their datatype
    tlab_trainer.params.learning_rate = float(tlab_trainer.params.learning_rate)
    tlab_trainer.params.num_train_epochs = int(tlab_trainer.params.num_train_epochs)
    tlab_trainer.params.max_steps = int(tlab_trainer.params.max_steps)

    # Process preference strategy
    preference_strategy = tlab_trainer.params.pref_loss
    if preference_strategy == "dpo":
        preference_strategy = "sigmoid"  # llama factory calls dpo "sigmoid"
    if preference_strategy not in ["sigmoid", "orpo", "simpo"]:
        print("Invalid preference strategy")
        raise ValueError("Invalid preference strategy. Must be one of: dpo, orpo, simpo.")

    # Load dataset
    dataset = tlab_trainer.load_dataset()
    dataset = dataset["train"]

    # Define dataset processing functions
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

    # Process the dataset
    try:
        # output dataset to a json file, row by row
        with open(f"{data_directory}/train.json", "w") as f:
            all_data = []
            for row in dataset:
                all_data.append(row)
            json.dump(all_data, f, indent=2)

        create_data_directory_in_llama_factory_format()
    except Exception as e:
        raise e

    # Generate YAML config
    yaml_config_path = f"{data_directory}/llama3_lora_dpo.yaml"
    today = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(tlab_trainer.params["output_dir"], f"job_{tlab_trainer.params['job_id']}_{today}")

    # First copy a template file to the data directory
    os.system(f"cp {plugin_dir}/LLaMA-Factory/examples/train_lora/llama3_lora_dpo.yaml {yaml_config_path}")

    # Now replace specific values in the file using PyYAML
    yml = {}
    with open(yaml_config_path, "r") as file:
        yml = yaml.safe_load(file)

    print("Template configuration:")
    print(yml)

    # Update the YAML config with parameters
    yml["pref_loss"] = preference_strategy
    yml["model_name_or_path"] = tlab_trainer.params.model_name
    yml["output_dir"] = tlab_trainer.params.adaptor_output_dir
    yml["logging_dir"] = output_dir
    yml["learning_rate"] = tlab_trainer.params.learning_rate
    yml["num_train_epochs"] = tlab_trainer.params.num_train_epochs
    yml["max_steps"] = tlab_trainer.params.max_steps
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

    # Train the model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    popen_command = [python_executable, "train", yaml_config_path]

    print("Running command:")
    print(popen_command)

    error_output = ""

    process = await asyncio.create_subprocess_exec(
        *popen_command,
        cwd=os.path.join(plugin_dir, "LLaMA-Factory"),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert process.stdout is not None
    training_step_has_started = False
    error_output = ""

    async for line_bytes in process.stdout:
        line = line_bytes.decode("utf-8", errors="replace")
        error_output += line

        if "***** Running training *****" in line:
            training_step_has_started = True

        if not training_step_has_started:
            continue

        pattern = r"(\d+)%\|.*\| (\d+)\/(\d+) \[(\d+):(\d+)<(\d+):(\d+),(\s*)(\d+\.\d+)(.+)]"
        match = re.search(pattern, line)
        if match:
            percentage = match.group(1)
            current = match.group(2)
            total = match.group(3)
            minutes = match.group(4)
            seconds = match.group(5)
            it_s = match.group(9)

            print(
                f"Percentage: {percentage}, Current: {current}, Total: {total}, Minutes: {minutes}, Seconds: {seconds}, It/s: {it_s}"
            )
            tlab_trainer.progress_update(round(float(percentage), 2))

        print(line, end="", flush=True)

    return_code = await process.wait()

    if (
        return_code != 0
        and "TypeError: DPOTrainer.create_model_card() got an unexpected keyword argument 'license'" not in error_output
    ):
        raise RuntimeError(f"Training failed: {error_output}")

    print("Finished training.")

    # Fuse the model with the base model
    await fuse_model()


async def fuse_model():
    """Fuse the adapter with the base model"""
    print("Now fusing the adaptor with the model.")

    data_directory = f"{WORKSPACE_DIR}/temp/llama_factory_reward/data"
    model_name = tlab_trainer.params.model_name
    adaptor_name = tlab_trainer.params.adaptor_name
    adaptor_output_dir = tlab_trainer.params.adaptor_output_dir

    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    fused_model_name = f"{model_name}_{adaptor_name}"
    fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)

    # Make the directory to save the fused model
    if not os.path.exists(fused_model_location):
        os.makedirs(fused_model_location)

    yaml_config_path = f"{data_directory}/merge_llama3_lora_sft.yaml"
    # Copy a template file to the data directory
    os.system(f"cp {plugin_dir}/LLaMA-Factory/examples/merge_lora/llama3_lora_sft.yaml {yaml_config_path}")

    yml = {}
    with open(yaml_config_path, "r") as file:
        yml = yaml.safe_load(file)

    yml["model_name_or_path"] = tlab_trainer.params.model_name
    yml["adapter_name_or_path"] = adaptor_output_dir
    yml["export_dir"] = fused_model_location
    yml["resize_vocab"] = True

    with open(yaml_config_path, "w") as file:
        yaml.dump(yml, file)
        print("Merge configuration:")
        print(yml)

    env = os.environ.copy()
    python_executable = get_python_executable(plugin_dir)
    env["PATH"] = python_executable.replace("/python", ":") + env["PATH"]

    if "venv" in python_executable:
        python_executable = python_executable.replace("venv/bin/python", "venv/bin/llamafactory-cli")

    fuse_popen_command = [python_executable, "export", yaml_config_path]

    process = await asyncio.create_subprocess_exec(
        *fuse_popen_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    assert process.stdout is not None
    async for line_bytes in process.stdout:
        print(line_bytes.decode("utf-8", errors="replace"), end="", flush=True)

    return_code = await process.wait()

    print("Return code: ", return_code)
    if return_code == 0:
        tlab_trainer.create_transformerlab_model(
            fused_model_name=fused_model_name,
            model_architecture=tlab_trainer.params.get("model_architecture", "llama"),
            json_data={
                "uniqueID": f"TransformerLab/{fused_model_name}",
                "name": "dpo_orpo_simpo_trainer_llama_factory",
                "description": f"Model generated using Llama Factory in Transformer Lab based on {tlab_trainer.params.model_name}",
                "architecture": tlab_trainer.params.get("model_architecture", "llama"),
                "huggingface_repo": "",
            },
            output_dir=os.path.join(WORKSPACE_DIR, "models"),
        )
        print("Finished fusing the adaptor with the model.")
    else:
        print("Fusing model with adaptor failed: ", return_code)
        raise RuntimeError(f"Fusing model with adaptor failed: {return_code}")


run_train()
