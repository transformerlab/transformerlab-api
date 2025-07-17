"""
Reinforcement Learning from AI Feedback (RLAIF) using MLX PPO adapted from https://github.com/andrew-silva/mlx-rlhf

This plugin runs PPO training for MLX models, using a custom dataset provided by TransformerLab.

"""

import os
import re
import json
import asyncio
from transformerlab.sdk.v1.train import tlab_trainer
from transformerlab.plugin import WORKSPACE_DIR, get_python_executable


@tlab_trainer.async_job_wrapper(wandb_project_name="TLab_RLAIF", manual_logging=True)
async def train_mlx_rlaif():
    plugin_dir = os.path.dirname(os.path.realpath(__file__))
    print("Plugin dir:", plugin_dir)

    # Extract configuration parameters
    model_name = tlab_trainer.params.model_name
    batch_size = str(tlab_trainer.params.get("batch_size", "2"))
    mini_batch_size = str(tlab_trainer.params.get("mini_batch_size", "2"))
    ppo_epochs = str(tlab_trainer.params.get("ppo_epochs", "4"))
    num_steps = str(tlab_trainer.params.get("num_steps", "5550"))
    adaptor_name = tlab_trainer.params.get("adaptor_name", "default")
    ground_truth_reward = tlab_trainer.params.get("ground_truth_reward", True)
    adap_kl_ctrl = tlab_trainer.params.get("adap_kl_ctrl", True)
    init_kl_coef = str(tlab_trainer.params.get("init_kl_coef", "0.2"))
    seed = str(tlab_trainer.params.get("seed", "42"))
    max_completion_length = str(tlab_trainer.params.get("max_completion_length", "256"))

    # Load datasets for training
    datasets = tlab_trainer.load_dataset(["train"])

    # Directory for storing temporary working files
    data_directory = f"{WORKSPACE_DIR}/plugins/mlx_rlaif_trainer/data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Write the custom dataset to a temporary JSON file
    custom_dataset_path = os.path.join(data_directory, "ppo_custom_dataset.json")
    with open(custom_dataset_path, "w") as f:
        for example in datasets["train"]:
            f.write(json.dumps(example) + "\n")
    print(f"Custom dataset written to {custom_dataset_path}")

    # Set output directory for the trained model
    model_output_dir = tlab_trainer.params.get("model_output_dir", "")
    if not model_output_dir:
        model_output_dir = os.path.join(WORKSPACE_DIR, "models", f"{model_name.split('/')[-1]}_{adaptor_name}_ppo")
        print(f"Using default model output directory: {model_output_dir}")
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Get Python executable (from venv if available)
    python_executable = get_python_executable(plugin_dir)
    env = os.environ.copy()
    env["PATH"] = python_executable.replace("/python", ":") + env["PATH"]

    # Prepare the command for MLX PPO training
    ppo_script_path = os.path.join(plugin_dir, "mlx-rlhf", "ppo_training.py")
    popen_command = [
        python_executable,
        ppo_script_path,
        f"--model={model_name}",
        f"--batch_size={batch_size}",
        f"--mini_batch_size={mini_batch_size}",
        f"--ppo_epoch={ppo_epochs}",
        f"--num_steps={num_steps}",
        f"--init_kl_coef={init_kl_coef}",
        f"--seed={seed}",
        f"--custom_hf_dataset={custom_dataset_path}",
        f"--max_completion_length={max_completion_length}",
        f"--output_dir={model_output_dir}",
    ]
    if ground_truth_reward:
        popen_command.append("--ground_truth_reward")
    if adap_kl_ctrl:
        popen_command.append("--adap_kl_ctrl=True")

    print("Running command:")
    print(" ".join(str(x) for x in popen_command))
    print(f"Model will be saved in: {model_output_dir}")

    total_steps = int(num_steps)
    current_step = 0

    process = await asyncio.create_subprocess_exec(
        *popen_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
    )

    assert process.stdout is not None
    current_step = 0

    async for line_bytes in process.stdout:
        line = line_bytes.decode("utf-8", errors="replace")
        print(line, end="", flush=True)

        # Progress parsing
        step_pattern = r"Step - (\d+)"
        step_match = re.search(step_pattern, line)
        if step_match:
            current_step = int(step_match.group(1))
            percent_complete = (current_step / total_steps) * 100
            tlab_trainer.progress_update(percent_complete)

        # Reward parsing
        reward_pattern = r"Reward: array\((.*?), dtype=float32\)"
        reward_match = re.search(reward_pattern, line)
        if reward_match:
            try:
                reward_str = reward_match.group(1).replace("...", "0, 0, 0")
                reward_values = [float(x) for x in reward_str.replace("[", "").replace("]", "").split(",") if x.strip()]
                if reward_values:
                    mean_reward = sum(reward_values) / len(reward_values)
                    tlab_trainer.log_metric("train/mean_reward", mean_reward, current_step)
                    tlab_trainer.log_metric("train/reward_batch_size", len(reward_values), current_step)
            except Exception as e:
                print(f"Error parsing rewards: {e}")

        # Other metrics
        metrics_pattern = r"(policy_loss|value_loss|kl_div|entropy): ([-+]?\d*\.\d+|\d+)"
        for match in re.finditer(metrics_pattern, line):
            metric_name = match.group(1)
            metric_value = float(match.group(2))
            tlab_trainer.log_metric(f"train/{metric_name}", metric_value, current_step)

    return_code = await process.wait()
    if return_code != 0:
        print("An error occurred before training completed.")
        raise RuntimeError("Training failed.")

    print("Finished training.")

    # Register the model with TransformerLab
    model_name_base = model_name.split("/")[-1]
    final_model_name = f"{model_name_base}_{adaptor_name}_ppo"
    json_data = {"description": f"An MLX model trained with PPO by Transformer Lab based on {model_name}"}
    tlab_trainer.create_transformerlab_model(
        fused_model_name=final_model_name, model_architecture="MLX", json_data=json_data
    )
    print(f"Model registered as {final_model_name}")
    return True


train_mlx_rlaif()
