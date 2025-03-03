# The following is adapted from
# https://www.philschmid.de/instruction-tune-llama-2

import argparse
import json
import os
import sqlite3
import sys
import subprocess
import time
from random import randrange

# Get all parameters provided to this script from Transformer Lab
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--launched_with_accelerate", action="store_true", 
                   help="Flag to prevent recursive subprocess launching")
args, unknown = parser.parse_known_args()

print("Arguments:")
print(args)

# Check if we should launch with accelerate
if not args.launched_with_accelerate:
    print("Launching training with accelerate for multi-GPU...")
    
    # Fix import issues by ensuring the parent directory is in PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
    
    # Create environment for subprocess with modified PYTHONPATH
    env = os.environ.copy()
    
    # Add the specific SDK path to PYTHONPATH
    tfl_source_dir = os.environ.get("_TFL_SOURCE_CODE_DIR")
    python_path = env.get("PYTHONPATH", "")
    
    # Create a list of paths to include
    paths_to_include = [api_dir]
    
    # If _TFL_SOURCE_CODE_DIR is available, construct the path to the plugin sdk
    if tfl_source_dir:
        tflab_sdk_path = os.path.join(tfl_source_dir, "transformerlab", "plugin_sdk")
        paths_to_include.append(tflab_sdk_path)
        print(f"Adding SDK path: {tflab_sdk_path}")
        
        # Also add the parent directory of the plugin_sdk
        plugin_parent = os.path.join(tfl_source_dir, "transformerlab")
        paths_to_include.append(plugin_parent)
        print(f"Adding plugin parent path: {plugin_parent}")
    
    # Add the existing PYTHONPATH if it exists
    if python_path:
        paths_to_include.append(python_path)
    
    # Join all paths with colons
    env["PYTHONPATH"] = ":".join(paths_to_include)
    
    print(f"Setting PYTHONPATH to: {env['PYTHONPATH']}")
    
    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        __file__,
        "--input_file", args.input_file,
        "--launched_with_accelerate"
    ]
    print(f"Running command: {' '.join(cmd)}")
    
    # Pass the modified environment to the subprocess
    result = subprocess.run(cmd, env=env)
    print(f"Subprocess completed with return code: {result.returncode}")
    sys.exit(result.returncode)

# Import dependencies after the subprocess check to ensure we're in the right environment
import torch
from datasets import load_dataset
from jinja2 import Environment
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator

# Get source code dir and print for debugging
tfl_source_dir = os.environ.get("_TFL_SOURCE_CODE_DIR")
print(f"TFL_SOURCE_CODE_DIR: {tfl_source_dir}")

# Ensure transformerlab can be imported
try:
    import transformerlab.plugin
    print("Successfully imported transformerlab.plugin")
except ImportError as e:
    print(f"Error importing transformerlab.plugin: {e}")
    # Add appropriate paths if needed
    print("Current sys.path:", sys.path)
    
    # Try multiple approaches to find the module
    if tfl_source_dir:
        tflab_sdk_path = os.path.join(tfl_source_dir, "transformerlab", "plugin_sdk")
        if os.path.exists(tflab_sdk_path):
            print(f"Adding {tflab_sdk_path} to sys.path")
            sys.path.append(tflab_sdk_path)
    
    # Also try the parent directory approach
    transformerlab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    print(f"Adding {transformerlab_path} to sys.path")
    sys.path.append(transformerlab_path)
    
    # If _TFL_SOURCE_CODE_DIR exists, try adding transformerlab directory
    if tfl_source_dir:
        plugin_parent = os.path.join(tfl_source_dir, "transformerlab")
        if os.path.exists(plugin_parent):
            print(f"Adding {plugin_parent} to sys.path")
            sys.path.append(plugin_parent)
    
    # Try importing again
    try:
        import transformerlab.plugin
        print("Successfully imported transformerlab.plugin after path adjustment")
    except ImportError as e:
        print(f"Still unable to import transformerlab.plugin: {e}")
        # Print directory structure to debug
        print("Listing potential transformerlab directories:")
        if tfl_source_dir:
            os.system(f"find {tfl_source_dir} -name plugin.py -type f")
        sys.exit(1)

# Initialize Accelerator only after the subprocess check and imports
accelerator = Accelerator()
print(f"Running with accelerate on {accelerator.num_processes} processes")

jinja_environment = Environment()

use_flash_attention = False

# Connect to the LLM Lab database
llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")
WORKSPACE_DIR: str | None = os.getenv("_TFL_WORKSPACE_DIR")
db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3")

# Rest of your code remains the same...
input_config = None
# open the input file that provides configs
with open(args.input_file) as json_file:
    input_config = json.load(json_file)
config = input_config["config"]
print("Input:")
print(input_config)

model_id = config["model_name"]
# model_id = "NousResearch/Llama-2-7b-hf"  # non-gated

JOB_ID = config["job_id"]

WANDB_LOGGING = config.get("log_to_wandb", None)

job = transformerlab.plugin.Job(config["job_id"])
job.update_progress(0)

# Get the dataset
# Datasets can be a huggingface ID or the name of a locally uploaded dataset
# Need to check the DB to figure out which because it changes how we load the dataset
# TODO: Refactor this to somehow simplify across training plugins
dataset_id = config["dataset_name"]
cursor = db.execute("SELECT location FROM dataset WHERE dataset_id = ?", (dataset_id,))
row = cursor.fetchone()
cursor.close()

# if no rows exist then the dataset hasn't been installed!
if row is None:
    print(f"No dataset named {dataset_id} installed.")
    job.set_job_completion_status("failed", f"No dataset named {dataset_id} installed.")
    raise RuntimeError(f"No dataset named {dataset_id} installed")

# dataset_location will be either "local" or "huggingface"
# (and if it's something else we're going to treat "huggingface" as default)
dataset_location = row[0]

# Load dataset - if it's local then pass it the path to the dataset directory
if dataset_location == "local":
    dataset_target = os.path.join(WORKSPACE_DIR, "datasets", dataset_id)

# Otherwise assume it is a Huggingface ID
else:
    dataset_target = dataset_id

dataset = load_dataset(dataset_target, split="train", trust_remote_code=True)

print(f"dataset size: {len(dataset)}")
print(dataset[randrange(len(dataset))])
print("formatting_template: " + config["formatting_template"])

template = jinja_environment.from_string(config["formatting_template"])


def format_instruction(mapping):
    return template.render(mapping)


print("formatted instruction: (example) ")
print(format_instruction(dataset[randrange(len(dataset))]))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
try:
    # For multi-GPU with quantization, auto is typically best
    device_map = None if accelerator.num_processes > 1 else "auto"
    print(f"Using device_map: {device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=use_flash_attention,
        device_map=device_map,
    )
    model.config.pretraining_tp = 1
    print(f"Successfully loaded model: {model_id}")
except Exception as e:
    print(f"Failed to load model: {e}")
    job.set_job_completion_status("failed", "Failed to load model")
    raise e

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
    job.set_job_completion_status("failed", "Failure to load tokenizer")
    raise e

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
    lora_alpha=int(config["lora_alpha"]),
    lora_dropout=float(config["lora_dropout"]),
    r=int(config["lora_r"]),
    bias="none",
    task_type="CAUSAL_LM",
)
print(f"LoRA config: alpha={config['lora_alpha']}, dropout={config['lora_dropout']}, r={config['lora_r']}")

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print("Model prepared for training with PEFT")

# This is where the tensorboard output is stored
output_dir: str = config["output_dir"]
print(f"Storing Tensorboard Output to: {output_dir}")

# In the json job_data column for this job, store the tensorboard output dir
db.execute(
    "UPDATE job SET job_data = json_insert(job_data, '$.tensorboard_output_dir', ?) WHERE id = ?",
    (output_dir, JOB_ID),
)
db.commit()

max_seq_length = int(config["maximum_sequence_length"])  # max sequence length for model and packing of the dataset
print(f"Maximum sequence length: {max_seq_length}")

report_to = ['tensorboard']

if WANDB_LOGGING:
    WANDB_LOGGING, report_to = transformerlab.plugin.test_wandb_login()
    if not WANDB_LOGGING:
        print("WANDB API Key not found. WANDB logging will be disabled. Please set the WANDB API Key in Settings.")

today = time.strftime("%Y%m%d-%H%M%S")
run_suffix = config.get("template_name", today)

args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=int(config["num_train_epochs"]),
    per_device_train_batch_size=int(config.get("batch_size", 4)),
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type=config.get("learning_rate_schedule", "constant"),
    max_seq_length=max_seq_length,
    disable_tqdm=False,  # disable tqdm since with packing values are in correct
    packing=True,
    run_name=f"job_{JOB_ID}_{run_suffix}",
    report_to=report_to,
    # Multi-GPU training parameters
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=True,
    # Let Accelerate handle device placement
    no_cuda=False,
)


class ProgressTableUpdateCallback(TrainerCallback):
    "A callback that prints updates progress percent in the TransformerLab DB"

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            # print(state.epoch)
            # print(state.global_step)
            # print(state.max_steps)
            # I think the following works but it may need to be
            # augmented by the epoch number
            progress = state.global_step / state.max_steps
            progress = int(progress * 100)
            print(f"Training progress: {progress}% (step {state.global_step}/{state.max_steps})")
            # db_job_id = JOB_ID
            # Write to jobs table in database, updating the
            # progress column:
            job.update_progress(progress)
            if job.should_stop:
                print("Job stop requested, terminating training...")
                control.should_training_stop = True
                return control

        return


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    formatting_func=format_instruction,
    args=args,
    callbacks=[ProgressTableUpdateCallback],
)
try:
    # train
    print("Starting training...")
    trainer.train()
    print("Training completed successfully")
except Exception as e:
    print(f"Training failed with error: {e}")
    job.set_job_completion_status("failed", "Failure during training")
    raise e


try:
    # save model
    print(f"Saving model to {config['adaptor_output_dir']}...")
    trainer.save_model(output_dir=config["adaptor_output_dir"])
    print("Model saved successfully")
except Exception as e:
    print(f"Failed to save model: {e}")
    job.set_job_completion_status("failed", "Failure to save model")
    raise e

job.set_job_completion_status("success", "Adaptor trained successfully")