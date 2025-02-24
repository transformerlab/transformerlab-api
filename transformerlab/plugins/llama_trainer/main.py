# The following is adapted from
# https://www.philschmid.de/instruction-tune-llama-2

import argparse
import json
import os
import sqlite3
import time
from random import randrange

import torch
from datasets import load_dataset
from jinja2 import Environment
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

import transformerlab.plugin

jinja_environment = Environment()

use_flash_attention = False

# Connect to the LLM Lab database
llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")
WORKSPACE_DIR: str | None = os.getenv("_TFL_WORKSPACE_DIR")
db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3")

# Get all parameters provided to this script from Transformer Lab
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args, unknown = parser.parse_known_args()

print("Arguments:")
print(args)

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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=use_flash_attention,
        device_map="auto",
    )
    model.config.pretraining_tp = 1
except Exception as e:
    job.set_job_completion_status("failed", "Failed to load model")
    raise e

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
except Exception as e:
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

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


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
print(max_seq_length)

if WANDB_LOGGING:
    # Test if WANDB API Key is available
    def test_wandb_login():
        import netrc
        from pathlib import Path

        netrc_path = Path.home() / (".netrc" if os.name != "nt" else "_netrc")
        if netrc_path.exists():
            auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
            if auth:
                return True
            else:
                return False
        else:
            return False

    if not test_wandb_login():
        print("WANDB API Key not found. WANDB logging will be disabled. Please set the WANDB API Key in Settings.")
        WANDB_LOGGING = False
        os.environ["WANDB_DISABLED"] = "true"
        report_to = ["tensorboard"]
    else:
        os.environ["WANDB_DISABLED"] = "false"
        report_to = ["tensorboard", "wandb"]
        os.environ["WANDB_PROJECT"] = "TFL_Training"

today = time.strftime("%Y%m%d-%H%M%S")

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
    run_name=f"job_{JOB_ID}_{today}",
    report_to=report_to,
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
            # db_job_id = JOB_ID
            # Write to jobs table in database, updating the
            # progress column:
            job.update_progress(progress)
            if job.should_stop:
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
    trainer.train()
except Exception as e:
    job.set_job_completion_status("failed", "Failure during training")
    raise e


try:
    # save model
    trainer.save_model(output_dir=config["adaptor_output_dir"])
except Exception as e:
    job.set_job_completion_status("failed", "Failure to save model")
    raise e

job.set_job_completion_status("success", "Adaptor trained successfully")
