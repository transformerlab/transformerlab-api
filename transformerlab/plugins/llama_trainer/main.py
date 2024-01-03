# The following is adapted from
# https://www.philschmid.de/instruction-tune-llama-2

import json
from random import randrange
import sqlite3
from string import Template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
import os

use_flash_attention = False

# Connect to the LLM Lab database
llmlab_root_dir = os.getenv('LLM_LAB_ROOT_PATH')
db = sqlite3.connect(llmlab_root_dir + "/workspace/llmlab.sqlite3")


# Get all parameters provided to this script from Transformer Lab
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
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

model_id = input_config["experiment"]["config"]["foundation"]
# model_id = "NousResearch/Llama-2-7b-hf"  # non-gated

# Load dataset from the hub
dataset = load_dataset(config['dataset_name'], split="train")

print(f"dataset size: {len(dataset)}")
print(dataset[randrange(len(dataset))])
print("formatting_template: " + config['formatting_template'])

# Takes in a template in the form of String.Template from Python's standard library
# https://docs.python.org/3.4/library/string.html#template-strings
# e.g. "$who likes $what"
template = Template(config['formatting_template'])


def format_instruction(mapping):
    return template.substitute(mapping)


print("formatted instruction: (example) ")
print(format_instruction(dataset[randrange(len(dataset))]))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=use_flash_attention,
    device_map="auto",
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
    lora_alpha=int(config['lora_alpha']),
    lora_dropout=float(config['lora_dropout']),
    r=int(config['lora_r']),
    bias="none",
    task_type="CAUSAL_LM",
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

JOB_ID = config["job_id"]

output_dir: str = config["output_dir"]

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=int(config['num_train_epochs']),
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=float(config['learning_rate']),
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=False,  # disable tqdm since with packing values are in correct
    report_to=["tensorboard"],
)

max_seq_length = 2048  # max sequence length for model and packing of the dataset


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
            db_job_id = JOB_ID
            # Write to jobs table in database, updating the
            # progress column:
            db.execute(
                "UPDATE job SET progress = ? WHERE id = ?",
                (progress, db_job_id),
            )
            db.commit()
        return


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
    callbacks=[ProgressTableUpdateCallback]
)

# train
trainer.train()

# save model
trainer.save_model()
