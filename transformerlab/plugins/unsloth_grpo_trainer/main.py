import json
import sqlite3
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer 
import argparse
from unsloth import FastLanguageModel, PatchFastRL
import torch
from transformers import BitsAndBytesConfig, TrainerCallback
import os
import transformerlab.plugin

from jinja2 import Environment

jinja_environment = Environment()

use_flash_attention = False

PatchFastRL("GRPO",FastLanguageModel)

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
dataset_id = config["dataset_name"]
max_seq_length = int(config.get("maximum_sequence_length", 1024))
max_completion_length = int(config.get("maximum_completion_length", 512))
lora_rank = int(config.get("lora_r", 16))
lora_alpha = int(config.get("lora_alpha",32))
learning_rate = float(config.get("learning_rate",0.005))
learning_rate_schedule = config.get("learning_rate_schedule","constant")
max_grad_norm = float(config.get("max_grad_norm",0.3))
batch_size = int(config.get("batch_size", 4))
num_epochs = int(config["num_train_epochs"])
weight_decay = float(config.get("weight_decay",0.0))
adam_beta1 = float(config.get("adam_beta1",0.9))
adam_beta2 = float(config.get("adam_beta2",0.999))
adam_epsilon = float(config.get("adam_epsilon",1e-8))

question_formatting_template = config.get("input_template","")
answer_formatting_template = config.get("output_template","")
system_prompt = config.get("instruction_template","")

start_thinking_string = config.get("start_thinking_string","<reasoning>")
end_thinking_string = config.get("end_thinking_string","</reasoning>")
start_answer_string = config.get("start_answer_string","<answer>")
end_answer_string = config.get("end_answer_string","</answer>")

output_dir: str = config["output_dir"]
JOB_ID = config["job_id"]

job = transformerlab.plugin.Job(JOB_ID)
job.update_progress(0)

cursor = db.execute("SELECT location FROM dataset WHERE dataset_id = ?", (dataset_id,))
row = cursor.fetchone()
cursor.close()

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

def format_instruction(template, mapping):
    return template.render(mapping)

question_template = jinja_environment.from_string(question_formatting_template)
answer_template = jinja_environment.from_string(answer_formatting_template)

dataset = dataset.map(lambda x: {
    'prompt': [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': format_instruction(question_template, x)}
    ],
    #most gsm8k datasets have the answer after 4 hashes, so I have this split here
    'answer' : format_instruction(answer_template, x).split("#### ")[-1] 
})

def extract_answer(text: str) -> str:
    answer = text.split(f"{start_answer_string}")[-1]
    answer = answer.split(f"{end_answer_string}")[0]
    return answer.strip()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def count_xml(text) -> float:
    count = 0.0
    if text.count(f"{start_thinking_string}\n") == 1:
        count += 0.125
    if text.count(f"\n{end_thinking_string}\n") == 1:
        count += 0.125
    if text.count(f"\n{start_answer_string}\n") == 1:
        count += 0.125
        count -= len(text.split(f"\n{end_answer_string}\n")[-1])*0.001
    if text.count(f"\n{end_answer_string}") == 1:
        count += 0.125
        count -= (len(text.split(f"\n{end_answer_string}")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

#print(f"dataset size: {len(dataset)}")
#print(dataset[randrange(len(dataset))])
#print("formatting_template: " + config["formatting_template"])

#print("formatted instruction: (example) ")
#print(format_instruction(dataset[randrange(len(dataset))]))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
try:
    model,tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length, 
        max_lora_rank = lora_rank, 
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=use_flash_attention,
        device_map="auto",
    )
    model.config.pretraining_tp = 1
except Exception as e:
    job.set_job_completion_status("failed", "Failed to load model")
    raise e

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_alpha,
    use_gradient_checkpointing = "unsloth"
)

# This is where the tensorboard output is stored
print(f"Storing Tensorboard Output to: {output_dir}")

# In the json job_data column for this job, store the tensorboard output dir
db.execute(
    "UPDATE job SET job_data = json_insert(job_data, '$.tensorboard_output_dir', ?) WHERE id = ?",
    (output_dir, JOB_ID),
)
db.commit()

print(max_seq_length)


args = GRPOConfig(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=learning_rate,
    bf16=True,
    tf32=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=0.03,
    max_completion_length = max_completion_length,
    lr_scheduler_type=learning_rate_schedule,
    adam_beta1 = adam_beta1,
    adam_beta2 = adam_beta2,
    adam_epsilon= adam_epsilon,
    disable_tqdm=False,  # disable tqdm since with packing values are in correct
    report_to=["tensorboard"],
)


class ProgressTableUpdateCallback(TrainerCallback):
    "A callback that prints updates progress percent in the TransformerLab DB"

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            progress = state.global_step / state.max_steps
            progress = int(progress * 100)
            job.update_progress(progress)
            if job.should_stop:
                control.should_training_stop = True
                return control

        return


trainer = GRPOTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        correctness_reward_func,
    ],
    args=args,
    callbacks=[ProgressTableUpdateCallback],
)

try:
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
