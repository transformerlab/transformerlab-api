import argparse
import json
import os
import sqlite3
import time

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from transformers import TrainerCallback

import transformerlab.plugin

# Connect to the LLM Lab database
llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")
WORKSPACE_DIR = os.getenv("_TFL_WORKSPACE_DIR")
db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3")

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args, unknown = parser.parse_known_args()

print("Arguments:")
print(args)

with open(args.input_file) as f:
    input_config = json.load(f)

config = input_config["config"]
print("Input config:")
print(json.dumps(config, indent=2))

model_id = config.get("model_name", "BAAI/bge-base-en-v1.5")  # Default embedding model
dataset_id = config.get("dataset_name", "sentence-transformers/all-nli")
job_id = config["job_id"]
output_dir = config.get("output_dir", "./output")

num_train_epochs = int(config.get("num_train_epochs", 3))
batch_size = int(config.get("batch_size", 16))
learning_rate = float(config.get("learning_rate", 2e-5))
warmup_ratio = float(config.get("warmup_ratio", 0.1))
fp16 = bool(config.get("fp16", False))
bf16 = bool(config.get("bf16", False))
max_samples = int(config.get("max_samples", -1))

# Matryoshka Dimensions: from largest to smallest
matryoshka_dims = config.get("matryoshka_dims", [768, 512, 256, 128, 64])

WANDB_LOGGING = config.get("log_to_wandb", False)

job = transformerlab.plugin.Job(job_id)
job.update_progress(0)

# Obtain dataset location
cursor = db.execute("SELECT location FROM dataset WHERE dataset_id = ?", (dataset_id,))
row = cursor.fetchone()
cursor.close()
if row is None:
    msg = f"No dataset named {dataset_id} installed."
    print(msg)
    job.set_job_completion_status("failed", msg)
    raise RuntimeError(msg)

dataset_location = row[0]  # "local" or "huggingface"
if dataset_location == "local":
    dataset_target = os.path.join(WORKSPACE_DIR, "datasets", dataset_id)
else:
    dataset_target = dataset_id  # huggingface ID

try:
    full_dataset = load_dataset(path=dataset_target, data_dir="pair", split="train", trust_remote_code=True)
except Exception as e:
    job.set_job_completion_status("failed", f"Failed to load dataset: {str(e)}")
    raise e

if max_samples > 0 and max_samples < len(full_dataset):
    full_dataset = full_dataset.select(range(max_samples))

train_dataset = full_dataset  # In this example, we use the entire data as training

# Prepare an IR evaluator for each dimension in matryoshka_dims
# We'll only do this if the dataset has 'id', 'anchor', and 'positive'
evaluator = None
has_evaluator = False

if all(col in train_dataset.column_names for col in ["id", "anchor", "positive"]):
    # Build queries, corpus, relevant doc mapping
    corpus = dict(zip(train_dataset["id"], train_dataset["positive"]))
    queries = dict(zip(train_dataset["id"], train_dataset["anchor"]))
    relevant_docs = {q_id: [q_id] for q_id in queries}

    # Build an IR evaluator for each dimension
    matryoshka_evaluators = []
    for dim in matryoshka_dims:
        ir_eval = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # <-- crucial for MRL evaluation
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_eval)

    if matryoshka_evaluators:
        evaluator = SequentialEvaluator(matryoshka_evaluators)
        has_evaluator = True

print(f"Loading Sentence Transformer model {model_id}")
try:
    model = SentenceTransformer(model_id, device=("cuda" if torch.cuda.is_available() else "cpu"))
except Exception as e:
    job.set_job_completion_status("failed", f"Could not load model: {str(e)}")
    raise e

# ===========================
# MATRYOSHKA REPRESENTATION LEARNING
# ===========================
# 1. Create an "inner" loss function
inner_train_loss = MultipleNegativesRankingLoss(model)

# 2. Wrap it in a MatryoshkaLoss
train_loss = MatryoshkaLoss(
    model=model,
    loss=inner_train_loss,
    matryoshka_dims=matryoshka_dims,  # from largest to smallest
)

# If user wants to log to W&B
if WANDB_LOGGING:
    import transformerlab.plugin

    wandb_logging_enabled, report_to = transformerlab.plugin.test_wandb_login()
    if not wandb_logging_enabled:
        print("W&B API Key not found. Disabling W&B logging.")
        report_to = []
else:
    report_to = []

training_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    fp16=fp16,
    bf16=bf16,
    warmup_ratio=warmup_ratio,
    learning_rate=learning_rate,
    load_best_model_at_end=False,  # change as desired
    evaluation_strategy="epoch" if has_evaluator else "no",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    report_to=report_to,
    run_name=f"job_{job_id}_mrl_plugin",
    # We'll track the dimension-128 NDCG@10, for example
    metric_for_best_model=f"eval_dim_128_cosine_ndcg@10",
    greater_is_better=True,
)


class ProgressUpdateCallback(TrainerCallback):
    """A callback that reports training progress to TransformerLab DB."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            total_steps = state.max_steps if state.max_steps else 1
            progress = int((state.global_step / total_steps) * 100)
            job.update_progress(progress)

            # If user requests stop
            if job.should_stop:
                control.should_training_stop = True
        return control


# Prepare the trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.select_columns(["anchor", "positive"])
    if all(col in train_dataset.column_names for col in ["anchor", "positive"])
    else train_dataset,
    loss=train_loss,
    evaluator=evaluator,
    callbacks=[ProgressUpdateCallback],
)

try:
    trainer.train()
except Exception as e:
    job.set_job_completion_status("failed", f"Failure during training: {str(e)}")
    raise e

try:
    trainer.save_model(output_dir)
except Exception as e:
    job.set_job_completion_status("failed", f"Failure to save model: {str(e)}")
    raise e

job.set_job_completion_status("success", "Embedding model (with Matryoshka) trained successfully")
