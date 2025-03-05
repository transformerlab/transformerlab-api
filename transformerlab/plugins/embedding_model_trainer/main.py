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
from sentence_transformers.losses import MatryoshkaLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from transformers import TrainerCallback

import transformerlab.plugin

# --- Utility Functions ---


def normalize_dataset_columns(dataset, dataset_type_str):
    """
    Rename the dataset columns to the lower-case names derived from the dataset_type.
    It excludes any column named 'id' (which is preserved).
    Assumes that the relevant text columns (in order) are the first columns
    that are not 'id'.
    """
    expected_names = [name.strip().lower() for name in dataset_type_str.split("|")]
    # Get all columns except 'id'
    cols = [col for col in dataset.column_names if col.lower() != "id"]
    if len(expected_names) > len(cols):
        raise ValueError(f"Dataset does not have enough columns to match the dataset type '{dataset_type_str}'")
    mapping = {}
    for i, new_name in enumerate(expected_names):
        mapping[cols[i]] = new_name
    return dataset.rename_columns(mapping)


def get_loss_function(loss_name, model):
    """
    Dynamically import and instantiate the loss function from sentence_transformers.losses.
    """
    loss_module = __import__("sentence_transformers.losses", fromlist=[loss_name])
    try:
        loss_cls = getattr(loss_module, loss_name)
        return loss_cls(model)
    except AttributeError:
        raise ValueError(f"Loss function '{loss_name}' is not available in sentence_transformers.losses.")


# Mapping from dataset type to allowed loss functions
ALLOWED_LOSSES = {
    "anchor | positive": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "MultipleNegativesSymmetricRankingLoss",
        "CachedMultipleNegativesSymmetricRankingLoss",
        "MegaBatchMarginLoss",
        "GISTEmbedLoss",
        "CachedGISTEmbedLoss",
    ],
    "anchor | positive | negative": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "TripletLoss",
        "CachedGISTEmbedLoss",
        "GISTEmbedLoss",
    ],
    "sentence_A | sentence_B | score": ["CoSENTLoss", "AnglELoss", "CosineSimilarityLoss"],
    "single sentences": ["ContrastiveTensionLoss", "DenoisingAutoEncoderLoss"],
    "single sentences | class": [
        "BatchAllTripletLoss",
        "BatchHardSoftMarginTripletLoss",
        "BatchHardTripletLoss",
        "BatchSemiHardTripletLoss",
    ],
    "anchor | anchor": ["ContrastiveTensionLossInBatchNegatives"],
    "damaged_sentence | original_sentence": ["DenoisingAutoEncoderLoss"],
    "sentence_A | sentence_B | class": ["SoftmaxLoss"],
    "anchor | positve/negative | class": ["ContrastiveLoss", "OnlineContrastiveLoss"],
    "anchor | positive | negative_1 | negative_2 | ... | negative_n": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "CachedGISTEmbedLoss",
    ],
    "id | anchor | positive": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "MultipleNegativesSymmetricRankingLoss",
        "CachedMultipleNegativesSymmetricRankingLoss",
        "MegaBatchMarginLoss",
        "GISTEmbedLoss",
        "CachedGISTEmbedLoss",
    ],
}

# --- Main Plugin Code ---

# Connect to the LLM Lab database
llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")
WORKSPACE_DIR = os.getenv("_TFL_WORKSPACE_DIR")
db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3")
db.row_factory = sqlite3.Row  # so we can access columns by name

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
fp16 = bool(config.get("fp16", True))
bf16 = bool(config.get("bf16", False))
max_samples = int(config.get("max_samples", -1))
matryoshka_dims = config.get("matryoshka_dims", [768, 512, 256, 128, 64])
WANDB_LOGGING = config.get("log_to_wandb", False)

# Get user-selected dataset type and loss function
user_dataset_type = config.get("dataset_type")
user_loss_function = config.get("loss_function")

# Validate loss function against dataset type
if user_dataset_type not in ALLOWED_LOSSES:
    raise ValueError(f"Dataset type '{user_dataset_type}' is not recognized.")
allowed = ALLOWED_LOSSES[user_dataset_type]
if user_loss_function not in allowed:
    raise ValueError(
        f"Selected loss function '{user_loss_function}' is not allowed for dataset type '{user_dataset_type}'. Allowed loss functions: {allowed}"
    )

# Get user-selected loss modifier (e.g., MatryoshkaLoss, AdaptiveLayerLoss, Matryoshka2dLoss)
loss_modifier_name = config.get("loss_modifier", "MatryoshkaLoss")

job = transformerlab.plugin.Job(job_id)
job.update_progress(0)

# Obtain dataset location from database
cursor = db.execute("SELECT * FROM dataset WHERE dataset_id = ?", (dataset_id,))
row = cursor.fetchone()
cursor.close()
if row is None:
    msg = f"No dataset named {dataset_id} installed."
    print(msg)
    job.set_job_completion_status("failed", msg)
    raise RuntimeError(msg)

dataset_location = row["location"]  # "local" or "huggingface"

if dataset_location == "local":
    dataset_target = os.path.join(WORKSPACE_DIR, "datasets", dataset_id)
else:
    dataset_target = dataset_id  # huggingface ID

json_data = json.loads(row["json_data"])
config_name = json_data.get("config_name", None)
if not config_name:
    msg = f"Dataset {dataset_id} does not have a 'config_name' in its json_data."
    print(msg)
    job.set_job_completion_status("failed", msg)
    raise RuntimeError(msg)

try:
    full_dataset = load_dataset(path=dataset_target, name=config_name, split="train", trust_remote_code=True)
except Exception as e:
    job.set_job_completion_status("failed", f"Failed to load dataset: {str(e)}")
    raise e

if max_samples > 0 and max_samples < len(full_dataset):
    full_dataset = full_dataset.select(range(max_samples))

# Normalize dataset columns according to the dataset type.
normalized_dataset = normalize_dataset_columns(full_dataset, user_dataset_type)

# Prepare an IR evaluator if the normalized dataset has "id", "anchor", and "positive"
evaluator = None
has_evaluator = False
if all(col in normalized_dataset.column_names for col in ["id", "anchor", "positive"]):
    corpus = dict(zip(normalized_dataset["id"], normalized_dataset["positive"]))
    queries = dict(zip(normalized_dataset["id"], normalized_dataset["anchor"]))
    relevant_docs = {q_id: [q_id] for q_id in queries}
    matryoshka_evaluators = []
    for dim in matryoshka_dims:
        ir_eval = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
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

# --- MATRYOSHKA REPRESENTATION LEARNING ---
# Dynamically load the user-selected inner loss function.
inner_train_loss = get_loss_function(user_loss_function, model)

# If a loss modifier is provided, dynamically load and wrap the inner loss.
if loss_modifier_name != "None":
    try:
        loss_modifier_module = __import__("sentence_transformers.losses", fromlist=[loss_modifier_name])
        loss_modifier_cls = getattr(loss_modifier_module, loss_modifier_name)
        if loss_modifier_name == "AdaptiveLayerLoss":
            # AdaptiveLayerLoss does not take matryoshka_dims as a parameter.
            train_loss = loss_modifier_cls(model=model, loss=inner_train_loss)
        else:
            train_loss = loss_modifier_cls(model=model, loss=inner_train_loss, matryoshka_dims=matryoshka_dims)
    except AttributeError:
        raise ValueError(f"Loss modifier '{loss_modifier_name}' is not available in sentence_transformers.losses.")
else:
    train_loss = inner_train_loss

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
    load_best_model_at_end=False,
    evaluation_strategy="epoch" if has_evaluator else "no",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    report_to=report_to,
    run_name=f"job_{job_id}_mrl_plugin",
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
            if job.should_stop:
                control.should_training_stop = True
        return control


# Determine which columns to pass to the trainer.
# If the normalized dataset contains 'anchor' and 'positive', use them; otherwise, use the whole dataset.
if all(col in normalized_dataset.column_names for col in ["anchor", "positive"]):
    train_data = normalized_dataset.select_columns(["anchor", "positive"])
else:
    train_data = normalized_dataset

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
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
