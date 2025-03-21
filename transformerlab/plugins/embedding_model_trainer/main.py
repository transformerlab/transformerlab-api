import argparse
import json
import os
import sqlite3
import time
import requests
import random

import torch
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from transformers import TrainerCallback
from werkzeug.utils import secure_filename

import transformerlab.plugin
from jinja2 import Environment

jinja_environment = Environment()

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

def add_noise(sentence):
    """Randomly removes some words to create a noised version."""
    words = sentence.split()
    if len(words) < 2:
        return sentence  # Skip short sentences
    num_words_to_remove = max(1, len(words) // 4)  # Remove 25% of words
    indices_to_remove = random.sample(range(len(words)), num_words_to_remove)
    noised_words = [w for i, w in enumerate(words) if i not in indices_to_remove]
    return " ".join(noised_words)

def load_dataset_column(dataset, column_name = "context"):
    """Load a specific column from a dataset and return the sentences as a list."""
    
    # Check if column exists
    if column_name not in dataset.column_names:
        raise ValueError(f"Column '{column_name}' not found in dataset. Available columns: {dataset.column_names}")
    
    sentences = dataset[column_name]
    print(f"Loaded {len(sentences)} sentences from column '{column_name}'.")
    return sentences


def prepare_training_data(sentences):
    data_pairs = [
        {"noised_text": add_noise(s), "original_text": s}
        for s in sentences if isinstance(s, str) and len(s) > 0
    ]
    return Dataset.from_list(data_pairs)

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

WORKSPACE_DIR = os.getenv("_TFL_WORKSPACE_DIR")

print(f"Workspace directory: {WORKSPACE_DIR}")

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

model_id = config.get("embedding_model", "BAAI/bge-base-en-v1.5")  # Default embedding model
model_file_path = config.get("embedding_model_file_path", None)
if model_file_path or model_file_path != "":
    model_id = model_file_path
    print(f"Using model file path: {model_file_path} as the primary model.")
dataset_id = config.get("dataset_name", "sentence-transformers/all-nli")
job_id = config["job_id"]

# Set final model name from model_id, template_name, and job_id
template_name = config.get("template_name", "Fine-tune-embed-" + time.strftime("%Y%m%d-%H%M%S"))
# model_id_output = model_id.replace("/", "~~~")
model_id_output = secure_filename(model_id)
final_model_name = f"{model_id_output}_{template_name}_{job_id}"

# Define OUTPUT_DIR using final model name
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "models", final_model_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

num_train_epochs = int(config.get("num_train_epochs", 3))
batch_size = int(config.get("batch_size", 16))
learning_rate = float(config.get("learning_rate", 2e-5))
warmup_ratio = float(config.get("warmup_ratio", 0.1))
fp16 = bool(config.get("fp16", False))
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

try:
    if config_name is None:
        full_dataset = load_dataset(path=dataset_target, split="train", trust_remote_code=True)
    else:
        full_dataset = load_dataset(path=dataset_target, name=config_name, split="train", trust_remote_code=True)
except Exception as e:
    job.set_job_completion_status("failed", "Failed to load dataset")
    raise e

if max_samples > 0 and max_samples < len(full_dataset):
    full_dataset = full_dataset.select(range(max_samples))

# Normalize dataset columns according to the dataset type.
if user_dataset_type != "single sentences":
    normalized_dataset = normalize_dataset_columns(full_dataset, user_dataset_type)
else:
    sentences = load_dataset_column(full_dataset, config.get("text_column_name", "context"))
    normalized_dataset = prepare_training_data(sentences)


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
    model = SentenceTransformer(
        model_id, device=("cuda" if torch.cuda.is_available() else "cpu"), local_files_only=os.path.exists(model_id)
    )
except Exception as e:
    print(f"Failed to load model {model_id}: {e}")
    job.set_job_completion_status("failed", "Could not load model")
    raise e

# --- LOSS MODIFIER SECTION ---

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

report_to = ["tensorboard"]
if WANDB_LOGGING:
    import transformerlab.plugin

    WANDB_LOGGING, report_to = transformerlab.plugin.test_wandb_login()
    if not WANDB_LOGGING:
        print("W&B API Key not found. Disabling W&B logging.")

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
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
    run_name=f"job_{job_id}_embedding_model_plugin",
    metric_for_best_model="eval_dim_128_cosine_ndcg@10",
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
    print("Training completed.")
except Exception as e:
    job.set_job_completion_status("failed", "Failure during training")
    raise e

try:
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
except Exception as e:
    job.set_job_completion_status("failed", "Failure to save model")
    raise e

try:
    # Load the model into Transformer Lab App
    response = requests.get("http://localhost:8338/model/import_from_local_path", params={"model_path": OUTPUT_DIR})
    if response.status_code != 200:
        print(f"Failed to import model to Transformer Lab: {response.text}")
    else:
        print("Model imported to Transformer Lab successfully.")

except Exception as e:
    job.set_job_completion_status("failed", "Failed to import model to Transformer Lab")
    raise e

job.set_job_completion_status("success", f"Embedding model {final_model_name} trained successfully")
