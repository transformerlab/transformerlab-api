import time
import os
import torch
import importlib

from unsloth import FastModel
from transformers import TrainingArguments, Trainer
from unsloth import is_bfloat16_supported
from datasets import Audio
from transformers import AutoProcessor, AutoConfig

from trainer import CsmAudioTrainer

from transformerlab.sdk.v1.train import tlab_trainer  # noqa: E402

def find_lora_target_modules(model):
    patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = set()
    for name, module in model.named_modules():
        for pattern in patterns:
            if pattern in name:
                found.add(pattern)
    return list(found) if found else patterns


@tlab_trainer.job_wrapper(wandb_project_name="TLab_Training", manual_logging=True)
def train_model():
    print(f"!!!!", tlab_trainer.params)
    # Configuration is loaded automatically when tlab_trainer methods are called
    datasets = tlab_trainer.load_dataset()
    dataset = datasets["train"]

    # Get configuration values
    lora_alpha = int(tlab_trainer.params.get("lora_alpha", 16))
    lora_dropout = float(tlab_trainer.params.get("lora_dropout", 0))
    lora_r = int(tlab_trainer.params.get("lora_r", 8))
    model_id = tlab_trainer.params.model_name

    max_seq_length = int(tlab_trainer.params.maximum_sequence_length)
    learning_rate = float(tlab_trainer.params.learning_rate)
    learning_rate_schedule = tlab_trainer.params.get("learning_rate_schedule", "constant")
    max_grad_norm = float(tlab_trainer.params.max_grad_norm)
    batch_size = int(tlab_trainer.params.batch_size)
    num_epochs = int(tlab_trainer.params.num_train_epochs)
    weight_decay = float(tlab_trainer.params.weight_decay)
    adam_beta1 = float(tlab_trainer.params.adam_beta1)
    adam_beta2 = float(tlab_trainer.params.adam_beta2)
    adam_epsilon = float(tlab_trainer.params.adam_epsilon)
    output_dir = tlab_trainer.params.output_dir
    report_to = tlab_trainer.report_to
    sampling_rate = int(tlab_trainer.params.get("sampling_rate", 24000))
    max_steps = int(tlab_trainer.params.get("max_steps", -1))
    

    model_trainer = CsmAudioTrainer(model_name=model_id)

    print("Loading model...")
    model, processor = model_trainer.load_model(max_seq_length=max_seq_length)


    # Setup LoRA - use direct attribute access with safe defaults
    print("Setting up LoRA...")
    try:
        model = FastModel.get_peft_model(
            model,
            r = lora_r,
            target_modules = find_lora_target_modules(model),
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
    )
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_trainable}")
    except Exception as e:
        print(f"Failed to set up LoRA: {str(e)}")
        return f"Failed to set up LoRA: {str(e)}"

    
    processor = AutoProcessor.from_pretrained(model_id)
    # Getting the speaker id is important for multi-speaker models and speaker consistency
    speaker_key = "source"
    if "source" not in dataset.column_names and "speaker_id" not in dataset.column_names:
        print("No speaker found, adding default \"source\" of 0 for all examples")
        new_column = ["0"] * len(dataset)
        dataset = dataset.add_column("source", new_column)
    elif "source" not in dataset.column_names and "speaker_id" in dataset.column_names:
        speaker_key = "speaker_id"

    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    max_audio_length = max(len(example["audio"]["array"]) for example in dataset)
    processed_ds = dataset.map(
        lambda example: preprocess_example(
            example,
            speaker_key=speaker_key,
            processor=processor,
            max_seq_length=max_seq_length,
            max_audio_length=max_audio_length,
            sampling_rate=sampling_rate
        ),
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset",
    )

    print(f"Processed dataset length: {len(processed_ds)}")

    # Create progress callback using tlab_trainer
    progress_callback = tlab_trainer.create_progress_callback(framework="huggingface")
    
    # Training run name
    today = time.strftime("%Y%m%d-%H%M%S")
    run_suffix = tlab_trainer.params.get("template_name", today)
    trainer = Trainer(
        model = model,
        train_dataset = processed_ds,
        callbacks = [progress_callback],
        args = TrainingArguments(
            logging_dir=os.path.join(output_dir, f"job_{tlab_trainer.params.job_id}_{run_suffix}"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = 2,
            gradient_checkpointing=True,
            warmup_ratio = 0.03,
            max_steps = max_steps,
            learning_rate = learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            save_strategy="epoch",
            weight_decay = weight_decay,
            lr_scheduler_type = learning_rate_schedule,
            max_grad_norm=max_grad_norm,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            disable_tqdm=False,
            seed = 3407,
            output_dir = output_dir,
            run_name=f"job_{tlab_trainer.params.job_id}_{run_suffix}",
            report_to = report_to
        ),
)
    # Train the model
    try:
        trainer.train()
    except Exception as e:
        raise e
    
    # Save the model
    try:
        trainer.save_model(output_dir=tlab_trainer.params.adaptor_output_dir)
    except Exception as e:
        raise e
    
    # Return success message
    return "Audio model trained successfully."

    
train_model()
