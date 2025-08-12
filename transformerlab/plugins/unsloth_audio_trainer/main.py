import time
import os
# from random import randrange
import torch
# import shutil
# from functools import partial

# HAS_AMD = False
# if shutil.which("rocminfo") is not None:
#     HAS_AMD = True
#     # AMD-specific optimizations
#     os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:128"
#     os.environ["HIP_VISIBLE_DEVICES"] = "0"
#     # Disable some problematic CUDA-specific features
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
# if torch.cuda.is_available():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from unsloth import FastModel
from transformers import TrainingArguments, Trainer
from unsloth import is_bfloat16_supported
from datasets import Audio
from transformers import AutoProcessor


from transformers import CsmForConditionalGeneration
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel  # noqa: E402
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Mxfp4Config  # noqa: E402
# from trl import SFTConfig, SFTTrainer  # noqa: E402
# import torch.nn as nn  # noqa: E402


# from transformerlab.plugin import WORKSPACE_DIR, format_template  # noqa: E402
from transformerlab.sdk.v1.train import tlab_trainer  # noqa: E402


def preprocess_example(example, speaker_key="source", processor=None):
    conversation = [
        {
            "role": str(example[speaker_key]),
            "content": [
                {"type": "text", "text": example["text"]},
                {"type": "audio", "path": example["audio"]["array"]},
            ],
        }
    ]

    try:
        model_inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            text_kwargs = {
                "padding": "max_length", # pad to the max_length
                "max_length": 256, # this should be the max length of audio
                "pad_to_multiple_of": 8,
                "padding_side": "right",
            },
            audio_kwargs = {
                "sampling_rate": 24_000,
                "max_length": 240001, # max input_values length of the whole dataset
                "padding": "max_length",
            },
            common_kwargs = {"return_tensors": "pt"},
        )
    except Exception as e:
        print(f"Error processing example with text '{example['text'][:50]}...': {e}")
        return None

    required_keys = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
    processed_example = {}
    # print(model_inputs.keys())
    for key in required_keys:
        if key not in model_inputs:
            print(f"Warning: Required key '{key}' not found in processor output for example.")
            return None

        value = model_inputs[key][0]
        processed_example[key] = value


    # Final check (optional but good)
    if not all(isinstance(processed_example[key], torch.Tensor) for key in processed_example):
        print(f"Error: Not all required keys are tensors in final processed example. Keys: {list(processed_example.keys())}")
        return None

    return processed_example


@tlab_trainer.job_wrapper(wandb_project_name="TLab_Training", manual_logging=True)
def train_text_to_speech_unsloth():
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

    print("Loading model...")
    try:
        # TODO: what parameters are needed to passed
        model, processor = FastModel.from_pretrained(
            model_name = model_id,
            max_seq_length= max_seq_length,
            # dtype = None, # Leave as None for auto-detection
            auto_model = CsmForConditionalGeneration, # TODO: check if this is needed!
            load_in_4bit = False, # Keep it false because voice models are small and we can keep the high quality result.
        )
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return f"Failed to load model: {str(e)}"


    # Setup LoRA

    # # Setup LoRA - use direct attribute access with safe defaults
    

    model = FastModel.get_peft_model(
        model,
        r = lora_r, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",], # TODO: implement something like find_lora_target_modules
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
)
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_trainable}")

    processor = AutoProcessor.from_pretrained(model_id)


    # Getting the speaker id is important for multi-speaker models and speaker consistency
    speaker_key = "source"
    if "source" not in dataset.column_names and "speaker_id" not in dataset.column_names:
        print("No speaker found, adding default \"source\" of 0 for all examples")
        new_column = ["0"] * len(dataset)
        dataset = dataset.add_column("source", new_column)
    elif "source" not in dataset.column_names and "speaker_id" in dataset.column_names:
        speaker_key = "speaker_id"

    target_sampling_rate = 24000
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

    processed_ds = dataset.map(
        lambda example: preprocess_example(example, speaker_key=speaker_key, processor=processor),
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
            warmup_rate = 0.03,
            max_steps = 60,
            learning_rate = learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit", # which one? "paged_adamw_32bit"
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

    
train_text_to_speech_unsloth()
