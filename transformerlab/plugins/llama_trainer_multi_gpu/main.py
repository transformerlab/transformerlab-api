import os
import subprocess
import time
from random import randrange

from transformerlab.sdk.v1.train import tlab_trainer

# Add custom arguments
tlab_trainer.add_argument(
    "--launched_with_accelerate", action="store_true", help="Flag to prevent recursive subprocess launching"
)


def setup_accelerate_environment():
    """Set up the environment for the accelerate launch subprocess"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
    env = os.environ.copy()
    tlab_source_dir = os.environ.get("_TFL_SOURCE_CODE_DIR")
    python_path = env.get("PYTHONPATH", "")
    paths_to_include = [api_dir]

    if tlab_source_dir:
        tlabab_sdk_path = os.path.join(tlab_source_dir, "transformerlab", "plugin_sdk")
        paths_to_include.append(tlabab_sdk_path)
        plugin_parent = os.path.join(tlab_source_dir, "transformerlab")
        paths_to_include.append(plugin_parent)

    if python_path:
        paths_to_include.append(python_path)

    env["PYTHONPATH"] = ":".join(paths_to_include)
    return env


@tlab_trainer.job_wrapper()
def train_model():
    """Main training function using TrainerTLabPlugin"""
    # Get configuration from tlab_trainer
    # Configuration is loaded automatically when tlab_trainer methods are called
    datasets = tlab_trainer.load_dataset()
    dataset = datasets["train"]

    # Set up accelerate configuration
    accelerate_config = {
        "cuda": "multi_gpu",
        "cpu": "cpu",
        "tpu": "tpu",
    }

    train_device = accelerate_config.get(tlab_trainer.params.train_device, "multi_gpu")
    print(f"Training setup for accelerate launch: {train_device}")

    # Configure GPU IDs
    gpu_ids = None
    if train_device == "multi_gpu":
        gpu_ids = tlab_trainer.params.gpu_ids
        if gpu_ids and gpu_ids != "auto":
            gpu_ids = str(gpu_ids)
        if gpu_ids == "auto":
            gpu_ids = None

    # Check if we need to launch with accelerate
    if tlab_trainer.params.get("launched_with_accelerate", False):
        print("Launching training with accelerate for multi-GPU...")
        env = setup_accelerate_environment()

        cmd = [
            "accelerate",
            "launch",
            f"--{train_device}",
            __file__,
            "--input_file",
            tlab_trainer.input_file,
            "--launched_with_accelerate",
        ]
        if gpu_ids:
            cmd.extend(["--gpu_ids", gpu_ids])

        result = subprocess.run(cmd, env=env)
        print(f"Subprocess completed with return code: {result.returncode}")
        return

    # Import dependencies after the subprocess check
    import torch
    from jinja2 import Environment
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    from accelerate import Accelerator

    # Initialize Accelerator
    accelerator = Accelerator()
    print(f"Running with accelerate on {accelerator.num_processes} processes")

    jinja_environment = Environment()
    use_flash_attention = False

    # Get model info
    model_id = tlab_trainer.params.model_name

    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])
    print("formatting_template: " + tlab_trainer.params.formatting_template)

    template = jinja_environment.from_string(tlab_trainer.params.formatting_template)

    def format_instruction(mapping):
        return template.render(mapping)

    print("formatted instruction: (example) ")
    print(format_instruction(dataset[randrange(len(dataset))]))

    # Model configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    device_map = None if accelerator.num_processes > 1 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=use_flash_attention,
        device_map=device_map,
    )
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=int(tlab_trainer.params.lora_alpha),
        lora_dropout=float(tlab_trainer.params.lora_dropout),
        r=int(tlab_trainer.params.lora_r),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Training configuration
    output_dir = tlab_trainer.params.get("output_dir", "./output")

    # Setup WandB - decorator would handle this check
    today = time.strftime("%Y%m%d-%H%M%S")
    run_suffix = tlab_trainer.params.get("template_name", today)
    max_seq_length = int(tlab_trainer.params.maximum_sequence_length)

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=int(tlab_trainer.params.num_train_epochs),
        per_device_train_batch_size=int(tlab_trainer.params.batch_size),
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=float(tlab_trainer.params.learning_rate),
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type=tlab_trainer.params.learning_rate_schedule,
        max_seq_length=max_seq_length,
        disable_tqdm=False,
        packing=True,
        run_name=f"job_{tlab_trainer.params.job_id}_{run_suffix}",
        report_to=tlab_trainer.report_to,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        no_cuda=False,
    )

    # Create progress callback using tlab_trainer
    progress_callback = tlab_trainer.create_progress_callback(framework="huggingface")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        formatting_func=format_instruction,
        args=args,
        callbacks=[progress_callback],
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model(output_dir=tlab_trainer.params.adaptor_output_dir)

    # Return success message
    return "Adaptor trained successfully"


train_model()
