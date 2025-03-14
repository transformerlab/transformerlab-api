import os
import subprocess
import time
from random import randrange

from transformerlab.tfl_decorators import tfl_trainer

# Add custom arguments
tfl_trainer.add_argument("--launched_with_accelerate", action="store_true",
                        help="Flag to prevent recursive subprocess launching")

def setup_accelerate_environment():
    """Set up the environment for the accelerate launch subprocess"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
    env = os.environ.copy()
    tfl_source_dir = os.environ.get("_TFL_SOURCE_CODE_DIR")
    python_path = env.get("PYTHONPATH", "")
    paths_to_include = [api_dir]
    
    if tfl_source_dir:
        tflab_sdk_path = os.path.join(tfl_source_dir, "transformerlab", "plugin_sdk")
        paths_to_include.append(tflab_sdk_path)
        plugin_parent = os.path.join(tfl_source_dir, "transformerlab")
        paths_to_include.append(plugin_parent)
    
    if python_path:
        paths_to_include.append(python_path)
    
    env["PYTHONPATH"] = ":".join(paths_to_include)
    return env

@tfl_trainer.job_wrapper(progress_start=0, progress_end=100)
def train_model():
    """Main training function using TrainerTFLPlugin"""
    # Get configuration from tfl_trainer
     # Configuration is loaded automatically when tfl_trainer methods are called
    datasets = tfl_trainer.load_dataset()
    dataset = datasets["train"]

    # Setup logging on WANDB and Tensorboard
    report_to = tfl_trainer.setup_train_logging()

    # Set up accelerate configuration
    accelerate_config = {
        "cuda": "multi_gpu",
        "cpu": "cpu",
        "tpu": "tpu",
    }
    
    train_device = accelerate_config.get(tfl_trainer.train_device, "multi_gpu")
    print(f"Training setup for accelerate launch: {train_device}")
    
    # Configure GPU IDs
    gpu_ids = None
    if train_device == "multi_gpu":
        gpu_ids = tfl_trainer.gpu_ids
        if gpu_ids and gpu_ids != "auto":
            gpu_ids = str(gpu_ids)
        if gpu_ids == "auto":
            gpu_ids = None
    
    # Check if we need to launch with accelerate
    if not getattr(tfl_trainer, "launched_with_accelerate", False):
        print("Launching training with accelerate for multi-GPU...")
        env = setup_accelerate_environment()
        
        cmd = [
            "accelerate", "launch",
            f"--{train_device}",
            __file__,
            "--input_file", tfl_trainer.input_file,
            "--launched_with_accelerate"
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
    model_id = tfl_trainer.model_name
    
    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])
    print("formatting_template: " + tfl_trainer.formatting_template)
    
    template = jinja_environment.from_string(tfl_trainer.formatting_template)
    
    def format_instruction(mapping):
        return template.render(mapping)
    
    print("formatted instruction: (example) ")
    print(format_instruction(dataset[randrange(len(dataset))]))
    
    # Model configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
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
        lora_alpha=int(tfl_trainer.lora_alpha),
        lora_dropout=float(tfl_trainer.lora_dropout),
        r=int(tfl_trainer.lora_r),
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Prepare model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Training configuration
    output_dir = getattr(tfl_trainer, "output_dir", "./output")
    tfl_trainer.add_job_data("tensorboard_output_dir", output_dir)
    
    # Setup WandB - decorator would handle this check
    today = time.strftime("%Y%m%d-%H%M%S")
    run_suffix = getattr(tfl_trainer, "template_name", today)
    max_seq_length = int(tfl_trainer.maximum_sequence_length)
        
    
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=int(tfl_trainer.num_train_epochs),
        per_device_train_batch_size=int(tfl_trainer.batch_size),
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=float(tfl_trainer.learning_rate),
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type=tfl_trainer.learning_rate_schedule,
        max_seq_length=max_seq_length,
        disable_tqdm=False,
        packing=True,
        run_name=f"job_{tfl_trainer.job_id}_{run_suffix}",
        report_to=report_to,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        no_cuda=False,
    )
    
    # Create progress callback using tfl_trainer
    progress_callback = tfl_trainer.create_progress_callback(framework="huggingface")
    
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
    trainer.save_model(output_dir=tfl_trainer.adaptor_output_dir)
    
    # Return success message
    return "Adaptor trained successfully"



train_model()