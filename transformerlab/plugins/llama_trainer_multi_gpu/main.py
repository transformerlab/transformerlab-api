print("Loading multi-GPU trainer plugin...")
import os
import subprocess
import time
import traceback
import copy
from random import randrange
import torch.nn as nn

from transformerlab.sdk.v1.train import tlab_trainer
from transformerlab.plugin import WORKSPACE_DIR, get_python_executable

print("IMPORTS DONE")

tlab_trainer.add_argument(
    "--launched_with_accelerate", action="store_true", help="Flag to prevent recursive subprocess launching"
)

tlab_trainer.add_argument(
    "--run_sweeps", action="store_true", help="Run hyperparameter sweeps"
)

print("ARGUMENTS ADDED")

def find_lora_target_modules(model, keyword="proj"):
    """
    Returns all submodule names (e.g., 'q_proj') suitable for LoRA injection.
    These can be passed directly to LoraConfig as `target_modules`.
    """
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and keyword in name:
            # Keep full relative module name, excluding the root prefix (e.g., "model.")
            cleaned_name = ".".join(name.split('.')[1:]) if name.startswith("model.") else name
            module_names.add(cleaned_name.split('.')[-1])  # Use just the relative layer name
    return sorted(module_names)

print("FIND LORA TARGET MODULES DONE")

def setup_accelerate_environment():
    """Set up the environment for the accelerate launch subprocess"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.dirname(os.path.realpath(__file__))
    python_executable = get_python_executable(plugin_dir)
    api_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
    env = os.environ.copy()
    env["PATH"] = python_executable.replace("/python", ":") + env["PATH"]
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
print("SETUP ACCELERATE ENVIRONMENT DONE")

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
            
    # Determine if we're doing a sweep
    run_sweep = tlab_trainer.params.get("run_sweeps", True)
    # run_sweep = True
    
    # Check if we need to launch with accelerate
    if not tlab_trainer.params.get("launched_with_accelerate", False):
        print("Launching training with accelerate for multi-GPU...")
        env = setup_accelerate_environment()

        cmd = [
            "accelerate",
            "launch",
            f"--{train_device}",
            __file__,
            "--input_file",
            tlab_trainer.params.input_file,
            "--launched_with_accelerate",
        ]
        
        # Add GPU IDs if specified
        if gpu_ids:
            cmd.extend(["--gpu_ids", gpu_ids])
            
        # Add sweep parameter if we're doing a sweep
        if run_sweep is not None:
            cmd.extend(["--run_sweeps", "true"])

        result = subprocess.run(cmd, env=env)
        print(f"Subprocess completed with return code: {result.returncode}")
        return
    
    # If we're running sweeps
    if run_sweep is not None:
        # Run hyperparameter sweep
        sweep_results = tlab_trainer.run_sweep(train_function)

        # If there's a best config, train a final model with it
        if tlab_trainer.params.get("train_final_model", True) and sweep_results["best_config"]:
            print("\n--- Training final model with best configuration ---")

            # Create parameters with the best configuration
            final_params = copy.deepcopy(tlab_trainer.params)
            for k, v in sweep_results["best_config"].items():
                final_params[k] = v

            # Run final training
            result = train_function(**final_params)
            return {**sweep_results, "final_model_metrics": result}

        return sweep_results
    else:
        # Run single training
        return train_function(**tlab_trainer.params)

def train_function(**params):
    """Train a model with given parameters and return metrics"""
    import torch
    from jinja2 import Environment
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
    from trl import SFTConfig, SFTTrainer
    from accelerate import Accelerator
    
    # Initialize Accelerator
    accelerator = Accelerator()
    print(f"Running with accelerate on {accelerator.num_processes} processes")

    jinja_environment = Environment()
    use_flash_attention = False

    # Extract or set parameters
    model_id = params.get("model_name")
    output_dir = params.get("output_dir")
    adaptor_output_dir = params.get("adaptor_output_dir")

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(adaptor_output_dir, exist_ok=True)

    # Get dataset from parameters or load it
    if "datasets" in params:
        dataset = params["datasets"]["train"]
    else:
        datasets = tlab_trainer.load_dataset()
        dataset = datasets["train"]

    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])
    
    # Setup template for formatting
    formatting_template = params.get("formatting_template")
    print("formatting_template: " + formatting_template)
    template = jinja_environment.from_string(formatting_template)

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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            use_cache=False,
            use_flash_attention_2=use_flash_attention,
            device_map=device_map,
            trust_remote_code=True
        )
        lora_target_modules = find_lora_target_modules(model)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                use_flash_attention_2=use_flash_attention,
                device_map=device_map,
                trust_remote_code=True,
            )
        lora_target_modules = find_lora_target_modules(model)
    except Exception as e:
        print(f"Model loading error: {str(e)}")
        raise e

    model.config.pretraining_tp = 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Get LoRA parameters
    lora_alpha = int(params.get("lora_alpha", 16))
    lora_dropout = float(params.get("lora_dropout", 0.05))
    lora_r = int(params.get("lora_r", 8))
    
    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model
    model = prepare_model_for_kbit_training(model)
    try:
        model = get_peft_model(model, peft_config)
    except ValueError as e:
        print(f"PEFT model preparation error: {str(e)}")
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, peft_config)
    
    # Get other training parameters
    run_name = f"job_{params.get('job_id')}_{params.get('template_name', time.strftime('%Y%m%d-%H%M%S'))}"
    max_seq_length = int(params.get("maximum_sequence_length", 2048))
    learning_rate = float(params.get("learning_rate", 2e-4))
    num_epochs = int(params.get("num_train_epochs", 3))
    batch_size = int(params.get("batch_size", 4))
    lr_schedule = params.get("learning_rate_schedule", "constant")
    gradient_accumulation_steps = int(params.get("gradient_accumulation_steps", 2))
    

    # Setup evaluation dataset - use 10% of the data if enough examples
    if len(dataset) >= 10:
        split_dataset = dataset.train_test_split(test_size=0.1)
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]
    else:
        train_data = dataset
        eval_data = None


    args = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        max_seq_length=max_seq_length,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type=params.get("learning_rate_schedule", "constant"),
        save_total_limit=1,
        report_to=tlab_trainer.report_to if hasattr(tlab_trainer, "report_to") else None,
        eval_strategy="epoch",
        do_eval=True,
        packing=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        no_cuda=False,
    )

    # Create progress callback using tlab_trainer if available
    callbacks = []
    if hasattr(tlab_trainer, "create_progress_callback"):
        progress_callback = tlab_trainer.create_progress_callback(framework="huggingface")
        callbacks.append(progress_callback)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=format_instruction,
        args=args,
        callbacks=callbacks,
    )

    # Train model
    try:
        trainer.train()
        
        # Extract metrics
        metrics = {}
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history") and trainer.state.log_history:
            # Get final metrics
            for entry in trainer.state.log_history:
                if "eval_loss" in entry:
                    metrics.update(
                        {
                            f"eval/{k.replace('eval_', '')}" if k.startswith("eval_") else k: v
                            for k, v in entry.items()
                            if k not in ["epoch", "step"]
                        }
                    )
        
        # Save model - don't save during sweep runs unless explicitly requested
        is_sweep_run = "run_id" in params
        if not is_sweep_run or params.get("save_sweep_models") is not None:
            trainer.save_model(output_dir=adaptor_output_dir)
            print(f"Model saved successfully to {adaptor_output_dir}")

        # Save the fused model if requested and not in a sweep run
        if not is_sweep_run and params.get("fuse_model", False):
            # Merge the model with the adaptor
            try:
                model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                model_architecture = model_config.architectures[0]
                # Load the base model again
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        use_cache=False,
                        use_flash_attention_2=use_flash_attention,
                        device_map=None,
                        trust_remote_code=True,
                    )
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        use_flash_attention_2=use_flash_attention,
                        device_map=None,
                        trust_remote_code=True,
                    )
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                model.to(device)
                if "/" in model_id:
                    model_id = model_id.split("/")[-1]
                adaptor_name = params.get("adaptor_name", "default")
                fused_model_name = f"{model_id}_{adaptor_name}"
                fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)
                peft_model = PeftModel.from_pretrained(model, adaptor_output_dir)
                merged_model = peft_model.merge_and_unload()
                merged_model.save_pretrained(fused_model_location)
                tokenizer.save_pretrained(fused_model_location)
                print(f"Model saved successfully to {fused_model_location}")
                json_data = {
                    "description": f"A model trained and generated by Transformer Lab based on {params.get('model_name')}"
                }
                tlab_trainer.create_transformerlab_model(fused_model_name, model_architecture, json_data)
            except Exception as e:
                print(f"Model merging error: {str(e)}")
        
        return metrics
        
    except Exception as e:
        error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        if hasattr(tlab_trainer, "job"):
            tlab_trainer.job.set_job_completion_status("failed", "Training failed")
            tlab_trainer.add_job_data("error", error_msg)
        raise


train_model()