import os
import json
import copy
import time
import torch
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import traceback
from typing import Dict, List, Any, Optional, Union
import torch.nn as nn
from jinja2 import Environment
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer, SFTConfig
import jinja2

from transformerlab.sdk.v1.train import tlab_trainer
from transformerlab.plugin import WORKSPACE_DIR

jinja_environment = Environment()

use_flash_attention = False

# Add the new parameters
tlab_trainer.add_argument("--run_sweeps", action="store_true", help="Run hyperparameter sweep instead of a single training job")
tlab_trainer.add_argument("--save_sweep_models", action="store_true", help="Save models for each sweep configuration")
tlab_trainer.add_argument("--sweep_metric", type=str, default="eval/loss", help="Metric to optimize in sweep")
tlab_trainer.add_argument("--lower_is_better", action="store_true", help="Whether lower values are better for the sweep metric")


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


def train_function(**params):
    """Train a model with given parameters and return metrics"""
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
    
    # Setup template for formatting
    template = jinja_environment.from_string(params.get("formatting_template"))
    
    def format_instruction(example):
        """Format the instruction using the template"""
        return template.render(example)
    
    # Set training parameters
    max_length = params.get("maximum_sequence_length", 2048)
    batch_size = params.get("batch_size", 4)
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 2)
    
    # Setup quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    print(f"Loading model: {model_id}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                use_cache=False,
                use_flash_attention_2=use_flash_attention,
                device_map="auto",
                trust_remote_code=True,
            )
        lora_target_modules = find_lora_target_modules(model)
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print(f"Model and tokenizer loaded successfully: {model_id}")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                use_flash_attention_2=use_flash_attention,
                device_map="auto",
                trust_remote_code=True,
            )
        lora_target_modules = find_lora_target_modules(model)

    except Exception as e:
        print(f"Model loading error: {str(e)}")
        raise


    model.config.pretraining_tp = 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    
    # Add padding token if needed
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup LoRA configuration
    lora_alpha = params.get("lora_alpha", 16)
    lora_dropout = params.get("lora_dropout", 0.05)
    lora_r = params.get("lora_r", 8)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    try:
        model = get_peft_model(model, lora_config)
    except ValueError as e:
        print(f"PEFT model preparation error. Determining target modules...")
        lora_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
    


    
    # Get other training parameters
    learning_rate = params.get("learning_rate", 2e-4)
    num_epochs = params.get("num_train_epochs", 3)
    
    # Create training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=10,
        # save_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=tlab_trainer.report_to if hasattr(tlab_trainer, 'report_to') else None,
        eval_strategy="epoch",
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # Create progress reporting callback
    callback = tlab_trainer.create_progress_callback() if hasattr(tlab_trainer, 'create_progress_callback') else None
    callbacks = [callback] if callback else []
    
    # Setup evaluation dataset - use 10% of the data if enough examples
    if len(dataset) >= 10:
        split_dataset = dataset.train_test_split(test_size=0.1)
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]
    else:
        train_data = dataset
        eval_data = None
    
    # Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=format_instruction,
        args=training_args,
        callbacks=callbacks
    )
    
    # Train the model
    try:
        trainer.train()
        print("Training completed successfully")
        
        # Extract metrics
        metrics = {}
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history") and trainer.state.log_history:
            # Get final metrics
            for entry in trainer.state.log_history:
                if 'eval_loss' in entry:
                    metrics.update({f"eval/{k.replace('eval_', '')}" if k.startswith('eval_') else k: v 
                                  for k, v in entry.items() if k not in ['epoch', 'step']})
        
        # Save the model - don't save during sweep runs unless explicitly requested
        is_sweep_run = "run_id" in params
        if not is_sweep_run or params.get("save_sweep_models", False):
            trainer.save_model(output_dir=adaptor_output_dir)
            print(f"Model saved successfully to {adaptor_output_dir}")

        # Save the tokenizer
        tokenizer.save_pretrained(adaptor_output_dir)
        print(f"Tokenizer saved successfully to {adaptor_output_dir}")

        # Save the fused model if train_final_model is set
        if not is_sweep_run:
            if params.get("fuse_model", False):
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
                                device_map="auto",
                                trust_remote_code=True,
                            )
                    except TypeError:
                        model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                quantization_config=bnb_config,
                                use_flash_attention_2=use_flash_attention,
                                device_map="auto",
                                trust_remote_code=True,
                            )

                    if "/" in model_id:
                        model_id = model_id.split("/")[-1]
                    adaptor_name = tlab_trainer.params.get("adaptor_name", "default")
                    fused_model_name = f"{model_id}_{adaptor_name}"
                    fused_model_location = os.path.join(WORKSPACE_DIR, "models", fused_model_name)
                    peft_model = PeftModel.from_pretrained(model, tlab_trainer.params.adaptor_output_dir)
                    merged_model = peft_model.merge_and_unload()
                    merged_model.save_pretrained(fused_model_location)
                    tokenizer.save_pretrained(fused_model_location)
                    print(f"Fused model saved successfully to {fused_model_location}")
                    json_data = {
                                "description": f"A model trained and generated by Transformer Lab based on {tlab_trainer.params.model_name}"
                            }
                    tlab_trainer.create_transformerlab_model(fused_model_name, model_architecture, json_data)
                except Exception as e:
                    print(f"Model merging error: {str(e)}")
                    raise

        return metrics 

    except Exception as e:
        error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        if hasattr(tlab_trainer, 'job'):
            tlab_trainer.job.set_job_completion_status("failed", "Training failed")
            tlab_trainer.add_job_data("error", error_msg)
        raise


@tlab_trainer.job_wrapper()
def run_plugin():
    """Main entry point for the training plugin"""
    tlab_trainer._ensure_args_parsed()
    
    # Initialize parameters
    tlab_trainer.params.lora_r = int(tlab_trainer.params.get("lora_r", 8))
    tlab_trainer.params.lora_alpha = int(tlab_trainer.params.get("lora_alpha", 16))
    tlab_trainer.params.lora_dropout = float(tlab_trainer.params.get("lora_dropout", 0.05))
    tlab_trainer.params.maximum_sequence_length = int(tlab_trainer.params.get("maximum_sequence_length", 2048))
    tlab_trainer.params.num_train_epochs = int(tlab_trainer.params.get("num_train_epochs", 3))
    tlab_trainer.params.batch_size = int(tlab_trainer.params.get("batch_size", 4))
    tlab_trainer.params.learning_rate = float(tlab_trainer.params.get("learning_rate", 2e-4))    

    # Determine if we're doing a sweep
    run_sweep = tlab_trainer.params.get("run_sweeps", False)
    
    if run_sweep:
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


# Execute the plugin when imported
run_plugin()