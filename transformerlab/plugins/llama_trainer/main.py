import os
import json
import copy
import time
import torch
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import traceback
from typing import Dict, List, Any, Optional, Union
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
from datasets import Dataset

try:
    from transformerlab.sdk.v1.train import tlab_trainer
    from sweep import HyperparameterSweep
except ImportError:
    # Handle import errors gracefully
    from transformerlab.sdk.v1.train import tlab_trainer
    # You may need to create sweep.py if it doesn't exist

jinja_environment = jinja2.Environment()

use_flash_attention = False

# Add the new parameters to your tlab_trainer
# tlab_trainer.add_argument("--run_sweeps", action="store_true", help="Run hyperparameter sweep instead of a single training job")
tlab_trainer.add_argument("--save_sweep_models", action="store_true", help="Save models for each sweep configuration")
tlab_trainer.add_argument("--sweep_metric", type=str, default="eval/loss", help="Metric to optimize in sweep")
tlab_trainer.add_argument("--lower_is_better", action="store_true", help="Whether lower values are better for the sweep metric")

@tlab_trainer.job_wrapper()
def train_model():
    """Main training function that handles both single runs and hyperparameter sweeps"""
    tlab_trainer._ensure_args_parsed()
    
    # # Load configuration if input file is provided
    # if getattr(tlab_trainer.params, "input_file", None):
    #     tlab_trainer.load_config()
    
    # # Update job status
    # tlab_trainer.job.update_status("RUNNING")
    # tlab_trainer.job.add_to_job_data("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))

    tlab_trainer.params.lora_r = int(tlab_trainer.params.get("lora_r", 8))
    tlab_trainer.params.lora_alpha = int(tlab_trainer.params.get("lora_alpha", 16))
    tlab_trainer.params.lora_dropout = float(tlab_trainer.params.get("lora_dropout", 0.05))
    tlab_trainer.params.maximum_sequence_length = int(tlab_trainer.params.get("maximum_sequence_length", 2048))
    tlab_trainer.params.num_train_epochs = int(tlab_trainer.params.get("num_train_epochs", 3))
    tlab_trainer.params.batch_size = int(tlab_trainer.params.get("batch_size", 4))
    tlab_trainer.params.learning_rate = float(tlab_trainer.params.get("learning_rate", 2e-4))    

    # Determine if we're doing a sweep
    run_sweep = tlab_trainer.params.get("run_sweeps", True)
    
    if run_sweep:
        result = run_hyperparameter_sweep()
    else:
        result = run_single_training()
    
    return result

def run_single_training():
    """Run a single training job with the current parameters"""
    # Load dataset
    datasets = tlab_trainer.load_dataset()
    dataset = datasets["train"]
    
    # Setup template for formatting
    template = jinja_environment.from_string(tlab_trainer.params.formatting_template)
    
    def format_instruction(example):
        """Format the instruction using the template"""
        return template.render(example)
    
    # Setup parameters
    model_id = tlab_trainer.params.model_name
    output_dir = tlab_trainer.params.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    adaptor_output_dir = tlab_trainer.params.adaptor_output_dir
    os.makedirs(adaptor_output_dir, exist_ok=True)
    
    max_length = tlab_trainer.params.get("maximum_sequence_length", 2048)
    batch_size = tlab_trainer.params.get("batch_size", 4)
    gradient_accumulation_steps = tlab_trainer.params.get("gradient_accumulation_steps", 2)
    
    # Setup quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    
    # Add padding token if needed
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA configuration
    lora_alpha = tlab_trainer.params.get("lora_alpha", 16)
    lora_dropout = tlab_trainer.params.get("lora_dropout", 0.05)
    lora_r = tlab_trainer.params.get("lora_r", 8)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Get other training parameters
    learning_rate = tlab_trainer.params.get("learning_rate", 2e-4)
    num_epochs = tlab_trainer.params.get("num_train_epochs", 3)
    
    # Create training arguments - keep the same configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to=tlab_trainer.report_to
    )

    
    # Create progress reporting callback
    callback = tlab_trainer.create_progress_callback()
    
    # Create SFTTrainer instead of manual tokenization and Trainer setup
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,  # Pass LoRA config directly
        processing_class=tokenizer,
        formatting_func=format_instruction,
        args=training_args,
        callbacks=[callback]  # Add the progress callback
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
                if 'loss' in entry:
                    metrics.update({k: v for k, v in entry.items() if k not in ['epoch', 'step']})
        
        # Save the model
        trainer.save_model(output_dir=adaptor_output_dir)
        print(f"Model saved successfully to {adaptor_output_dir}")
        
        # # Create TransformerLab model
        # tlab_trainer.create_transformerlab_model(
        #     fused_model_name=f"{tlab_trainer.params.model_name}_{tlab_trainer.params.adaptor_name}",
        #     model_architecture="llama",
        #     json_data={
        #         "base_model": model_id,
        #         "adaptor": os.path.basename(adaptor_output_dir),
        #         "adaptor_type": "lora",
        #         "configuration": {
        #             "lora_alpha": lora_alpha,
        #             "lora_dropout": lora_dropout,
        #             "lora_r": lora_r
        #         }
        #     },
        #     output_dir=tlab_trainer.params.output_dir
        # )
            # Train the model

        # Save the model
        try:
            trainer.save_model(output_dir=adaptor_output_dir)
            print(f"Model saved successfully to {adaptor_output_dir}")
        except Exception as e:
            print(f"Model saving error: {str(e)}")
            raise

        if tlab_trainer.params.get("fuse_model", False):
            # Merge the model with the adaptor
            try:
                model_config = AutoConfig.from_pretrained(model_id)
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
            
            return metrics
        
    except Exception as e:
        error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        tlab_trainer.job.set_job_completion_status("failed", "Training failed")
        tlab_trainer.add_job_data("error", error_msg)
        raise

def run_hyperparameter_sweep():
    """Run a hyperparameter sweep with the specified parameter ranges"""
    print("Starting hyperparameter sweep")
    
    # Load dataset once for all runs
    datasets = tlab_trainer.load_dataset()
    dataset = datasets["train"]    
    # Get sweep parameter definitions
    sweep_config = tlab_trainer.params.get("sweep_config", {})
    if not sweep_config:
        # Default sweep parameters if none specified
        sweep_config = {
            "learning_rate": [1e-4, 3e-4, 5e-4],
            "lora_alpha": [8, 16],
            "lora_r": [8, 16, 32],
        }
        print(f"Using default sweep configuration: {json.dumps(sweep_config, indent=2)}")
    
    # Setup the sweeper with a copy of current parameters
    base_params = copy.deepcopy(tlab_trainer.params)
    print("Base parameters for sweep:")
    print(json.dumps(base_params, indent=2))
    sweeper = HyperparameterSweep(base_params)
    
    # Add parameters to sweep
    for param_name, values in sweep_config.items():
        sweeper.add_parameter(param_name, values)
    
    # Generate all configurations
    configs = sweeper.generate_configs()
    total_configs = len(configs)
    
    print(f"Generated {total_configs} configurations for sweep")
    
    # Create a directory for sweep results
    sweep_dir = os.path.join(tlab_trainer.params.output_dir, f"sweep_{tlab_trainer.params.job_id}")
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Setup logging for the sweep
    sweep_log_path = os.path.join(sweep_dir, "sweep_results.json")
    tlab_trainer.add_job_data("sweep_log_path", sweep_log_path)
    tlab_trainer.add_job_data("sweep_configs", str(total_configs))
    
    # Process the dataset once for all runs
    template = jinja_environment.from_string(tlab_trainer.params.formatting_template)
    
    def format_instruction(example):
        """Format the instruction using the template"""
        return template.render(example)
    
    def process_dataset(example):
        """Process each example in the dataset"""
        prompt = format_instruction(example)
        return {"text": prompt}
    
    processed_dataset = dataset.map(process_dataset)
    
    # Run each configuration
    for i, config in enumerate(configs):
        run_id = f"run_{i+1}_of_{total_configs}"
        
        # Update job progress based on sweep progress
        overall_progress = int((i / total_configs) * 100)
        tlab_trainer.progress_update(overall_progress)
        
        # # Update job data with current run info
        # tlab_trainer.add_job_data("current_run", i+1)
        # tlab_trainer.add_job_data("current_config", {k: config[k] for k in sweep_config.keys()})
        
        print(f"\n--- Starting sweep {run_id} ---")
        print(f"Configuration: {json.dumps({k: config[k] for k in sweep_config.keys()}, indent=2)}")
        
        # Create run-specific output directories
        run_output_dir = os.path.join(sweep_dir, run_id)
        run_adaptor_dir = os.path.join(run_output_dir, "adaptor")
        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(run_adaptor_dir, exist_ok=True)
        
        # Store original params
        original_params = tlab_trainer.params
        
        try:
            # Create a new params object for this run
            run_params = copy.deepcopy(config)
            run_params["output_dir"] = run_output_dir
            run_params["adaptor_output_dir"] = run_adaptor_dir
            run_params["run_id"] = run_id
            
            # Replace the params temporarily
            tlab_trainer.params = DotDict(run_params)
            
            # Run training with this configuration
            metrics = run_training_for_sweep(processed_dataset)
            
            # Add result to sweeper
            sweeper.add_result(config, metrics)
            
            print(f"Run {i+1} completed with metrics: {json.dumps(metrics, indent=2)}")
            
        except Exception as e:
            error_msg = f"Error in sweep run {i+1}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            sweeper.add_result(config, {}, "failed")
        finally:
            # Restore original params
            tlab_trainer.params = original_params
        
        # Save intermediate sweep results
        with open(sweep_log_path, "w") as f:
            json.dump({
                "sweep_config": sweep_config,
                "results": sweeper.results,
                "completed_runs": i+1,
                "total_runs": total_configs
            }, f, indent=2)
    
    # Find best configuration
    metric_name = tlab_trainer.params.get("sweep_metric", "eval/loss")
    lower_is_better = tlab_trainer.params.get("lower_is_better", True)
    best_result = sweeper.get_best_config(metric_name, lower_is_better)
    
    if best_result:
        print("\n--- Sweep completed ---")
        print("Best configuration:")
        print(json.dumps(best_result["params"], indent=2))
        print("Metrics:")
        print(json.dumps(best_result["metrics"], indent=2))
        
        # Add best result to job data
        tlab_trainer.add_job_data("best_config", str(best_result["params"]))
        tlab_trainer.add_job_data("best_metrics", str(best_result["metrics"]))
        
        # Train the final model with the best configuration
        if tlab_trainer.params.get("train_final_model", True):
            print("\n--- Training final model with best configuration ---")
            
            # Create a new params object with the best config
            final_params = copy.deepcopy(original_params)
            for k, v in best_result["params"].items():
                final_params[k] = v
                
            # final_params["output_dir"] = os.path.join(sweep_dir, "final_model")
            # final_params["adaptor_output_dir"] = os.path.join(final_params["output_dir"], "adaptor")
            # final_params["template_name"] = f"{original_params.template_name}_best"
            
            # # Create directories
            # os.makedirs(final_params["output_dir"], exist_ok=True)
            # os.makedirs(final_params["adaptor_output_dir"], exist_ok=True)
            
            # Store original params
            original_params = tlab_trainer.params
            
            try:
                # Replace the params temporarily
                tlab_trainer.params = DotDict(final_params)
                
                # Run training with the best configuration
                run_single_training()
                
                print("Final model trained successfully")
                
            except Exception as e:
                error_msg = f"Error training final model: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                tlab_trainer.add_job_data("final_model_error", error_msg)
            finally:
                # Restore original params
                tlab_trainer.params = original_params
    
    # Return all results
    return {
        "sweep_results": sweeper.results,
        "best_config": best_result["params"] if best_result else None,
        "best_metrics": best_result["metrics"] if best_result else None,
        "sweep_log_path": sweep_log_path
    }

def run_training_for_sweep(processed_dataset):
    """Run a single training job for the sweep with the pre-processed dataset"""
    max_length = tlab_trainer.params.get("maximum_sequence_length", 2048)
    batch_size = tlab_trainer.params.get("batch_size", 4)
    gradient_accumulation_steps = tlab_trainer.params.get("gradient_accumulation_steps", 2)
    
    # Load model
    print("Loading model for sweep run")
    model_id = tlab_trainer.params.model_name
    if model_id is None:
        raise ValueError("Model ID is required for training")
    
    # Setup quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=False,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    
    # Add padding token if needed
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA configuration
    lora_alpha = tlab_trainer.params.get("lora_alpha", 16)
    lora_dropout = tlab_trainer.params.get("lora_dropout", 0.05)
    lora_r = tlab_trainer.params.get("lora_r", 8)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Get training parameters
    learning_rate = tlab_trainer.params.get("learning_rate", 2e-4)
    num_epochs = tlab_trainer.params.get("num_train_epochs", 3)
    output_dir = tlab_trainer.params.output_dir
    
    # Define formatting function for SFTTrainer
    template = jinja_environment.from_string(tlab_trainer.params.formatting_template)
    def format_instruction(example):
        """Format the instruction using the template"""
        return template.render(example)
    
    # Create training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,  # Save less during sweep to save space
        report_to=tlab_trainer.report_to,
        eval_strategy="epoch",
        do_eval=True,  # Enable evaluation during training
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="loss",  # Use loss as the metric
        greater_is_better=False,  # Lower loss is better
    )

    # # Get the original dataset from processed_dataset
    # # For sweep, we need to extract the original dataset before SFTTrainer formatting
    # if 'train' in processed_dataset.dataset:
    #     # If processed_dataset is a subset with train/test split
    #     original_dataset = processed_dataset.dataset['train']
    # else:
    #     # If processed_dataset is the whole dataset
    original_dataset = processed_dataset
    
    # Create evaluation dataset - use 10% of the data
    if len(original_dataset) >= 10:
        split_dataset = original_dataset.train_test_split(test_size=0.1)
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]
    else:
        train_data = original_dataset
        eval_data = None
    
    # Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=format_instruction,
        args=training_args
    )
    
    # Train the model
    try:
        trainer.train()
        
        # Extract metrics
        metrics = {}
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history") and trainer.state.log_history:
            # Get evaluation metrics
            for entry in trainer.state.log_history:
                if 'eval_loss' in entry:
                    metrics.update({f"eval/{k.replace('eval_', '')}" if k.startswith('eval_') else k: v 
                                  for k, v in entry.items() if k not in ['epoch', 'step']})
        
        # Save the model if requested
        if tlab_trainer.params.get("save_sweep_models", False):
            trainer.save_model(output_dir=tlab_trainer.params.adaptor_output_dir)
        
        return metrics
        
    except Exception as e:
        print(f"Training error in sweep run: {str(e)}")
        raise

# Define a DotDict class for easier parameter handling
class DotDict(dict):
    """Dictionary subclass that allows attribute access to dictionary keys"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Add the train_model function to make the plugin importable
train_model()