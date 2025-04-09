import xmlrpc.client
import json
import time
import os
import sys
import torch
from datasets import load_dataset
from pprint import pprint
from datetime import datetime
from tqdm import tqdm


class TransformerLabClient:
    """Client for reporting training progress to TransformerLab via XML-RPC"""

    def __init__(self, server_url="http://localhost:8338/trainer_rpc"):
        """Initialize the XML-RPC client"""
        self.server = xmlrpc.client.ServerProxy(server_url)
        self.job_id = None
        self.config = {}
        self.last_report_time = 0
        self.report_interval = 1  # seconds

    def start_job(self, config):
        """Register job with TransformerLab and get a job ID"""
        result = self.server.start_training(json.dumps(config))
        if result["status"] == "started":
            self.job_id = result["job_id"]
            self.config = config
            print(f"Registered job with TransformerLab. Job ID: {self.job_id}")
            return self.job_id
        else:
            raise Exception(f"Failed to start job: {result['message']}")


    def report_progress(self, progress, metrics=None):
        """Report training progress to TransformerLab"""
        if not self.job_id:
            return True

        # Rate limit reports
        current_time = time.time()
        if current_time - self.last_report_time < self.report_interval:
            return True

        self.last_report_time = current_time

        try:
            # Since get_training_status doesn't accept metrics directly,
            # we need to store them in the job data or use another method
            status = self.server.get_training_status(self.job_id, int(progress))
            
            # If metrics are important, consider logging them separately
            if metrics and hasattr(self.server, 'log_metrics'):
                self.server.log_metrics(self.job_id, json.dumps(metrics))
            
            if status.get("status") == "STOPPED":
                print("Job was stopped remotely. Terminating training...")
                return False
            return True
        except Exception as e:
            print(f"Error reporting progress: {e}")
            # Still return True to continue training despite reporting error
            return True

    def complete_job(self, status="COMPLETE", message="Training completed successfully"):
        """Mark job as complete in TransformerLab"""
        if not self.job_id:
            return
            
        try:
            # Use the dedicated complete_job method if it exists
            if hasattr(self.server, 'complete_job'):
                self.server.complete_job(self.job_id, status, message)
            else:
                # Fall back to using get_training_status with 100% progress
                self.report_progress(100)
                self.server.get_training_status(self.job_id, 100)
        except Exception as e:
            print(f"Error completing job: {e}")


def train():
    """Main training function that runs locally but reports to TransformerLab"""

    # Training configuration
    training_config = {
        "experiment_id": "alpha",
        "model_name": "HuggingFaceTB/SmolLM-135M-Instruct",
        "dataset": "Trelis/touch-rugby-rules",
        "template_name": "llama3instruct",
        "output_dir": "./output",
        "log_to_wandb": False,
        "_config": {
            "dataset_name": "Trelis/touch-rugby-rules",
            "lr": 2e-5,
            "num_train_epochs": 1,
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "max_seq_length": 512,
        },
    }

    # Initialize TransformerLab client
    tlab_client = TransformerLabClient()
    job_id = tlab_client.start_job(training_config)

    # Create output directory
    os.makedirs(training_config["output_dir"], exist_ok=True)

    try:
        # Log start time
        start_time = datetime.now()
        print(f"Training started at {start_time}")

        # Load the dataset
        print("Loading dataset...")
        dataset = load_dataset(training_config["dataset"])
        print(f"Loaded dataset with {len(dataset['train'])} training examples")

        # Report progress to TransformerLab
        tlab_client.report_progress(10, {"status": "dataset_loaded"})

        # Load tokenizer and model
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading model: {training_config['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(training_config["model_name"])
        model = AutoModelForCausalLM.from_pretrained(
            training_config["model_name"],
            device_map="auto",
        )

        # Configure tokenizer
        if not tokenizer.pad_token_id:
            tokenizer.pad_token = tokenizer.eos_token

        # Report progress
        tlab_client.report_progress(20, {"status": "model_loaded"})

        # Process dataset
        def format_instruction(example):
            """Format instruction and response using template"""
            instruction = example["prompt"]
            response = example["completion"]

            # Simple Llama-3 instruction template
            if training_config["template_name"] == "llama3instruct":
                formatted = f"<|begin_of_text|><|prompt|>{instruction}<|response|>{response}<|end_of_text|>"
            else:
                # Default simple template
                formatted = f"Instruction: {instruction}\n\nResponse: {response}"

            return {"formatted_text": formatted}

        tokenized_dataset = dataset.map(format_instruction)

        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["formatted_text"],
                padding="max_length",
                truncation=True,
                max_length=training_config["_config"]["max_seq_length"],
                return_tensors="pt",
            )

        processed_dataset = tokenized_dataset.map(
            tokenize_function, batched=True, remove_columns=tokenized_dataset["train"].column_names
        )

        # Report progress
        tlab_client.report_progress(30, {"status": "dataset_processed"})

        # Setup training arguments
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

        training_args = TrainingArguments(
            output_dir=os.path.join(training_config["output_dir"], f"job_{job_id}"),
            learning_rate=training_config["_config"]["lr"],
            num_train_epochs=training_config["_config"]["num_train_epochs"],
            per_device_train_batch_size=training_config["_config"]["batch_size"],
            gradient_accumulation_steps=training_config["_config"]["gradient_accumulation_steps"],
            warmup_ratio=training_config["_config"]["warmup_ratio"],
            weight_decay=training_config["_config"]["weight_decay"],
            logging_steps=20,
            save_steps=500,
            save_total_limit=2,
            report_to=[],  # We'll handle reporting to TransformerLab ourselves
        )

        # Define a custom callback to report progress to TransformerLab
        from transformers import TrainerCallback

        class TLabProgressCallback(TrainerCallback):
            def __init__(self, tlab_client):
                self.tlab_client = tlab_client

            def on_step_end(self, args, state, control, **kwargs):
                if state.is_local_process_zero:
                    if state.max_steps > 0:
                        # Calculate progress percentage (30-90%)
                        progress = 30 + ((state.global_step / state.max_steps) * 90)
                        metrics = {
                            "step": state.global_step,
                            "loss": state.log_history[-1]["loss"] if state.log_history else None,
                        }
                        # Report progress to TransformerLab
                        if not self.tlab_client.report_progress(progress, metrics):
                            # Job was stopped remotely
                            control.should_training_stop = True

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    metrics = {"step": state.global_step, "loss": logs["loss"]}
                    # Add other metrics if available
                    for key, value in logs.items():
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                    self.tlab_client.report_progress(30 + ((state.global_step / state.max_steps) * 90), metrics)

        # Setup trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[TLabProgressCallback(tlab_client)],
        )

        # Train the model
        print("Starting training...")
        trainer.train()

        # Save the final model
        print("Saving model...")
        trainer.save_model(os.path.join(training_config["output_dir"], f"final_model_{job_id}"))
        tokenizer.save_pretrained(os.path.join(training_config["output_dir"], f"final_model_{job_id}"))

        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        print(f"Training completed in {training_duration}")

        # Complete the job in TransformerLab
        tlab_client.complete_job("COMPLETE", "Training completed successfully")

        return {
            "status": "success",
            "job_id": job_id,
            "duration": str(training_duration),
            "output_dir": os.path.join(training_config["output_dir"], f"final_model_{job_id}"),
        }

    except KeyboardInterrupt:
        print("Training interrupted by user")
        tlab_client.complete_job("STOPPED", "Training stopped by user")
        return {"status": "stopped", "job_id": job_id}

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback

        traceback.print_exc()
        tlab_client.complete_job("STOPPED", f"Training failed: {str(e)}")
        return {"status": "error", "job_id": job_id, "error": str(e)}


if __name__ == "__main__":
    result = train()
    pprint(result)
