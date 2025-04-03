import time
import os
from random import randrange

import torch
from jinja2 import Environment
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from trl import SFTConfig, SFTTrainer


from transformerlab.plugin import WORKSPACE_DIR, generate_model_json
from transformerlab.sdk.v1.train import tlab_trainer

use_flash_attention = False
# Initialize Jinja environment
jinja_environment = Environment()

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@tlab_trainer.job_wrapper()
def train_model():
    # Configuration is loaded automatically when tlab_trainer methods are called
    datasets = tlab_trainer.load_dataset()
    dataset = datasets["train"]

    print(f"Dataset loaded successfully with {len(dataset)} examples")
    print(dataset[randrange(len(dataset))])

    # Setup template for formatting
    print("Formatting template: " + tlab_trainer.params.formatting_template)
    template = jinja_environment.from_string(tlab_trainer.params.formatting_template)

    def format_instruction(mapping):
        return template.render(mapping)

    print("Formatted instruction example:")
    print(format_instruction(dataset[randrange(len(dataset))]))

    # Setup quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model_id = tlab_trainer.params.model_name

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            use_cache=False,
            use_flash_attention_2=use_flash_attention,
            device_map="auto",
        )
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print(f"Model and tokenizer loaded successfully: {model_id}")
    except Exception as e:
        print(f"Model loading error: {str(e)}")
        raise

    # Setup LoRA - use direct attribute access with safe defaults
    lora_alpha = int(tlab_trainer.params.get("lora_alpha", 16))
    lora_dropout = float(tlab_trainer.params.get("lora_dropout", 0.05))
    lora_r = int(tlab_trainer.params.get("lora_r", 8))

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Get output directories - use direct attribute access
    output_dir = tlab_trainer.params.get("output_dir", "./output")
    adaptor_output_dir = tlab_trainer.params.get("adaptor_output_dir", "./adaptor")

    # Setup training arguments - use direct attribute access
    max_seq_length = int(tlab_trainer.params.get("maximum_sequence_length", 2048))
    num_train_epochs = int(tlab_trainer.params.get("num_train_epochs", 3))
    batch_size = int(tlab_trainer.params.get("batch_size", 4))
    learning_rate = float(tlab_trainer.params.get("learning_rate", 2e-4))
    lr_scheduler = tlab_trainer.params.get("learning_rate_schedule", "constant")

    today = time.strftime("%Y%m%d-%H%M%S")
    run_suffix = tlab_trainer.params.get("template_name", today)

    # Setup training configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=learning_rate,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type=lr_scheduler,
        max_seq_length=max_seq_length,
        disable_tqdm=False,
        packing=True,
        run_name=f"job_{tlab_trainer.params.job_id}_{run_suffix}",
        report_to=tlab_trainer.report_to,
    )

    # Create progress callback
    progress_callback = tlab_trainer.create_progress_callback(framework="huggingface")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=format_instruction,
        args=training_args,
        callbacks=[progress_callback],
    )

    # Train the model
    try:
        trainer.train()
        print("Training completed successfully")
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise

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
            architecture = model_config.architectures[0]
            # Load the base model again
            model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    use_cache=False,
                    use_flash_attention_2=use_flash_attention,
                    device_map="auto",
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
            print(f"Model saved successfully to {fused_model_location}")
            json_data = {
                        "description": f"A model trained and generated by Transformer Lab based on {tlab_trainer.params.model_name}"
                    }
            generate_model_json(fused_model_name, architecture, json_data=json_data)

        except Exception as e:
            print(f"Model merging error: {str(e)}")


train_model()
