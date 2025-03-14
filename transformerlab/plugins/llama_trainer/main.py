import torch
from jinja2 import Environment
from random import randrange
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from transformerlab.tfl_decorators import tfl_trainer

use_flash_attention = False
# Initialize Jinja environment
jinja_environment = Environment()


@tfl_trainer.job_wrapper(progress_start=0, progress_end=100)
def train_model():
    # Configuration is loaded automatically when tfl_trainer methods are called
    datasets = tfl_trainer.load_dataset()
    dataset = datasets["train"]

    report_to = tfl_trainer.setup_train_logging()

    print(f"Dataset loaded successfully with {len(dataset)} examples")
    print(dataset[randrange(len(dataset))])

    # Setup template for formatting
    print("Formatting template: " + tfl_trainer.formatting_template)
    template = jinja_environment.from_string(tfl_trainer.formatting_template)

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
    model_id = tfl_trainer.model_name

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
    lora_alpha = int(getattr(tfl_trainer, "lora_alpha", 16))
    lora_dropout = float(getattr(tfl_trainer, "lora_dropout", 0.05))
    lora_r = int(getattr(tfl_trainer, "lora_r", 8))

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
    output_dir = getattr(tfl_trainer, "output_dir", "./output")
    adaptor_output_dir = getattr(tfl_trainer, "adaptor_output_dir", "./adaptor")

    print(f"Storing Tensorboard Output to: {output_dir}")
    tfl_trainer.add_job_data("tensorboard_output_dir", output_dir)

    print("TFL Data:", tfl_trainer.output_dir)

    # Setup training arguments - use direct attribute access
    max_seq_length = int(getattr(tfl_trainer, "maximum_sequence_length", 2048))
    num_train_epochs = int(getattr(tfl_trainer, "num_train_epochs", 3))
    batch_size = int(getattr(tfl_trainer, "batch_size", 4))
    learning_rate = float(getattr(tfl_trainer, "learning_rate", 2e-4))
    lr_scheduler = getattr(tfl_trainer, "learning_rate_schedule", "constant")

    # Create unique run name
    import time

    today = time.strftime("%Y%m%d-%H%M%S")
    run_suffix = getattr(tfl_trainer, "template_name", today)

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
        run_name=f"job_{tfl_trainer.job_id}_{run_suffix}",
        report_to=report_to,
    )

    # Create progress callback
    progress_callback = tfl_trainer.create_progress_callback(framework="huggingface")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
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

    return {"status": "success", "message": "Adaptor trained successfully"}


train_model()
