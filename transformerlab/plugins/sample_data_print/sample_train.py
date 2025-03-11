from transformerlab.tfl_decorators import tfl
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer

# Add custom arguments
tfl.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
tfl.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")


@tfl.job_wrapper(progress_start=0, progress_end=100)
@tfl.load_dataset(dataset_types=["train", "validation"])
def main(datasets):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(tfl.model_name)
    tokenizer = AutoTokenizer.from_pretrained(tfl.model_name)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = {
        "train": datasets["train"].map(tokenize_function, batched=True),
        "validation": datasets["validation"].map(tokenize_function, batched=True),
    }

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=tfl.num_epochs,
        learning_rate=tfl.learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
    )

    # Create TFL progress callback
    tfl_callback = tfl.create_progress_callback(framework="huggingface")

    # Create trainer with our callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        callbacks=[tfl_callback],
    )

    # Run training
    trainer.train()

    # Save model
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")

    return {"model_path": "./final_model"}


if __name__ == "__main__":
    main()
