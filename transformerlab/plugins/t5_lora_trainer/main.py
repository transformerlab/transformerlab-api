# The following was adapted from
# https://www.philschmid.de/fine-tune-flan-t5-peft
# and works with T5 models using
# transformers.Seq2SeqTrainer
# It is designed to only work with datasets that look
# like the samsum dataset. If you see the function called
# preprocess_function you will see it assumes specific
# column names in the dataset and it ignores the
# formatting_template parameter. If you want to use
# this to train T5 on a different dataset, edit the
# preprocess function

import argparse
import json
import os
import sqlite3
import time

from dataclasses import dataclass, field
from random import randrange
from typing import List, Optional

import evaluate
import numpy as np
import transformers
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

# Connect to the LLM Lab database
llmlab_root_dir = os.getenv("LLM_LAB_ROOT_PATH")
WORKSPACE_DIR: str | None = os.getenv("_TFL_WORKSPACE_DIR")
db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3")


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "v"])
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    peft_name: str = ""


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/flan_t5_small")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    num_data: int = -1
    preprocessed_path: str = field(default=None, metadata={"help": "Path to the preprocessed training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


class Trainer:
    def __init__(self):
        self.model = None
        self.model_id = None
        self.peft_model_id = None

        self.tokenizer = None
        self.trainer = None

        self.dataset = None
        self.dataset_id = None

        self.should_abort = False

        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

        self.tokenized_dataset = None

        self.max_source_length = None
        self.max_target_length = None

    def set_model(self, model_id):
        self.model_id = model_id
        return

    def load_dataset(self, dataset_id):
        assert dataset_id is not None

        if dataset_id == self.dataset_id:
            return

        # Load dataset from the hub
        self.dataset = load_dataset(dataset_id, trust_remote_code=True)

        print(f"Train dataset size: {len(self.dataset['train'])}")
        print(f"Test dataset size: {len(self.dataset['test'])}")

        self.train_dataset = load_dataset(dataset_id, split="train[:100%]", trust_remote_code=True)
        self.test_dataset = load_dataset(dataset_id, split="test[:100%]", trust_remote_code=True)
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

        self.dataset = DatasetDict({"train": self.train_dataset, "test": self.test_dataset})
        print(f"Train dataset size: {len(self.dataset['train'])}")
        print(f"Test dataset size: {len(self.dataset['test'])}")

        self.dataset_id = dataset_id
        return

    def tokenize_dataset(self):
        assert self.model_id is not None

        # Load tokenizer of current model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # The maximum total input sequence length after tokenization.
        # Sequences longer than this will be truncated, sequences shorter will be padded.
        tokenized_inputs = concatenate_datasets([self.train_dataset, self.test_dataset]).map(
            lambda x: self.tokenizer(x["dialogue"], truncation=True),
            batched=True,
            remove_columns=["dialogue", "summary"],
        )
        input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
        # take 85 percentile of max length for better utilization
        self.max_source_length = int(np.percentile(input_lenghts, 85))
        print(f"Max source length: {self.max_source_length}")

        # The maximum total sequence length for target text after tokenization.
        # Sequences longer than this will be truncated, sequences shorter will be padded."
        tokenized_targets = concatenate_datasets([self.train_dataset, self.test_dataset]).map(
            lambda x: self.tokenizer(x["summary"], truncation=True),
            batched=True,
            remove_columns=["dialogue", "summary"],
        )
        target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
        # take 90 percentile of max length for better utilization
        self.max_target_length = int(np.percentile(target_lenghts, 90))
        print(f"Max target length: {self.max_target_length}")
        return

    def __preprocess_function(self, sample, padding="max_length"):
        # add prefix to the input for t5
        inputs = ["summarize: " + item for item in sample["dialogue"]]
        # tokenize inputs
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(
            text_target=sample["summary"],
            max_length=self.max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the
        # labels by -100 when we want to ignore padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(id if id != self.tokenizer.pad_token_id else -100) for id in input_ids]
                for input_ids in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenized_dataset = self.dataset.map(
            self.__preprocess_function,
            batched=True,
            remove_columns=["dialogue", "summary", "id"],
        )
        print(f"Keys of tokenized dataset: {list(self.tokenized_dataset['train'].features)}")

        # save datasets to disk for later easy loading
        self.tokenized_dataset["train"].save_to_disk("workspace/temp/data/train")
        self.tokenized_dataset["test"].save_to_disk("workspace/temp/data/eval")

    def load_model(self):
        # load model from the hub
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, load_in_8bit=True, device_map="auto")

    def train_lora(
        self,
        peft_model_id,
        lora_r,
        lora_alpha,
        lora_dropout,
        learning_rate,
        num_train_epochs,
        logging_steps,
        job_id,
        output_dir,
        adaptor_output_dir,
        wandb_logging=None,
    ):
        self.peft_model_id = peft_model_id

        # Right now the training args and lora args are ignored. Later on we should look at them

        # Define LoRA Config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        # prepare int-8 model for training
        self.model = prepare_model_for_int8_training(self.model)

        # add LoRA adaptor
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # we want to ignore tokenizer pad token in the loss
        label_pad_token_id = -100
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )

        print(f"Storing Tensorboard Output to: {output_dir}")
        # In the json job_data column for this job, store the tensorboard output dir
        db.execute(
            "UPDATE job SET job_data = json_insert(job_data, '$.tensorboard_output_dir', ?) WHERE id = ?",
            (output_dir, JOB_ID),
        )
        db.commit()

        if wandb_logging:
            # Test if WANDB API Key is available
            def test_wandb_login():
                import netrc
                from pathlib import Path

                netrc_path = Path.home() / (".netrc" if os.name != "nt" else "_netrc")
                if netrc_path.exists():
                    auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
                    if auth:
                        return True
                    else:
                        return False
                else:
                    return False

            if not test_wandb_login():
                print(
                    "WANDB API Key not found. WANDB logging will be disabled. Please set the WANDB API Key in Settings."
                )
                wandb_logging = False
                os.environ["WANDB_DISABLED"] = "true"
                report_to = ["tensorboard"]
            else:
                os.environ["WANDB_DISABLED"] = "false"
                report_to = ["tensorboard", "wandb"]
                os.environ["WANDB_PROJECT"] = "TFL_Training"

        today = time.strftime("%Y%m%d-%H%M%S")
        # Define training args
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=learning_rate,  # higher learning rate
            num_train_epochs=num_train_epochs,
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=logging_steps,
            save_strategy="no",
            run_name=f"job_{JOB_ID}_{today}",
            report_to=report_to,
        )

        class ProgressTableUpdateCallback(TrainerCallback):
            "A callback that prints updates progress percent in the Transformer Lab DB"

            def on_step_end(self, args, state, control, **kwargs):
                if state.is_local_process_zero:
                    # print(state.epoch)
                    # print(state.global_step)
                    # print(state.max_steps)
                    # I think the following works but it may need to be
                    # augmented by the epoch number
                    progress = state.global_step / state.max_steps
                    progress = int(progress * 100)
                    db_job_id = JOB_ID
                    # Write to jobs table in database, updating the
                    # progress column:
                    db.execute(
                        "UPDATE job SET progress = ? WHERE id = ?",
                        (progress, db_job_id),
                    )
                    db.commit()
                return

        # Create Trainer instance
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.tokenized_dataset["train"],
            callbacks=[ProgressTableUpdateCallback],
        )
        # silence the warnings. Please re-enable for inference!
        self.model.config.use_cache = False

        # train model
        self.trainer.train()

        self.trainer.model.save_pretrained(adaptor_output_dir)
        self.tokenizer.save_pretrained(adaptor_output_dir)

    def load_peft(self):
        # Load peft config for pre-trained checkpoint etc.
        config = PeftConfig.from_pretrained(self.peft_model_id)

        # load base LLM model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path, load_in_8bit=True, device_map={"": 0}
        )
        AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        # Load the Lora model
        model = PeftModel.from_pretrained(model, self.peft_model_id, device_map={"": 0})
        model.eval()

        print("Peft model loaded")

    def evaluate(self, evaluation_id):
        # Load dataset from the hub and get a sample
        dataset = load_dataset(self.dataset_id, trust_remote_code=True)
        sample = dataset["test"][randrange(len(dataset["test"]))]

        input_ids = self.tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
        # with torch.inference_mode():
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
        print(f"input sentence: {sample['dialogue']}\n{'---' * 20}")

        print(f"summary:\n{self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")

        # Metric
        metric = evaluate.load(evaluation_id)

        def evaluate_peft_model(sample, max_target_length=50):
            # generate summary
            outputs = self.model.generate(
                input_ids=sample["input_ids"].unsqueeze(0).cuda(),
                do_sample=True,
                top_p=0.9,
                max_new_tokens=max_target_length,
            )
            prediction = self.tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
            # decode eval sample
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(sample["labels"] != -100, sample["labels"], self.tokenizer.pad_token_id)
            labels = self.tokenizer.decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            return prediction, labels

        # load test dataset from disk
        test_dataset = load_from_disk("workspace/temp/data/eval/").with_format("torch")

        # run predictions
        # this can take ~45 minutes
        predictions, references = [], []
        for sample in tqdm(test_dataset):
            p, r = evaluate_peft_model(sample)
            predictions.append(p)
            references.append(r)

        # compute metric
        rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

        # print results
        print(f"Rogue1: {rogue['rouge1'] * 100:2f}%")
        print(f"rouge2: {rogue['rouge2'] * 100:2f}%")
        print(f"rougeL: {rogue['rougeL'] * 100:2f}%")
        print(f"rougeLsum: {rogue['rougeLsum'] * 100:2f}%")
        return

    def unload_model(self):
        self.model = None
        self.model_id = None
        return

    def unload_dataset(self):
        self.dataset = None
        self.dataset_id = None
        return

    def train(
        self,
        peft_model_id,
        model_name_or_path,
        data_path,
        lora_r,
        lora_alpha,
        lora_dropout,
        learning_rate,
        num_train_epochs,
        logging_steps,
        job_id,
        output_dir,
        adaptor_output_dir,
        wandb_logging=None,
    ):
        self.set_model(model_name_or_path)

        self.load_dataset(data_path, trust_remote_code=True)
        self.tokenize_dataset()
        self.preprocess()

        self.load_model()

        self.train_lora(
            peft_model_id=peft_model_id,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            logging_steps=logging_steps,
            job_id=job_id,
            output_dir=output_dir,
            adaptor_output_dir=adaptor_output_dir,
            wandb_logging=wandb_logging,
        )

        # # t.load_peft()
        # # t.evaluate('rouge')

        self.unload_model()
        self.unload_dataset()


# Store the LLM_LAB JOB ID that is related to this trainign task
# This is set by argument passed to the script
JOB_ID = None


def main():
    global JOB_ID

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)

    args, unknown = parser.parse_known_args()
    print("args:")
    print(args)

    input_config = None
    # open the input file that provides configs
    with open(args.input_file) as json_file:
        input_config = json.load(json_file)
    config = input_config["config"]
    print("Input:")
    print(input_config)

    JOB_ID = config["job_id"]

    WANDB_LOGGING = config.get("log_to_wandb", None)

    t = Trainer()
    t.train(
        peft_model_id=config["adaptor_name"],
        model_name_or_path=config["model_name"],
        data_path=config["dataset_name"],
        lora_r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=int(config["num_train_epochs"]),
        logging_steps=100,
        job_id=JOB_ID,
        output_dir=config["output_dir"],
        adaptor_output_dir=config["adaptor_output_dir"],
        wandb_logging=WANDB_LOGGING,
    )


if __name__ == "__main__":
    main()
