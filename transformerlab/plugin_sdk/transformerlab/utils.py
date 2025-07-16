import json
from jinja2 import Environment
from transformers import AutoTokenizer


def prepare_dataset_files(
    data_directory,
    datasets,
    formatting_template=None,
    chat_template=None,
    model_name=None,
    chat_column="messages"
):
    tokenizer = None
    if chat_template:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for split_name in datasets:
        dataset_split = datasets[split_name]
        print(f"Processing {split_name} dataset with {len(dataset_split)} examples.")

        output_file = f"{data_directory}/{split_name}.jsonl"
        with open(output_file, "w") as f:
            for i in range(len(dataset_split)):
                example = dataset_split[i]
                try:
                    rendered_text = format_template(
                        example=example,
                        formatting_template=formatting_template,
                        chat_template=chat_template,
                        tokenizer=tokenizer,
                        chat_column=chat_column
                    )
                    rendered_text = rendered_text.replace("\n", "\\n").replace("\r", "\\r")
                    f.write(json.dumps({"text": rendered_text}) + "\n")
                except Exception:
                        print(f"Warning: Failed to process example {i} in '{split_name}'. Skipping.")
                        continue # Skip problematic examples

        # Print one example from the written jsonl file
        try:
            with open(output_file, "r") as f:
                first_line = f.readline()
                if first_line:
                    parsed = json.loads(first_line)
                    print(f"Example from {split_name} split:")
                    print(parsed.get("text", first_line))
                else:
                    print(f"Example from {split_name} split: file is empty.")
        except Exception as e:
            print(f"Error reading example from {output_file}: {e}")

def format_template(
    example,
    formatting_template=None,
    chat_template=None,
    tokenizer=None,
    chat_column="messages"
):
    if chat_template and tokenizer:
        return tokenizer.apply_chat_template(
                        example[chat_column],
                        tokenize=False,
                        add_generation_prompt=False,
                        chat_template=chat_template
                    )
    
    if formatting_template:
        jinja_env = Environment()
        formatting_template = jinja_env.from_string(formatting_template)
        return formatting_template.render(example)
    raise ValueError("Either formatting_template or chat_template must be provided.")


    