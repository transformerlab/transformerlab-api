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

    jinja_env = Environment()
    if formatting_template:
        formatting_template = jinja_env.from_string(formatting_template)

    for split_name in datasets:
        dataset_split = datasets[split_name]
        print(f"Processing {split_name} dataset with {len(dataset_split)} examples.")

        output_file = f"{data_directory}/{split_name}.jsonl"
        with open(output_file, "w") as f:
            for i in range(len(dataset_split)):
                example = dataset_split[i]
                if formatting_template:
                    data_line = dict(example)
                    line = formatting_template.render(data_line)
                    line = line.replace("\n", "\\n").replace("\r", "\\r")
                    o = {"text": line}
                    f.write(json.dumps(o) + "\n")
                elif chat_template:
                    rendered = tokenizer.apply_chat_template(
                        example[chat_column],
                        tokenize=False,
                        add_generation_prompt=False,
                        chat_template=chat_template
                    )
                    rendered = rendered.replace("\n", "\\n").replace("\r", "\\r")
                    f.write(json.dumps({"text": rendered}) + "\n")
                else:
                    raise ValueError("No formatting template found.")

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
