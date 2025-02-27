import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

import transformerlab.plugin

try:
    parser = argparse.ArgumentParser(description="Run DeepEval metrics for LLM-as-judge evaluation.")
    parser.add_argument("--run_name", default="evaluation", type=str)
    parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Model to use for evaluation.")
    parser.add_argument("--input_col", default="input", type=str, help="Input column name in the dataset.")
    parser.add_argument("--output_col", default="output", type=str, help="Output column name in the dataset.")
    parser.add_argument("--experiment_name", default="", type=str)
    parser.add_argument("--eval_name", default="", type=str)
    parser.add_argument("--predefined_tasks", default="", type=str)
    parser.add_argument("--tasks", default="", type=str)
    parser.add_argument(
        "--model_adapter",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
    )
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--job_id", default=None, type=str)
    parser.add_argument("--limit", default=None, type=float)

    args, other = parser.parse_known_args()

    # sys.exit(1)

    # Processing tasks to be a list of json
    try:
        args.tasks = eval(args.tasks)
    except Exception as e:
        print(f"Error {e} occurred while parsing the tasks: {args.tasks}.")
        sys.exit(1)

except Exception as e:
    print("Error occurred while parsing the arguments.")
    print(e)
    sys.exit(1)


if args.job_id:
    job = transformerlab.plugin.Job(args.job_id)
    job.update_progress(0)
else:
    print("Job ID not provided.")
    sys.exit(1)

today = time.strftime("%Y%m%d-%H%M%S")
tensorboard_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "experiments", args.experiment_name, "tensorboards")
# Find directory to put in based on eval name
combined_tensorboard_dir = None
for dir in os.listdir(tensorboard_dir):
    if args.run_name == dir or args.run_name == dir.lower():
        combined_tensorboard_dir = os.path.join(tensorboard_dir, dir)
if combined_tensorboard_dir is None:
    combined_tensorboard_dir = os.path.join(tensorboard_dir, args.run_name)
output_dir = os.path.join(combined_tensorboard_dir, f"evaljob_{args.job_id}_{today}")
os.makedirs(output_dir, exist_ok=True)
writer = SummaryWriter(output_dir)
job.set_tensorboard_output_dir(output_dir)
print("Writing tensorboard logs to", output_dir)


print("ARGS", args)
try:
    job.add_to_job_data("config", str(args))
    job.add_to_job_data("template_name", args.run_name)
    job.add_to_job_data("model_name", args.model_name)
    job.add_to_job_data("start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print(f"An error occurred while adding job data: {e}")
    job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    job.set_job_completion_status("failed", "An error occurred while adding job data.")
    sys.exit(1)


# Get Predefined tasks
pre_defined = {
    "Is Valid JSON": {
        "expression": None,
        "return_type": "json",
        "name": "Is Valid JSON",
    },
    "Word Count": {
        "expression": r"\w+",
        "return_type": "number",
        "name": "Word Count",
    },
    "Contains bulleted lists": {
        "expression": r"\n([*-])\s",
        "return_type": "boolean",
        "name": "Contains bulleted lists",
    },
    "Contains headings": {
        "expression": r"#+\s+.+",
        "return_type": "boolean",
        "name": "Contains headings",
    },
    "Contains URLs": {
        "expression": r"https?://[-a-zA-Z0-9&@#/%?=+~_|!:,.;]*[-a-zA-Z0-9&@#/%=+~_|]",
        "return_type": "boolean",
        "name": "Contains URLs",
    },
    "Contains code blocks": {
        "expression": r"```",
        "return_type": "boolean",
        "name": "Contains code blocks",
    },
    "Contains tables": {
        "expression": r"\|",
        "return_type": "boolean",
        "name": "Contains tables",
    },
    "Contains images": {
        "expression": r"!\[.*\]\(.*\)",
        "return_type": "boolean",
        "name": "Contains images",
    },
    "Contains numbered lists": {
        "expression": r"\n([0-9]+)\.\s",
        "return_type": "boolean",
        "name": "Contains numbered lists",
    },
    "Contains bold text": {
        "expression": r"\*\*",
        "return_type": "boolean",
        "name": "Contains bold text",
    },
    "Contains italic text": {
        "expression": r"\*",
        "return_type": "boolean",
        "name": "Contains italic text",
    },
    "Contains underline text": {
        "expression": r"_",
        "return_type": "boolean",
        "name": "Contains underline text",
    },
    "Contains strikethrough text": {
        "expression": r"~~",
        "return_type": "boolean",
        "name": "Contains strikethrough text",
    },
    "Contains blockquotes": {
        "expression": r">",
        "return_type": "boolean",
        "name": "Contains blockquotes",
    },
    "Contains inline code": {
        "expression": r"`",
        "return_type": "boolean",
        "name": "Contains inline code",
    },
    "Contains emojis": {
        "expression": r"(:\w+:)",
        "return_type": "boolean",
        "name": "Contains emojis",
    },
    "Contains email addresses": {
        "expression": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "return_type": "boolean",
        "name": "Contains email addresses",
    },
    "Contains phone numbers": {
        "expression": r"\+?([0-9]{1,3})\)?([0-9]{3})\)?([0-9]{3})\)?([0-9]{4})",
        "return_type": "boolean",
        "name": "Contains phone numbers",
    },
    "Contains dates": {
        "expression": r"\d{2}[-/]\d{2}[-/]\d{4}",
        "return_type": "boolean",
        "name": "Contains dates",
    },
    "Contains times": {
        "expression": r"\d{2}:\d{2}(:\d{2})?",
        "return_type": "boolean",
        "name": "Contains times",
    },
    "Contains numbers": {
        "expression": r"\d+",
        "return_type": "boolean",
        "name": "Contains numbers",
    },
}

if args.predefined_tasks and args.predefined_tasks.strip() != "":
    try:
        pre_defined_tasks = args.predefined_tasks.split(",")
        for task in pre_defined_tasks:
            if task in pre_defined:
                args.tasks.append(pre_defined[task])
            else:
                print(f"Predefined task {task} not found.")
    except Exception as e:
        print(f"Error {e} occurred while parsing the predefined tasks: {args.predefined_tasks}.")
        sys.exit(1)


def get_tflab_dataset():
    try:
        dataset_target = transformerlab.plugin.get_dataset_path(args.dataset_name)
    except Exception as e:
        job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        job.set_job_completion_status("failed", "Failure to get dataset")
        raise e
    dataset = {}
    dataset_types = ["train"]
    for dataset_type in dataset_types:
        try:
            dataset[dataset_type] = load_dataset(dataset_target, split=dataset_type, trust_remote_code=True)

        except Exception as e:
            job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            job.set_job_completion_status("failed", "Failure to load dataset")
            raise e
    # Convert the dataset to a pandas dataframe
    df = dataset["train"].to_pandas()
    return df


def get_output_file_path():
    experiment_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "experiments", args.experiment_name)
    p = os.path.join(experiment_dir, "evals", args.eval_name)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f"detailed_output_{args.job_id}.csv")


def execute_custom_function_regexp(output_text: str, expression: str, return_type: str):
    if return_type.lower() == "number":
        # Occurrence: Count all matches with the global-like flag
        matches = re.findall(expression, output_text.strip())
        return len(matches) if matches is not None else 0
    elif return_type.lower() == "boolean":
        # Existence: Check if at least one match exists
        match = re.search(expression, output_text.strip())
        return True if match is not None else False
    elif return_type.lower() == "json":
        # Check if the output is a valid JSON
        try:
            json.loads(output_text.strip())
            return True
        except Exception:
            return False
    elif return_type.lower() == "isequals":
        # Check if the output is equal to the expression
        return output_text.strip() == expression
    elif return_type.lower() == "contains":
        # Check if the output contains the expression
        return expression in output_text.strip()
    else:
        print("Invalid return type.")
        return None


def run_evaluation():
    try:
        # Load the csv file
        # df = pd.read_csv(args.dataset_path)
        df = get_tflab_dataset()
        # Check if `input`, `output` and `expected_output` columns exist
        assert args.input_col in df.columns, (
            f"Input column not found in the dataset. Please make sure the column name is {args.input_col}"
        )
        assert args.output_col in df.columns, (
            f"Output column not found in the dataset. Please make sure the column name is {args.output_col}"
        )

        if args.limit and float(args.limit) != 1.0:
            num_samples = int(len(df) * float(args.limit))
            if num_samples < 1:
                num_samples = 1
            df_limited = df.iloc[:num_samples].copy()
        else:
            df_limited = df.copy()

        print(f"Test cases loaded successfully: {len(df_limited)}")

        for task in tqdm(args.tasks):
            df_limited[f"eval_{task['name']}"] = df_limited[args.output_col].apply(
                lambda x: execute_custom_function_regexp(x, task["expression"], task["return_type"].lower())
            )

        job.update_progress(20)
        metrics = []
        # Calculate metrics for each test case
        for task in args.tasks:
            metric = task["name"]
            if task["return_type"] == "number":
                score_colour_sensitivity = 0
            else:
                score_colour_sensitivity = 1
            try:
                for idx, row in df_limited.iterrows():
                    metrics.append(
                        {
                            "test_case_id": f"test_case_{idx}",
                            "metric_name": metric,
                            "score": [int(row[f"eval_{metric}"]), score_colour_sensitivity],
                            "input": row[args.input_col],
                            "output": row[args.output_col],
                        }
                    )
            except Exception as e:
                print(f"Error occurred while calculating metrics for {metric}.")
                print(e)

        job.update_progress(60)
        # Save the metrics to a csv file
        metrics_df = pd.DataFrame(metrics)
        output_path = get_output_file_path()
        metrics_df.to_csv(output_path, index=False)
        job.update_progress(80)

        print(f"Metrics saved to {output_path}")
        print("Evaluation completed.")
        job.update_progress(100)
        score_list = []
        for task in args.tasks:
            metric = task["name"]
            writer.add_scalar(
                f"eval/{metric}",
                metrics_df[metrics_df["metric_name"] == metric]["score"].apply(lambda x: x[0]).mean(),
                1,
            )
            score_list.append(
                {
                    "type": metric,
                    "score": round(
                        metrics_df[metrics_df["metric_name"] == metric]["score"].apply(lambda x: x[0]).mean(), 2
                    ),
                }
            )
            print(
                f"Average {metric} score: {metrics_df[metrics_df['metric_name'] == metric]['score'].apply(lambda x: x[0]).mean()}"
            )

        job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        job.set_job_completion_status(
            "success",
            "Evaluation completed successfully.",
            score=score_list,
            additional_output_path=output_path,
        )
    except Exception as e:
        print("Error occurred while running the evaluation.")
        job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        job.set_job_completion_status("failed", str(e))
        print(e)
        traceback.print_exc()


try:
    run_evaluation()
except Exception as e:
    print("Error occurred while running the evaluation.")
    job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    job.set_job_completion_status("failed", str(e))
    print(e)
    traceback.print_exc()
