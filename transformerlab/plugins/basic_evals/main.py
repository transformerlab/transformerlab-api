import json
import re

import pandas as pd

from transformerlab.tfl_decorators import tfl_evals


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


def execute_custom_function_regexp(output_text: str, expression: str, return_type: str):
    """Execute custom regex-based evaluation functions"""
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


@tfl_evals.job_wrapper(progress_start=0, progress_end=100)
def run_evaluation():
    """Run basic evaluations using regex and simple metrics"""
    # Setup evaluation logging
    tfl_evals.setup_eval_logging()

    # Type casting for limit
    tfl_evals.limit = float(tfl_evals.limit) if tfl_evals.limit else 1.0

    # Parse tasks
    tasks = []

    # First try to parse the tasks JSON
    try:
        if hasattr(tfl_evals, "tasks") and tfl_evals.tasks and tfl_evals.tasks.strip() != "":
            tasks = eval(tfl_evals.tasks)
    except Exception as e:
        print(f"Error parsing tasks JSON: {e}")
        raise ValueError(f"Invalid tasks JSON format: {tfl_evals.tasks}")

    # Add predefined tasks if specified
    if hasattr(tfl_evals, "predefined_tasks") and tfl_evals.predefined_tasks.strip() != "":
        pre_defined_tasks = tfl_evals.predefined_tasks.split(",")
        for task in pre_defined_tasks:
            if task in pre_defined:
                tasks.append(pre_defined[task])
            else:
                print(f"Predefined task {task} not found.")

    if not tasks:
        raise ValueError("No tasks specified. Please provide tasks or predefined_tasks.")

    tfl_evals.progress_update(10)

    # Load dataset
    try:
        dataset = tfl_evals.load_dataset()
        df = dataset["train"].to_pandas()
        print("Dataset loaded successfully")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise ValueError(f"Failed to load dataset {tfl_evals.dataset_name}: {str(e)}")

    # Verify required columns exist
    if tfl_evals.input_col not in df.columns:
        raise ValueError(f"Input column '{tfl_evals.input_col}' not found in the dataset.")

    if tfl_evals.output_col not in df.columns:
        raise ValueError(f"Output column '{tfl_evals.output_col}' not found in the dataset.")

    # Apply limit if specified
    if tfl_evals.limit and float(tfl_evals.limit) != 1.0:
        num_samples = max(int(len(df) * float(tfl_evals.limit)), 1)
        df_limited = df.iloc[:num_samples].copy()
    else:
        df_limited = df.copy()

    print(f"Test cases loaded successfully: {len(df_limited)}")
    tfl_evals.progress_update(20)

    # Apply evaluations
    for task in tasks:
        df_limited[f"eval_{task['name']}"] = df_limited[tfl_evals.output_col].apply(
            lambda x: execute_custom_function_regexp(x, task["expression"], task["return_type"].lower())
        )

    tfl_evals.progress_update(40)

    # Generate metrics data
    metrics = []

    for task in tasks:
        metric_name = task["name"]
        metric_avg = df_limited[f"eval_{metric_name}"].mean()

        # Log metric to TensorBoard
        tfl_evals.log_metric(metric_name, metric_avg)

        # Add to scores list for job data

        # Create individual metrics entries
        for idx, row in df_limited.iterrows():
            metrics.append(
                {
                    "test_case_id": f"test_case_{idx}",
                    "metric_name": metric_name,
                    "score": int(row[f"eval_{metric_name}"]),
                    "input": row[tfl_evals.input_col],
                    "output": row[tfl_evals.output_col],
                }
            )

    tfl_evals.progress_update(60)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Save results using the plugin's method
    output_path, plot_data_path = tfl_evals.save_evaluation_results(metrics_df)

    tfl_evals.progress_update(100)
    print(f"Metrics saved to {output_path}")
    print(f"Plotting data saved to {plot_data_path}")
    print("Evaluation completed.")

    return True


# Run the evaluation when script is executed
run_evaluation()
