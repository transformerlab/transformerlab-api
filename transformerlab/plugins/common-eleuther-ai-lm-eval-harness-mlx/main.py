import os
import re
import subprocess
import pandas as pd
import torch
import traceback

from transformerlab.tfl_decorators import tfl_evals


def get_detailed_file_names(output_file_path, prefix="samples_", suffix=".jsonl"):
    """This function is necessary to fetch all the .jsonl files that EleutherAI LM Evaluation Harness generates so we can make a metrics_df out of the results for each test case"""
    try:
        matching_files = []
        for root, dirs, files in os.walk(output_file_path):
            for file in files:
                if file.startswith(prefix) and file.endswith(suffix):
                    matching_files.append(os.path.join(root, file))
        return matching_files
    except Exception as e:
        print(f"An error occurred while getting the output file name: {e}")
        return []


@tfl_evals.job_wrapper(progress_start=0, progress_end=100)
def run_evaluation():
    """Run the MLX Evaluation Harness"""
    try:
        # Setup evaluation logging
        tfl_evals.setup_eval_logging()

        # Validate parameters
        if not tfl_evals.model_name or tfl_evals.model_name == "":
            raise ValueError("No model provided. Please re-run after setting a Foundation model.")

        if hasattr(tfl_evals, "limit") and tfl_evals.limit:
            limit_val = float(tfl_evals.limit)
            if limit_val < 0:
                raise ValueError("Limit must be a positive number.")
            if limit_val > 1:
                raise ValueError("Limit should be between 0 and 1.")
            if limit_val == 1:
                limit_val = 1.0

        # Use model_path as model_name if provided
        model_name = tfl_evals.model_name
        if hasattr(tfl_evals, "model_path") and tfl_evals.model_path.strip() != "":
            model_name = tfl_evals.model_path
            print(f"Model path provided. Using model path as model name: {model_name}")

        # Check if CUDA is available
        if torch.cuda.is_available():
            raise ValueError("CUDA is available. Please use the `eleuther-ai-lm-evaluation-harness-plugin`.")

        # Get plugin directory
        plugin_dir = os.path.realpath(os.path.dirname(__file__))

        # Prepare output directory for lm-eval
        output_path = tfl_evals.get_output_file_path(dir_only=True)

        # Build model args
        model_args = f"model={model_name},trust_remote_code=True"

        # Build command
        command = ["lm-eval", "--model", "mlx", "--model_args", model_args, "--tasks", tfl_evals.tasks, "--log_samples"]

        # Add limit if provided
        if hasattr(tfl_evals, "limit") and tfl_evals.limit and float(tfl_evals.limit) != 1.0:
            command.extend(["--limit", str(tfl_evals.limit)])

        # Add output path
        command.extend(["--output_path", output_path])

        print("Running command: $ " + " ".join(command))
        print("--Beginning to run evaluations (please wait)...")

        # Run subprocess
        with subprocess.Popen(
            command,
            cwd=plugin_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        ) as process:
            for line in process.stdout:
                print(line.strip())
                # Parse progress from output
                pattern = r"^Running.*?(\d+)%\|"
                match = re.search(pattern, line)
                if match:
                    tfl_evals.progress_update(int(match.group(1)))

        # Get detailed report files
        detailed_report_files = get_detailed_file_names(output_path)

        # Process results
        metrics_list = []
        scores_list = []

        # Extract metrics from detailed reports
        for task_name in tfl_evals.tasks.split(","):
            for file in detailed_report_files:
                if task_name in file:
                    df = pd.read_json(file, lines=True)
                    avg_score = df["acc"].mean()

                    # Log to tensorboard
                    tfl_evals.log_metric(task_name, avg_score)
                    scores_list.append({"type": task_name, "score": avg_score})

                    # Build metrics dataframe
                    for _, row in df.iterrows():
                        metrics_list.append(
                            {
                                "test_case_id": f"test_case_{row['doc_id']}",
                                "metric_name": task_name,
                                "score": row["acc"],
                                "input": row["doc"],
                                "expected_output": row.get("target", ""),
                            }
                        )

        # Create metrics DataFrame
        metrics_df = pd.DataFrame(metrics_list)

        # Save results using plugin's method
        tfl_evals.save_evaluation_results(metrics_df)

        print("Evaluation completed successfully.")

        return True

    except Exception as e:
        print(f"An error occurred while running the evaluation harness: {e}")
        traceback.print_exc()
        raise


# Run the evaluation when script is executed
run_evaluation()
