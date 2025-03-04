import argparse
import os
import re
import subprocess
import sys
import time
import pandas as pd
from datetime import datetime

import torch
from tensorboardX import SummaryWriter

import transformerlab.plugin

parser = argparse.ArgumentParser(description="Run Eleuther AI LM Evaluation Harness.")
parser.add_argument("--run_name", default="evaluation", type=str)
parser.add_argument("--job_id", default=None, type=str)
parser.add_argument("--model_name", default="gpt-j-6b", type=str, help="Model to use for evaluation.")
parser.add_argument("--model_path", default="", type=str, help="Path to model to use for evaluation.")
parser.add_argument("--model_type", default="hf-causal", type=str, help="Type of model to use for evaluation.")
parser.add_argument("--experiment_name", default="", type=str)
parser.add_argument("--eval_name", default="", type=str)
parser.add_argument("--tasks", default="", type=str)
parser.add_argument(
    "--model_adapter",
    default=None,
    type=str,
)
parser.add_argument("--limit", default=None, type=float)

args, other = parser.parse_known_args()

args.limit = str(float(args.limit))

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

if args.model_path.strip() != "":
    args.model_name = args.model_path
    print(f"Model path provided. Using model path as model name: {args.model_name}")

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


if float(args.limit) < 0:
    print("Limit must be a positive number.")
    job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    job.set_job_completion_status("failed", "Limit must be a positive number.")
    sys.exit(1)

if float(args.limit) == 1:
    args.limit = 1.0

if float(args.limit) > 1:
    print("Limit should be between 0 and 1.")
    job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    job.set_job_completion_status("failed", "Limit should be between 0 and 1.")
    sys.exit(1)

root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
plugin_dir = os.path.realpath(os.path.dirname(__file__))

task = args.tasks

if not args.model_name or args.model_name == "":
    print("No model provided. Please re-run after setting a Foundation model.")
    sys.exit(1)

job.update_progress(0.5)


def get_detailed_file_names(output_file_path, prefix="samples_", suffix=".jsonl"):
    try:
        matching_files = []
        for root, dirs, files in os.walk(output_file_path):
            for file in files:
                if file.startswith(prefix) and file.endswith(suffix):
                    matching_files.append(os.path.join(root, file))
        return matching_files
    except Exception as e:
        print(f"An error occurred while getting the output file name: {e}")
        return None


def get_output_file_path():
    experiment_dir = os.path.join(os.environ["_TFL_WORKSPACE_DIR"], "experiments", args.experiment_name)
    p = os.path.join(experiment_dir, "evals", args.job_id)
    os.makedirs(p, exist_ok=True)
    return p


def get_output_file_name(output_file_path):
    # Go to the end of the path and get the file name
    try:
        for root, dirs, files in os.walk(output_file_path):
            for file in files:
                if file.endswith(".json"):
                    return os.path.join(root, file)
            for dir in dirs:
                return get_output_file_name(os.path.join(root, dir))
    except Exception as e:
        print(f"An error occurred while getting the output file name: {e}")
        return None


def get_plotting_data(metrics_df, output_file_name):
    file_path = output_file_name.replace(".csv", "_plotting.json")
    metrics_data = metrics_df[["test_case_id", "metric_name", "score"]].copy()
    metrics_data["metric_name"] = metrics_data["metric_name"].apply(lambda x: x.replace("_", " ").title())
    metrics_data.to_json(file_path, orient="records", lines=False)

    return file_path


try:
    # Call the evaluation harness using HTTP if the platform is not CUDA
    if not torch.cuda.is_available():
        # model name is the first item in the list:
        model_name = args.model_name
        # lm_eval --model local-completions --tasks gsm8k --model_args model=mlx-community/Llama-3.2-1B-Instruct-4bit,base_url=http://localhost:8338/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False
        model_args = "model=" + model_name + ",trust_remote_code=True"
        # if args.model_adapter:
        #     adapter_path = os.path.join(
        #         os.environ["_TFL_WORKSPACE_DIR"], "adaptors", args.model_name, args.model_adapter
        #     )
        #     # model_args += f",peft={adapter_path}"
        command = ["lm-eval", "--model", "mlx", "--model_args", model_args, "--tasks", task, "--log_samples"]
        # Add limit if provided
        if float(args.limit) != 1.0:
            command.extend(["--limit", args.limit])
        # Adding output file path
        command.extend(["--output_path", get_output_file_path()])
        scores_list = []
        print("Running command: $ " + " ".join(command))
        print("--Beginning to run evaluations (please wait)...")
        try:
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
                    pattern = r"^Running.*?(\d+)%\|"
                    match = re.search(pattern, line)
                    if match:
                        job.update_progress(int(match.group(1)))

                    if job.should_stop:
                        print("Stopping job because of user interruption.")
                        job.update_status("STOPPED")
                        process.terminate()

            output_file_path = get_output_file_path()
            output_file_name = get_output_file_name(output_file_path)
            detailed_report_files = get_detailed_file_names(output_file_path)
            print(f"Saving output at {output_file_name}")

            job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            job.add_to_job_data("detailed_output_files", str(detailed_report_files))
            scores_list = []
            metrics_list = []

            # Updating score in a standard way for Eleuther AI LM Evaluation Harness
            task_list = task.split(",")
            for task_name in task.split(","):
                for file in detailed_report_files:
                    if task_name in file:
                        df = pd.read_json(file, lines=True)
                        writer.add_scalar(f"eval/{task_name}", df["acc"].mean(), 1)
                        scores_list.append({"type": task_name, "score": df["acc"].mean()})
                        for index, row in df.iterrows():
                            metrics_list.append(
                                {
                                    "test_case_id": f"test_case_{row['doc_id']}",
                                    "metric_name": task_name,
                                    "score": row["acc"],
                                    "input": row["doc"],
                                    "expected_output": row["target"],
                                }
                            )
            metrics_df = pd.DataFrame(metrics_list)
            additional_output_path = os.path.join(output_file_path, f"detailed_output_{args.job_id}.csv")
            metrics_df.to_csv(additional_output_path, index=False)
            plot_data_path = get_plotting_data(metrics_df, additional_output_path)
            print(f"Saving output at {additional_output_path}")

            job.set_job_completion_status(
                "success",
                "Evaluation task completed successfully.",
                score=scores_list,
                additional_output_path=additional_output_path,
                plot_data_path=plot_data_path,
            )

        except Exception as e:
            print(f"An error occurred while running the subprocess: {e}")
            job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            job.set_job_completion_status("failed", "An error occurred while running the subprocess.")
        print("--Evaluation task complete")

    else:
        print("CUDA is available. Please use the `eleuther-ai-lm-evaluation-harness-plugin`.")
        job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        job.set_job_completion_status(
            "failed", "CUDA is available. Please use the `eleuther-ai-lm-evaluation-harness-plugin`."
        )
        sys.exit(1)

except Exception as e:
    print(f"An error occurred while running the evaluation harness: {e}")
    job.add_to_job_data("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    job.set_job_completion_status("failed", "An error occurred while running the evaluation harness.")
    sys.exit(1)
