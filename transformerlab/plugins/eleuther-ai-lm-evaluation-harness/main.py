import argparse
import subprocess
import sys
import os
import transformerlab.plugin
import re
import requests
import torch

parser = argparse.ArgumentParser(
    description='Run Eleuther AI LM Evaluation Harness.')
parser.add_argument('--job_id', default=None, type=str)
parser.add_argument('--model_name', default='gpt-j-6b', type=str,
                    help='Model to use for evaluation.')
parser.add_argument('--model_type', default='hf-causal',
                    type=str, help='Type of model to use for evaluation.')
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--eval_name', default='', type=str)
parser.add_argument('--task', default='', type=str)
parser.add_argument('--model_adapter', default=None, type=str)
parser.add_argument('--limit', default=None, type=float)


args, other = parser.parse_known_args()

args.limit = float(args.limit)

# print("Calling Eleuther AI LM Evaluation Harness with args:")
# print(args)
if args.job_id:
    job = transformerlab.plugin.Job(args.job_id)
    job.update_progress(0)
else:
    print("Job ID not provided.")
    sys.exit(1)

if float(args.limit) < 0:
    print("Limit must be a positive number.")
    job.set_completion_status(
        "failed", "Limit must be a positive number.")
    sys.exit(1)

if float(args.limit) == 1:
    args.limit = None

if float(args.limit) > 1:
    print("Limit should be between 0 and 1.")
    job.set_completion_status(
        "failed", "Limit should be between 0 and 1.")
    sys.exit(1)


root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
plugin_dir = os.path.realpath(os.path.dirname(__file__))

# example command from https://github.com/EleutherAI/lm-evaluation-harness
# python main.py \
#    --model hf-causal \
#    --model_args pretrained=EleutherAI/gpt-j-6B \
#    --tasks hellaswag \
#    --device cuda:0

# type = args.model_type

model_args = 'pretrained=' + args.model_name
task = args.task

# Exiting if model name is not provided
if not args.model_name or args.model_name == '':
    print('No model provided. Please re-run after setting a Foundation model.')
    job.set_completion_status(
        "failed", "No model provided. Please re-run after setting a Foundation model.")
    sys.exit(1)

# Call the evaluation harness using HTTP if the platform is not CUDA
if not torch.cuda.is_available():
    # print("CUDA is not available. Running eval using the MLX Plugin.")
    print("CUDA is not available. Please use the `eleuther-ai-lm-evaluation-harness-mlx-plugin` if using a Mac.")

    # model name is the first item in the list:
    model_name = args.model_name

    # lm_eval --model local-completions --tasks gsm8k --model_args model=mlx-community/Llama-3.2-1B-Instruct-4bit,base_url=http://localhost:8338/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False
    model_args = 'model=' + model_name + \
        ',trust_remote_code=True'
    if args.model_adapter:
        adapter_path = os.path.join(
            os.environ["_TFL_WORKSPACE_DIR"], 'adaptors', args.model_name, args.model_adapter)
        model_args += f",peft={adapter_path}"

    command = ["lm-eval", '--model', 'hf',
               '--model_args', model_args, '--tasks', task]

else:
    if args.model_adapter:
        adapter_path = os.path.join(
            os.environ["_TFL_WORKSPACE_DIR"], 'adaptors', args.model_name, args.model_adapter)
        model_args += f",peft={adapter_path}"

    command = ["lm-eval",
               '--model_args', model_args, '--tasks', task, '--device', 'cuda:0', '--trust_remote_code']

# Add limit if provided
if float(args.limit) != 1.0:
    command.extend(['--limit', str(args.limit)])
print('Running command: $ ' + ' '.join(command))
print("--Beginning to run evaluations (please wait)...")
try:
    with subprocess.Popen(command, cwd=plugin_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:
        # process = subprocess.Popen(
        #     command,
        #     cwd=plugin_dir,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.STDOUT
        # )
        for line in process.stdout:
            print(line.strip())
            pattern = r'^Running.*?(\d+)%\|'
            match = re.search(pattern, line)
            if match:
                job.update_progress(int(match.group(1)))

            if job.should_stop:
                print("Stopping job because of user interruption.")
                job.update_status("STOPPED")
                process.terminate()

    job.set_job_completion_status(
        "success", "Evaluation task completed successfully.")
except Exception as e:
    print(f"An error occurred while running the subprocess: {e}")
print('--Evaluation task complete')

# subprocess.Popen(
#     command,
#     cwd=plugin_dir,
# )
