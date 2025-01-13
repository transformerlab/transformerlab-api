import argparse
import subprocess
import sys
import os

import requests
import torch


parser = argparse.ArgumentParser(
    description='Run Eleuther AI LM Evaluation Harness.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str,
                    help='Model to use for evaluation.')
parser.add_argument('--model_type', default='hf-causal',
                    type=str, help='Type of model to use for evaluation.')
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--eval_name', default='', type=str)
parser.add_argument('--task', default='', type=str)


args, other = parser.parse_known_args()

# print("Calling Eleuther AI LM Evaluation Harness with args:")
# print(args)

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

# Call the evaluation harness using HTTP if the platform is not CUDA
if not torch.cuda.is_available():
    print("CUDA is not available. Running eval using HTTP.")

    # first check http://localhost:8338/healthz
    response = requests.get('http://localhost:8338/server/worker_healthz')
    print(response.json())
    if response.status_code != 200 or not isinstance(response.json(), list) or len(response.json()) == 0:
        print("Local completions server is not running. Please start it before running the evaluation.")
        sys.exit(0)

    # model name is the first item in the list:
    model_name = args.model_name

    # lm_eval --model local-completions --tasks gsm8k --model_args model=mlx-community/Llama-3.2-1B-Instruct-4bit,base_url=http://localhost:8338/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False
    model_args = 'model=' + model_name + \
        ',base_url=http://localhost:8338/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False'
    command = ["lm-eval", '--model', 'local-completions',
               '--model_args', model_args, '--tasks', task]
    print('Running command: $ ' + ' '.join(command))
    print("--Beginning to run evaluations (please wait)...")
    try:
        process = subprocess.Popen(
            command,
            cwd=plugin_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        for line in process.stdout:
            print(line.decode('utf-8').strip())
    except Exception as e:
        print(f"An error occurred while running the subprocess: {e}")
    print('--Evaluation task complete')
else:
    command = ["lm-eval",
               '--model_args', model_args, '--tasks', task, '--device', 'cuda:0', '--trust_remote_code']

    subprocess.Popen(
        command,
        cwd=plugin_dir,
    )
