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
parser.add_argument("--model_adapter", default=None, type=str,)


args, other = parser.parse_known_args()


root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
plugin_dir = os.path.realpath(os.path.dirname(__file__))

task = args.task

# Call the evaluation harness using HTTP if the platform is not CUDA
if not torch.cuda.is_available():

    # model name is the first item in the list:
    model_name = args.model_name

    # lm_eval --model local-completions --tasks gsm8k --model_args model=mlx-community/Llama-3.2-1B-Instruct-4bit,base_url=http://localhost:8338/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False
    model_args = 'model=' + model_name + \
        ',trust_remote_code=True'
    if args.model_adapter:
        adapter_path = os.path.join(
            os.environ["_TFL_WORKSPACE_DIR"], 'adaptors', args.model_name, args.model_adapter)
        model_args += f",peft={adapter_path}"
    command = ["lm-eval", '--model', 'mlx',
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
    print("CUDA is available. Please use the `eleuther-ai-lm-evaluation-harness-plugin`.")
