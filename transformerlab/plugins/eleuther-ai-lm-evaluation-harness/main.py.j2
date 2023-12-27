import argparse
import subprocess
import sys
import os


parser = argparse.ArgumentParser(
    description='Run Eleuther AI LM Evaluation Harness.')
parser.add_argument('--model_name', default='gpt-j-6b', type=str,
                    help='Model to use for evaluation.')
parser.add_argument('--model_type', default='hf-causal',
                    type=str, help='Type of model to use for evaluation.')
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--eval_name', default='', type=str)

args, other = parser.parse_known_args()

print(args)

root_dir = os.environ.get("LLM_LAB_ROOT_PATH")
output_file_path = root_dir + "workspace/experiments/" + args.experiment_name + \
    "/scripts/evals/" + args.eval_name + "/output.txt"

# example command from https://github.com/EleutherAI/lm-evaluation-harness
# python main.py \
#    --model hf-causal \
#    --model_args pretrained=EleutherAI/gpt-j-6B \
#    --tasks hellaswag \
#    --device cuda:0

# type = args.model_type
# type = "hf-causal" # hardcoded for now
type = "hf-seq2seq"  # hardcoded for now
model_args = 'pretrained=' + args.model_name
task = "{{ task }}"

with open(output_file_path, 'w') as fd:
    subprocess.Popen(
        [sys.executable, root_dir + '/workspace/plugins/eleuther-ai-lm-evaluation-harness/lm-evaluation-harness/main.py',
            '--model', type, '--model_args', model_args, '--tasks', task, '--device', 'cuda:0'],
        cwd=root_dir + "/workspace",
        stdout=fd
    )
