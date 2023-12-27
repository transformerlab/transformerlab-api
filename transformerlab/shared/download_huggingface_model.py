from huggingface_hub import snapshot_download, hf_hub_download
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--model_filename', type=str, required=False)

args, other = parser.parse_known_args()

model = args.model_name
model_filename = args.model_filename

print("starting script")

if model_filename is not None:
    # make the directory if it doesn't exist
    print("downloading model to workspace/models using filename mode")
    os.makedirs("workspace/models", exist_ok=True)
    hf_hub_download(repo_id=model, filename=model_filename,
                    resume_download=True, local_dir="workspace/models", local_dir_use_symlinks=True)
else:
    print("disabled for now")
    # snapshot_download(repo_id=model, resume_download=True)
