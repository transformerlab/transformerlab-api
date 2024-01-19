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
    # Filename mode means we download just one file from the repo, not the whole repo
    # This is useful for downloading GGUF repos which contain multiple versions of the model
    # make the directory if it doesn't exist
    print("downloading model to workspace/models using filename mode")
    location = f"workspace/models/{model_filename}"
    os.makedirs(location, exist_ok=True)
    hf_hub_download(repo_id=model, filename=model_filename,
                    resume_download=True, local_dir=location, local_dir_use_symlinks=True)
else:
    # Download the whole repo
    try:
        snapshot_download(repo_id=model, resume_download=True)
    except Exception as e:
        print("Failed to download model")
        print(e)
        exit(1)
