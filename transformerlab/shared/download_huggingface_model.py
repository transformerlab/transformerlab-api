import json
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
    # Right now the logo is hard coded to assuming if you are downloading one file, you are looking
    # at GGUF
    print("downloading model to workspace/models using filename mode")
    location = f"workspace/models/{model_filename}"
    os.makedirs(location, exist_ok=True)
    hf_hub_download(repo_id=model, filename=model_filename,
                    resume_download=True, local_dir=location, local_dir_use_symlinks=True)
    # create a file in that same directory called info.json:
    info = [{
        "model_id": model_filename,
        "model_filename": model_filename,
        "name": model_filename,
        "stored_in_filesystem": True,
        "json_data": {
            "uniqueId": f"gguf/{model_filename}",
            "name": model_filename,
            "description": "A GGUF model downloaded from the HuggingFace Hub",
            "architecture": "GGUF",
            "huggingface_repo": model,
            "logo": "https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png"
        }
    }]
    with open(f"{location}/info.json", "w") as f:
        f.write(json.dumps(info, indent=2))
else:
    # Download the whole repo
    try:
        snapshot_download(repo_id=model, resume_download=True)
    except Exception as e:
        print("Failed to download model")
        print(e)
        exit(1)
