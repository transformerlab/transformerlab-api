import json
import sqlite3
from threading import Thread, Event
from time import sleep
from huggingface_hub import hf_hub_download, snapshot_download
import argparse
import os
from pathlib import Path

WORKSPACE_DIR = os.environ.get("_TFL_WORKSPACE_DIR")

if WORKSPACE_DIR is None:
    WORKSPACE_DIR = Path.home() / ".transformerlab" / "workspace"
    print(f"Using default workspace directory: {WORKSPACE_DIR}")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_filename', type=str, required=False)
parser.add_argument('--job_id', type=str, required=True)
parser.add_argument('--total_size_of_model_in_mb', type=float, required=True)

args, other = parser.parse_known_args()
model = args.model_name
model_filename = args.model_filename
job_id = args.job_id

print(f"Downloading model {model} with job_id {job_id}")


def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


total_size_of_model_in_mb = args.total_size_of_model_in_mb
hf_home = os.getenv("HF_HOME")
if hf_home:
    cache_dir = Path(hf_home) / "hub"
else:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

print("starting script with progressbar updater")


# This was an attempt to override tqdm. This works BUT huggingface doesn't pass
# the tqdm_class to the secondary progress bars and if the file is one big file
# then the progress bar doesn't report any progress.
# I leave the code here in case we want to try to override a different
# tqdm class in the future.
# class tqdm_transformerlab_database(tqdm):
#     def __init__(self, *args, **kwargs):
#         if not kwargs.get('disable'):
#             kwargs = kwargs.copy()
#             logging.getLogger("HTTPClient").setLevel(logging.WARNING)
#             kwargs['mininterval'] = max(1.5, kwargs.get('mininterval', 1.5))
#         super(tqdm_transformerlab_database, self).__init__(*args, **kwargs)

#     def display(self, **kwargs):
#         super(tqdm_transformerlab_database, self).display(**kwargs)
#         fmt = self.format_dict
#         print("Status:")
#         print(f"{fmt['n']} of {fmt['total']}")
#         print(fmt['n'] / fmt['total'] * 100)

#     def clear(self, *args, **kwargs):
#         super(tqdm_transformerlab_database, self).clear(*args, **kwargs)
#         if not self.disable:
#             self.sio.write("")

#     def tsrange(*args, **kwargs):
#         """Shortcut for `tqdm.contrib.slack.tqdm(range(*args), **kwargs)`."""
#         return tqdm_transformerlab_database(range(*args), **kwargs)


def download_blocking(model_is_downloaded):
    print("Downloading model")
    db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3")
    db.execute(
        "UPDATE job SET progress=? WHERE id=?", (0, job_id))
    db.commit()
    if model_filename is not None:
        # Filename mode means we download just one file from the repo, not the whole repo
        # This is useful for downloading GGUF repos which contain multiple versions of the model
        # make the directory if it doesn't exist
        # Right now the logo is hard coded to assuming if you are downloading one file, you are looking
        # at GGUF
        print("downloading model to workspace/models using filename mode")
        location = f"{WORKSPACE_DIR}/models/{model_filename}"
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
        try:
            snapshot_download(
                repo_id=model, resume_download=True)
            # sleep(20)
        except Exception as e:
            print(f"Error downloading model: {e}")

    model_is_downloaded.set()
    print("Downloaded model")


def check_disk_size(model_is_downloaded: Event):
    # Recursively checks the size of the huggingface cache
    # which is stored at ~/.cache/huggingface/hub
    db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3")

    starting_size_of_huggingface_cache_in_mb = get_dir_size(
        cache_dir) / 1024 / 1024
    starting_size_of_model_dir_in_mb = get_dir_size(
        str(WORKSPACE_DIR) + "/models") / 1024 / 1024
    starting_size_of_cache = starting_size_of_huggingface_cache_in_mb + \
        starting_size_of_model_dir_in_mb

    counter = 0

    while True:
        hf_size = get_dir_size(path=cache_dir) / 1024 / 1024
        model_dir_size = get_dir_size(
            str(WORKSPACE_DIR) + "/models") / 1024 / 1024
        cache_size_growth = (hf_size + model_dir_size) - starting_size_of_cache
        progress = cache_size_growth / total_size_of_model_in_mb * 100
        print(f"\nModel Download Progress: {progress:.2f}%\n")
        # Write to jobs table in database, updating the
        # progress column:
        job_data = json.dumps({"downloaded": cache_size_growth})
        db.execute(
            "UPDATE job SET progress=?, job_data=json(?) WHERE id=?", (progress, job_data, job_id))
        print(f"flag:  {model_is_downloaded.is_set()}")
        db.commit()

        if (model_is_downloaded.is_set()):
            print("I was told to quit")
            break

        counter += 1
        if counter > 5000:  # around 3 hours
            print("I've been running for a while, I'm going to quit")
            break

        sleep(2)


def main():
    model_is_downloaded = Event()  # A threadsafe flag to coordinate the two threads
    print(f"flag:  {model_is_downloaded.is_set()}")

    p1 = Thread(target=check_disk_size, args=(model_is_downloaded,))
    p2 = Thread(target=download_blocking, args=(
        model_is_downloaded,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


if __name__ == '__main__':
    main()
