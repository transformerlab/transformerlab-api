import json
import sqlite3
from threading import Thread, Event
from time import sleep
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, EntryNotFoundError
import argparse
import os
from pathlib import Path
from werkzeug.utils import secure_filename

returncode = 0
error_msg = False

WORKSPACE_DIR = os.environ.get("_TFL_WORKSPACE_DIR")
if WORKSPACE_DIR is None:
    WORKSPACE_DIR = Path.home() / ".transformerlab" / "workspace"

parser = argparse.ArgumentParser()
parser.add_argument("--peft", type=str, required=True)
parser.add_argument("--local_model_id", type=str, required=True)
parser.add_argument("--job_id", type=str, required=True)
parser.add_argument("--total_size_of_model_in_mb", type=float, required=True)
args, other = parser.parse_known_args()

peft = args.peft
local_model_id = args.local_model_id
job_id = args.job_id

# Sanitize both model_id and peft
safe_model_id = secure_filename(local_model_id)
safe_peft = secure_filename(peft)

# Always set target_dir to WORKSPACE_DIR/adaptors/local_model_id
target_dir = os.path.join(WORKSPACE_DIR, "adaptors", safe_model_id, safe_peft)
if not os.path.commonpath([target_dir, WORKSPACE_DIR]) == os.path.abspath(WORKSPACE_DIR):
    raise ValueError("Invalid path after sanitization. Potential security risk.")

os.makedirs(target_dir, exist_ok=True)


def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_symlink():
                pass
            elif entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


total_size_of_model_in_mb = args.total_size_of_model_in_mb


def check_disk_size(model_is_downloaded: Event):
    # Recursively checks the size of the huggingface cache
    # which is stored at ~/.cache/huggingface/hub

    starting_size_of_huggingface_cache_in_mb = get_dir_size(target_dir) / 1024 / 1024
    starting_size_of_model_dir_in_mb = get_dir_size(str(WORKSPACE_DIR) + "/models") / 1024 / 1024
    starting_size_of_cache = starting_size_of_huggingface_cache_in_mb + starting_size_of_model_dir_in_mb

    counter = 0

    while True:
        hf_size = get_dir_size(path=target_dir) / 1024 / 1024
        model_dir_size = get_dir_size(str(WORKSPACE_DIR) + "/models") / 1024 / 1024
        cache_size_growth = (hf_size + model_dir_size) - starting_size_of_cache
        adjusted_total_size = total_size_of_model_in_mb if total_size_of_model_in_mb > 0 else 7000
        progress = cache_size_growth / adjusted_total_size * 100
        print(f"\nModel Download Progress: {progress:.2f}%\n")

        # Need to set these PRAGMAs every time as they get reset per connection
        # Not sure if we should reconnect over and over in the loop like this
        # But leaving it to reduce the chance of leaving a connection open if this
        # thread gets interrupted?
        db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3", isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=normal")
        db.execute("PRAGMA busy_timeout=5000")

        try:
            db.execute(
                "UPDATE job SET job_data=json_set(job_data, '$.downloaded', ?),  progress=? WHERE id=?",
                (cache_size_growth, progress, job_id),
            )
        except sqlite3.OperationalError:
            # Bit of a hack: We were having DB lock errors and this update isn't crucial
            # So for now just skip if something goes wrong.
            print(f"Failed to update download progress in DB ({progress}%). Skipping.")
        db.close()

        print(f"flag:  {model_is_downloaded.is_set()}")

        if model_is_downloaded.is_set():
            print("Model is downloaded, exiting check_disk_size thread")
            break

        counter += 1
        if counter > 5000:  # around 3 hours
            print("Model is not yet downloaded, but check disk size thread is exiting after running for 3 hours.")
            break

        sleep(2)


def download_blocking(adapter_is_downloaded):
    global error_msg, returncode

    job_data = json.dumps(
        {
            "downloaded": 0,
            "model": peft,
            "total_size_in_mb": total_size_of_model_in_mb,
            "total_size_of_model_in_mb": total_size_of_model_in_mb,
        }
    )

    db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3", isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=normal")
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("UPDATE job SET progress=?, job_data=json(?) WHERE id=?", (0, job_data, job_id))
    db.close()

    try:
        snapshot_download(repo_id=peft, local_dir=target_dir, repo_type="model")
        adapter_is_downloaded.set()
    except GatedRepoError:
        returncode = 77
        error_msg = f"{peft} is a gated adapter. Please accept the license."
    except EntryNotFoundError:
        returncode = 1
        error_msg = f"{peft} does not contain a config.json or is not available."
    except Exception as e:
        returncode = 1
        error_msg = f"{type(e).__name__}: {e}"


def main():
    global returncode
    adapter_is_downloaded = Event()
    p1 = Thread(target=check_disk_size, args=(adapter_is_downloaded,))
    p2 = Thread(target=download_blocking, args=(adapter_is_downloaded,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3", isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=normal")
    db.execute("PRAGMA busy_timeout=5000")

    if error_msg:
        status = "FAILED"
        if returncode == 77:
            status = "UNAUTHORIZED"
        try:
            db.execute(
                "UPDATE job SET status=?, job_data=json(?) WHERE id=?",
                (status, json.dumps({"error_msg": str(error_msg)}), job_id),
            )
        except sqlite3.OperationalError:
            returncode = 74
    else:
        try:
            db.execute(
                "UPDATE job SET status=?, job_data=json(?) WHERE id=?",
                ("SUCCESS", json.dumps({"success_msg": f"Adapter {peft} installed successfully"}), job_id),
            )
        except sqlite3.OperationalError:
            returncode = 74

    db.close()
    exit(returncode)


if __name__ == "__main__":
    main()
