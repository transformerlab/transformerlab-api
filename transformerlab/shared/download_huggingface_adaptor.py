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
args, other = parser.parse_known_args()

peft = args.peft
local_model_id = args.local_model_id
job_id = args.job_id

# Sanitize both model_id and peft
safe_model_id = secure_filename(local_model_id)
safe_peft = secure_filename(peft)

# Always set target_dir to WORKSPACE_DIR/adaptors/local_model_id
target_dir = os.path.join(WORKSPACE_DIR, "adaptors", safe_model_id, safe_peft)
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


def check_disk_size(adapter_is_downloaded: Event):
    starting_size = get_dir_size(target_dir) / 1024 / 1024
    counter = 0

    while True:
        current_size = get_dir_size(target_dir) / 1024 / 1024
        cache_size_growth = current_size - starting_size
        progress = min(100.0, cache_size_growth * 10)

        db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3", isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=normal")
        db.execute("PRAGMA busy_timeout=5000")
        try:
            db.execute(
                "UPDATE job SET job_data=json_set(job_data, '$.downloaded', ?), progress=? WHERE id=?",
                (cache_size_growth, progress, job_id),
            )
        except sqlite3.OperationalError:
            print(f"Failed to update progress: {progress}%")
        db.close()

        if adapter_is_downloaded.is_set():
            break

        counter += 1
        if counter > 5000:
            break

        sleep(2)


def download_blocking(adapter_is_downloaded):
    global error_msg, returncode
    db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3", isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=normal")
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("UPDATE job SET progress=?, job_data=json(?) WHERE id=?", (0, json.dumps({"downloaded": 0}), job_id))
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
        except sqlite3.OperationalError as e:
            returncode = 74

    db.close()
    exit(returncode)


if __name__ == "__main__":
    main()
