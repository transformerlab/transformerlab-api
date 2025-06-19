import json
import sqlite3
from threading import Thread, Event
from time import sleep
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import GatedRepoError, EntryNotFoundError
import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Process, Queue
from werkzeug.utils import secure_filename
from contextlib import contextmanager

# If there is an error set returncode and error_msg
# returncode is used by API to know about errors and
# error_msg will get passed back in API response.
# We're using the following exit codes by convention
# based on Stack Overflow advice:
# 0 = success
# 1 = general failure
# 77 = permission denied (GatedRepoError)
returncode = 0
error_msg = False


WORKSPACE_DIR = os.environ.get("_TFL_WORKSPACE_DIR")
if WORKSPACE_DIR is None:
    raise EnvironmentError("Environment variable _TFL_WORKSPACE_DIR is not set!")


@contextmanager
def get_db_connection():
    """Context manager for database connections with proper pragma settings"""
    db = sqlite3.connect(f"{WORKSPACE_DIR}/llmlab.sqlite3", isolation_level=None)
    try:
        # Set these PRAGMAs every time as they get reset per connection
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=normal")
        db.execute("PRAGMA busy_timeout=5000")
        yield db
    finally:
        db.close()


def safe_db_update(query, params, error_message="Database update failed"):
    """Safely execute a database update with error handling and retry logic"""
    max_retries = 3
    retry_delay = 0.1  # 100ms
    
    for attempt in range(max_retries):
        try:
            with get_db_connection() as db:
                db.execute(query, params)
                return True
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                print(f"Database locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            print(f"{error_message}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected database error: {e}")
            return False
    
    return False


def get_job_data(job_id):
    """Get job data from database with retry logic"""
    max_retries = 3
    retry_delay = 0.1  # 100ms
    
    for attempt in range(max_retries):
        try:
            with get_db_connection() as db:
                result = db.execute("SELECT job_data FROM job WHERE id=?", (job_id,)).fetchone()
                return json.loads(result[0]) if result else None
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                print(f"Database locked during job data query, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            print(f"Warning: Failed to get job data: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Warning: Failed to get job data: {e}", file=sys.stderr)
            return None
    
    return None

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["model", "adaptor"], default="model")
parser.add_argument("--job_id", type=str, required=True)
parser.add_argument("--total_size_of_model_in_mb", type=float, required=True)

# Args for mode=model
parser.add_argument("--model_name", type=str)
parser.add_argument("--model_filename", type=str, required=False)
parser.add_argument("--allow_patterns", type=str, required=False)

# Args for mode=adaptor
parser.add_argument("--peft", type=str)
parser.add_argument("--local_model_id", type=str)

args, other = parser.parse_known_args()
mode = args.mode
print(f"MODE IS: {mode}")

if mode == "adaptor":
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
    print(f"DOWNLOADING TO: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading adaptor {peft} with job_id {job_id}")

else:
    model = args.model_name
    model_filename = args.model_filename
    job_id = args.job_id

    # Models can have custom allow_patterns filters
    # Start with a default set of allow_patterns
    # but if we are able to read a list from the passed parameter use that instead
    allow_patterns = ["*.json", "*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "*.npz", "*.bin"]
    if args.allow_patterns:
        allow_patterns_json = args.allow_patterns
        try:
            converted_json = json.loads(allow_patterns_json)
            if isinstance(converted_json, list):
                allow_patterns = converted_json
        except Exception:
            pass

    print(f"Downloading model {model} with job_id {job_id}")


def do_download(repo_id, queue, allow_patterns=None, mode="model"):
    try:
        if mode == "model":
            snapshot_download(repo_id, allow_patterns=allow_patterns)
        else:
            snapshot_download(repo_id=peft, local_dir=target_dir, repo_type="model")
        queue.put("done")
    except Exception as e:
        queue.put(f"error: {str(e)}")


def cancel_check():
    job_data = get_job_data(job_id)
    return job_data and job_data.get("status") == "cancelled"


def launch_snapshot_with_cancel(repo_id, allow_patterns=None):
    queue = Queue()
    if mode == "model":
        p = Process(target=do_download, args=(repo_id, queue, allow_patterns, "model"))
    else:
        p = Process(target=do_download, args=(repo_id, queue, None, "adaptor"))
    p.start()

    while p.is_alive():
        if cancel_check():
            print("Cancellation detected. Terminating download...", file=sys.stderr)
            p.terminate()
            p.join()
            return "cancelled"
        sys.stdout.flush()

    result = queue.get()
    return result


def get_dir_size(path):
    total = 0
    if not os.path.exists(path):
        return total
    with os.scandir(path) as it:
        for entry in it:
            # Skip symlinks to avoid double counting
            if entry.is_symlink():
                pass
            elif entry.is_file():
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


def download_blocking(model_is_downloaded):
    global error_msg, returncode
    print("Downloading")

    # NOTE: For now storing size in two fields.
    # Will remove total_size_of_model_in_mb in the future.
    if mode == "adaptor":
        job_data = json.dumps(
            {
                "downloaded": 0,
                "model": peft,
                "total_size_in_mb": total_size_of_model_in_mb,
                "total_size_of_model_in_mb": total_size_of_model_in_mb,
            }
        )
    else:
        job_data = json.dumps(
            {
                "downloaded": 0,
                "model": model,
                "total_size_in_mb": total_size_of_model_in_mb,
                "total_size_of_model_in_mb": total_size_of_model_in_mb,
            }
        )

    print(job_data)

    # Initialize the job in the database
    safe_db_update(
        "UPDATE job SET progress=?, job_data=json(?) WHERE id=?",
        (0, job_data, job_id),
        "Failed to initialize job status"
    )

    if mode == "adaptor":
        try:
            launch_snapshot_with_cancel(repo_id=peft)
            model_is_downloaded.set()
        except GatedRepoError:
            returncode = 77
            error_msg = f"{peft} is a gated adapter. Please accept the license."
        except EntryNotFoundError:
            returncode = 1
            error_msg = f"{peft} does not contain a config.json or is not available."
        except Exception as e:
            returncode = 1
            error_msg = f"{type(e).__name__}: {e}"
    else:
        if model_filename is not None:
            # Filename mode means we download just one file from the repo, not the whole repo
            # This is useful for downloading GGUF repos which contain multiple versions of the model
            # make the directory if it doesn't exist
            # Right now the logo is hard coded to assuming if you are downloading one file, you are looking
            # at GGUF
            print("downloading model to workspace/models using filename mode")
            location = f"{WORKSPACE_DIR}/models/{model_filename}"
            os.makedirs(location, exist_ok=True)
            hf_hub_download(
                repo_id=model,
                filename=model_filename,
                local_dir=location,
                local_dir_use_symlinks=True,
            )
            # create a file in that same directory called info.json:
            info = [
                {
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
                        "logo": "https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png",
                    },
                }
            ]
            with open(f"{location}/info.json", "w") as f:
                f.write(json.dumps(info, indent=2))
        else:
            try:
                # snapshot_download(repo_id=model, allow_patterns=allow_patterns)
                launch_snapshot_with_cancel(repo_id=model, allow_patterns=allow_patterns)

            except GatedRepoError:
                returncode = 77
                error_msg = f"{model} is a gated HuggingFace model. \
    To continue downloading, you must agree to the terms \
    on the model's Huggingface page."

            except Exception as e:
                returncode = 1
                error_msg = f"{type(e).__name__}: {e}"

        model_is_downloaded.set()
    print("Download complete")


def check_disk_size(model_is_downloaded: Event):
    # Recursively checks the size of the huggingface cache
    # which is stored at ~/.cache/huggingface/hub

    starting_size_of_huggingface_cache_in_mb = get_dir_size(cache_dir) / 1024 / 1024
    starting_size_of_model_dir_in_mb = get_dir_size(str(WORKSPACE_DIR) + "/models") / 1024 / 1024
    starting_size_of_cache = starting_size_of_huggingface_cache_in_mb + starting_size_of_model_dir_in_mb

    counter = 0
    last_db_update = 0
    db_update_interval = 10  # Update database every 10 iterations (20 seconds)

    while True:
        hf_size = get_dir_size(path=cache_dir) / 1024 / 1024
        model_dir_size = get_dir_size(str(WORKSPACE_DIR) + "/models") / 1024 / 1024
        cache_size_growth = (hf_size + model_dir_size) - starting_size_of_cache
        adjusted_total_size = total_size_of_model_in_mb if total_size_of_model_in_mb > 0 else 7000
        progress = cache_size_growth / adjusted_total_size * 100
        print(f"\nModel Download Progress: {progress:.2f}%\n")

        # Only update database every 20 seconds to reduce lock contention
        if counter - last_db_update >= db_update_interval:
            safe_db_update(
                "UPDATE job SET job_data=json_set(job_data, '$.downloaded', ?), progress=? WHERE id=?",
                (cache_size_growth, progress, job_id),
                f"Failed to update download progress ({progress:.2f}%). Skipping."
            )
            last_db_update = counter

        print(f"flag:  {model_is_downloaded.is_set()}")

        if model_is_downloaded.is_set():
            print("Model is downloaded, exiting check_disk_size thread")
            # Final database update when download completes
            safe_db_update(
                "UPDATE job SET job_data=json_set(job_data, '$.downloaded', ?), progress=? WHERE id=?",
                (cache_size_growth, progress, job_id),
                f"Failed to update final download progress ({progress:.2f}%). Skipping."
            )
            break

        counter += 1
        if counter > 5000:  # around 3 hours
            print("Model is not yet downloaded, but check disk size thread is exiting after running for 3 hours.")
            break

        sleep(2)


def main():
    global returncode

    model_is_downloaded = Event()  # A threadsafe flag to coordinate the two threads
    print(f"flag:  {model_is_downloaded.is_set()}")

    p1 = Thread(target=check_disk_size, args=(model_is_downloaded,))
    p2 = Thread(target=download_blocking, args=(model_is_downloaded,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    if error_msg:
        print(f"Error downloading: {error_msg}")

        # save to job database
        job_data = json.dumps({"error_msg": str(error_msg)})
        status = "FAILED"
        if returncode == 77:
            status = "UNAUTHORIZED"

        # Update final job status
        if not safe_db_update(
            "UPDATE job SET status=?, job_data=json(?) WHERE id=?",
            (status, job_data, job_id),
            f"Failed to save download job status {status}"
        ):
            # If we fail to write to the database the app won't get the right error message
            print(f"Failed to save download job status {status}:")
            print(error_msg)
            returncode = 74  # IOERR

        exit(returncode)


if __name__ == "__main__":
    main()
