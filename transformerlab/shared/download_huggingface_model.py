import json
import sqlite3
import time
from threading import Thread, Event
from huggingface_hub import hf_hub_download, snapshot_download, HfFileSystem, list_repo_files
from huggingface_hub.utils import GatedRepoError, EntryNotFoundError
import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Process, Queue
from werkzeug.utils import secure_filename

from lab import HOME_DIR, WORKSPACE_DIR


DATABASE_FILE_NAME = f"{HOME_DIR}/llmlab.sqlite3"


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


# Global variables for cache-based progress tracking
_cache_stop_monitoring = False

def get_repo_file_metadata(repo_id, allow_patterns=None):
    """
    Get metadata for all files in a HuggingFace repo.
    Returns dict with filename -> size mapping.
    """
    try:
        print(f"Fetching file metadata for {repo_id}...")
        
        # Get list of files in the repo
        files = list_repo_files(repo_id)
        
        # Filter out git files
        files = [f for f in files if not f.startswith('.git')]
        
        # Filter by allow_patterns if provided
        if allow_patterns:
            import fnmatch
            filtered_files = []
            for file in files:
                if any(fnmatch.fnmatch(file, pattern) for pattern in allow_patterns):
                    filtered_files.append(file)
            files = filtered_files
        
        # Get file sizes using HfFileSystem
        fs = HfFileSystem()
        file_metadata = {}
        total_size = 0
        
        for file in files:
            try:
                # Get file info including size
                file_info = fs.info(f"{repo_id}/{file}")
                file_size = file_info.get('size', 0)
                file_metadata[file] = file_size
                total_size += file_size
                print(f"  {file}: {file_size / 1024 / 1024:.1f} MB")
            except Exception as e:
                print(f"  Warning: Could not get size for {file}: {e}")
                file_metadata[file] = 0
        
        print(f"Total repo size: {total_size / 1024 / 1024:.1f} MB ({len(files)} files)")
        return file_metadata, total_size
        
    except Exception as e:
        print(f"Error getting repo metadata: {e}")
        return {}, 0

def get_cache_dir_for_repo(repo_id):
    """Get the HuggingFace cache directory for a specific repo"""
    from huggingface_hub.constants import HF_HUB_CACHE
    
    # Convert repo_id to cache-safe name (same logic as huggingface_hub)
    # repo_name = re.sub(r'[^\w\-_.]', '-', repo_id)
    # Replace / with --
    repo_name = repo_id.replace("/", "--")

    return os.path.join(HF_HUB_CACHE, f"models--{repo_name}")

def get_downloaded_size_from_cache(repo_id, file_metadata):
    """
    Check HuggingFace cache directory to see which files exist and their sizes.
    Returns total downloaded bytes.
    """
    try:
        cache_dir = get_cache_dir_for_repo(repo_id)
                
        if not os.path.exists(cache_dir):
            return 0
        
        # Look in the snapshots directory for the latest commit
        snapshots_dir = os.path.join(cache_dir, "snapshots")
        if not os.path.exists(snapshots_dir):
            return 0
        
        # Get the most recent snapshot (highest timestamp or lexicographically last)
        try:
            commits = os.listdir(snapshots_dir)
            if not commits:
                return 0
            
            # Use the lexicographically last commit (usually the latest)
            latest_commit = sorted(commits)[-1]
            snapshot_path = os.path.join(snapshots_dir, latest_commit)
        except Exception:
            return 0
        
        downloaded_size = 0
        
        # Check each expected file
        for filename, expected_size in file_metadata.items():
            file_path = os.path.join(snapshot_path, filename)
            
            if os.path.exists(file_path):
                try:
                    actual_size = os.path.getsize(file_path)
                    # Use the smaller of expected and actual size to be conservative
                    downloaded_size += min(actual_size, expected_size)
                except Exception:
                    pass
        
        return downloaded_size
        
    except Exception as e:
        print(f"Error checking cache: {e}")
        return 0

def update_database_progress(job_id, workspace_dir, model_name, downloaded_bytes, total_bytes):
    """Update progress in the database"""
    try:
        db = sqlite3.connect(DATABASE_FILE_NAME, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=normal")
        db.execute("PRAGMA busy_timeout=30000")
        
        downloaded_mb = downloaded_bytes / 1024 / 1024
        total_mb = total_bytes / 1024 / 1024
        progress_pct = (downloaded_bytes / total_bytes * 100) if total_bytes > 0 else 0
        
        job_data = json.dumps({
            "downloaded": downloaded_mb,
            "model": model_name,
            "total_size_in_mb": total_mb,
            "total_size_of_model_in_mb": total_mb,
            "progress_pct": progress_pct,
            "bytes_downloaded": downloaded_bytes,
            "total_bytes": total_bytes,
            "monitoring_type": "cache_based"
        })
        
        db.execute(
            "UPDATE job SET job_data=json(?), progress=? WHERE id=?",
            (job_data, progress_pct, job_id)
        )
        db.close()
        
        print(f"Cache Progress: {progress_pct:.2f}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)")
        
    except Exception as e:
        print(f"Failed to update database progress: {e}")

def cache_progress_monitor(job_id, workspace_dir, model_name, repo_id, file_metadata, total_bytes):
    """
    Monitor cache directory for download progress.
    Runs in a separate thread.
    """
    global _cache_stop_monitoring
    
    while not _cache_stop_monitoring:
        try:
            downloaded_bytes = get_downloaded_size_from_cache(repo_id, file_metadata)
            
            # Update database
            update_database_progress(job_id, workspace_dir, model_name, downloaded_bytes, total_bytes)

                        
            # Check if download is complete
            if downloaded_bytes >= total_bytes * 0.99:  # 99% complete
                print("Download appears to be complete")
                break
                
            time.sleep(2)  # Check every 2 seconds
            
        except Exception as e:
            print(f"Error in progress monitor: {e}")
            time.sleep(5)  # Wait longer on error
    
    print("Progress monitoring stopped")


def check_model_gated(repo_id):
    """
    Check if a model is gated by trying to read config.json or model_index.json
    using HuggingFace Hub filesystem.

    Args:
        repo_id (str): The repository ID to check

    Raises:
        GatedRepoError: If the model is gated and requires authentication/license acceptance
    """
    fs = HfFileSystem()

    # List of config files to check
    config_files = ["config.json", "model_index.json"]

    # Try to read each config file
    for config_file in config_files:
        file_path = f"{repo_id}/{config_file}"
        try:
            # Try to open and read the file
            with fs.open(file_path, "r") as f:
                f.read(1)  # Just read a byte to check accessibility
            # If we can read any config file, the model is not gated
            return
        except GatedRepoError:
            # If we get a GatedRepoError, the model is definitely gated
            raise GatedRepoError(f"Model {repo_id} is gated and requires authentication or license acceptance")
        except Exception:
            # If we get other errors (like file not found), continue to next file
            continue

    # If we couldn't read any config file due to non-gated errors,
    # we'll let the main download process handle it
    return


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
            # Download without custom progress bar (we'll monitor cache instead)
            snapshot_download(repo_id, allow_patterns=allow_patterns)
        else:
            # Download without custom progress bar (we'll monitor cache instead)
            snapshot_download(repo_id=peft, local_dir=target_dir, repo_type="model")
        queue.put("done")
    except Exception as e:
        queue.put(f"error: {str(e)}")


def cancel_check():
    try:
        db = sqlite3.connect(DATABASE_FILE_NAME, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=normal")
        db.execute("PRAGMA busy_timeout=30000")
        status = db.execute("SELECT job_data FROM job WHERE id=?", (job_id,)).fetchone()
        db.close()

        if status:
            job_data = json.loads(status[0])
            return job_data.get("status") == "cancelled"
    except Exception as e:
        print(f"Warning: cancel_check() failed: {e}", file=sys.stderr)
    return False


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
    global error_msg, returncode, _cache_stop_monitoring
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

    # Connect to the DB to start the job and then close
    # Need to set these PRAGMAs every time as they get reset per connection
    db = sqlite3.connect(DATABASE_FILE_NAME, isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=normal")
    db.execute("PRAGMA busy_timeout=30000")
    db.execute("UPDATE job SET progress=?, job_data=json(?) WHERE id=?", (0, job_data, job_id))
    db.close()

    # Check if model is gated before starting download
    if mode == "adaptor":
        repo_to_check = peft
    else:
        repo_to_check = model

    try:
        check_model_gated(repo_to_check)
    except GatedRepoError:
        returncode = 77
        if mode == "adaptor":
            error_msg = f"{peft} is a gated adapter. Please accept the license on the model's HuggingFace page."
        else:
            error_msg = f"{model} is a gated HuggingFace model. To continue downloading, you must agree to the terms on the model's HuggingFace page."
        model_is_downloaded.set()
        return

    if mode == "adaptor":
        try:
            # Get file metadata before starting download
            file_metadata, actual_total_size = get_repo_file_metadata(peft)
            
            # Start progress monitoring thread
            progress_thread = Thread(
                target=cache_progress_monitor,
                args=(job_id, WORKSPACE_DIR, peft, peft, file_metadata, actual_total_size),
                daemon=True
            )
            progress_thread.start()
            
            result = launch_snapshot_with_cancel(repo_id=peft)
            if result == "cancelled":
                returncode = 1
                error_msg = "Download was cancelled"
            
            # Stop progress monitoring
            _cache_stop_monitoring = True
            progress_thread.join(timeout=5)
            
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
            # Get metadata for single file
            try:
                fs = HfFileSystem()
                file_info = fs.info(f"{model}/{model_filename}")
                file_size = file_info.get('size', total_size_of_model_in_mb * 1024 * 1024)
                file_metadata = {model_filename: file_size}
            except Exception:
                file_metadata = {model_filename: total_size_of_model_in_mb * 1024 * 1024}
                file_size = total_size_of_model_in_mb * 1024 * 1024
            
            # Start progress monitoring thread
            progress_thread = Thread(
                target=cache_progress_monitor,
                args=(job_id, WORKSPACE_DIR, model_filename, model, file_metadata, file_size),
                daemon=True
            )
            progress_thread.start()
            
            hf_hub_download(
                repo_id=model,
                filename=model_filename,
                local_dir=location
            )
            
            # Stop progress monitoring
            _cache_stop_monitoring = True
            progress_thread.join(timeout=5)
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
                # Get file metadata before starting download
                file_metadata, actual_total_size = get_repo_file_metadata(model, allow_patterns)
                
                # Start progress monitoring thread
                progress_thread = Thread(
                    target=cache_progress_monitor,
                    args=(job_id, WORKSPACE_DIR, model, model, file_metadata, actual_total_size),
                    daemon=True
                )
                progress_thread.start()
                
                result = launch_snapshot_with_cancel(repo_id=model, allow_patterns=allow_patterns)
                if result == "cancelled":
                    returncode = 1
                    error_msg = "Download was cancelled"
                
                # Stop progress monitoring
                _cache_stop_monitoring = True
                progress_thread.join(timeout=5)

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


def main():
    global returncode

    model_is_downloaded = Event()  # A threadsafe flag to coordinate threads
    print(f"flag:  {model_is_downloaded.is_set()}")

    # Simple approach: just run the download with built-in progress tracking
    p2 = Thread(target=download_blocking, args=(model_is_downloaded,))
    p2.start()
    p2.join()

    if error_msg:
        print(f"Error downloading: {error_msg}")

        # save to job database
        job_data = json.dumps({"error_msg": str(error_msg)})
        status = "FAILED"
        if returncode == 77:
            status = "UNAUTHORIZED"

        # Need to set these PRAGMAs every time as they get reset per connection
        db = sqlite3.connect(DATABASE_FILE_NAME, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=normal")
        db.execute("PRAGMA busy_timeout=30000")

        # If the error is that the database is locked then this call might also fail
        # for the same reason! Better catch and at least print a message.
        try:
            db.execute(
                "UPDATE job SET status=?, job_data=json(?)\
                    WHERE id=?",
                (status, job_data, job_id),
            )
        except sqlite3.OperationalError:
            # NOTE: If we fail to write to the database the app won't get
            # the right error message. So set a different
            print(f"Failed to save download job status {status}:")
            print(error_msg)
            returncode = 74  # IOERR

        db.close()
        exit(returncode)


if __name__ == "__main__":
    main()
