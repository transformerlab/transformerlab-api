# Root dir is the parent of the parent of this current directory:

import os
from lab import HOME_DIR, dirs
from transformerlab.db.db import experiment_get

from werkzeug.utils import secure_filename

"""
TFL_SOURCE_CODE_DIR is the directory where the source code is stored.
By default, it is set to TFL_HOME_DIR/src
This directory stores code but shouldn't store any data because it is erased and replaced
on updates.

You can set any of the above using environment parameters and it will override the defaults.

ROOT_DIR is a legacy variable that we should replace with the above, eventually.
"""

FASTCHAT_LOGS_DIR = os.path.join(dirs.WORKSPACE_DIR, "logs")
if not os.path.exists(FASTCHAT_LOGS_DIR):
    os.makedirs(FASTCHAT_LOGS_DIR)


# TFL_STATIC_FILES_DIR is TFL_HOME_DIR/webapp
STATIC_FILES_DIR = os.path.join(HOME_DIR, "webapp")
os.makedirs(name=STATIC_FILES_DIR, exist_ok=True)
# if there is no index.html file in the static directory, create blank one
if not os.path.exists(os.path.join(STATIC_FILES_DIR, "index.html")):
    with open(os.path.join(STATIC_FILES_DIR, "index.html"), "w") as f:
        f.write(
            "<html><body><p>Transformer Lab Cloud App Files Missing. Run <pre>curl https://raw.githubusercontent.com/transformerlab/transformerlab-api/main/install.sh | bash</pre> to install.</p></body></html>"
        )

# TFL_SOURCE_CODE_DIR
api_py_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if api_py_dir != os.path.join(HOME_DIR, "src"):
    print(f"We are working from {api_py_dir} which is not {os.path.join(HOME_DIR, 'src')}")
    print(
        "That means you are probably developing in a different location so we will set source dir to the current directory"
    )
    TFL_SOURCE_CODE_DIR = api_py_dir
else:
    print(f"Source code directory is set to: {os.path.join(HOME_DIR, 'src')}")
    TFL_SOURCE_CODE_DIR = os.path.join(HOME_DIR, "src")

# ROOT_DIR (deprecate later)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def experiment_dir_by_id(experiment_id: int) -> str:
    if experiment_id is not None:
        experiment = await experiment_get(experiment_id)
    else:
        print("Error: experiment_id is None")
        experiments_dir = dirs.get_experiments_dir()
        return os.path.join(experiments_dir, "error")

    experiment_name = experiment["name"]
    return dirs.experiment_dir_by_name(experiment_name)


# PLUGIN_PRELOADED_GALLERY
PLUGIN_PRELOADED_GALLERY = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab", "plugins")

PLUGIN_SDK_DIR = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab", "plugin_sdk")
PLUGIN_HARNESS = os.path.join(PLUGIN_SDK_DIR, "plugin_harness.py")


# Galleries cache directory
GALLERIES_LOCAL_FALLBACK_DIR = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab/galleries/")


# TEMPORARY: We want to move jobs back into the root directory instead of under experiment
# But for now we need to leave this here.


def experiment_dir_by_name(experiment_name: str) -> str:
    experiments_dir = dirs.get_experiments_dir()
    return os.path.join(experiments_dir, experiment_name)


def job_dir_by_experiment_and_id(experiment_name: str, job_id: str) -> str:
    """
    Get the job directory path for a given experiment and job ID.
    Uses new structure: WORKSPACE_DIR/experiments/experiment_name/jobs/job_id/
    Args:
        experiment_name: Name of the experiment
        job_id: Job ID (will be sanitized with secure_filename)
    Returns:
        Path to the job directory
    """
    job_id_safe = secure_filename(str(job_id))
    experiment_dir = experiment_dir_by_name(experiment_name)
    job_dir = os.path.join(experiment_dir, "jobs", job_id_safe)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def get_job_output_dir(experiment_name: str, job_id: str) -> str:
    """
    Get the job output directory, with backward compatibility.
    First tries new structure (under experiment),
    then falls back to old structure if needed.
    Args:
        experiment_name: Name of the experiment
        job_id: Job ID
    Returns:
        Path to the job output directory
    """
    # Try new structure first
    new_job_dir = job_dir_by_experiment_and_id(experiment_name, job_id)
    if os.path.exists(new_job_dir):
        return new_job_dir

    # Fall back to old structure for backward compatibility
    job_id_safe = secure_filename(str(job_id))
    old_job_dir = os.path.join(dirs.WORKSPACE_DIR, "jobs", job_id_safe)
    if os.path.exists(old_job_dir):
        return old_job_dir

    # If neither exists, return new structure (will be created when needed)
    return new_job_dir
