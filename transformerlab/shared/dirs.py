# Root dir is the parent of the parent of this current directory:

import os
from pathlib import Path
import transformerlab.db as db


"""
TFL_HOME_DIR is the directory that is the parent of the src and workspace directories.
By default, it is set to ~/.transformerlab

TFL_WORKSPACE_DIR is the directory where all the experiments, plugins, and models are stored.
By default, it is set to TFL_HOME_DIR/workspace

TFL_SOURCE_CODE_DIR is the directory where the source code is stored.
By default, it is set to TFL_HOME_DIR/src
This directory stores code but shouldn't store any data because it is erased and replaced
on updates.

You can set any of the above using environment parameters and it will override the defaults.

ROOT_DIR is a legacy variable that we should replace with the above, eventually.
"""


# TFL_HOME_DIR
if "TFL_HOME_DIR" in os.environ:
    HOME_DIR = os.environ["TFL_HOME_DIR"]
    if not os.path.exists(HOME_DIR):
        print(f"Error: Home directory {HOME_DIR} does not exist")
        exit(1)
    print(f"Home directory is set to: {HOME_DIR}")
else:
    HOME_DIR = Path.home() / ".transformerlab"
    os.makedirs(name=HOME_DIR, exist_ok=True)
    print(f"Using default home directory: {HOME_DIR}")

# TFL_WORKSPACE_DIR
if "TFL_WORKSPACE_DIR" in os.environ:
    WORKSPACE_DIR = os.environ["TFL_WORKSPACE_DIR"]
    if not os.path.exists(WORKSPACE_DIR):
        print(f"Error: Workspace directory {WORKSPACE_DIR} does not exist")
        exit(1)
    print(f"Workspace is set to: {WORKSPACE_DIR}")
else:
    WORKSPACE_DIR = os.path.join(HOME_DIR, "workspace")
    os.makedirs(name=WORKSPACE_DIR, exist_ok=True)
    print(f"Using default workspace directory: {WORKSPACE_DIR}")

# TFL_SOURCE_CODE_DIR
api_py_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if (api_py_dir != os.path.join(HOME_DIR, "src")):
    print(
        f"We are working from {api_py_dir} which is not {os.path.join(HOME_DIR, 'src')}")
    print("That means you are probably developing in a different location so we will set source dir to the current directory")
    TFL_SOURCE_CODE_DIR = api_py_dir
else:
    print(f"Source code directory is set to: {os.path.join(HOME_DIR, 'src')}")
    TFL_SOURCE_CODE_DIR = os.path.join(HOME_DIR, "src")

# EXPERIMENTS_DIR
EXPERIMENTS_DIR: str = os.path.join(WORKSPACE_DIR, "experiments")
os.makedirs(name=EXPERIMENTS_DIR, exist_ok=True)

# GLOBAL_LOG_PATH
GLOBAL_LOG_PATH = os.path.join(HOME_DIR, "transformerlab.log")

# OTHER LOGS DIR:
LOGS_DIR = os.path.join(HOME_DIR, "logs")
os.makedirs(name=LOGS_DIR, exist_ok=True)

# ROOT_DIR (deprecate later)
ROOT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


def experiment_dir_by_name(experiment_name: str) -> str:
    return os.path.join(EXPERIMENTS_DIR, experiment_name)


async def experiment_dir_by_id(experiment_id: str) -> str:
    if (experiment_id is not None and experiment_id != "undefined"):
        experiment = await db.experiment_get(experiment_id)
    else:
        print("Error: experiment_id is None or undefined")
        return os.path.join(EXPERIMENTS_DIR, "error")

    experiment_name = experiment['name']
    return os.path.join(EXPERIMENTS_DIR, experiment_name)

# PLUGIN_PRELOADED_GALLERY
PLUGIN_PRELOADED_GALLERY = os.path.join(
    TFL_SOURCE_CODE_DIR, "transformerlab", "plugins")

# PLUGIN_DIR
PLUGIN_DIR = os.path.join(WORKSPACE_DIR, "plugins")


def plugin_dir_by_name(plugin_name: str) -> str:
    return os.path.join(PLUGIN_DIR, plugin_name)


PLUGIN_SDK_DIR = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab", "plugin_sdk")
PLUGIN_HARNESS = os.path.join(PLUGIN_SDK_DIR, "plugin_harness.py")

# MODELS_DIR
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")

# DATASETS_DIR
DATASETS_DIR = os.path.join(WORKSPACE_DIR, "datasets")
os.makedirs(name=DATASETS_DIR, exist_ok=True)


def dataset_dir_by_id(dataset_id: str) -> str:
    return os.path.join(DATASETS_DIR, dataset_id)


TEMP_DIR = os.path.join(WORKSPACE_DIR, "temp")
os.makedirs(name=TEMP_DIR, exist_ok=True)


# Prompt Templates Dir:
PROMPT_TEMPLATES_DIR = os.path.join(WORKSPACE_DIR, "prompt_templates")
os.makedirs(name=PROMPT_TEMPLATES_DIR, exist_ok=True)

# Tools Dir:
TOOLS_DIR = os.path.join(WORKSPACE_DIR, "tools")
os.makedirs(name=TOOLS_DIR, exist_ok=True)


# Galleries cache directory
GALLERIES_SOURCE_PATH = "transformerlab/galleries/"
GALLERIES_CACHE_DIR = os.path.join(WORKSPACE_DIR, "galleries")
os.makedirs(name=GALLERIES_CACHE_DIR, exist_ok=True)
