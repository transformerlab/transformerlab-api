# Root dir is the parent of the parent of this current directory:

import os
import transformerlab.shared.dirs_workspace as dirs_workspace
import transformerlab.db.db as db

from werkzeug.utils import secure_filename

WORKSPACE_DIR = dirs_workspace.WORKSPACE_DIR
HOME_DIR = dirs_workspace.HOME_DIR

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


FASTCHAT_LOGS_DIR = os.path.join(WORKSPACE_DIR, "logs")
if not os.path.exists(FASTCHAT_LOGS_DIR):
    os.makedirs(FASTCHAT_LOGS_DIR)

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

# TFL_STATIC_FILES_DIR is TFL_HOME_DIR/webapp
STATIC_FILES_DIR = os.path.join(HOME_DIR, "webapp")
os.makedirs(name=STATIC_FILES_DIR, exist_ok=True)
# if there is no index.html file in the static directory, create blank one
if not os.path.exists(os.path.join(STATIC_FILES_DIR, "index.html")):
    with open(os.path.join(STATIC_FILES_DIR, "index.html"), "w") as f:
        f.write(
            "<html><body><p>Transformer Lab Cloud App Files Missing. Run <pre>curl https://raw.githubusercontent.com/transformerlab/transformerlab-api/main/install.sh | bash</pre> to install.</p></body></html>"
        )

# EXPERIMENTS_DIR
EXPERIMENTS_DIR: str = os.path.join(WORKSPACE_DIR, "experiments")
os.makedirs(name=EXPERIMENTS_DIR, exist_ok=True)

# GLOBAL_LOG_PATH
GLOBAL_LOG_PATH = os.path.join(HOME_DIR, "transformerlab.log")

# OTHER LOGS DIR:
LOGS_DIR = os.path.join(HOME_DIR, "logs")
os.makedirs(name=LOGS_DIR, exist_ok=True)

# ROOT_DIR (deprecate later)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def experiment_dir_by_name(experiment_name: str) -> str:
    return os.path.join(EXPERIMENTS_DIR, experiment_name)


async def experiment_dir_by_id(experiment_id: str) -> str:
    if experiment_id is not None and experiment_id != "undefined":
        experiment = await db.experiment_get(experiment_id)
    else:
        print("Error: experiment_id is None or undefined")
        return os.path.join(EXPERIMENTS_DIR, "error")

    experiment_name = experiment["name"]
    return os.path.join(EXPERIMENTS_DIR, experiment_name)


# PLUGIN_PRELOADED_GALLERY
PLUGIN_PRELOADED_GALLERY = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab", "plugins")

# PLUGIN_DIR
PLUGIN_DIR = os.path.join(WORKSPACE_DIR, "plugins")


def plugin_dir_by_name(plugin_name: str) -> str:
    plugin_name = secure_filename(plugin_name)
    return os.path.join(PLUGIN_DIR, plugin_name)


PLUGIN_SDK_DIR = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab", "plugin_sdk")
PLUGIN_HARNESS = os.path.join(PLUGIN_SDK_DIR, "plugin_harness.py")

# MODELS_DIR
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
os.makedirs(name=MODELS_DIR, exist_ok=True)

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

# Batched Prompts Dir:
BATCHED_PROMPTS_DIR = os.path.join(WORKSPACE_DIR, "batched_prompts")
os.makedirs(name=BATCHED_PROMPTS_DIR, exist_ok=True)

# Galleries cache directory
GALLERIES_LOCAL_FALLBACK_DIR = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab/galleries/")
GALLERIES_CACHE_DIR = os.path.join(WORKSPACE_DIR, "galleries")
os.makedirs(name=GALLERIES_CACHE_DIR, exist_ok=True)

# Evals output file:


async def eval_output_file(experiment_name: str, eval_name: str) -> str:
    experiment_dir = experiment_dir_by_name(experiment_name)
    eval_name = secure_filename(eval_name)
    p = os.path.join(experiment_dir, "evals", eval_name)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, "output.txt")


async def generation_output_file(experiment_name: str, generation_name: str) -> str:
    experiment_dir = experiment_dir_by_name(experiment_name)
    generation_name = secure_filename(generation_name)
    p = os.path.join(experiment_dir, "generations", generation_name)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, "output.txt")
