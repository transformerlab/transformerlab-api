# Root dir is the parent of the parent of this current directory:

import os
from lab import HOME_DIR, dirs
from transformerlab.db.db import experiment_get


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
        return os.path.join(dirs.EXPERIMENTS_DIR, "error")

    experiment_name = experiment["name"]
    return dirs.experiment_dir_by_name(experiment_name)


# PLUGIN_PRELOADED_GALLERY
PLUGIN_PRELOADED_GALLERY = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab", "plugins")

PLUGIN_SDK_DIR = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab", "plugin_sdk")
PLUGIN_HARNESS = os.path.join(PLUGIN_SDK_DIR, "plugin_harness.py")


# Galleries cache directory
GALLERIES_LOCAL_FALLBACK_DIR = os.path.join(TFL_SOURCE_CODE_DIR, "transformerlab/galleries/")
