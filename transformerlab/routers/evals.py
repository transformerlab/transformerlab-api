import json
import os
from typing import Annotated
import transformerlab.db as db
from fastapi import APIRouter, Body
from fastapi.responses import FileResponse
from fastchat.model.model_adapter import get_conversation_template
from huggingface_hub import snapshot_download
from transformerlab.shared import dirs

import urllib.parse

router = APIRouter(prefix="/evals", tags=["evals"])


@router.get("/list")
async def eval_local_list():
    """Get the list of local evals"""
    eval_plugins = await db.get_plugins_of_type('EVALUATION')

    result = []

    # for each eval_plugin, check if it has saved local files:
    for eval_plugin in eval_plugins:
        name = eval_plugin['name']
        info_file = f"{dirs.plugin_dir_by_name(name)}/index.json"
        print(info_file)
        info = {}
        # check if info_file exists:
        if os.path.exists(info_file):
            print("info_file exists")
            with open(info_file, "r") as f:
                info = json.load(f)
        else:
            print('info_file does not exist')

        result.append({"name": name, "info": info})

    return result
