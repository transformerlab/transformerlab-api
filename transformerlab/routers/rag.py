import asyncio
import json
import os
import subprocess
import sys
from typing import Annotated
import transformerlab.db as db
from fastapi import APIRouter, Body
from fastapi.responses import FileResponse
from fastchat.model.model_adapter import get_conversation_template
from huggingface_hub import snapshot_download
from transformerlab.shared import dirs

import urllib.parse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/query")
async def query(experimentId: str, query: str, settings: str = None):
    """Query the RAG engine"""
    experiment_dir = await dirs.experiment_dir_by_id(experimentId)
    documents_dir = os.path.join(experiment_dir, "documents")
    experiment_details = await db.experiment_get(id=experimentId)
    experiment_config = json.loads(experiment_details["config"])
    model = experiment_config.get("foundation")

    print("Querying RAG with model " + model +
          " and query " + query + " and settings " + settings)

    plugin = experiment_config.get("rag_engine")

    if plugin is None or plugin == "":
        return "Error: No RAG Engine has been assigned to this experiment."

    # Check if it exists in workspace/plugins:
    plugin_path = os.path.join(dirs.PLUGIN_DIR, plugin)
    if not os.path.exists(plugin_path):
        return f"Plugin {plugin} does not exist on the filesystem -- you must install or reinstall this plugin."
    # Call main.py which is at plugin_path/main.py
    plugin_main = os.path.join(plugin_path, "main.py")
    print(f"Calling plugin {plugin_main}" +
          " with model " + model + " and query " + query)
    process = await asyncio.create_subprocess_exec(
        sys.executable, plugin_main, "--model_name", model, "--query", query, "--documents_dir", documents_dir, "--settings", settings,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    print(stderr)

    # if the process is erroring, return the error message
    if process.returncode != 0:
        output = stderr.decode()
        return_object = {"response": output}
        return return_object

    output = stdout.decode()

    # if output is json, convert it to an object:
    try:
        output = json.loads(output)
    except:
        pass

    return output
