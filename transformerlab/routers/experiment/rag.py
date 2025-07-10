import asyncio
import json
import os
import subprocess
import sys
from transformerlab.db.db import experiment_get
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from transformerlab.shared import dirs
from pydantic import BaseModel


class EmbedRequest(BaseModel):
    experiment_id: int
    text: str


router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/query")
async def query(experimentId: int, query: str, settings: str = None, rag_folder: str = "rag"):
    """Query the RAG engine"""

    experiment_dir = await dirs.experiment_dir_by_id(experimentId)
    documents_dir = os.path.join(experiment_dir, "documents")
    documents_dir = os.path.join(documents_dir, rag_folder)
    documents_dir = os.path.abspath(documents_dir)
    if not documents_dir.startswith(os.path.abspath(experiment_dir)):
        return "Error: Invalid RAG folder path"
    if not os.path.exists(documents_dir):
        return "Error: The RAG folder does not exist in the documents directory"
    experiment_details = await experiment_get(id=experimentId)
    experiment_config = json.loads(experiment_details["config"])
    model = experiment_config.get("foundation")
    embedding_model = experiment_config.get("embedding_model")
    if embedding_model is None or embedding_model == "":
        print("No embedding model found in experiment config, using default")
        embedding_model = "BAAI/bge-base-en-v1.5"
    else:
        embedding_model_file_path = experiment_config.get("embedding_model_filename")
        if embedding_model_file_path is not None and embedding_model_file_path != "":
            embedding_model = embedding_model_file_path

    print(
        "Querying RAG with model "
        + model
        + " and query "
        + query
        + " and settings "
        + settings
        + " and embedding model "
        + embedding_model
    )

    plugin = experiment_config.get("rag_engine")

    if plugin is None or plugin == "":
        return "Error: No RAG Engine has been assigned to this experiment."

    # Check if it exists in workspace/plugins:
    plugin_path = os.path.join(dirs.PLUGIN_DIR, plugin)
    if not os.path.exists(plugin_path):
        return f"Plugin {plugin} does not exist on the filesystem -- you must install or reinstall this plugin."

    # Call plug by passing plugin_path to plugin harness
    params = [
        dirs.PLUGIN_HARNESS,
        "--plugin_dir",
        plugin_path,
        "--model_name",
        model,
        "--embedding_model",
        embedding_model,
        "--query",
        query,
        "--documents_dir",
        documents_dir,
        "--settings",
        settings,
    ]

    print(f"Calling plugin {plugin_path}" + " with model " + model + " and query " + query)
    venv_path = os.path.join(plugin_path, "venv")
    if os.path.exists(venv_path) and os.path.isdir(venv_path):
        print(f">Plugin has virtual environment, activating venv from {venv_path}")
        venv_python = os.path.join(venv_path, "bin", "python")
        command = [venv_python, *params]
    else:
        print(">Using system python interpreter")
        command = [sys.executable, *params]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    print(stderr)

    # if the process is erroring, return the error message
    if process.returncode != 0:
        output = stderr.decode()
        return_object = {"error": output}
        return return_object

    output = stdout.decode()

    # if output is json, convert it to an object:
    try:
        output = json.loads(output)
    except Exception:
        pass

    return output


@router.get("/reindex")
async def reindex(experimentId: int, rag_folder: str = "rag"):
    """Reindex the RAG engine"""

    experiment_dir = await dirs.experiment_dir_by_id(experimentId)
    documents_dir = os.path.join(experiment_dir, "documents")
    documents_dir = os.path.join(documents_dir, rag_folder)
    if not os.path.exists(documents_dir):
        return "Error: The RAG folder does not exist in the documents directory."

    experiment_details = await experiment_get(id=experimentId)
    experiment_config = json.loads(experiment_details["config"])
    model = experiment_config.get("foundation")
    embedding_model = experiment_config.get("embedding_model")
    if embedding_model is None or embedding_model == "":
        print("No embedding model found in experiment config, using default")
        embedding_model = "BAAI/bge-base-en-v1.5"
    else:
        embedding_model_file_path = experiment_config.get("embedding_model_filename")
        if embedding_model_file_path is not None and embedding_model_file_path.strip() != "":
            embedding_model = embedding_model_file_path

    print("Reindexing RAG with embedding model " + embedding_model)

    plugin = experiment_config.get("rag_engine")

    if plugin is None or plugin == "":
        return "Error: No RAG Engine has been assigned to this experiment."

    # Check if it exists in workspace/plugins:
    plugin_path = os.path.join(dirs.PLUGIN_DIR, plugin)
    if not os.path.exists(plugin_path):
        return f"Plugin {plugin} does not exist on the filesystem -- you must install or reinstall this plugin."

    # Call plug by passing plugin_path to plugin harness
    params = [
        dirs.PLUGIN_HARNESS,
        "--plugin_dir",
        plugin_path,
        "--model_name",
        model,
        "--embedding_model",
        embedding_model,
        "--index",
        "True",
        "--documents_dir",
        documents_dir,
    ]
    print(f"Calling plugin {plugin_path}" + " with model " + model + " and reindex")
    venv_path = os.path.join(plugin_path, "venv")
    if os.path.exists(venv_path) and os.path.isdir(venv_path):
        print(f">Plugin has virtual environment, activating venv from {venv_path}")
        venv_python = os.path.join(venv_path, "bin", "python")
        command = [venv_python, *params]
    else:
        print(">Using system python interpreter")
        command = [sys.executable, *params]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    print(stderr)

    # if the process is erroring, return the error message
    if process.returncode != 0:
        output = stderr.decode()
        return_object = {"error": output}
        return return_object

    output = stdout.decode()

    # if output is json, convert it to an object:
    try:
        output = json.loads(output)
    except Exception:
        pass

    return output


@router.post("/embed")
async def embed_text(request: EmbedRequest):
    """Embed text using the embedding model using sentence transformers"""
    from sentence_transformers import SentenceTransformer

    experiment_details = await experiment_get(id=request.experiment_id)
    experiment_config = json.loads(experiment_details["config"])
    embedding_model = experiment_config.get("embedding_model")
    if embedding_model is None or embedding_model == "":
        print("No embedding model found in experiment config, using default")
        embedding_model = "BAAI/bge-base-en-v1.5"
    else:
        embedding_model_file_path = experiment_config.get("embedding_model_filename")
        if embedding_model_file_path is not None and embedding_model_file_path != "":
            embedding_model = embedding_model_file_path
    print("Using Embedding model: " + embedding_model)
    model = SentenceTransformer(embedding_model)
    text_list = request.text.split("\n")
    embeddings = model.encode(text_list)

    return JSONResponse(content={"embeddings": embeddings.tolist()})
