from collections import namedtuple
import json
import shutil
from typing import Annotated
import transformerlab.db as db
from fastapi import APIRouter, Body
from fastapi.responses import FileResponse
from fastchat.model.model_adapter import get_conversation_template
from huggingface_hub import hf_hub_download, HfFileSystem, model_info
from huggingface_hub import snapshot_download

import os

from transformerlab.shared import shared
from transformerlab.shared import dirs

from transformerlab.models import model_helper
from transformerlab.models import basemodel
from transformerlab.models import localmodel

router = APIRouter(tags=["model"])


def get_models_dir():
    """
    Helper function to get the models directory and create it if it doesn't exist
    models are stored in separate subdirectories under workspace/models
    """
    models_dir = dirs.MODELS_DIR

    # make models directory if it does not exist:
    if not os.path.exists(f"{models_dir}"):
        os.makedirs(f"{models_dir}")

    return models_dir


def get_model_dir(model_id: str):
    """
    Helper function gets the directory for a model ID
    model_id may be in Hugging Face format
    """
    models_dir = get_models_dir()
    model_id_without_author = model_id.split("/")[-1]
    return os.path.join(models_dir, model_id_without_author)


def get_model_details_from_gallery(model_id: str):
    """
    Given a model ID this returns the associated data from the model gallery file.
    Returns None if no such value found.
    """
    with open(f"{dirs.TFL_SOURCE_CODE_DIR}/transformerlab/galleries/model-gallery.json") as f:
        gallery = json.load(f)

    result = None

    for model in gallery:
        if model['uniqueID'] == model_id or model['huggingface_repo'] == model_id:
            result = model
            break

    return result


@router.get("/healthz")  # TODO: why isn't this /model/helathz?
async def healthz():
    return {"message": "OK"}


@router.get("/model/gallery")
async def model_gallery_list_all():
    with open(f"{dirs.TFL_SOURCE_CODE_DIR}/transformerlab/galleries/model-gallery.json") as f:
        gallery = json.load(f)

    local_models = await db.model_local_list()
    local_model_names = set(model['model_id'] for model in local_models)

    # Mark which models have been downloaded already. The huggingfacerepo is our model_id.
    for model in gallery:
        model['downloaded'] = True if model['huggingface_repo'] in local_model_names else False

    return gallery


@router.get("/model/gallery/{model_id}")
async def model_gallery(model_id: str):

    # convert "~~~"" in string to "/":
    model_id = model_id.replace("~~~", "/")

    return get_model_details_from_gallery(model_id)


@router.get("/model/local/{model_id}")
async def model_details_from_source(model_id: str):

    # convert "~~~"" in string to "/":
    model_id = model_id.replace("~~~", "/")

    # Try to get from huggingface first
    model = model_helper.get_model_by_source_id("huggingface", model_id)

    # If there is no model then try looking in the filesystem
    if not model:
        model = model_details_from_filesystem(model_id)

    return model


@router.get("/model/details/{model_id}")
async def model_details_from_filesystem(model_id: str):

    # convert "~~~"" in string to "/":
    model_id = model_id.replace("~~~", "/")

    # TODO: Refactor this code with models/list function
    # see if the model exists locally
    model_path = get_model_dir(model_id)
    if (os.path.isdir(model_path)):

        # Look for model information in info.json
        info_file = os.path.join(model_path, "info.json")
        try:
            with open(info_file, "r") as f:
                filedata = json.load(f)
                f.close()

                # NOTE: In some places info.json may be a list and in others not
                # Once info.json format is finalized we can remove this
                if isinstance(filedata, list):
                    filedata = filedata[0]

                # Some models are a single file (possibly of many in a directory, e.g. GGUF)
                # For models that have model_filename set we should link directly to that specific file
                if ("json_data" in filedata and filedata["json_data"]):
                    return filedata["json_data"]

        except FileNotFoundError:
            # do nothing: file doesn't exist
            pass

    return {}


@router.get(path="/model/login_to_huggingface")
async def login_to_huggingface():
    from huggingface_hub import login
    token = await db.config_get("HuggingfaceUserAccessToken")

    if token is None:
        return {"message": "HuggingfaceUserAccessToken not set"}

    # Note how login() works. When you login, huggingface_hub saves your token as a file to ~/.huggingface/token
    # and it is there forever, until you delete it. So you only need to login once and it
    # persists across sessions.
    # https://huggingface.co/docs/huggingface_hub/v0.19.3/en/package_reference/login#huggingface_hub.login

    try:
        login(token=token)
        return {"message": "OK"}
    except:
        return {"message": "Login failed"}


def get_model_details_from_huggingface(hugging_face_id: str):
    """
    Gets model config details from hugging face and convert in to gallery format

    This function can raise several Exceptions from HuggingFace:
        RepositoryNotFoundError: invalid model ID or private repo without access
        GatedRepoError: Model exists but this user is not on the authorized lsit
        EntryNotFoundError: this model doesn't have a config.json
        HfHubHTTPError: Catch all for everything else
    """

    # Use the Hugging Face Hub API to download the config.json file for this model
    # This may throw an exception if the model doesn't exist or we don't have access rights
    hf_hub_download(repo_id=hugging_face_id, filename="config.json")

    # Also get model info for metadata and license details
    # Similar to hf_hub_download this can throw exceptions
    # Some models don't have a model card (mostly models that have been deprecated)
    # In that case just set model_card_data to an empty object
    hf_model_info = model_info(hugging_face_id)
    try:
        model_card = hf_model_info.card_data
        model_card_data = model_card.data.to_dict()
    except AttributeError:
        model_card_data = {}

    # Use Hugging Face file system API and let it figure out which file we should be reading
    fs = HfFileSystem()
    filename = os.path.join(hugging_face_id, "config.json")
    with fs.open(filename) as f:
        filedata = json.load(f)

        # config.json stores a list of architectures but we only store one so just take the first!
        architecture_list = filedata.get("architectures", [])
        architecture = architecture_list[0] if architecture_list else ""

        # Oh except that GGUF and MLX aren't listed as architectures, we have to look in library_name
        library_name = getattr(hf_model_info, "library", "")
        if (library_name == "MLX" or library_name == "GGUF"):
            architecture = library_name

        # TODO: Context length definition seems to vary by architecture. May need conditional logic here.
        context_size = filedata.get("max_position_embeddings", "")

        # TODO: Figure out description, paramters, model size
        newmodel = basemodel.BaseModel(hugging_face_id)
        config = newmodel.json_data
        config = {
            "uniqueID": hugging_face_id,
            "name": filedata.get("name", hugging_face_id),
            "description": f"Downloaded by TransformerLab from Hugging Face at {hugging_face_id}",
            "parameters": "",
            "context": context_size,
            "private": getattr(hf_model_info, "private", False),
            "gated": getattr(hf_model_info, "gated", False),
            "architecture": architecture,
            "huggingface_repo": hugging_face_id,
            "model_type": filedata.get("model_type", ""),
            "library_name": library_name,
            "transformers_version": filedata.get("transformers_version", ""),
            "license": model_card_data.get("license", ""),
            "logo": ""
        }
        return config

    # Something did not go to plan
    return None


async def download_huggingface_model(hugging_face_id: str, model_details: str = {}, job_id: int | None = None):
    """
    Tries to download a model with the id hugging_face_id
    model_details is the object created from the gallery json
        or a similarly-formatted object containing the fields:
        - name (display name)
        - size_of_model_in_mb (for progress meter)
        - huggingface_filename (for models with many files like GGUF)

    Returns an object with the following fields:
    - status: "success" or "error"
    - message: error message if status is "error"
    """
    if job_id is None:
        job_id = await db.job_create(type="DOWNLOAD_MODEL", status="STARTED",
                                     job_data='{}')
    else:
        await db.job_update(job_id=job_id, type="DOWNLOAD_MODEL", status="STARTED")

    # try to figure out model details from model_details object
    # default is empty object so can't assume any of this exists
    name = model_details.get("name", hugging_face_id)
    model_size = str(model_details.get("size_of_model_in_mb", -1))
    hugging_face_filename = model_details.get("huggingface_filename", None)

    args = [f"{dirs.TFL_SOURCE_CODE_DIR}/transformerlab/shared/download_huggingface_model.py",
            "--model_name", hugging_face_id,
            "--job_id", str(job_id),
            "--total_size_of_model_in_mb", model_size
            ]

    if hugging_face_filename is not None:
        args += ["--model_filename", hugging_face_filename]

    try:
        process = await shared.async_run_python_script_and_update_status(python_script=args, job_id=job_id, begin_string="Model Download Progress")
        exitcode = process.returncode

        if (exitcode != 0):
            error_msg = await db.job_get_error_msg(job_id)
            if not error_msg:
                error_msg = f"Exit code {exitcode}"
                await db.job_update_status(job_id, "FAILED", error_msg)
            return {"status": "error", "message": error_msg}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        await db.job_update_status(job_id, "FAILED", error_msg)
        return {"status": "error", "message": error_msg}

    if hugging_face_filename is None:
        # only save to local database if we are downloading the whole repo
        await model_local_create(id=hugging_face_id, name=name, json_data=model_details)

    return {"status": "success", "message": "success", "model": model_details, "job_id": job_id}


@router.get(path="/model/download_from_huggingface")
async def download_model_by_huggingface_id(model: str):
    """Takes a specific model string that must match huggingface ID to download
    This function will not be able to infer out description etc of the model
    since it is not in the gallery"""

    # Get model details from Hugging Face
    # If None then that means either the model doesn't exist
    # Or we don't have proper Hugging Face authentication setup
    try:
        model_details = get_model_details_from_huggingface(model)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        return {"status": "error", "message": error_msg}

    if model_details is None:
        error_msg = f"Error reading config for model with ID {model}"
        return {"status": "error", "message": error_msg}

    return await download_huggingface_model(model, model_details)


@router.get(path="/model/download_model_from_gallery")
async def download_model_from_gallery(gallery_id: str, job_id: int | None = None):
    """Provide a reference to a model in the gallery, and we will download it
    from huggingface

    You can manually specify a pre-created job_id if you want to track the progress of the download with
    a defined job_id provided by the API using /job/createId"""

    # Get model details from the gallery
    # If None then return an error
    gallery_entry = get_model_details_from_gallery(gallery_id)
    if gallery_entry is None:
        return {"status": "error", "message": "Model not found in gallery"}

    # Need to use huggingface repo to download - not always the same as uniqueID
    huggingface_id = gallery_entry.get('huggingface_repo', gallery_id)
    return await download_huggingface_model(huggingface_id, gallery_entry, job_id)


@router.get("/model/get_conversation_template")
async def get_model_prompt_template(model: str):
    # Below we grab the conversation template from FastChat's model adapter
    # solution by passing in the model name
    return get_conversation_template(model)


@router.get("/model/list")
async def model_local_list():

    # the model list is a combination of downloaded hugging face models and locally generated models
    # start with the list of downloaded models which is stored in the db
    models = await db.model_local_list()

    # now generate a list of local models by reading the filesystem
    models_dir = get_models_dir()

    # now iterate through all the subdirectories in the models directory
    with os.scandir(models_dir) as dirlist:
        for entry in dirlist:
            if entry.is_dir():

                # Look for model information in info.json
                info_file = os.path.join(models_dir, entry, "info.json")
                try:
                    with open(info_file, "r") as f:
                        filedata = json.load(f)
                        f.close()

                        # NOTE: In some places info.json may be a list and in others not
                        # Once info.json format is finalized we can remove this
                        if isinstance(filedata, list):
                            filedata = filedata[0]

                        # tells the app this model was loaded from workspace directory
                        filedata["stored_in_filesystem"] = True

                        # Set local_path to the filesystem location
                        # this will tell Hugging Face to not try downloading
                        filedata["local_path"] = os.path.join(
                            models_dir, entry)

                        # Some models are a single file (possibly of many in a directory, e.g. GGUF)
                        # For models that have model_filename set we should link directly to that specific file
                        if ("model_filename" in filedata and filedata["model_filename"]):
                            filedata["local_path"] = os.path.join(
                                filedata["local_path"], filedata["model_filename"])

                        models.append(filedata)

                except FileNotFoundError:
                    # do nothing: just ignore this directory
                    pass

    return models


@router.get("/model/create")
async def model_local_create(id: str, name: str, json_data={}):
    await db.model_local_create(model_id=id, name=name, json_data=json_data)
    return {"message": "model created"}


@router.get("/model/delete")
async def model_local_delete(model_id: str):
    # If this is a locally generated model then actually delete from filesystem
    # Check for the model stored in a directory based on the model name (i.e. the part after teh slash)
    root_models_dir = get_models_dir()
    model_dir = model_id.rsplit('/', 1)[-1]
    info_file = os.path.join(root_models_dir, model_dir, "info.json")
    if (os.path.isfile(info_file)):
        model_path = os.path.join(root_models_dir, model_dir)
        print(f"Deleteing {model_path}")
        shutil.rmtree(model_path)

    else:
        # If this is a hugging face model then delete from the database but leave in the cache
        print(
            f"Deleting model {model_id}. Note that this will not free up space because it remains in the HuggingFace cache.")
        print("If you want to delete the model from the HuggingFace cache, you must delete it from:")
        print("~/.cache/huggingface/hub/")

    # Delete from the database
    await db.model_local_delete(model_id=model_id)
    return {"message": "model deleted"}


@router.post("/model/pefts")
async def model_gets_pefts(model_id: Annotated[str, Body()],):
    workspace_dir = dirs.WORKSPACE_DIR
    adaptors_dir = f"{workspace_dir}/adaptors/{model_id}"
    adaptors = []
    if (os.path.exists(adaptors_dir)):
        adaptors = os.listdir(adaptors_dir)
    return adaptors


@router.get("/model/delete_peft")
async def model_delete_peft(model_id: str, peft: str):
    workspace_dir = dirs.WORKSPACE_DIR
    adaptors_dir = f"{workspace_dir}/adaptors/{model_id}"
    peft_path = f"{adaptors_dir}/{peft}"
    shutil.rmtree(peft_path)
    return {"message": "success"}


@router.get(path="/model/get_local_hfconfig")
async def get_local_hfconfig(model_id: str):
    """
    Returns the config.json file for a model stored in the local filesystem
    """
    try:
        local_file = snapshot_download(model_id, local_files_only=True)
        config_json = os.path.join(local_file, "config.json")
        contents = "{}"
        with open(config_json) as f:
            contents = f.read()
        d = json.loads(contents)
    except:
        # failed to open config.json so create an empty config
        d = {}

    return d


async def get_model_from_db(model_id: str):
    return await db.model_local_get(model_id)


@router.get("/model/list_local_uninstalled")
async def models_list_local_uninstalled(path: str = ""):

    # first search and get a list of BaseModel objects 
    found_models = []
    if path is not None and path != "":
        if os.path.isfile(path):
            found_models = []
        elif os.path.isdir(path):
            found_models = await localmodel.list_models(path)
        else:
            return {"status": "error", "message": "Invalid path"}

    # If a folder wasn't given then search known sources for uninstalled models
    else:
        found_models = await models_search_for_local_uninstalled()

    # Then iterate through models and return appropriate details
    response_models = []
    for found_model in found_models:
        # Figure out if this model is supported in TransformerLab
        architecture = found_model.architecture
        supported = model_helper.model_architecture_is_supported(architecture)
        if (found_model.status != "OK"):
            status = f"❌ {found_model.status}"
        elif found_model.architecture == "unknown" or found_model.architecture == "":
            status = "❌ Unknown architecture"
        elif not supported:
            status = f"❌ {architecture}"
        else:
            status = f"✅ {architecture}"

        new_model = {
            "id": found_model.id,
            "name": found_model.name,
            "architecture": architecture,
            "source": found_model.model_source,
            "installed": False,
            "status": status,
            "supported": supported
        }
        response_models.append(new_model)

    return {"status": "success", "data": response_models}

async def models_search_for_local_uninstalled():
    # iterate through each model source and look for uninstalled models
    modelsources = model_helper.list_model_sources()
    models = []
    for source in modelsources:
        source_models = await model_helper.list_models_from_source(source, uninstalled_only=True)
        models += source_models

    return models


@router.get("/model/import_local")
async def model_import_local(model_source: str, model_id: str):

    if model_source not in model_helper.list_model_sources():
        return {"status": "error", "message": f"Invalid model source {model_source}."}

    model = model_helper.get_model_by_source_id(model_source, model_id)

    # Only add a row for uninstalled and supported repos
    if not model:
        return {"status": "error", "message": f"{model_id} not found in {model_source}."}
    if model.status != "OK":
        return {"status": "error", "message": model.status}
    if model.architecture == "unknown" or model.architecture == "":
        return {"status": "error", "message": f"Unable to determine model architecture."}
    if not model_helper.model_architecture_is_supported(model.architecture):
        return {"status": "error", "message": f"Architecture {model.architecture} not supported."}
    if await model.is_installed():
        return {"status": "error", "message": f"{model_id} is already installed."}

    print(f"Importing {model_id}...")

    await model_local_create(id=model_id, name=model.name, json_data=model.json_data)

    return {"status": "success", "data": model_id}
