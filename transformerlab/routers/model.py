import json
import shutil
from typing import Annotated
import transformerlab.db as db
from fastapi import APIRouter, Body
from fastapi.responses import FileResponse
from fastchat.model.model_adapter import get_conversation_template
import os

from transformerlab.shared import shared

router = APIRouter(tags=["model"])


@router.get("/healthz")  # TODO: why isn't this /model/helathz?
async def healthz():
    return {"message": "OK"}


@router.get("/model/gallery")
async def model_gallery_list_all():
    with open("transformerlab/galleries/model-gallery.json") as f:
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

    with open("transformerlab/galleries/model-gallery.json") as f:
        gallery = json.load(f)

    result = None

    for model in gallery:
        if model['huggingface_repo'] == model_id:
            result = model
            break

    return result


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


@router.get(path="/model/download_from_huggingface")
async def download_model_from_huggingface(model: str):
    """specify a specific model from huggingface to download
    This function will not be able to infer out description etc of the model
    since it is not in the gallery"""
    job_id = await db.job_create(type="DOWNLOAD_MODEL", status="STARTED",
                                 job_data='{}')

    args = ["transformerlab/shared/download_huggingface_model.py",
            "--model_name", model]

    try:
        await shared.async_run_python_script_and_update_status(python_script=args, job_id=job_id, begin_string="Fetching")
    except Exception as e:
        await db.job_update(job_id=job_id, status="FAILED")
        return {"message": "Failed to download model"}

    # Now save this to the local database
    await model_local_create(id=model, name=model)
    return {"message": "success", "model": model, "job_id": job_id}


@router.get(path="/model/download_model_from_gallery")
async def download_model_from_gallery(gallery_id: str):
    """Provide a reference to a model in the gallery, and we will download it
    from huggingface"""

    # get all models from gallery
    with open("transformerlab/galleries/model-gallery.json") as f:
        gallery = json.load(f)

    gallery_entry = None

    # for each entry in the gallery, check if the model_id matches
    for model in gallery:
        if model['uniqueID'] == gallery_id:
            gallery_entry = model
            break
    else:
        return {"message": "Model not found in gallery"}

    hugging_face_id = gallery_entry['huggingface_repo']

    print(gallery_entry)
    hugging_face_filename = gallery_entry.get("huggingface_filename", None)

    name = gallery_entry['name']

    job_id = await db.job_create(type="DOWNLOAD_MODEL", status="STARTED",
                                 job_data='{}')

    args = ["transformerlab/shared/download_huggingface_model.py",
            "--model_name", hugging_face_id,
            ]

    if hugging_face_filename is not None:
        args += ["--model_filename", hugging_face_filename]

    try:
        await shared.async_run_python_script_and_update_status(python_script=args, job_id=job_id, begin_string="Fetching")
    except Exception as e:
        await db.job_update(job_id=job_id, status="FAILED")
        return {"message": "Failed to download model"}

    # Now save this to the local database
    await model_local_create(id=hugging_face_id, name=name, json_data=gallery_entry)
    return {"message": "success", "model": model, "job_id": job_id}


@router.get("/model/get_conversation_template")
async def get_model_prompt_template(model: str):
    # Below we grab the conversation template from FastChat's model adapter
    # solution by passing in the model name
    return get_conversation_template(model)


@router.get("/model/list")
async def model_local_list():
    models = await db.model_local_list()
    return models


@router.get("/model/create")
async def model_local_create(id: str, name: str, json_data={}):
    await db.model_local_create(model_id=id, name=name, json_data=json_data)
    return {"message": "model created"}


@router.post("/model/pefts")
async def model_gets_pefts(model_id: Annotated[str, Body()],):
    workspace_dir = shared.WORKSPACE_DIR
    adaptors_dir = f"{workspace_dir}/adaptors/{model_id}"
    adaptors = []
    if (os.path.exists(adaptors_dir)):
        adaptors = os.listdir(adaptors_dir)
    return adaptors


@router.get("/model/delete_peft")
async def model_delete_peft(model_id: str, peft: str):
    workspace_dir = shared.WORKSPACE_DIR
    adaptors_dir = f"{workspace_dir}/adaptors/{model_id}"
    peft_path = f"{adaptors_dir}/{peft}"
    shutil.rmtree(peft_path)
    return {"message": "success"}
