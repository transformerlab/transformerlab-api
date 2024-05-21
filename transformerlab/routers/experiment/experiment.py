import json
import os
from pathlib import Path

from typing import Annotated

from fastapi import APIRouter, Body

import transformerlab.db as db
from transformerlab.shared import shared
from transformerlab.shared import dirs
from transformerlab.routers.experiment import rag, documents, plugins, conversations, export, evals


router = APIRouter(prefix="/experiment")

router.include_router(
    router=rag.router, prefix="/{experimentId}", tags=["rag"])
router.include_router(
    router=documents.router, prefix="/{experimentId}", tags=["documents"])
router.include_router(
    router=plugins.router, prefix="/{id}", tags=["plugins"])
router.include_router(
    router=conversations.router, prefix="/{experimentId}", tags=["conversations"])
router.include_router(
    router=export.router, prefix="/{id}", tags=["export"])
router.include_router(
    router=evals.router, prefix="/{experimentId}", tags=["evals"])


EXPERIMENTS_DIR: str = dirs.EXPERIMENTS_DIR


async def convert_experiment_id_to_name_if_needed(id: int | str) -> str:
    """As we move to make the experiment name the best way to fetch an experiment, we use this function to
    convert an experiment id to an experiment name if needed.

    Later on we can remove this function, once we have updated all the code to use the experiment name instead of the id."""
    if isinstance(id, int):
        data = await db.experiment_get(id)
        if data is None:
            return id
        return data["name"]
    return id


async def convert_experiment_name_to_id_if_needed(name: int | str) -> int:
    """As we move to make the experiment name the best way to fetch an experiment, we use this function to
    convert an experiment name to an experiment id if needed.

    Later on we can remove this function, once we have updated all the code to use the experiment name instead of the id."""
    if isinstance(name, str):
        data = await db.experiment_get_by_name(name)
        if data is None:
            return name
        return data["id"]
    return name


@router.get("/", summary="Get all Experiments", tags=["experiment"])
async def experiments_get_all():
    """Get a list of all experiments"""
    return await db.experiment_get_all()


@router.get("/create", summary="Create Experiment", tags=["experiment"])
async def experiments_create(name: str):
    newid = await db.experiment_create(name, "{}")
    return newid


@router.get("/{id}", summary="Get Experiment by ID", tags=["experiment"])
async def experiment_get(id: str | int):
    id = await convert_experiment_name_to_id_if_needed(id)

    data = await db.experiment_get(id)

    if data is None:
        return {"status": "error", "message": f"Experiment {id} does not exist"}

    # convert the JSON string called config to json object
    data["config"] = json.loads(data["config"])
    return data


@router.get("/{id}/delete", tags=["experiment"])
async def experiments_delete(id: str | int):
    id = await convert_experiment_name_to_id_if_needed(id)
    await db.experiment_delete(id)
    return {"message": f"Experiment {id} deleted"}


@router.get("/{id}/update", tags=["experiment"])
async def experiments_update(id: str | int, name: str):
    id = await convert_experiment_name_to_id_if_needed(id)
    await db.experiment_update(id, name)
    return {"message": f"Experiment {id} updated to {name}"}


@router.get("/{id}/update_config", tags=["experiment"])
async def experiments_update_config(id: str | int, key: str, value: str):
    id = await convert_experiment_name_to_id_if_needed(id)
    await db.experiment_update_config(id, key, value)
    return {"message": f"Experiment {id} updated"}


@router.post("/{id}/prompt", tags=["experiment"])
async def experiments_save_prompt_template(id: str | int, template: Annotated[str, Body()]):
    id = await convert_experiment_name_to_id_if_needed(id)
    await db.experiment_save_prompt_template(id, template)
    return {"message": f"Experiment {id} prompt template saved"}


@router.post("/{id}/save_file_contents", tags=["experiment"])
async def experiment_save_file_contents(id: str | int, filename: str, file_contents: Annotated[str, Body()]):
    id = await convert_experiment_name_to_id_if_needed(id)
    # first get the experiment name:
    data = await db.experiment_get(id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    # remove file extension from file:
    [filename, file_ext] = os.path.splitext(filename)

    if (file_ext != '.py') and (file_ext != '.ipynb') and (file_ext != '.md'):
        return {"message": f"File extension {file_ext} not supported"}

    # clean the file name:
    filename = shared.slugify(filename)

    # make directory if it does not exist:
    if not os.path.exists(f"{EXPERIMENTS_DIR}/{experiment_name}"):
        os.makedirs(f"{EXPERIMENTS_DIR}/{experiment_name}")

    # now save the file contents, overwriting if it already exists:
    with open(f"{EXPERIMENTS_DIR}/{experiment_name}/{filename}{file_ext}", "w") as f:
        f.write(file_contents)

    return {"message": f"{EXPERIMENTS_DIR}/{experiment_name}/{filename}{file_ext} file contents saved"}


@router.get("/{id}/file_contents", tags=["experiment"])
async def experiment_get_file_contents(id: str | int, filename: str):
    id = await convert_experiment_name_to_id_if_needed(id)

    # first get the experiment name:
    data = await db.experiment_get(id)

    # if the experiment does not exist, return an error:
    if data is None:
        return {"message": f"Experiment {id} does not exist"}

    experiment_name = data["name"]

    # remove file extension from file:
    [filename, file_ext] = os.path.splitext(filename)

    allowed_extensions = ['.py', '.ipynb', '.md', '.txt']

    if file_ext not in allowed_extensions:
        return {"message": f"File extension {file_ext} for {filename} not supported"}

    # clean the file name:
    # filename = shared.slugify(filename)

    # The following prevents path traversal attacks:
    experiment_dir = dirs.experiment_dir_by_name(experiment_name)
    final_path = Path(experiment_dir).joinpath(
        filename + file_ext).resolve().relative_to(experiment_dir)

    final_path = experiment_dir + "/" + str(final_path)
    print("Listing Contents of File: " + final_path)

    # now get the file contents
    try:
        with open(final_path, "r") as f:
            file_contents = f.read()
    except FileNotFoundError:
        return ""

    return file_contents
