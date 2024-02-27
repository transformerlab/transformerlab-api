import os
import re
import shutil
import unicodedata

import aiofiles
from datasets import load_dataset, load_dataset_builder
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

import transformerlab.db as db
from transformerlab.shared import dirs

from transformerlab.shared.shared import slugify

router = APIRouter(prefix="/data", tags=["datasets"])

# Get list of datasets that we have in our hardcoded gallery


@router.get("/gallery", summary="Display the datasets available in the dataset gallery.")
async def dataset_gallery():
    file_location = os.path.join(dirs.TFL_SOURCE_CODE_DIR,
                                 "transformerlab", "galleries", "data-gallery.json")
    return FileResponse(file_location)


# Get info on dataset from huggingface


@router.get("/info", summary="Fetch the details of a particular dataset.")
async def dataset_info(dataset_id: str):
    d = await db.get_dataset(dataset_id)

    if d is None:
        return {}

    r = {}

    if d["location"] == "local":
        dataset = load_dataset(path=dirs.dataset_dir_by_id(dataset_id))
        # print(dataset['train'].features)
        r["features"] = dataset["train"].features
    else:
        ds_builder = load_dataset_builder(dataset_id)
        r = {
            "description": ds_builder.info.description,
            "features": ds_builder.info.features,
            "dataset_size": ds_builder.info.dataset_size,
            "download_size": ds_builder.info.download_size,
            "citation": ds_builder.info.citation,
            "homepage": ds_builder.info.homepage,
            "license": ds_builder.info.license,
            "splits": ds_builder.info.splits,
            "supervised_keys": ds_builder.info.supervised_keys,
            "version": ds_builder.info.version,
        }
    return r


@router.get("/preview", summary="Preview the contents of a dataset.")
async def dataset_preview(dataset_id: str):
    d = await db.get_dataset(dataset_id)

    result = []

    print(d)

    if d["location"] == "local":
        # TODO: This can fail in many ways considering this is user generated input.
        # Need to catch exception and return error
        dataset = load_dataset(path=dirs.dataset_dir_by_id(dataset_id))
        # print(dataset['train'].features)

        # convert dataset to array of dicts
        for i in range(0, min(10, len(dataset["train"]))):
            result.append(dataset["train"][i])

    else:
        dataset = load_dataset(dataset_id)
        for i in range(0, 10):
            result.append(dataset["train"][i])
    return result


@router.get("/download", summary="Download a dataset from the HuggingFace Hub to the LLMLab server.")
async def dataset_download(dataset_id: str):
    # Check to make sure we don't have a dataset with this name
    # Possibly we want to allow redownloading in the future but for we can't add duplicate dataset_id to the DB
    row = await db.get_dataset(dataset_id)
    if row is not None:
        return {"status": "error", "message": f"A dataset with the name {dataset_id} already exists"}

    ds_builder = load_dataset_builder(dataset_id)
    try:
        dataset = load_dataset(dataset_id)
    except Exception as e:
        return {"status": "error", "message":  str(e)}

    await db.create_huggingface_dataset(
        dataset_id, ds_builder.info.description, ds_builder.info.download_size
    )

    return {"status": "success"}


@router.get("/list", summary="List available datasets.")
async def dataset_list():
    list = await db.get_datasets()
    return list  # convert list to JSON object


@router.get("/new", summary="Create a new dataset.")
async def dataset_new(dataset_id: str):
    dataset_id = slugify(dataset_id)

    # Check to make sure we don't have a dataset with this name
    row = await db.get_dataset(dataset_id)
    if row is not None:
        return {"status": "error", "message": f"A dataset with the name {dataset_id} already exists"}

    # Create a new dataset in the database
    await db.create_local_dataset(dataset_id)

    # Now make a directory that maps to the above dataset_id
    # Check if the directory already exists
    if not os.path.exists(dirs.dataset_dir_by_id(dataset_id)):
        os.makedirs(dirs.dataset_dir_by_id(dataset_id))
    return {"status": "success", "dataset_id": dataset_id}


@router.get("/delete", summary="Delete a dataset.")
async def dataset_delete(dataset_id: str):
    await db.delete_dataset(dataset_id)

    # delete directory and contents. ignore_errors because we don't care if the directory doesn't exist
    shutil.rmtree(dirs.dataset_dir_by_id(dataset_id), ignore_errors=True)

    return {"status": "success"}


@router.post("/fileupload", summary="Upload the contents of a dataset.")
async def create_upload_file(dataset_id: str, file: UploadFile):
    print("uploading filename is: " + str(file.filename))

    # # ensure filename is in the format <something>_train.jsonl or <something>_eval.jsonl
    # if not re.match(r"^.+_(train|eval).jsonl$", str(file.filename)):
    #     raise HTTPException(
    #         status_code=403, detail=f"The filenames must be named EXACTLY: {dataset_id}_train.jsonl and {dataset_id}_eval.jsonl")

    # ensure the filename is exactly {dataset_id}_train.jsonl or {dataset_id}_eval.jsonl
    if not re.match(rf"^{dataset_id}_(train|eval).jsonl$", str(file.filename)):
        raise HTTPException(
            status_code=403, detail=f"The filenames must be named EXACTLY: {dataset_id}_train.jsonl and {dataset_id}_eval.jsonl")

    dataset_id = slugify(dataset_id)

    # Save the file to the dataset directory
    try:
        content = await file.read()
        newfilename = os.path.join(
            dirs.dataset_dir_by_id(dataset_id), str(file.filename))
        async with aiofiles.open(newfilename, "wb") as out_file:
            await out_file.write(content)
    except Exception:
        raise HTTPException(
            status_code=403, detail="There was a problem uploading the file")

    return {"status": "success", "filename": file.filename}
