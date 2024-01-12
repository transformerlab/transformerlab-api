import os
import re
import unicodedata

import aiofiles
from datasets import load_dataset, load_dataset_builder
from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse

import transformerlab.db as db
from transformerlab.shared.shared import slugify

router = APIRouter(prefix="/data", tags=["datasets"])

# Get list of datasets that we have in our hardcoded gallery


@router.get("/gallery", summary="Display the models available for LLMLab to download.")
async def model_gallery():
    return FileResponse("transformerlab/galleries/data-gallery.json")


# Get info on dataset from huggingface


@router.get("/info", summary="Fetch the details of a particular dataset.")
async def dataset_info(dataset_id: str):
    d = await db.get_dataset(dataset_id)

    if d is None:
        return {}

    r = {}

    if d["location"] == "local":
        dataset = load_dataset(path=f"workspace/datasets/{dataset_id}")
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
        dataset = load_dataset(path=f"workspace/datasets/{dataset_id}")
        # print(dataset['train'].features)

        # convert dataset to array of dicts
        for i in range(0, 10):
            result.append(dataset["train"][i])

    else:
        dataset = load_dataset(dataset_id)
        for i in range(0, 10):
            result.append(dataset["train"][i])
    return result


# Load a specified dataset from the HuggingFace Hub


@router.get("/download", summary="Download a dataset to the LLMLab server.")
async def dataset_download(dataset_id: str):
    ds_builder = load_dataset_builder(dataset_id)
    await db.create_huggingface_dataset(
        dataset_id, ds_builder.info.description, ds_builder.info.download_size
    )

    load_dataset(dataset_id)
    return {"message": "OK"}


@router.get("/list", summary="List available datasets.")
async def dataset_list():
    list = await db.get_datasets()
    return list  # convert list to JSON object


@router.get("/new", summary="Create a new dataset.")
async def dataset_new(dataset_id: str):
    dataset_id = slugify(dataset_id)
    # Create a new dataset in the database
    await db.create_local_dataset(dataset_id)
    print(dataset_id)
    # Now make a directory that maps to the above dataset_id
    os.makedirs(f"workspace/datasets/{dataset_id}")
    return {"message": "OK", "dataset_id": dataset_id}


@router.get("/delete", summary="Delete a dataset.")
async def dataset_delete(dataset_id: str):
    await db.delete_dataset(dataset_id)
    return {"message": "OK"}


@router.post("/fileupload", summary="Upload the contents of a dataset.")
async def create_upload_file(dataset_id: str, file: UploadFile):
    dataset_id = slugify(dataset_id)
    # Save the file to the dataset directory
    async with aiofiles.open(
        f"workspace/datasets/{dataset_id}/{file.filename}", "wb"
    ) as out_file:
        content = await file.read()
        await out_file.write(content)

    return {"filename": file.filename}
