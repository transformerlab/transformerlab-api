import contextlib
import os
import shutil
import json
import aiofiles
from PIL import Image as PILImage
from datasets import load_dataset, load_dataset_builder, Image
from fastapi import APIRouter, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
from io import BytesIO
import base64
import hashlib
from pathlib import Path
import transformerlab.db as db
from transformerlab.shared import dirs
from datasets.data_files import EmptyDatasetError
from transformerlab.shared.shared import slugify
from transformerlab.shared import galleries


from werkzeug.utils import secure_filename

from jinja2 import Environment
from jinja2.sandbox import SandboxedEnvironment
import logging


jinja_environment = Environment()
sandboxed_jinja2_evironment = SandboxedEnvironment()

logging.basicConfig(level=logging.ERROR)


# Configure logging
GLOBAL_LOG_PATH = dirs.GLOBAL_LOG_PATH


def log(msg):
    with open(GLOBAL_LOG_PATH, "a") as f:
        f.write(msg + "\n")


def hash_image_content(image_obj, is_path=False):
    """
    Accepts either a file path (str/Path) or a base64 data URL string.
    Produces the base64-encoded image field exactly as in the original encoding,
    then hashes it using SHA256.
    """
    try:
        if isinstance(image_obj, str):
            if is_path:
                # It's a file path - open and encode as data URL base64
                with PILImage.open(image_obj) as image:
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG")
                    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{encoded}"
                    image_bytes = data_url.encode("utf-8")
            else:
                # It's a base64 data URL string from edit["image"]
                image_bytes = image_obj.encode("utf-8")
        else:
            return None

        return hashlib.sha256(image_bytes).hexdigest()
    except Exception as e:
        log(f"Error hashing image content: {e}")
        return None


# logging.basicConfig(filename=GLOBAL_LOG_PATH, level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


router = APIRouter(prefix="/data", tags=["datasets"])

# Get list of datasets that we have in our hardcoded gallery


class SuccessResponse(BaseModel):
    status: str
    data: Dict[str, Any]


class ErrorResponse(BaseModel):
    status: str
    message: str


@router.get(
    "/gallery",
    summary="Display the datasets available in the dataset gallery.",
    responses={
        200: {
            "model": SuccessResponse,
            "description": "Successful response. Data is a list of column names followed by data, which can be of any datatype.",
        },
        400: {"model": ErrorResponse},
    },
)
async def dataset_gallery() -> Any:
    gallery = galleries.get_data_gallery()

    local_datasets = await db.get_datasets()

    local_dataset_names = set(str(dataset["dataset_id"]) for dataset in local_datasets)
    for dataset in gallery:
        dataset["downloaded"] = True if dataset["huggingfacerepo"] in local_dataset_names else False
    return {"status": "success", "data": gallery}


@router.get("/info", summary="Fetch the details of a particular dataset.")
async def dataset_info(dataset_id: str):
    d = await db.get_dataset(dataset_id)
    if d is None:
        return {"status": "error", "message": "Dataset not found."}

    r = {}

    if d["location"] == "local":
        dataset_dir = dirs.dataset_dir_by_id(slugify(dataset_id))

        try:
            parquet_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(dataset_dir, followlinks=False)
                for f in files
                if f.lower().endswith(".parquet")
            ]
            if parquet_files:
                # Load using Parquet
                dataset = load_dataset("parquet", data_files=parquet_files)
                splits = list(dataset.keys())
                split = splits[0]
                features = dataset[split].features
                is_image_dataset = any(
                    isinstance(feature.dtype, Image.__class__) or (feature._type and feature._type.lower() == "image")
                    for feature in features.values()
                )
                r["features"] = features
                r["splits"] = splits
                r["is_parquet"] = True
                r["is_image"] = is_image_dataset
                return r
            dataset = load_dataset(path=dataset_dir)
            splits = list(dataset.keys())
            split = splits[0]
            features = dataset[split].features

            is_image_dataset = any(isinstance(f, Image) for f in features.values())
            if is_image_dataset:
                dataset = load_dataset("imagefolder", data_dir=dataset_dir)
                splits = list(dataset.keys())
                split = splits[0]
                features = dataset[split].features

                r["is_image"] = True
                r["features"] = features
                r["splits"] = splits

                label_set = set()
                for root, dirs_, files in os.walk(dataset_dir, followlinks=False):
                    for file in files:
                        if str(file).lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            image_path = os.path.join(root, file)
                            parent_folder = Path(image_path).parent.name
                            if parent_folder.lower() not in ["train", "test"]:
                                label_set.add(parent_folder)
                available_labels = sorted(label_set)
                r["labels"] = available_labels
            else:
                r["is_image"] = False
                r["features"] = features
        except EmptyDatasetError:
            return {"status": "error", "message": "The dataset is empty."}
        except Exception as e:
            log(f"Exception occurred: {type(e).__name__}: {e}")
            return {"status": "error"}

    else:
        dataset_config = d.get("json_data", {}).get("dataset_config", None)
        config_name = d.get("json_data", {}).get("config_name", None)
        try:
            if dataset_config is not None:
                ds_builder = load_dataset_builder(dataset_id, dataset_config, trust_remote_code=True)
            elif config_name is not None:
                ds_builder = load_dataset_builder(path=dataset_id, name=config_name, trust_remote_code=True)
            else:
                ds_builder = load_dataset_builder(dataset_id, trust_remote_code=True)
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
        except Exception as e:
            log(f"Exception occurred: {type(e).__name__}: {e}")
            return {"status": "error"}
    r["is_parquet"] = False
    return r


@router.get(
    "/preview",
    summary="Preview the contents of a dataset.",
    responses={
        200: {
            "model": SuccessResponse,
            "description": "Successful response. Data is a list of column names followed by data, which can be of any datatype.",
        },
        400: {"model": ErrorResponse},
    },
)
async def dataset_preview(
    dataset_id: str = Query(
        description="The ID of the dataset to preview. This can be a HuggingFace dataset ID or a local dataset ID."
    ),
    offset: int = Query(0, description="The starting index from where to fetch the data.", ge=0),
    split: str = Query(None, description="The split to preview. This can be train, test, or validation."),
    limit: int = Query(10, description="The maximum number of data items to fetch.", ge=1, le=1000),
    streaming: bool = False,
) -> Any:
    d = await db.get_dataset(dataset_id)
    dataset_len = 0
    result = {}

    try:
        if d["location"] == "local":
            dataset = load_dataset(path=dirs.dataset_dir_by_id(dataset_id), streaming=streaming)
        else:
            dataset_config = d.get("json_data", {}).get("dataset_config", None)
            config_name = d.get("json_data", {}).get("config_name", None)
            if dataset_config is not None:
                dataset = load_dataset(dataset_id, dataset_config, trust_remote_code=True, streaming=streaming)
            elif config_name is not None:
                dataset = load_dataset(path=dataset_id, name=config_name, trust_remote_code=True, streaming=streaming)
            else:
                dataset = load_dataset(dataset_id, trust_remote_code=True, streaming=streaming)
    except Exception as e:
        log(f"Exception occurred: {type(e).__name__}: {e}")
        return {"status": "error", "message": "An internal error has occurred."}

    if split is None or split == "":
        splits = list(dataset.keys())
        if len(splits) == 0:
            return {"status": "error", "message": "No splits available in the dataset."}
        split = splits[0]

    if streaming:
        dataset_len = -1
        dataset = dataset[split].skip(offset)
        result["rows"] = list(dataset.take(limit))
        result["splits"] = None
    else:
        if d["location"] != "local" and split not in dataset.keys():
            return {"status": "error", "message": f"Split '{split}' does not exist in the dataset."}
        dataset_len = len(dataset[split])
        result["columns"] = dataset[split][offset : min(offset + limit, dataset_len)]
        result["splits"] = list(dataset.keys())

    result["len"] = dataset_len
    return {"status": "success", "data": result}


@router.get(
    "/preview_with_template",
    summary="Preview the contents of a dataset after applying a jinja template to it.",
    responses={
        200: {"model": SuccessResponse, "description": "Preview data with column names and rows"},
        400: {"model": ErrorResponse},
    },
)
async def dataset_preview_with_template(
    dataset_id: str = Query(description="Dataset ID (HuggingFace or local)"),
    template: str = "",
    offset: int = Query(0, ge=0, description="Starting index"),
    limit: int = Query(10, ge=1, le=1000, description="Max items to fetch"),
) -> Any:
    d = await db.get_dataset(dataset_id)
    dataset_dir = dirs.dataset_dir_by_id(slugify(dataset_id))
    dataset_len = 0

    try:
        if d["location"] == "local":
            is_image_dataset = False
            parquet_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(dataset_dir, followlinks=False)
                for f in files
                if f.lower().endswith(".parquet")
            ]
            if parquet_files:
                dataset = load_dataset("parquet", data_files=parquet_files)
            else:
                dataset = load_dataset(path=dirs.dataset_dir_by_id(dataset_id))

                split_name = list(dataset.keys())[0]
                features = dataset[split_name].features
                is_image_dataset = any(isinstance(f, Image) for f in features.values())

                if not is_image_dataset:
                    dataset_len = len(dataset["train"])
                    return {
                        "status": "success",
                        "data": {
                            "columns": dataset["train"][offset : min(offset + limit, dataset_len)],
                            "len": dataset_len,
                            "offset": offset,
                            "limit": limit,
                        },
                    }
        else:
            dataset_config = d.get("json_data", {}).get("dataset_config", None)
            config_name = d.get("json_data", {}).get("config_name", None)
            if dataset_config:
                dataset = load_dataset(dataset_id, dataset_config, trust_remote_code=True)
            elif config_name:
                dataset = load_dataset(path=dataset_id, name=config_name, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_id, trust_remote_code=True)

    except Exception as e:
        log(f"Exception occurred: {type(e).__name__}: {e}")
        return {"status": "error"}

    dataset_len = sum(len(split) for split in dataset.values())

    jinja_template = sandboxed_jinja2_evironment.from_string(template)
    rows = []
    index = 0

    for split_name, split_data in dataset.items():
        for i in range(len(split_data)):
            if index < offset:
                index += 1
                continue
            if len(rows) >= limit:
                break

            row = dict(split_data[i])
            row["__index__"] = index
            index += 1
            row["split"] = split_name

            if d["location"] == "local" and (is_image_dataset or parquet_files):
                image = row["image"]
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                row["image"] = f"data:image/jpeg;base64,{encoded}"

            row["__formatted__"] = jinja_template.render(row)
            rows.append(row)

    column_names = list(rows[0].keys()) if rows else []

    return {
        "status": "success",
        "data": {"columns": column_names, "rows": rows, "len": dataset_len, "offset": offset, "limit": limit},
    }


@router.post("/save_metadata", summary="Update metadata entries by __index__, caption field, and split.")
async def save_metadata(dataset_id: str, file: UploadFile):
    try:
        # Get and resolve the dataset directory
        dataset_dir = Path(dirs.dataset_dir_by_id(slugify(dataset_id))).resolve()
        # Read uploaded file content
        content = await file.read()
        edits = json.loads(content)

        # Collect valid JSON/JSONL metadata files within dataset_dir
        metadata_files = []
        for root, dirs_, files in os.walk(dataset_dir, followlinks=False):
            root_path = Path(root).resolve()
            if not str(root_path).startswith(str(dataset_dir)):
                continue  # Skip files outside dataset_dir
            for f in files:
                if f.lower().endswith((".json", ".jsonl")):
                    metadata_files.append(root_path / f)

        if not metadata_files:
            return JSONResponse(status_code=404, content={"status": "error", "message": "No metadata files found."})

        edits_applied = 0

        # Possible fields for caption/description
        possible_caption_fields = ["text", "caption", "description"]

        # Process each metadata file
        for metadata_path in metadata_files:
            with open(metadata_path, "r", encoding="utf-8") as f:
                if metadata_path.suffix == ".jsonl":
                    data = [json.loads(line) for line in f]
                elif metadata_path.suffix == ".json":
                    data = json.load(f)
                else:
                    continue

            updated = False
            for edit in edits:
                for row in data:
                    # Find the caption field dynamically
                    row_caption = next((row.get(field) for field in possible_caption_fields if field in row), None)

                    # Hash edit image
                    if row_caption == edit.get("previous_caption") and row.get("label") == edit.get("label"):
                        # Get the image path from the first element of row
                        image_key = next(iter(row))
                        image_path = row[image_key]
                        image_full_path = metadata_path.parent / image_path

                        # Hash row image content
                        row_image_hash = hash_image_content(str(image_full_path), is_path=True)

                        # Hash edit image content from base64
                        try:
                            edit_image_hash = hash_image_content(edit.get("image"))
                        except Exception as e:
                            log(f"Error decoding base64 image: {e}")
                            edit_image_hash = None

                        if row_image_hash == edit_image_hash:
                            for field in possible_caption_fields:
                                if field in row:
                                    row[field] = edit.get("text")
                                    break
                            edits_applied += 1
                            updated = True
                            break

            if updated:
                with open(metadata_path, "w", encoding="utf-8") as f_out:
                    if metadata_path.suffix == ".jsonl":
                        for row in data:
                            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    else:
                        json.dump(data, f_out, ensure_ascii=False, indent=2)

        return {"status": "success", "message": f"Updated {edits_applied} row(s)."}

    except Exception as e:
        log(f"Exception occurred: {type(e).__name__}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Exception"})


@router.get("/download", summary="Download a dataset from the HuggingFace Hub to the LLMLab server.")
async def dataset_download(dataset_id: str, config_name: str = None):
    # Check to make sure we don't have a dataset with this name
    # Possibly we want to allow redownloading in the future but for we can't add duplicate dataset_id to the DB
    row = await db.get_dataset(dataset_id)
    if row is not None:
        return {"status": "error", "message": f"A dataset with the name {dataset_id} already exists"}

    # Try to get the dataset info from the gallery
    gallery = []
    json_data = {}
    gallery = galleries.get_data_gallery()
    for dataset in gallery:
        if dataset["huggingfacerepo"] == dataset_id:
            json_data = dataset

    try:
        dataset_config = json_data.get("dataset_config", None)
        config_name = json_data.get("config_name", config_name)
        if dataset_config is not None:
            ds_builder = load_dataset_builder(dataset_id, dataset_config, trust_remote_code=True)
        elif config_name is not None:
            ds_builder = load_dataset_builder(path=dataset_id, name=config_name, trust_remote_code=True)
        else:
            ds_builder = load_dataset_builder(dataset_id, trust_remote_code=True)
        log(f"Dataset builder loaded for dataset_id: {dataset_id}")

    except ValueError as e:
        log(f"ValueError occurred: {type(e).__name__}: {e}")
        if "Config name is missing" in str(e):
            return {"status": "error", "message": "Please enter the folder_name of the dataset from huggingface"}
        else:
            return {"status": "error", "message": "An internal error has occurred!"}

    except Exception as e:
        log(f"Exception occurred: {type(e).__name__}: {e}")
        return {"status": "error", "message": "An internal error has occurred!"}

    dataset_size = ds_builder.info.download_size
    if not dataset_size:
        dataset_size = -1

    if json_data == {}:
        json_data = {
            "name": ds_builder.info.dataset_name,
            "huggingfacerepo": dataset_id,
            "config_name": config_name,
            "description": ds_builder.info.description,
            "dataset_size": dataset_size,
            "citation": ds_builder.info.citation,
            "homepage": ds_builder.info.homepage,
            "license": ds_builder.info.license,
            "version": str(ds_builder.info.version),
        }

    await db.create_huggingface_dataset(dataset_id, ds_builder.info.description, dataset_size, json_data)
    log(f"Dataset created in database for dataset_id: {dataset_id}")

    # Download the dataset
    # Later on we can move this to a job
    async def load_dataset_thread(dataset_id, config_name=None):
        logFile = open(GLOBAL_LOG_PATH, "a")
        flushLogFile = FlushFile(logFile)
        with contextlib.redirect_stdout(flushLogFile), contextlib.redirect_stderr(flushLogFile):
            try:
                if config_name is not None:
                    dataset = load_dataset(path=dataset_id, name=config_name, trust_remote_code=True)
                else:
                    dataset = load_dataset(dataset_id, trust_remote_code=True)
                print(f"Dataset downloaded for dataset_id: {dataset_id}")
                return dataset

            except ValueError as e:
                error_msg = f"{type(e).__name__}: {e}"
                print(error_msg)
                raise ValueError(e)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                print(error_msg)
                raise

    try:
        dataset = await load_dataset_thread(dataset_id, config_name)

    except ValueError as e:
        log(f"Exception occurred while downloading dataset: {type(e).__name__}: {e}")
        if "Config name is missing" in str(e):
            return {"status": "error", "message": "Please enter the folder_name of the dataset from huggingface"}
        else:
            return {"status": "error", "message": "An internal error has occurred!"}

    except Exception as e:
        log(f"Exception occurred while downloading dataset: {type(e).__name__}: {e}")
        return {"status": "error", "message": "An internal error has occurred!"}

    return {"status": "success"}


@router.get("/list", summary="List available datasets.")
async def dataset_list(generated: bool = True):
    dataset_list = await db.get_datasets()
    if generated:
        return dataset_list

    final_list = []
    for entry in dataset_list:
        json_data = json.loads(entry.get("json_data", {}))
        if not generated and not json_data.get("generated", False):
            final_list.append(entry)

    return final_list


@router.get("/generated_datasets_list", summary="List available generated datasets.")
async def generated_datasets_list():
    list = await db.get_generated_datasets()
    return list


@router.get("/new", summary="Create a new dataset.")
async def dataset_new(dataset_id: str, generated: bool = False):
    dataset_id = slugify(dataset_id)

    # Check to make sure we don't have a dataset with this name
    row = await db.get_dataset(dataset_id)
    if generated:
        json_data = {"generated": True}
    else:
        json_data = None
    if row is not None:
        return {"status": "error", "message": f"A dataset with the name {dataset_id} already exists"}
    if json_data is None:
        # Create a new dataset in the database
        await db.create_local_dataset(dataset_id)
    else:
        await db.create_local_dataset(dataset_id, json_data=json_data)

    # Now make a directory that maps to the above dataset_id
    # Check if the directory already exists
    if not os.path.exists(dirs.dataset_dir_by_id(dataset_id)):
        os.makedirs(dirs.dataset_dir_by_id(dataset_id))
    return {"status": "success", "dataset_id": dataset_id}


@router.get("/delete", summary="Delete a dataset.")
async def dataset_delete(dataset_id: str):
    await db.delete_dataset(dataset_id)

    dataset_id = secure_filename(dataset_id)

    # delete directory and contents. ignore_errors because we don't care if the directory doesn't exist
    shutil.rmtree(dirs.dataset_dir_by_id(dataset_id), ignore_errors=True)

    return {"status": "success"}


@router.post("/fileupload", summary="Upload the contents of a dataset.")
async def create_upload_file(dataset_id: str, files: list[UploadFile]):
    for file in files:
        print("uploading filename is: " + str(file.filename))

        # # ensure filename is in the format <something>_train.jsonl or <something>_eval.jsonl
        # if not re.match(r"^.+_(train|eval).jsonl$", str(file.filename)):
        #     raise HTTPException(
        #         status_code=403, detail=f"The filenames must be named EXACTLY: {dataset_id}_train.jsonl and {dataset_id}_eval.jsonl")

        # ensure the filename is exactly {dataset_id}_train.jsonl or {dataset_id}_eval.jsonl

        # if not re.match(rf"^{dataset_id}_(train|eval).jsonl$", str(file.filename)):
        #     raise HTTPException(
        #         status_code=403, detail=f"The filenames must be named EXACTLY: {dataset_id}_train.jsonl and {dataset_id}_eval.jsonl")

        dataset_id = slugify(dataset_id)

        try:
            content = await file.read()
            target_path = os.path.join(dirs.dataset_dir_by_id(dataset_id), str(file.filename))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)  # 🔥 Create parent dirs
            async with aiofiles.open(target_path, "wb") as out_file:
                await out_file.write(content)
        except Exception:
            raise HTTPException(status_code=403, detail="There was a problem uploading the file")

    return {"status": "success"}


@router.post("/duplicate_dataset", summary="Duplicate an existing dataset.")
async def duplicate_dataset(dataset_id: str, new_dataset_id: str):
    dataset_id_slug = slugify(dataset_id)
    new_dataset_id_slug = slugify(new_dataset_id)

    # Check if the original dataset exists
    d = await db.get_dataset(dataset_id_slug)
    if d is None:
        print("DB NOT FOUND")
        return {"status": "error", "message": f"Dataset {dataset_id} not found."}

    # Check if the target dataset name already exists
    existing = await db.get_dataset(new_dataset_id_slug)
    if existing is not None:
        print("DB EXISTS")
        return {"status": "error", "message": f"Dataset {new_dataset_id} already exists."}

    # Create a new dataset entry
    await db.create_local_dataset(new_dataset_id_slug)

    # Create the directory for the new dataset
    src_dir = dirs.dataset_dir_by_id(dataset_id_slug)
    dest_dir = dirs.dataset_dir_by_id(new_dataset_id_slug)

    try:
        if not os.path.exists(src_dir):
            print("Source dataset directory not found.")
            return {"status": "error", "message": "Source dataset directory not found."}

        if os.path.exists(dest_dir):
            print("Target dataset directory already exists.")
            return {"status": "error", "message": "Target dataset directory already exists."}

        # Copy the directory contents
        shutil.copytree(src_dir, dest_dir)
        print(f"Dataset duplicated from {dataset_id} to {new_dataset_id}.")
        return {"status": "success", "message": f"Dataset duplicated from {dataset_id} to {new_dataset_id}."}

    except Exception as e:
        print(f"Exception occurred: {type(e).__name__}: {e}")
        log(f"Exception occurred: {type(e).__name__}: {e}")
        return {"status": "error", "message": "An error occurred"}


class FlushFile:
    def __init__(self, file):
        self.file = file

    def write(self, data):
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
