import contextlib
import os
import shutil
import json
import aiofiles
from PIL import Image as PILImage
from datasets import load_dataset, load_dataset_builder
from fastapi import APIRouter, HTTPException, UploadFile, Query
import csv
from pydantic import BaseModel
from typing import Dict, Any
from io import BytesIO
import base64
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
        return {}
    r = {}
    # This means it is a custom dataset the user uploaded
    if d["location"] == "local":
        try:
            dataset = load_dataset(path=dirs.dataset_dir_by_id(dataset_id))
        except EmptyDatasetError:
            return {"status": "error", "message": "The dataset is empty."}
        split = list(dataset.keys())[0]
        r["features"] = dataset[split].features

        # Add image column check
        is_image = any(
            f._type.lower() == "image" or str(f).lower() == "image" for f in dataset[split].features.values()
        )
        r["is_image"] = is_image

    else:
        dataset_config = d.get("json_data", {}).get("dataset_config", None)
        config_name = d.get("json_data", {}).get("config_name", None)
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
        logging.error(f"Exception occurred: {type(e).__name__}: {e}")
        return {"status": "error", "message": "An internal error has occurred."}

    if split is None or split == "":
        splits = list(dataset.keys())
        if len(splits) == 0:
            return {"status": "error", "message": "No splits available in the dataset."}
        split = splits[0]

    if streaming:
        dataset_len = -1
        dataset = dataset[split].skip(offset)
        rows = list(dataset.take(limit))
        # Serialize rows
        result["rows"] = [serialize_row(row) for row in rows]
        result["splits"] = None
    else:
        if d["location"] != "local" and split not in dataset.keys():
            return {"status": "error", "message": f"Split '{split}' does not exist in the dataset."}
        dataset_len = len(dataset[split])
        columns = dataset[split][offset : min(offset + limit, dataset_len)]
        # Serialize each value in the columns dict, preserving the columnar format
        if isinstance(columns, dict):
            result["columns"] = {k: [serialize_row(v) for v in vals] for k, vals in columns.items()}
        else:
            result["columns"] = columns
        result["splits"] = list(dataset.keys())

    result["len"] = dataset_len
    return {"status": "success", "data": result}


def serialize_row(row):
    """Convert PIL Images in a row to base64 strings, preserving original structure."""
    if isinstance(row, dict):
        return {k: serialize_row(v) for k, v in row.items()}
    elif isinstance(row, list):
        return [serialize_row(v) for v in row]
    elif isinstance(row, PILImage.Image):
        buffered = BytesIO()
        row.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    else:
        return row


@router.get(
    "/preview_with_template",
    summary="Preview the contents of a dataset after applying a jinja template to it.",
    responses={
        200: {
            "model": SuccessResponse,
            "description": "Successful response. Data is a list of column names followed by data, which can be of any datatype.",
        },
        400: {"model": ErrorResponse},
    },
)
async def dataset_preview_with_template(
    dataset_id: str = Query(
        description="The ID of the dataset to preview. This can be a HuggingFace dataset ID or a local dataset ID."
    ),
    template: str = "",
    offset: int = Query(0, description="The starting index from where to fetch the data.", ge=0),
    limit: int = Query(10, description="The maximum number of data items to fetch.", ge=1, le=1000),
) -> Any:
    d = await db.get_dataset(dataset_id)
    dataset_len = 0
    result = {}
    # This means it is a custom dataset the user uploaded
    if d["location"] == "local":
        try:
            dataset = load_dataset(path=dirs.dataset_dir_by_id(dataset_id))
        except Exception as e:
            logging.error(f"Error loading dataset: {type(e).__name__}: {e}")
            return {"status": "error", "message": "An internal error has occurred."}
        dataset_len = len(dataset["train"])
        result["columns"] = dataset["train"][offset : min(offset + limit, dataset_len)]
    else:
        dataset_config = d.get("json_data", {}).get("dataset_config", None)
        config_name = d.get("json_data", {}).get("config_name", None)
        if dataset_config is not None:
            dataset = load_dataset(dataset_id, dataset_config, trust_remote_code=True)
        elif config_name is not None:
            dataset = load_dataset(path=dataset_id, name=config_name, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_id, trust_remote_code=True)
        dataset_len = len(dataset["train"])
        result["columns"] = dataset["train"][offset : min(offset + limit, dataset_len)]
    result["len"] = dataset_len

    jinja_template = sandboxed_jinja2_evironment.from_string(template)

    column_names = list(result["columns"].keys())

    rows = []
    # now iterate over all columns and rows, do not use offset or len because we've already
    # sliced the dataset
    for i in range(0, len(result["columns"][column_names[0]])):
        row = {}
        row["__index__"] = i + offset
        for key in result["columns"].keys():
            row[key] = serialize_row(result["columns"][key][i])

        # Apply the template to a new key in row called __formatted__
        row["__formatted__"] = jinja_template.render(row)
        # row['__template__'] = template
        rows.append(row)

    return {
        "status": "success",
        "data": {"columns": column_names, "rows": rows, "len": dataset_len, "offset": offset, "limit": limit},
    }


@router.get(
    "/edit_with_template",
    summary="Preview and edit dataset with template, loading from metadata files and local images.",
)
async def dataset_edit_with_template(
    dataset_id: str = Query(..., description="Dataset ID"),
    template: str = Query("", description="Optional Jinja template"),
    offset: int = Query(0, ge=0, description="Starting index"),
    limit: int = Query(10, ge=1, le=1000, description="Max items to fetch"),
):
    dataset_dir = dirs.dataset_dir_by_id(slugify(dataset_id))
    if not os.path.exists(dataset_dir):
        return {"status": "error", "message": "Dataset directory not found"}

    rows = []
    index = 0

    for root, _, files in os.walk(dataset_dir, followlinks=False):
        for file in files:
            if file.lower().endswith((".json", ".jsonl", ".csv")):
                metadata_path = Path(root) / file
                try:
                    if file.endswith(".jsonl"):
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            data = [json.loads(line) for line in f]
                    elif file.endswith(".json"):
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                data = [data]
                    elif file.endswith(".csv"):
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            data = [row for row in reader]
                    else:
                        continue
                except Exception as e:
                    logging.error(f"Failed to read metadata from {metadata_path}: {e}")
                    return {"status": "error", "message": "Failed to read metadata file!"}

                for entry in data:
                    split = entry.get("split")
                    if not split:
                        path_parts = Path(root).parts
                        split = next(
                            (part for part in reversed(path_parts) if part.lower() in ("train", "test")), "train"
                        )

                    image_rel_path = entry.get("file_name")
                    if not image_rel_path:
                        continue

                    image_path = (Path(root) / image_rel_path).resolve()
                    if not str(image_path).startswith(str(dataset_dir)):
                        continue

                    if not image_path.exists():
                        log(f"Image not found: {image_path}")
                        continue

                    try:
                        with PILImage.open(image_path) as img:
                            buffer = BytesIO()
                            img.save(buffer, format="JPEG")
                            encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                            image_data_url = f"data:image/jpeg;base64,{encoded_img}"
                    except Exception as e:
                        logging.error(f"Failed to process image {image_path}: {e}")
                        return {"status": "error", "message": "Failed to process images!"}

                    row = {
                        "__index__": index,
                        "file_name": str(image_rel_path),
                        "image": image_data_url,
                        "text": entry.get("text") or entry.get("caption") or entry.get("description", ""),
                        "label": entry.get("label", ""),
                        "split": split,
                    }
                    if template:
                        try:
                            jinja_template = sandboxed_jinja2_evironment.from_string(template)
                            row["__formatted__"] = jinja_template.render(row)
                        except Exception as e:
                            row["__formatted__"] = f"Template Error: {e}"

                    rows.append(row)
                    index += 1

                    if len(rows) >= offset + limit:
                        break
        if len(rows) >= offset + limit:
            break

    paginated_rows = rows[offset : offset + limit]
    column_names = list(paginated_rows[0].keys()) if paginated_rows else []

    return {
        "status": "success",
        "data": {
            "columns": column_names,
            "rows": paginated_rows,
            "len": len(rows),
            "offset": offset,
            "limit": limit,
        },
    }


@router.post(
    "/save_metadata",
    summary="Save edited metadata and create a new dataset with reorganized files and updated metadata.",
)
async def save_metadata(dataset_id: str, new_dataset_id: str, file: UploadFile):
    old_dataset_dir = dirs.dataset_dir_by_id(slugify(dataset_id))
    if not os.path.exists(old_dataset_dir):
        return {"status": "error", "message": "Source dataset not found"}

    new_dataset_id = slugify(new_dataset_id)
    new_dataset_dir = dirs.dataset_dir_by_id(new_dataset_id)

    if os.path.exists(new_dataset_dir):
        return {"status": "error", "message": "New dataset already exists"}

    os.makedirs(new_dataset_dir, exist_ok=True)

    # Read updates
    updates_raw = await file.read()
    try:
        updates = json.loads(updates_raw.decode("utf-8"))
    except Exception as e:
        logging.error(f"Invalid JSON file: {e}")
        return {"status": "error", "message": "Invalid JSON file!"}

    # Scan source metadata
    source_map = {}
    for root, _, files in os.walk(old_dataset_dir, followlinks=False):
        for f in files:
            if f.lower().endswith((".json", ".jsonl", ".csv")):
                metadata_path = Path(root) / f
                try:
                    if f.endswith(".jsonl"):
                        with open(metadata_path, "r", encoding="utf-8") as meta_file:
                            data = [json.loads(line) for line in meta_file]
                    elif f.endswith(".json"):
                        with open(metadata_path, "r", encoding="utf-8") as meta_file:
                            data = json.load(meta_file)
                            if isinstance(data, dict):
                                data = [data]
                    elif f.endswith(".csv"):
                        with open(metadata_path, "r", encoding="utf-8") as meta_file:
                            reader = csv.DictReader(meta_file)
                            data = [row for row in reader]
                    else:
                        continue

                    for entry in data:
                        file_name = entry.get("file_name")
                        if not file_name:
                            continue
                        split = entry.get("split")
                        if not split:
                            path_parts = Path(root).parts
                            split = next((p for p in reversed(path_parts) if p.lower() in ("train", "test")), "train")
                        label = entry.get("label", "")
                        key = file_name
                        source_map[key] = {
                            "file_name": file_name,
                            "split": split,
                            "label": label,
                            "metadata_root": root,
                        }
                except Exception as e:
                    logging.error(f"Error reading metadata {metadata_path}: {e}")
                    return {"status": "error", "message": "Failed to read metadata!"}

    metadata_accumulator = {}

    for row in updates:
        file_name = row.get("file_name")
        final_split = row.get("split") if row.get("split") != "" else row.get("previous_split")
        final_label = row.get("label") if row.get("label") != "" else row.get("previous_label")
        final_caption = row.get("caption") if row.get("caption") != "" else row.get("previous_caption")
        if final_split not in ["train", "test"]:
            final_split = "train"
        source_info = source_map.get(file_name)
        if not source_info:
            log(f"Warning: Source info not found for {file_name}, skipping")
            continue

        source_path = Path(source_info["metadata_root"]) / file_name

        if not source_path.exists():
            log(f"Warning: Source image file not found {source_path}, skipping")
            continue

        dest_folder = Path(new_dataset_dir) / final_split / final_label
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = dest_folder / Path(file_name).name
        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            logging.error(f"Failed to copy {source_path} to {dest_path}: {e}")
            return {"status": "error", "message": "Failed to copy from source to destination"}

        key = (final_split, final_label)
        metadata_entry = {
            "file_name": Path(file_name).name,
            "caption": final_caption,
            "label": final_label,
            "split": final_split,
        }
        metadata_accumulator.setdefault(key, []).append(metadata_entry)

    for (split, label), entries in metadata_accumulator.items():
        folder = Path(new_dataset_dir) / split / label
        metadata_file = folder / "metadata.jsonl"
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logging.error(f"Failed to write metadata file {metadata_file}: {e}")
            return {"status": "error", "message": "Failed to write metadata file!"}

    result = await dataset_new(dataset_id=new_dataset_id, generated=False)
    if result.get("status") != "success":
        return {"status": "error", "message": "Failed to register new dataset"}

    return {
        "status": "success",
        "message": f"Dataset '{new_dataset_id}' created with updated metadata and files",
        "dataset_id": new_dataset_id,
    }


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
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            async with aiofiles.open(target_path, "wb") as out_file:
                await out_file.write(content)
        except Exception:
            raise HTTPException(status_code=403, detail="There was a problem uploading the file")

    return {"status": "success"}


class FlushFile:
    def __init__(self, file):
        self.file = file

    def write(self, data):
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
