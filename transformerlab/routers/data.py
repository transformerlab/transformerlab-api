import contextlib
import os
import shutil
import json
import aiofiles
import hashlib
from datasets import load_dataset, load_dataset_builder, Image
from fastapi import APIRouter, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
from io import BytesIO
from PIL import Image as PILImage
import base64
import csv
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
    r = {}

    if d["location"] == "local":
        dataset_dir = dirs.dataset_dir_by_id(slugify(dataset_id))

        try:
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

                r["features"] = features
                r["splits"] = splits

                label_set = set()
                for root, dirs_, files in os.walk(dataset_dir):
                    for file in files:
                        if str(file).lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            image_path = os.path.join(root, file)
                            parent_folder = Path(image_path).parent.name
                            if parent_folder.lower() not in ["train", "test"]:
                                label_set.add(parent_folder)
                available_labels = sorted(label_set)
                r["labels"] = available_labels
            else:
                r["features"] = features
        except EmptyDatasetError:
            return {"status": "error", "message": "The dataset is empty."}
        except Exception:
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
        except Exception:
            return {"status": "error"}

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
    dataset_dir = dirs.dataset_dir_by_id(dataset_id)
    dataset_len = 0

    def compute_base64_and_hash(image_path, dataset_root):
        # Resolve absolute paths
        resolved_image_path = Path(image_path).resolve()
        resolved_dataset_root = Path(dataset_root).resolve()

        # Check if resolved_image_path is within dataset_root
        if not str(resolved_image_path).startswith(str(resolved_dataset_root)):
            raise ValueError(f"Access denied for path: {resolved_image_path}")

        with open(resolved_image_path, "rb") as f:
            img = PILImage.open(f)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            hash_val = hashlib.sha256(encoded.encode()).hexdigest()
        return encoded, hash_val

    try:
        if d["location"] == "local":
            dataset = load_dataset(path=dirs.dataset_dir_by_id(dataset_id))

            split_name = list(dataset.keys())[0]
            features = dataset[split_name].features
            is_image_dataset = any(isinstance(f, Image) for f in features.values())

            metadata_map = {}
            if is_image_dataset:
                dataset = load_dataset("imagefolder", data_dir=dataset_dir)
                dataset_root = Path(dataset_dir).resolve()

                for root, dirs_, files in os.walk(dataset_dir, followlinks=False):
                    resolved_root = Path(root).resolve()

                    # Check if resolved root is within dataset_dir
                    if not str(resolved_root).startswith(str(dataset_root)):
                        print(f"Skipping potentially unsafe directory: {resolved_root}")
                        continue

                    for file in files:
                        if str(file).endswith((".json", ".jsonl", ".csv")):
                            file_path = resolved_root / file
                            path = file_path.resolve()

                            # Check if file is within dataset_dir
                            if not str(path).startswith(str(dataset_root)):
                                print(f"Skipping potentially unsafe file: {path}")
                                continue
                            if path.is_symlink():
                                print(f"Skipping symlink file: {path}")
                                continue
                            with open(path, "r", encoding="utf-8") as f:
                                if str(path).endswith(".json"):
                                    rows = [list(row.values()) for row in json.load(f)]
                                elif str(path).endswith(".jsonl"):
                                    rows = [list(json.loads(line).values()) for line in f]
                                elif str(path).endswith(".csv"):
                                    reader = csv.reader(f)
                                    rows = list(reader)
                                else:
                                    continue
                                for row in rows:
                                    # Inside metadata processing
                                    if len(row) < 2:
                                        continue
                                    file_name = row[0]
                                    caption = row[1]

                                    # Construct the full image path securely
                                    file_path = Path(root) / file_name
                                    resolved_file_path = file_path.resolve()

                                    # Ensure the resolved path is inside dataset_root
                                    if not str(resolved_file_path).startswith(str(dataset_root)):
                                        print(f"Skipping potentially unsafe image path: {resolved_file_path}")
                                        continue

                                    if resolved_file_path.exists():
                                        encoded, img_hash = compute_base64_and_hash(resolved_file_path, dataset_root)
                                        metadata_map[(img_hash, caption)] = {
                                            "file_name": file_name,
                                            "previous_caption": caption,
                                            "full_image_path": str(resolved_file_path),
                                        }
            else:
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

    except Exception:
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

            if d["location"] == "local" and is_image_dataset:
                image = row["image"]
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                img_hash = hashlib.sha256(encoded.encode()).hexdigest()
                # Dynamically get the caption field name
                caption_col = next(
                    (col for col in dataset[split_name].features.keys() if col != "image" and col != "label"), None
                )
                row_caption = row.get(caption_col) if caption_col else ""
                meta = metadata_map.get((img_hash, row_caption))

                if meta:
                    row["file_name"] = meta["file_name"]
                    row["previous_caption"] = meta["previous_caption"]
                    parent_folder = Path(meta["full_image_path"]).parent.name
                    if "label" not in row or row["label"] is None:
                        if parent_folder.lower() not in ["train", "test"]:
                            row["label"] = parent_folder
                        else:
                            row["label"] = ""
                else:
                    row["file_name"] = None
                    row["previous_caption"] = ""
                    row["label"] = ""

                row["image"] = f"data:image/jpeg;base64,{encoded}"

            row["__formatted__"] = jinja_template.render(row)
            rows.append(row)

    column_names = list(rows[0].keys()) if rows else []

    return {
        "status": "success",
        "data": {"columns": column_names, "rows": rows, "len": dataset_len, "offset": offset, "limit": limit},
    }


@router.post("/save_metadata", summary="Update caption fields by file_name.")
async def save_metadata(dataset_id: str, file: UploadFile):
    try:
        dataset_dir = dirs.dataset_dir_by_id(slugify(dataset_id))

        content = await file.read()
        edits = json.loads(content)

        metadata_files = []
        for root, dirs_, files in os.walk(dataset_dir):
            for f in files:
                if str(f).endswith((".json", ".jsonl", ".csv")):
                    path = os.path.join(root, f)
                    metadata_files.append(path)

        if not metadata_files:
            return JSONResponse(status_code=404, content={"status": "error", "message": "No metadata files found."})

        edits_applied = 0

        for edit in edits:
            edit_file_name = edit.get("file_name")
            edit_basename = os.path.basename(edit_file_name) if edit_file_name else None
            prev_caption = edit.get("previous_caption")
            new_caption = edit.get("text")

            if not edit_basename or prev_caption is None:
                print("Edit missing required fields. Skipping.")
                continue

            for metadata_path in metadata_files:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    header = []
                    if str(metadata_path).endswith(".csv"):
                        reader = csv.reader(f)
                        all_rows = list(reader)
                        if not all_rows:
                            continue
                        header = all_rows[0]
                        original = all_rows[1:]
                    elif str(metadata_path).endswith(".jsonl"):
                        original = [json.loads(line) for line in f]
                    elif str(metadata_path).endswith(".json"):
                        original = json.load(f)
                    else:
                        continue

                updated = False
                for row in original:
                    if isinstance(row, list):
                        row_file_basename = row[0]
                        row_caption = row[1]
                    elif isinstance(row, dict):
                        keys = list(row.keys())
                        row_file_basename = row[keys[0]]
                        row_caption = row[keys[1]]
                    else:
                        continue

                    if row_file_basename == edit_basename and row_caption == prev_caption:
                        if isinstance(row, list):
                            row[1] = new_caption
                        elif isinstance(row, dict):
                            key_to_update = list(row.keys())[1]
                            row[key_to_update] = new_caption
                        edits_applied += 1
                        updated = True
                        break

                if updated:
                    with open(metadata_path, "w", encoding="utf-8", newline="") as f_out:
                        if str(metadata_path).endswith(".csv"):
                            writer = csv.writer(f_out)
                            writer.writerow(header)
                            writer.writerows(original)
                        elif str(metadata_path).endswith(".jsonl"):
                            for row in original:
                                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        elif str(metadata_path).endswith(".json"):
                            json.dump(original, f_out, ensure_ascii=False, indent=2)
                    break
        return {"status": "success", "message": f"Updated {edits_applied} row(s)."}

    except Exception:
        return JSONResponse(status_code=500, content={"status": "error"})


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


class FlushFile:
    def __init__(self, file):
        self.file = file

    def write(self, data):
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
