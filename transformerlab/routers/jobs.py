import asyncio
from fnmatch import fnmatch
import json
import os
import csv
import pandas as pd
import logging
from fastapi import APIRouter, Body, Response
from fastapi.responses import StreamingResponse, FileResponse

from transformerlab.shared import shared
from transformerlab.shared import dirs
from typing import Annotated
from json import JSONDecodeError

from werkzeug.utils import secure_filename

from transformerlab.routers.serverinfo import watch_file

from transformerlab.db.db import get_training_template
from transformerlab.db.db import experiment_get
from datetime import datetime

import transformerlab.db.jobs as db_jobs

router = APIRouter(prefix="/jobs", tags=["train"])


@router.get("/list")
async def jobs_get_all(type: str = "", status: str = ""):
    jobs = await db_jobs.jobs_get_all(type=type, status=status)
    return jobs


@router.get("/delete/{job_id}")
async def job_delete(job_id: str):
    await db_jobs.job_delete(job_id)
    return {"message": "OK"}


@router.get("/create")
async def job_create(type: str = "UNDEFINED", status: str = "CREATED", data: str = "{}", experiment_id: int = -1):
    jobid = await db_jobs.job_create(type=type, status=status, job_data=data, experiment_id=experiment_id)
    return jobid


async def job_create_task(script: str, job_data: str = "{}", experiment_id: int = -1):
    jobid = await db_jobs.job_create(type="UNDEFINED", status="CREATED", job_data=job_data, experiment_id=experiment_id)
    return jobid


@router.get("/update/{job_id}")
async def job_update(job_id: str, status: str):
    await db_jobs.job_update_status(job_id, status)
    return {"message": "OK"}


@router.get("/start_next")
async def start_next_job():
    num_running_jobs = await db_jobs.job_count_running()
    if num_running_jobs > 0:
        return {"message": "A job is already running"}
    nextjob = await db_jobs.jobs_get_next_queued_job()
    if nextjob:
        print(f"Starting Next Job in Queue: {nextjob}")
        print("Starting job: " + str(nextjob["id"]))
        nextjob_data = nextjob["job_data"]
        if not isinstance(nextjob_data, dict):
            job_config = json.loads(nextjob["job_data"])
        else:
            job_config = nextjob_data
        experiment_id = nextjob["experiment_id"]
        data = await experiment_get(experiment_id)
        if data is None:
            # mark the job as failed
            await db_jobs.job_update_status(nextjob["id"], "FAILED")
            return {"message": f"Experiment {id} does not exist"}
        # config = json.loads(data["config"])

        experiment_name = data["name"]
        await shared.run_job(
            job_id=nextjob["id"], job_config=job_config, experiment_name=experiment_name, job_details=nextjob
        )
        return nextjob
    else:
        return {"message": "No jobs in queue"}


@router.get("/{job_id}/stop")
async def stop_job(job_id: str):
    # The way a job is stopped is simply by adding "stop: true" to the job_data
    # This will be checked by the plugin as it runs
    await db_jobs.job_stop(job_id)
    return {"message": "OK"}


@router.get("/delete_all")
async def job_delete_all():
    await db_jobs.job_delete_all()
    return {"message": "OK"}


@router.get("/{job_id}")
async def get_training_job(job_id: str):
    return await db_jobs.job_get(job_id)


@router.get("/{job_id}/output")
async def get_training_job_output(job_id: str, sweeps: bool = False):
    # First get the template Id from this job:
    job = await db_jobs.job_get(job_id)
    if job is None:
        return {"checkpoints": []}
    job_data = job["job_data"]

    if not isinstance(job_data, dict):
        try:
            job_data = json.loads(job_data)
        except JSONDecodeError:
            print(f"Error decoding job_data for job {job_id}. Using empty job_data.")
            job_data = {}

    if sweeps:
        output_file = job_data.get("sweep_output_file", None)
        if output_file is not None and os.path.exists(output_file):
            with open(output_file, "r") as f:
                output = f.read()
            return output

    if "template_id" not in job_data:
        return {"error": "true"}

    template_id = job_data["template_id"]
    # Then get the template:
    template = await get_training_template(template_id)
    # Then get the plugin name from the template:
    if not isinstance(template["config"], dict):
        template_config = json.loads(template["config"])
    else:
        template_config = template["config"]
    if "plugin_name" not in template_config:
        return {"error": "true"}

    plugin_name = template_config["plugin_name"]

    # Now we need the current experiment id from the job:
    # experiment_id = job["experiment_id"]
    # Then get the experiment name:
    # experiment = await db_jobs.experiment_get(experiment_id)
    # experiment_name = experiment["name"]

    # Now we can get the output.txt from the plugin which is stored in
    # /workspace/experiments/{experiment_name}/plugins/{plugin_name}/output.txt
    output_file = f"{dirs.plugin_dir_by_name(plugin_name)}/output.txt"
    with open(output_file, "r") as f:
        output = f.read()
    return output


# Templates


@router.get("/template/{template_id}")
async def get_train_template(template_id: str):
    return await get_training_template(template_id)


@router.put("/template/update")
async def update_training_template(
    template_id: str, name: str, description: str, type: str, config: Annotated[str, Body(embed=True)]
):
    try:
        configObject = json.loads(config)
        datasets = configObject["dataset_name"]
        await db_jobs.update_training_template(template_id, name, description, type, datasets, config)
    except JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return {"status": "error", "message": "An error occurred while processing the request."}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"status": "error", "message": "An internal error has occurred."}
    return {"status": "success"}


async def get_output_file_name(job_id: str):
    """
    Get the output file name for a job with comprehensive fallback logic.
    Adapted from train router for better robustness.
    """
    try:
        # First get the job data
        job = await db_jobs.job_get(job_id)
        job_data = job["job_data"]

        # Handle both dict and JSON string formats
        if not isinstance(job_data, dict):
            try:
                job_data = json.loads(job_data)
            except JSONDecodeError:
                logging.error(f"Error decoding job_data for job {job_id}. Using empty job_data.")
                job_data = {}

        # Check if job_data has an explicit output file path
        if job_data.get("output_file_path") is not None:
            return job_data["output_file_path"]

        # Try to get plugin name from job_data directly (newer format)
        plugin_name = job_data.get("plugin")

        # If not found, try to get it from template (legacy format)
        if not plugin_name and "template_id" in job_data:
            template_id = job_data["template_id"]
            template = await get_training_template(template_id)
            if template:
                template_config = template.get("config", {})
                if not isinstance(template_config, dict):
                    try:
                        template_config = json.loads(template_config)
                    except JSONDecodeError:
                        template_config = {}
                plugin_name = template_config.get("plugin_name")

        # If still no plugin name and we have config in job_data
        if not plugin_name and "config" in job_data:
            template_config = job_data["config"]
            if not isinstance(template_config, dict):
                try:
                    template_config = json.loads(template_config)
                except JSONDecodeError:
                    template_config = {}
            plugin_name = template_config.get("plugin_name")

        if not plugin_name:
            raise ValueError(f"Plugin name not found in job data for job {job_id}")

        plugin_dir = dirs.plugin_dir_by_name(plugin_name)
        job_id_safe = secure_filename(str(job_id))

        # Define potential output file locations in order of preference
        jobs_dir_output_file_name = os.path.join(dirs.WORKSPACE_DIR, "jobs", job_id_safe, f"output_{job_id_safe}.txt")
        plugin_job_output_file = os.path.join(plugin_dir, f"output_{job_id_safe}.txt")
        plugin_legacy_output_file = os.path.join(plugin_dir, "output.txt")

        # Check files in order of preference
        if os.path.exists(jobs_dir_output_file_name):
            return jobs_dir_output_file_name
        elif os.path.exists(plugin_job_output_file):
            return plugin_job_output_file
        elif os.path.exists(plugin_legacy_output_file):
            return plugin_legacy_output_file
        else:
            raise ValueError(f"No output file found for job {job_id}")

    except Exception as e:
        raise e


@router.get("/{job_id}/stream_output")
async def stream_job_output(job_id: str, sweeps: bool = False):
    """
    Stream job output with robust error handling and retry logic.
    Enhanced version combining the best of both train and jobs routers.
    """
    try:
        job = await db_jobs.job_get(job_id)
        job_data = job["job_data"]

        # Handle both dict and JSON string formats
        if not isinstance(job_data, dict):
            try:
                job_data = json.loads(job_data)
            except JSONDecodeError:
                logging.error(f"Error decoding job_data for job {job_id}. Using empty job_data.")
                job_data = {}

        job_id_safe = secure_filename(str(job_id))

        # Handle sweeps case first
        if sweeps:
            output_file = job_data.get("sweep_output_file", None)
            if output_file is not None and os.path.exists(output_file):
                output_file_name = output_file
            else:
                # Fall back to regular output file logic
                output_file_name = await get_output_file_name(job_id)
        else:
            # Try to get output file name with fallback logic
            output_file_name = await get_output_file_name(job_id)

    except ValueError as e:
        # If the value error starts with "No output file found for job" then wait 4 seconds and try again
        # because the file might not have been created yet
        if str(e).startswith("No output file found for job"):
            logging.info(f"Output file not found for job {job_id}, retrying in 4 seconds...")
            await asyncio.sleep(4)
            try:
                output_file_name = await get_output_file_name(job_id)
            except Exception as retry_e:
                # If still no file after retry, create an empty one in the jobs directory
                logging.warning(
                    f"Still no output file found for job {job_id} after retry, creating empty file: {retry_e}"
                )
                job_id_safe = secure_filename(str(job_id))
                new_output_dir = os.path.join(dirs.WORKSPACE_DIR, "jobs", job_id_safe)
                if not os.path.exists(new_output_dir):
                    os.makedirs(new_output_dir)
                output_file_name = os.path.join(new_output_dir, f"output_{job_id_safe}.txt")
                with open(output_file_name, "w") as f:
                    f.write("")
        else:
            logging.error(f"ValueError in stream_job_output: {e}")
            return StreamingResponse(
                iter(["data: Error: An internal error has occurred!\n\n"]),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
            )
    except Exception as e:
        # Handle general error
        logging.error(f"Error in stream_job_output: {e}")
        return StreamingResponse(
            iter(["data: Error: An internal error has occurred!\n\n"]),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
        )

    return StreamingResponse(
        # we force polling because i can't get this to work otherwise -- changes aren't detected
        watch_file(output_file_name, start_from_beginning=True, force_polling=True),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )


@router.get("/{job_id}/stream_detailed_json_report")
async def stream_detailed_json_report(job_id: str, file_name: str):
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        return "File not found", 404

    return StreamingResponse(
        # we force polling because i can't get this to work otherwise -- changes aren't detected
        watch_file(file_name, start_from_beginning=True, force_polling=False),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )


@router.get("/{job_id}/get_additional_details")
async def stream_job_additional_details(job_id: str, task: str = "view"):
    job = await db_jobs.job_get(job_id)
    job_data = job["job_data"]
    file_path = job_data["additional_output_path"]
    if file_path.endswith(".csv"):
        file_format = "text/csv"
        filename = f"report_{job_id}.csv"
    elif file_path.endswith(".json"):
        file_format = "application/json"
        filename = f"report_{job_id}.json"
    if task == "download":
        return FileResponse(file_path, filename=filename, media_type=file_format)

    if not os.path.exists(file_path):
        return Response("No additional details found for this evaluation", media_type="text/csv")

    # convert csv to JSON, but do not assume that \n marks the end of a row as cells can
    # contain fields that start and end with " and contain \n. Use a CSV parser instead.
    with open(file_path, "r") as csvfile:
        contents = csv.reader(csvfile, delimiter=",", quotechar='"')
        # convert the csv to a JSON object
        csv_content = {"header": [], "body": []}
        for i, row in enumerate(contents):
            if i == 0:
                csv_content["header"] = row
            else:
                csv_content["body"].append(row)
    return csv_content


@router.get("/{job_id}/get_figure_json")
async def get_figure_path(job_id: str):
    job = await db_jobs.job_get(job_id)
    job_data = job["job_data"]
    file_path = job_data.get("plot_data_path", None)

    if file_path is None or not os.path.exists(file_path):
        return Response("No plot data found for this evaluation", media_type="text/csv")

    content = json.loads(open(file_path, "r").read())
    return content


@router.get("/{job_id}/get_generated_dataset")
async def get_generated_dataset(job_id: str):
    job = await db_jobs.job_get(job_id)
    # Get experiment name
    job_data = job["job_data"]

    # Check if the job has additional output path
    if "additional_output_path" in job_data.keys() and job_data["additional_output_path"]:
        json_file_path = job_data["additional_output_path"]
    else:
        return Response("No dataset found for this evaluation", media_type="text/csv")

    if not os.path.exists(json_file_path):
        return Response("No dataset found for this evaluation", media_type="text/csv")
    else:
        json_content = json.loads(open(json_file_path, "r").read())

        df = pd.DataFrame(json_content)

        content = {"header": df.columns.tolist(), "body": df.values.tolist()}

        return content


@router.get("/{job_id}/get_eval_images")
async def get_eval_images(job_id: str):
    """Get list of evaluation images for a job"""
    job = await db_jobs.job_get(job_id)
    job_data = job["job_data"]

    # Check if the job has eval_images_dir
    if "eval_images_dir" not in job_data or not job_data["eval_images_dir"]:
        return {"images": []}

    images_dir = job_data["eval_images_dir"]

    if not os.path.exists(images_dir):
        return {"images": []}

    # Supported image extensions
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}

    images = []
    try:
        for filename in os.listdir(images_dir):
            file_path = os.path.join(images_dir, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in image_extensions:
                    # Get file stats for additional metadata
                    stat = os.stat(file_path)
                    images.append(
                        {
                            "filename": filename,
                            "path": f"/jobs/{job_id}/image/{filename}",  # API endpoint path
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                        }
                    )
    except OSError as e:
        logging.error(f"Error reading images directory {images_dir}: {e}")
        return {"images": []}

    # Sort by filename for consistent ordering
    images.sort(key=lambda x: x["filename"])

    return {"images": images}


@router.get("/{job_id}/image/{filename}")
async def get_eval_image(job_id: str, filename: str):
    """Serve individual evaluation image files"""
    job = await db_jobs.job_get(job_id)
    job_data = job["job_data"]

    # Check if the job has eval_images_dir
    if "eval_images_dir" not in job_data or not job_data["eval_images_dir"]:
        return Response("No images directory found for this job", status_code=404)

    images_dir = job_data["eval_images_dir"]

    if not os.path.exists(images_dir):
        return Response("Images directory not found", status_code=404)

    # Secure the filename to prevent directory traversal
    filename = secure_filename(filename)
    file_path = os.path.join(images_dir, filename)

    # Ensure the file exists and is within the images directory
    if not os.path.exists(file_path) or not os.path.commonpath([images_dir, file_path]) == images_dir:
        return Response("Image not found", status_code=404)

    # Determine media type based on file extension
    _, ext = os.path.splitext(filename.lower())
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }

    media_type = media_type_map.get(ext, "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
    )


@router.get("/{job_id}/checkpoints")
async def get_checkpoints(job_id: str):
    if job_id is None or job_id == "" or job_id == -1:
        return {"checkpoints": []}

    """Get list of checkpoints for a job"""
    job = await db_jobs.job_get(job_id)
    job_data = job["job_data"]

    # Check if the job has a supports_checkpoints flag
    # if "supports_checkpoints" not in job_data or not job_data["supports_checkpoints"]:
    #     return {"checkpoints": []}

    # By default we assume the training type is an adaptor training
    # and the checkpoints are stored alongside the adaptors
    # this maps to how mlx lora works, which will be the first use case
    # but we will have to abstract this further in the future
    config = job_data.get("config", {})
    if not isinstance(config, dict):
        try:
            config = json.loads(config)
        except Exception:
            config = {}
    model_name = config.get("model_name", "")
    adaptor_name = config.get("adaptor_name", "adaptor")
    default_adaptor_dir = os.path.join(dirs.WORKSPACE_DIR, "adaptors", secure_filename(model_name), adaptor_name)

    # print(f"Default adaptor directory: {default_adaptor_dir}")

    checkpoints_dir = job_data.get("checkpoints_dir", default_adaptor_dir)
    if not checkpoints_dir or not os.path.exists(checkpoints_dir):
        # print(f"Checkpoints directory does not exist: {checkpoints_dir}")
        return {"checkpoints": []}

    checkpoints_file_filter = job_data.get("checkpoints_file_filter", "*_adapters.safetensors")
    if not checkpoints_file_filter:
        checkpoints_file_filter = "*_adapters.safetensors"

    # print(f"Checkpoints directory: {checkpoints_dir}")
    # print(f"Checkpoints file filter: {checkpoints_file_filter}")

    checkpoints = []
    try:
        for filename in os.listdir(checkpoints_dir):
            if fnmatch(filename, checkpoints_file_filter):
                file_path = os.path.join(checkpoints_dir, filename)
                try:
                    stat = os.stat(file_path)
                    modified_time = stat.st_mtime
                    filesize = stat.st_size
                    # Format the timestamp as ISO 8601 string
                    formatted_time = datetime.fromtimestamp(modified_time).isoformat()
                except Exception as e:
                    logging.error(f"Error getting stat for file {file_path}: {e}")
                    formatted_time = None
                    filesize = None
                checkpoints.append({"filename": filename, "date": formatted_time, "size": filesize})
    except OSError as e:
        logging.error(f"Error reading checkpoints directory {checkpoints_dir}: {e}")

    # Sort checkpoints by filename in reverse (descending) order for consistent ordering
    checkpoints.sort(key=lambda x: x["filename"], reverse=True)
    # print(f"Sorted checkpoints: {checkpoints}")

    return {
        "checkpoints": checkpoints,
        "model_name": model_name,
        "adaptor_name": adaptor_name,
    }
