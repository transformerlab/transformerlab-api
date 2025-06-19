import json
import os
import csv
import pandas as pd
import logging
from fastapi import APIRouter, Body, Response
from fastapi.responses import StreamingResponse, FileResponse

import transformerlab.db as db
from transformerlab.shared import shared
from transformerlab.shared import dirs
from typing import Annotated
from json import JSONDecodeError

from werkzeug.utils import secure_filename

from transformerlab.routers.serverinfo import watch_file

router = APIRouter(prefix="/jobs", tags=["train"])


@router.get("/list")
async def jobs_get_all(type: str = "", status: str = ""):
    jobs = await db.jobs_get_all(type=type, status=status)
    return jobs


@router.get("/delete/{job_id}")
async def job_delete(job_id: str):
    await db.job_delete(job_id)
    return {"message": "OK"}


@router.get("/create")
async def job_create(type: str = "UNDEFINED", status: str = "CREATED", data: str = "{}", experiment_id: str = "-1"):
    jobid = await db.job_create(type=type, status=status, job_data=data, experiment_id=experiment_id)
    return jobid


async def job_create_task(script: str, job_data: str = "{}", experiment_id: str = "-1"):
    jobid = await db.job_create(type="UNDEFINED", status="CREATED", job_data=job_data, experiment_id=experiment_id)
    return jobid


@router.get("/update/{job_id}")
async def job_update(job_id: str, status: str):
    await db.job_update_status(job_id, status)
    return {"message": "OK"}


@router.get("/start_next")
async def start_next_job():
    num_running_jobs = await db.job_count_running()
    if num_running_jobs > 0:
        return {"message": "A job is already running"}
    nextjob = await db.jobs_get_next_queued_job()
    if nextjob:
        print(f"Starting Next Job in Queue: {nextjob}")
        print("Starting job: " + str(nextjob["id"]))
        job_config = json.loads(nextjob["job_data"])
        experiment_id = nextjob["experiment_id"]

        # Check if this job should be executed remotely
        target_machine_id = job_config.get("target_machine_id")
        if target_machine_id:
            return await _dispatch_remote_job(nextjob, job_config, target_machine_id)

        # Local execution (existing logic)
        data = await db.experiment_get(experiment_id)
        if data is None:
            # mark the job as failed
            await db.job_update_status(nextjob["id"], "FAILED")
            return {"message": f"Experiment {experiment_id} does not exist"}

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
    await db.job_stop(job_id)
    return {"message": "OK"}


@router.get("/delete_all")
async def job_delete_all():
    await db.job_delete_all()
    return {"message": "OK"}


@router.get("/{job_id}")
async def get_training_job(job_id: str):
    return await db.job_get(job_id)


@router.get("/{job_id}/output")
async def get_training_job_output(job_id: str, sweeps: bool = False):
    # First get the template Id from this job:
    job = await db.job_get(job_id)

    job_data = json.loads(job["job_data"])

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
    template = await db.get_training_template(template_id)
    # Then get the plugin name from the template:

    template_config = json.loads(template["config"])
    if "plugin_name" not in template_config:
        return {"error": "true"}

    plugin_name = template_config["plugin_name"]

    # Now we need the current experiment id from the job:
    # experiment_id = job["experiment_id"]
    # Then get the experiment name:
    # experiment = await db.experiment_get(experiment_id)
    # experiment_name = experiment["name"]

    # Now we can get the output.txt from the plugin which is stored in
    # /workspace/experiments/{experiment_name}/plugins/{plugin_name}/output.txt
    output_file = f"{dirs.plugin_dir_by_name(plugin_name)}/output.txt"
    with open(output_file, "r") as f:
        output = f.read()
    return output


# Templates


@router.get("/template/{template_id}")
async def get_training_template(template_id: str):
    return await db.get_training_template(template_id)


@router.put("/template/update")
async def update_training_template(
    template_id: str, name: str, description: str, type: str, config: Annotated[str, Body(embed=True)]
):
    try:
        configObject = json.loads(config)
        datasets = configObject["dataset_name"]
        await db.update_training_template(template_id, name, description, type, datasets, config)
    except JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return {"status": "error", "message": "An error occurred while processing the request."}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"status": "error", "message": "An internal error has occurred."}
    return {"status": "success"}


@router.get("/{job_id}/stream_output")
async def stream_job_output(job_id: str, sweeps: bool = False):
    job = await db.job_get(job_id)
    job_data = job["job_data"]

    job_id = secure_filename(job_id)

    plugin_name = job_data["plugin"]
    plugin_dir = dirs.plugin_dir_by_name(plugin_name)
    new_output_dir = os.path.join(dirs.WORKSPACE_DIR, "jobs", str(job_id))
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)

    output_file_name = os.path.join(plugin_dir, f"output_{job_id}.txt")
    jobs_dir_output_file_name = os.path.join(new_output_dir, f"output_{job_id}.txt")

    if sweeps:
        output_file = job_data.get("sweep_output_file", None)
        if output_file is not None and os.path.exists(output_file):
            output_file_name = output_file

    if os.path.exists(jobs_dir_output_file_name):
        output_file_name = jobs_dir_output_file_name
    elif not os.path.exists(jobs_dir_output_file_name) and not os.path.exists(output_file_name):
        with open(jobs_dir_output_file_name, "w") as f:
            f.write("")
    elif not os.path.exists(jobs_dir_output_file_name) and os.path.exists(output_file_name):
        output_file_name = output_file_name

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
    job = await db.job_get(job_id)
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
    job = await db.job_get(job_id)
    job_data = job["job_data"]
    file_path = job_data.get("plot_data_path", None)

    if file_path is None or not os.path.exists(file_path):
        return Response("No plot data found for this evaluation", media_type="text/csv")

    content = json.loads(open(file_path, "r").read())
    return content


@router.get("/{job_id}/get_generated_dataset")
async def get_generated_dataset(job_id: str):
    job = await db.job_get(job_id)
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
    job = await db.job_get(job_id)
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
    job = await db.job_get(job_id)
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


async def _dispatch_remote_job(job, job_config, target_machine_id):
    """Helper function to dispatch a job to a remote machine."""
    import httpx

    try:
        # Get machine details
        machine = await db.network_machine_get(target_machine_id)
        if not machine:
            await db.job_update_status(job["id"], "FAILED", error_msg="Target machine not found")
            return {"error": "Target machine not found"}

        if machine["status"] != "online":
            await db.job_update_status(job["id"], "FAILED", error_msg="Target machine is not online")
            return {"error": "Target machine is not online"}

        # Extract dependencies from job config
        plugins_required = job_config.get("plugins_required", [])
        models_required = job_config.get("models_required", [])
        datasets_required = job_config.get("datasets_required", [])

        # Build URL for dispatch
        base_url = f"http://{machine['host']}:{machine['port']}"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Prepare dependencies first
            if plugins_required or models_required or datasets_required:
                deps_response = await client.post(
                    f"{base_url}/network/prepare_dependencies",
                    json={"plugins": plugins_required, "models": models_required, "datasets": datasets_required},
                    headers=headers,
                )
                if deps_response.status_code != 200:
                    await db.job_update_status(job["id"], "FAILED", error_msg="Failed to prepare dependencies")
                    return {"error": "Failed to prepare dependencies on remote machine"}

            # Dispatch the job
            job_response = await client.post(
                f"{base_url}/network/execute_job",
                json={
                    "job_id": str(job["id"]),
                    "job_data": job_config,
                    "job_type": job.get("type"),
                    "experiment_id": str(job.get("experiment_id")),
                    "origin_machine": await _get_this_machine_info(),
                },
                headers=headers,
            )

            if job_response.status_code != 200:
                await db.job_update_status(job["id"], "FAILED", error_msg="Failed to start job on remote machine")
                return {"error": "Failed to start job on remote machine"}

            # Update job status to indicate it's running remotely
            await db.job_update_status(job["id"], "RUNNING_REMOTE")
            await db.job_update_job_data_insert_key_value(
                job["id"], "execution_host", f"{machine['host']}:{machine['port']}"
            )
            await db.job_update_job_data_insert_key_value(job["id"], "remote_execution", True)

            return {
                "status": "success",
                "message": f"Job dispatched to {machine['name']}",
                "job_id": job["id"],
                "target_machine": machine["name"],
            }

    except Exception as e:
        await db.job_update_status(job["id"], "FAILED", error_msg=f"Remote dispatch failed: {str(e)}")
        return {"error": f"Failed to dispatch remote job: {str(e)}"}


async def _get_this_machine_info():
    """Get information about this machine for identification purposes."""
    import socket
    import platform

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        return {"hostname": hostname, "ip": local_ip, "platform": platform.system(), "architecture": platform.machine()}
    except Exception:
        return {"hostname": "unknown", "ip": "unknown"}


@router.get("/list_by_machine")
async def jobs_get_by_machine(machine_id: int = None):
    """Get jobs filtered by target machine."""
    jobs = await db.jobs_get_all()

    if machine_id is not None:
        # Filter jobs that have target_machine_id in job_data
        filtered_jobs = []
        for job in jobs:
            job_data = job.get("job_data", {})
            if job_data.get("target_machine_id") == machine_id:
                filtered_jobs.append(job)
        return filtered_jobs

    return jobs


@router.get("/{job_id}/remote_status")
async def get_remote_job_status(job_id: str):
    """Get status of a job that may be running remotely."""
    job = await db.job_get(job_id)
    if not job:
        return {"error": "Job not found"}

    job_data = job.get("job_data", {})
    target_machine_id = job_data.get("target_machine_id")

    if not target_machine_id:
        # Local job, return local status
        return {"status": "success", "job_status": job, "location": "local"}

    # Remote job, query the remote machine
    try:
        machine = await db.network_machine_get(target_machine_id)
        if not machine:
            return {"error": "Target machine not found"}

        import httpx

        base_url = f"http://{machine['host']}:{machine['port']}"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/jobs/{job_id}", headers=headers)

            if response.status_code == 200:
                remote_job_data = response.json()
                return {
                    "status": "success",
                    "job_status": remote_job_data,
                    "location": "remote",
                    "machine": machine["name"],
                }
            else:
                return {"error": "Failed to get remote job status"}

    except Exception as e:
        return {"error": f"Failed to query remote machine: {str(e)}"}
