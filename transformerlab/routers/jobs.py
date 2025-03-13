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
        data = await db.experiment_get(experiment_id)
        if data is None:
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
async def get_training_job_output(job_id: str):
    # First get the template Id from this job:
    job = await db.job_get(job_id)

    job_data = json.loads(job["job_data"])
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
async def stream_job_output(job_id: str):
    job = await db.job_get(job_id)
    job_data = job["job_data"]

    job_id = secure_filename(job_id)

    plugin_name = job_data["plugin"]
    plugin_dir = dirs.plugin_dir_by_name(plugin_name)

    output_file_name = os.path.join(plugin_dir, f"output_{job_id}.txt")

    if not os.path.exists(output_file_name):
        with open(output_file_name, "w") as f:
            f.write("")

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
