import asyncio
import json
from fastapi import APIRouter

import transformerlab.db as db
from transformerlab.shared import shared
from transformerlab.shared import dirs


router = APIRouter(prefix="/jobs", tags=["train"])


@router.get("/list")
async def jobs_get_all(type: str = '', status: str = ''):
    jobs = await db.jobs_get_all(type=type, status=status)
    return jobs


@router.get("/delete/{job_id}")
async def job_delete(job_id: str):
    await db.job_delete(job_id)
    return {"message": "OK"}


@router.get("/create")
async def job_create(type: str = 'UNDEFINED', status: str = 'UNDEFINED', data: str = '{}', experiment_id: str = '-1'):
    jobid = await db.job_create(type=type, status=status, job_data=data, experiment_id=experiment_id)
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
        print(nextjob)
        print("Starting job: " + str(nextjob['id']))
        print(nextjob['job_data'])
        job_config = json.loads(nextjob['job_data'])
        print(job_config["template_id"])
        experiment_id = nextjob["experiment_id"]
        data = await db.experiment_get(experiment_id)
        if data is None:
            return {"message": f"Experiment {id} does not exist"}
        config = json.loads(data["config"])

        experiment_name = data["name"]
        await shared.run_job(job_id=nextjob['id'], job_config=job_config, experiment_name=experiment_name)
        return nextjob
    else:
        return {"message": "No jobs in queue"}


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
        return {"error": 'true'}

    template_id = job_data["template_id"]
    # Then get the template:
    template = await db.get_training_template(template_id)
    # Then get the plugin name from the template:

    template_config = json.loads(template["config"])
    if "plugin_name" not in template_config:
        return {"error": 'true'}

    plugin_name = template_config["plugin_name"]

    # Now we need the current experiment id from the job:
    experiment_id = job["experiment_id"]
    # Then get the experiment name:
    experiment = await db.experiment_get(experiment_id)
    experiment_name = experiment["name"]

    # Now we can get the output.txt from the plugin which is stored in
    # /workspace/experiments/{experiment_name}/plugins/{plugin_name}/output.txt
    output_file = f"{dirs.plugin_dir_by_name(plugin_name)}/output.txt"
    with open(output_file, "r") as f:
        output = f.read()
    return output
