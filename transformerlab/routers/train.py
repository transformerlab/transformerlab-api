import json
import os
import subprocess
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body

import transformerlab.db as db
from transformerlab.shared import shared
from transformerlab.shared import dirs


# @TODO hook this up to an endpoint so we can cancel a finetune


def abort_fine_tune():
    print("Aborting training...")
    return "abort"


router = APIRouter(prefix="/train", tags=["train"])


# @router.post("/finetune_lora")
# def finetune_lora(
#     model: str,
#     adaptor_name: str,
#     text: Annotated[str, Body()],
#     background_tasks: BackgroundTasks,
# ):
#     background_tasks.add_task(finetune, model, text, adaptor_name)

#     return {"message": "OK"}


@router.post("/template/create")
async def create_training_template(
    name: str,
    description: str,
    type: str,
    config: Annotated[str, Body(embed=True)],
):
    configObject = json.loads(config)
    datasets = configObject["dataset_name"]
    await db.create_training_template(name, description, type, datasets, config)
    return {"message": "OK"}


@router.get("/templates")
async def get_training_templates():
    return await db.get_training_templates()


@router.get("/template/{template_id}")
async def get_training_template(template_id: str):
    return await db.get_training_template(template_id)


@router.get("/template/{template_id}/delete")
async def delete_training_template(template_id: str):
    await db.delete_training_template(template_id)
    return {"message": "OK"}


@router.get("/jobs")
async def jobs_get_all():
    jobs = await db.training_jobs_get_all()
    return jobs


@router.get("/job/delete/{job_id}")
async def job_delete(job_id: str):
    await db.job_delete(job_id)
    return {"message": "OK"}


@router.get("/job/create")
async def job_create(template_id: str, description: str, experiment_id, config: str = "{}"):
    print("template_id: " + template_id)
    jobid = await db.training_job_create(template_id=template_id, description=description, experiment_id=experiment_id)
    return jobid


@router.get("/job/update/{job_id}")
async def job_update(job_id: str, status: str):
    await db.job_update_status(job_id, status)
    return {"message": "OK"}


@router.get("/job/start_next")
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


@router.get("/job/delete_all")
async def job_delete_all():
    await db.job_delete_all()
    return {"message": "OK"}


@router.get("/job/{job_id}")
async def get_training_job(job_id: str):
    return await db.job_get(job_id)


@router.get("/job/{job_id}/output")
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


tensorboard_process = None


@router.get("/tensorboard/stop")
async def stop_tensorboard():
    global tensorboard_process

    if tensorboard_process:
        print("Stopping Tensorboard")
        tensorboard_process.terminate()
    return {"message": "OK"}


@router.get("/tensorboard/start")
async def start_tensorboard(job_id: str):
    await spawn_tensorboard(job_id)
    return {"message": "OK"}


async def spawn_tensorboard(job_id: str):
    global tensorboard_process

    # call stop to ensure that if there is thread running we kill it first
    # otherwise it will dangle and we won't be able to grab the port
    await stop_tensorboard()

    print("Starting tensorboard")

    os.makedirs(
        f"{dirs.WORKSPACE_DIR}/tensorboards/job{job_id}", exist_ok=True)

    # hardcoded for now, later on we should get the information from the job id in SQLITE
    # and use the config of the job to determine the logdir
    logdir = f"{dirs.WORKSPACE_DIR}/tensorboards/job{job_id}"

    tensorboard_process = subprocess.Popen(
        ["tensorboard", "--logdir", logdir, "--host", "0.0.0.0"]
    )
