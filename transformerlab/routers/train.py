import json
import os
import subprocess
from typing import Annotated

from fastapi import APIRouter, Body
import logging
import transformerlab.db.db as db
import transformerlab.db.jobs as db_jobs
from transformerlab.shared import shared
from lab import dirs, Experiment

from werkzeug.utils import secure_filename

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


@router.get("/template/{template_id}/delete")
async def delete_training_template(template_id: str):
    await db.delete_training_template(template_id)
    return {"message": "OK"}


@router.get("/job/{job_id}")
async def get_training_job(job_id: str):
    return await db_jobs.job_get(job_id)


@router.get("/job/{job_id}/output")
async def get_training_job_output(job_id: str, sweeps: bool = False):
    try:
        if sweeps:
            job = await db_jobs.job_get(job_id)
            job_data = json.loads(job["job_data"])
            output_file = job_data.get("sweep_output_file", None)
            if output_file is not None and os.path.exists(output_file):
                with open(output_file, "r") as f:
                    output = f.read()
                return output
            else:
                # Get experiment information for new job directory structure
                experiment_id = job["experiment_id"]
                experiment = await db.experiment_get(experiment_id)
                experiment_name = experiment["name"]
                output_file_name = await shared.get_job_output_file_name(job_id, experiment_name=experiment_name)

        else:
            # Get experiment information for new job directory structure
            experiment_id = job["experiment_id"]
            experiment = await db.experiment_get(experiment_id)
            experiment_name = experiment["name"]
            output_file_name = await shared.get_job_output_file_name(job_id, experiment_name=experiment_name)

        with open(output_file_name, "r") as f:
            output = f.read()
        return output
    except ValueError as e:
        # Handle specific error
        logging.error(f"ValueError: {e}")
        return "An internal error has occurred!"
    except Exception as e:
        # Handle general error
        logging.error(f"Error: {e}")
        return "An internal error has occurred!"


@router.get("/job/{job_id}/sweep_results")
async def sweep_results(job_id: str):
    try:
        job = await db_jobs.job_get(job_id)
        job_data = job.get("job_data", {})

        output_file = job_data.get("sweep_results_file", None)
        if output_file and os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    output = json.load(f)
                return {"status": "success", "data": output}
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error for job {job_id}: {e}")
                return {"status": "error", "message": "Invalid JSON format in sweep results file."}
        else:
            logging.warning(f"Sweep results file not found for job {job_id}: {output_file}")
            return {"status": "error", "message": "Sweep results file not found."}

    except Exception as e:
        logging.error(f"Error loading sweep results for job {job_id}: {e}")
        return {"status": "error", "message": "An internal error has occurred!"}


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

    job = await db_jobs.job_get(job_id)
    # First get the experiment name from the job
    experiment_id = job["experiment_id"]
    exp_obj = Experiment(experiment_id)
    experiment_dir = exp_obj.get_dir()
    job_data = job["job_data"]

    if "template_name" not in job_data.keys():
        raise ValueError("Template Name not found in job data")

    template_name = job_data["template_name"]
    template_name = secure_filename(template_name)

    os.makedirs(f"{experiment_dir}/tensorboards/{template_name}", exist_ok=True)

    logdir = f"{experiment_dir}/tensorboards/{template_name}"

    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", logdir, "--host", "0.0.0.0"])
