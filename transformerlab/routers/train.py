import json
import os
import asyncio
from typing import Annotated

from fastapi import APIRouter, Body
import logging
import transformerlab.db.db as db
import transformerlab.db.jobs as db_jobs
from transformerlab.shared import dirs

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


async def get_output_file_name(job_id: str):
    try:
        # First get the template Id from this job:
        job = await db_jobs.job_get(job_id)

        job_data = job["job_data"]
        if "template_id" not in job_data:
            if job_data.get("output_file_path") is not None:
                # if the job data has an output file path, use that
                return job_data["output_file_path"]
            raise ValueError("Template ID not found in job data")

        template_config = job_data["config"]
        if "plugin_name" not in template_config:
            raise ValueError("Plugin name not found in template config")

        # get the output.txt from the plugin which is stored in
        plugin_name = template_config["plugin_name"]
        plugin_dir = dirs.plugin_dir_by_name(plugin_name)

        job_id = secure_filename(job_id)

        # job output is stored in separate files with a job number in the name...
        jobs_dir_output_file_name = os.path.join(dirs.WORKSPACE_DIR, "jobs", str(job_id))

        # job output is stored in separate files with a job number in the name...
        if os.path.exists(os.path.join(jobs_dir_output_file_name, f"output_{job_id}.txt")):
            output_file = os.path.join(jobs_dir_output_file_name, f"output_{job_id}.txt")
        elif os.path.exists(os.path.join(plugin_dir, f"output_{job_id}.txt")):
            output_file = os.path.join(plugin_dir, f"output_{job_id}.txt")

        # but it used to be all stored in a single file called output.txt, so check that as well
        elif os.path.exists(os.path.join(plugin_dir, "output.txt")):
            output_file = os.path.join(plugin_dir, "output.txt")
        else:
            raise ValueError(f"No output file found for job {job_id}")

        return output_file
    except Exception as e:
        raise e


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
                output_file_name = await get_output_file_name(job_id)

        else:
            output_file_name = await get_output_file_name(job_id)

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
        if tensorboard_process.returncode is None:  # Process is still running
            print("Stopping Tensorboard")
            tensorboard_process.kill()
            await tensorboard_process.wait()
        else:
            print("Tensorboard process already exited")
        tensorboard_process = None

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
    data = await db.experiment_get(experiment_id)
    if data is None:
        return {"message": f"Experiment {experiment_id} does not exist"}

    experiment_dir = dirs.experiment_dir_by_name(data["name"])

    job_data = job["job_data"]

    if "template_name" not in job_data.keys():
        raise ValueError("Template Name not found in job data")

    template_name = job_data["template_name"]
    template_name = secure_filename(template_name)

    os.makedirs(f"{experiment_dir}/tensorboards/{template_name}", exist_ok=True)

    logdir = f"{experiment_dir}/tensorboards/{template_name}"

    tensorboard_process = await asyncio.create_subprocess_exec("tensorboard", "--logdir", logdir, "--host", "0.0.0.0")
