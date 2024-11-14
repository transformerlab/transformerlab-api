import json
import yaml
import os
import subprocess
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body
from fastapi.responses import PlainTextResponse

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


@router.get("/template/{template_id}/delete")
async def delete_training_template(template_id: str):
    await db.delete_training_template(template_id)
    return {"message": "OK"}


@router.post("/template/import")
async def import_recipe(name: str, recipe_yaml: str = Body(...)):

    # TODO: Probably there is a way to do YAML validation automatically
    print(recipe_yaml)

    try:
        recipe = yaml.safe_load(recipe_yaml)
    except yaml.YAMLError as e:
        print(e)
        return {"status": "error", "message": e}

    # Get top level sections of recipe
    # TODO: Is it an error if any of these don't exist?
    metadata = recipe.get("metadata", {})
    model = recipe.get("model", {})
    datasets = recipe.get("datasets", {})
    training = recipe.get("training", {})

    # Get fields needed to save template
    # TODO: Is it an error if any of these are missing?
    description = metadata.get("description", "")
    type = training.get("type", "LoRA")
    datasets = datasets.get("path", "")
    config = training.get("config_json", {})

    print("CREATING TEMPLATE")
    print("Name:", name)
    print("Description:", description)
    print("Type:", type)
    print("Datasets:", datasets)
    print("Config:", config)

    # TODO: Throw proper error if template name exists
    await db.create_training_template(name, description, type, datasets, config)
    return {"message": "OK"}


@router.get("/template/{template_id}/export", response_class=PlainTextResponse)
async def export_recipe(template_id: str):
    
    # Read in training template from DB and parse config JSON
    training_template = await db.get_training_template(template_id)
    if not training_template:
        return ""

    template_config_json = training_template.get("config", {})
    try :
        template_config = json.loads(template_config_json)
    except:
        print("Error reading template config:")
        print(template_config_json)
        template_config = {}

    # Construct recipe object
    recipe = {}
    
    # TODO: Could remove for now but thought let's leave placeholder
    metadata = {
        "author": "",
        "name": training_template.get("name", ""),
        "description": training_template.get("description", "")
    }

    model = {
        "name": template_config.get("model_name", ""),
        "path": template_config.get("model_name", "")
    }

    datasets = {
        "name": training_template.get("datasets", ""),
        "path": training_template.get("datasets", "")
    }

    # TODO: Read in the type from the DB!
    training = {
        "type" : "LoRA",
        "plugin": template_config.get("plugin_name", ""),
        "formatting_template" : template_config.get("formatting_template", ""),
        "config_json": template_config_json
    }

    recipe["schemaVersion"] = 0.1
    recipe["metadata"] = metadata
    recipe["model"] = model
    recipe["datasets"] = datasets
    recipe["training"] = training
    recipe["test"] = {}

    # Convert recipe to YAML
    recipe_yaml = yaml.dump(recipe, sort_keys=False)
    print(recipe_yaml)
    return recipe_yaml


# @router.get("/jobs")
# async def jobs_get_all():
#     jobs = await db.training_jobs_get_all()
#     return jobs


# @router.get("/job/delete/{job_id}")
# async def job_delete(job_id: str):
#     await db.job_delete(job_id)
#     return {"message": "OK"}


# @router.get("/job/update/{job_id}")
# async def job_update(job_id: str, status: str):
#     await db.job_update_status(job_id, status)
#     return {"message": "OK"}


# @router.get("/job/start_next")
# async def start_next_job():
#     num_running_jobs = await db.job_count_running()
#     if num_running_jobs > 0:
#         return {"message": "A job is already running"}
#     nextjob = await db.jobs_get_next_queued_job()
#     if nextjob:
#         print(nextjob)
#         print("Starting job: " + str(nextjob['id']))
#         print(nextjob['job_data'])
#         job_config = json.loads(nextjob['job_data'])
#         print(job_config["template_id"])
#         experiment_id = nextjob["experiment_id"]
#         data = await db.experiment_get(experiment_id)
#         if data is None:
#             return {"message": f"Experiment {id} does not exist"}
#         config = json.loads(data["config"])

#         experiment_name = data["name"]
#         await shared.run_job(job_id=nextjob['id'], job_config=job_config, experiment_name=experiment_name)
#         return nextjob
#     else:
#         return {"message": "No jobs in queue"}


# @router.get("/job/delete_all")
# async def job_delete_all():
#     await db.job_delete_all()
#     return {"message": "OK"}


@router.get("/job/{job_id}")
async def get_training_job(job_id: str):
    return await db.job_get(job_id)


@router.get("/job/{job_id}/output")
async def get_training_job_output(job_id: str):
    # First get the template Id from this job:
    job = await db.job_get(job_id)

    job_data = job["job_data"]
    if "template_id" not in job_data:
        return {"status": "error", "error": 'true'}

    template_id = job_data["template_id"]
    # Then get the template:
    template = await db.get_training_template(template_id)
    # Then get the plugin name from the template:

    template_config = json.loads(template["config"])
    if "plugin_name" not in template_config:
        return {"status": "error", "error": 'true'}

    # get the output.txt from the plugin which is stored in
    plugin_name = template_config["plugin_name"]
    plugin_dir = dirs.plugin_dir_by_name(plugin_name)

    # job output is stored in separate files with a job number in the name...
    if os.path.exists(os.path.join(plugin_dir, f"output_{job_id}.txt")):
        output_file = os.path.join(plugin_dir, f"output_{job_id}.txt")

    # but it used to be all stored in a single file called output.txt, so check that as well
    elif os.path.exists(os.path.join(plugin_dir, "output.txt")):
        output_file = os.path.join(plugin_dir, "output.txt")

    else:
        return {"status": "error", "message": f"No output file found for job {job_id}"}

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
