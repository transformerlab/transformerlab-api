import json
import time
import os

from fastapi import APIRouter, Body
from werkzeug.utils import secure_filename

import transformerlab.db.db as db
from transformerlab.db.datasets import get_datasets
from transformerlab.db.jobs import job_create
from transformerlab.models import model_helper

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/list", summary="Returns all the tasks")
async def tasks_get_all():
    tasks = await db.tasks_get_all()
    return tasks


@router.get("/{task_id}/get", summary="Gets all the data for a single task")
async def tasks_get_by_id(task_id: int):
    tasks = await db.tasks_get_all()
    for task in tasks:
        if task["id"] == task_id:
            return task
    return {"message": "NOT FOUND"}


@router.get("/list_by_type", summary="Returns all the tasks of a certain type, e.g TRAIN")
async def tasks_get_by_type(type: str):
    tasks = await db.tasks_get_by_type(type)
    return tasks


@router.get(
    "/list_by_type_in_experiment", summary="Returns all the tasks of a certain type in a certain experiment, e.g TRAIN"
)
async def tasks_get_by_type_in_experiment(type: str, experiment_id: int):
    tasks = await db.tasks_get_by_type_in_experiment(type, experiment_id)
    return tasks


@router.put("/{task_id}/update", summary="Updates a task with new information")
async def update_task(task_id: int, new_task: dict = Body()):
    # Perform secure_filename before updating the task
    new_task["name"] = secure_filename(new_task["name"])
    await db.update_task(task_id, new_task)
    return {"message": "OK"}


@router.get("/{task_id}/delete", summary="Deletes a task")
async def delete_task(task_id: int):
    await db.delete_task(task_id)
    return {"message": "OK"}


@router.put("/new_task", summary="Create a new task")
async def add_task(new_task: dict = Body()):
    # Perform secure_filename before adding the task
    new_task["name"] = secure_filename(new_task["name"])
    await db.add_task(
        new_task["name"],
        new_task["type"],
        new_task["inputs"],
        new_task["config"],
        new_task["plugin"],
        new_task["outputs"],
        new_task["experiment_id"],
    )
    if new_task["type"] == "TRAIN":
        if not isinstance(new_task["config"], dict):
            new_task["config"] = json.loads(new_task["config"])
        config = new_task["config"]
        # Get the dataset info from the config
        datasets = config.get("_tlab_recipe_datasets", {})
        datasets = datasets.get("path", "")

        # Get the model info from the config
        model = config.get("_tlab_recipe_models", {})
        model_path = model.get("path", "")

        if datasets == "" and model_path == "":
            return {"message": "OK"}

        # Check if the model and dataset are installed
        # For model: get a list of local models to determine what has been downloaded already
        model_downloaded = await model_helper.is_model_installed(model_path)

        # Repeat for dataset
        dataset_downloaded = False
        local_datasets = await get_datasets()
        for dataset in local_datasets:
            if dataset["dataset_id"] == datasets:
                dataset_downloaded = True

        # generate a repsonse to tell if model and dataset need to be downloaded
        response = {}

        # Dataset info - including whether it needs to be downloaded or not
        dataset_status = {}
        dataset_status["path"] = datasets
        dataset_status["downloaded"] = dataset_downloaded
        response["dataset"] = dataset_status

        # Model info - including whether it needs to be downloaded or not
        model_status = {}
        model_status["path"] = model_path
        model_status["downloaded"] = model_downloaded
        response["model"] = model_status

        return {"status": "OK", "data": response}

    return {"message": "OK"}


@router.get("/delete_all", summary="Wipe the task table")
async def tasks_delete_all():
    await db.tasks_delete_all()
    return {"message": "OK"}


# These functions convert templates to tasks so that we can do a migration in dev without breaking main for users
# Right now it can do trains, evals, and generates
@router.get("/convert_training_template_to_task", summary="Convert a specific training template to a task")
async def convert_training_template_to_task(template_id: int, experiment_id: int):
    template = await db.get_training_template(template_id)
    template_config = json.loads(template["config"])
    inputs = {}
    if "model_name" in template_config.keys():
        inputs = {
            "model_name": template_config["model_name"],
            "model_architecture": template_config["model_architecture"],
            "dataset_name": template_config["dataset_name"],
        }
    if "embedding_model_name" in template_config.keys():
        inputs = {
            "embedding_model_name": template_config["embedding_model_name"],
            "embedding_model_architecture": template_config["embedding_model_architecture"],
            "dataset_name": template_config["dataset_name"],
        }

    outputs = {}
    if "adaptor_name" in template_config.keys():
        outputs = {"adaptor_name": template_config.get("adaptor_name", "adaptor")}
    try:
        await db.add_task(
            template["name"],
            "TRAIN",
            json.dumps(inputs),
            template["config"],
            template_config["plugin_name"],
            json.dumps(outputs),
            experiment_id,
        )
    except Exception:
        return {"message": "ERROR: unable to convert template to task."}
    return {"message": "OK"}


@router.get("/convert_eval_to_task", summary="Convert a specific eval template to a task")
async def convert_eval_to_task(eval_name: str, experiment_id: int):
    experiment_evaluations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["evaluations"])
    for eval in experiment_evaluations:
        if eval["name"] == eval_name:
            await db.add_task(eval["name"], "EVAL", "{}", json.dumps(eval), eval["plugin"], "{}", experiment_id)
    return {"message": "OK"}


@router.get("/convert_generate_to_task", summary="Convert a specific generation template to a task")
async def convert_generate_to_task(generate_name: str, experiment_id: int):
    experiment_generations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["generations"])
    for generation in experiment_generations:
        if generation["name"] == generate_name:
            await db.add_task(
                generation["name"], "GENERATE", "{}", json.dumps(generation), generation["plugin"], "{}", experiment_id
            )
    return {"message": "OK"}


@router.get("/convert_export_to_task", summary="Convert export request to a task")
async def convert_export_to_task(
    experiment_id: int,
    plugin_name: str,
    plugin_architecture: str,
    plugin_params: str = "{}",
):
    """
    Create an EXPORT task that can be queued and run later.
    This replaces the immediate execution in run_exporter_script.
    """

    # Load experiment details into config
    experiment_details = await db.experiment_get(id=experiment_id)
    if experiment_details is None:
        return {"message": f"Experiment {experiment_id} does not exist"}

    # Get input model parameters
    config = json.loads(experiment_details["config"])
    input_model_id = config["foundation"]
    input_model_id_without_author = input_model_id.split("/")[-1]
    input_model_architecture = config["foundation_model_architecture"]

    # The exporter plugin needs to know where to find the model
    input_model_path = config.get("foundation_filename", "")
    if not input_model_path:
        input_model_path = input_model_id

    # Convert JSON parameters
    params = json.loads(plugin_params)
    q_type = ""
    if "outtype" in params:
        q_type = params["outtype"]
    elif "q_bits" in params:
        q_type = str(params["q_bits"]) + "bit"

    # Generate output model details
    conversion_time = int(time.time())
    output_model_architecture = plugin_architecture
    output_model_id = f"{output_model_architecture}-{input_model_id_without_author}-{conversion_time}"
    if len(q_type) > 0:
        output_model_id = f"{output_model_id}-{q_type}"
    output_model_name = f"{input_model_id_without_author} - {output_model_architecture}"
    if len(q_type) > 0:
        output_model_name = f"{output_model_name} - {q_type}"
    output_filename = ""

    # GGUF is special: it generates a different format with only one file
    # For everything to work we need the model ID and output filename to match
    if output_model_architecture == "GGUF":
        output_model_id = f"{input_model_id_without_author}-{conversion_time}.gguf"
        if len(q_type) > 0:
            output_model_id = f"{input_model_id_without_author}-{conversion_time}-{q_type}.gguf"
        output_filename = output_model_id

    # Figure out plugin and model output directories
    from transformerlab.shared import dirs

    script_directory = dirs.plugin_dir_by_name(plugin_name)
    output_model_id = secure_filename(output_model_id)
    output_path = os.path.join(dirs.MODELS_DIR, output_model_id)

    # Create task inputs (what the task needs to run)
    inputs = {
        "input_model_id": input_model_id,
        "input_model_path": input_model_path,
        "input_model_architecture": input_model_architecture,
        "plugin_name": plugin_name,
        "plugin_architecture": plugin_architecture,
    }

    # Create task config (all the configuration needed for export)
    task_config = {
        "plugin_name": plugin_name,
        "input_model_id": input_model_id,
        "input_model_path": input_model_path,
        "input_model_architecture": input_model_architecture,
        "output_model_id": output_model_id,
        "output_model_architecture": output_model_architecture,
        "output_model_name": output_model_name,
        "output_model_path": output_path,
        "output_filename": output_filename,
        "script_directory": script_directory,
        "params": params,
    }

    # Create task outputs (what the task will produce)
    outputs = {
        "exported_model_path": output_path,
        "output_model_id": output_model_id,
        "export_status": "pending",
    }

    # Create the export task
    await db.add_task(
        name=f"Export {input_model_id_without_author} to {plugin_architecture}",
        Type="EXPORT",
        inputs=json.dumps(inputs),
        config=json.dumps(task_config),
        plugin=plugin_name,
        outputs=json.dumps(outputs),
        experiment_id=experiment_id,
    )

    return {"message": "OK"}


# this function is the "convert all" function so that its just 1 api call
@router.get("/{experiment_id}/convert_all_to_tasks", summary="Convert all templates to tasks")
async def convert_all_to_tasks(experiment_id):
    # train templates
    train_templates = await db.get_training_templates()
    for template in train_templates:
        await convert_training_template_to_task(template[0], experiment_id)
    exp = await db.experiment_get(experiment_id)
    if not isinstance(exp["config"], dict):
        experiment_config = json.loads((await db.experiment_get(experiment_id))["config"])
    else:
        experiment_config = exp["config"]
    # evals
    if "evaluations" in experiment_config.keys():
        experiment_evaluations = json.loads(experiment_config["evaluations"])
        for eval in experiment_evaluations:
            await convert_eval_to_task(eval["name"], experiment_id)
    # generations
    if "generations" in experiment_config:
        experiment_generations = json.loads(experiment_config["generations"])
        for generation in experiment_generations:
            await convert_generate_to_task(generation["name"], experiment_id)
    return {"message": "OK"}


@router.get("/{task_id}/queue", summary="Queue a task to run")
async def queue_task(task_id: int, input_override: str = "{}", output_override: str = "{}"):
    task_to_queue = await db.tasks_get_by_id(task_id)
    job_type = task_to_queue["type"]
    job_status = "QUEUED"
    job_data = {}
    # these are the input and output configs from the task
    if not isinstance(task_to_queue["inputs"], dict):
        task_to_queue["inputs"] = json.loads(task_to_queue["inputs"])
    if not isinstance(task_to_queue["outputs"], dict):
        task_to_queue["outputs"] = json.loads(task_to_queue["outputs"])

    inputs = task_to_queue["inputs"]
    outputs = task_to_queue["outputs"]

    # these are the in runtime changes that will override the input and output config from the task
    if not isinstance(input_override, dict):
        input_override = json.loads(input_override)
    if not isinstance(output_override, dict):
        output_override = json.loads(output_override)

    if not isinstance(task_to_queue["config"], dict):
        task_to_queue["config"] = json.loads(task_to_queue["config"])

    if job_type == "TRAIN":
        job_data["config"] = task_to_queue["config"]
        job_data["model_name"] = inputs["model_name"]
        job_data["dataset"] = inputs["dataset_name"]
        if "type" not in job_data["config"].keys():
            job_data["config"]["type"] = "LoRA"
        # sets the inputs and outputs from the task
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
        for key in outputs.keys():
            job_data["config"][key] = outputs[key]

        # overrides the inputs and outputs based on the runtime changes requested
        for key in input_override.keys():
            if key == "model_name":
                job_data["model_name"] = input_override["model_name"]
            if key == "dataset":
                job_data["dataset"] = input_override["dataset_name"]
            job_data["config"][key] = input_override[key]
        for key in output_override.keys():
            job_data["config"][key] = output_override[key]

        job_data["template_id"] = task_to_queue["id"]
        job_data["template_name"] = task_to_queue["name"]
    elif job_type == "EVAL":
        job_data["evaluator"] = task_to_queue["name"]
        job_data["config"] = task_to_queue["config"]
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
            job_data["config"]["script_parameters"][key] = inputs[key]
        for key in input_override.keys():
            job_data["config"][key] = input_override[key]
            job_data["config"]["script_parameters"][key] = input_override[key]

        job_data["plugin"] = task_to_queue["plugin"]
    elif job_type == "GENERATE":
        job_data["generator"] = task_to_queue["name"]
        job_data["config"] = task_to_queue["config"]
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
            job_data["config"]["script_parameters"][key] = inputs[key]
        for key in input_override.keys():
            job_data["config"][key] = input_override[key]
            job_data["config"]["script_parameters"][key] = input_override[key]

        for key in outputs.keys():
            job_data["config"][key] = outputs[key]
            job_data["config"]["script_parameters"][key] = outputs[key]
        for key in output_override.keys():
            job_data["config"][key] = output_override[key]
            job_data["config"]["script_parameters"][key] = output_override[key]
        job_data["plugin"] = task_to_queue["plugin"]
    elif job_type == "EXPORT":
        job_data["exporter"] = task_to_queue["name"]
        job_data["config"] = task_to_queue["config"]
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
        for key in input_override.keys():
            job_data["config"][key] = input_override[key]

        for key in outputs.keys():
            job_data["config"][key] = outputs[key]
        for key in output_override.keys():
            job_data["config"][key] = output_override[key]
        job_data["plugin"] = task_to_queue["plugin"]
    job_id = await job_create("EXPORT" if job_type == "EXPORT" else job_type, job_status, json.dumps(job_data), task_to_queue["experiment_id"])
    return {"id": job_id}
