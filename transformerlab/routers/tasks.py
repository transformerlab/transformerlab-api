import json
from fastapi import APIRouter, Body

import transformerlab.db as db

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

@router.put("/{task_id}/update", summary="Updates a task with new information")
async def update_task(task_id: int, new_task:dict = Body()):
    await db.update_task(task_id, new_task)
    return {"message":"OK"}

@router.get("/{task_id}/delete", summary="Deletes a task")
async def delete_task(task_id: int):
    await db.delete_task(task_id)
    return {"message":"OK"}

@router.put("/new_task", summary="Create a new task")
async def add_task(new_task: dict = Body()):
    await db.add_task(new_task["name"], new_task["type"], new_task["input_config"], new_task["config"], new_task["plugin"], new_task["output_config"], new_task["experiment_id"])
    return {"message":"OK"}

@router.get("/delete_all", summary="Wipe the task table")
async def tasks_delete_all():
    await db.tasks_delete_all()
    return {"message":"OK"}

# These functions convert templates to tasks so that we can do a migration in dev without breaking main for users
# Right now it can do trains, evals, and generates
@router.get("/convert_training_template_to_task", summary="Convert a specific training template to a task")
async def convert_training_template_to_task(template_id: int, experiment_id: int):
    template = await db.get_training_template(template_id)
    template_config = json.loads(template["config"])
    input_config = {"model_name":template_config["model_name"],"model_architecture":template_config["model_architecture"],"dataset_name":template_config["dataset_name"]} 
    output_config = {"adaptor_name":template_config["adaptor_name"]}
    await db.add_task(template["name"], "TRAIN", json.dumps(input_config), template["config"], template_config["plugin_name"], json.dumps(output_config), experiment_id)
    return {"message":"OK"}


@router.get("/convert_eval_to_task", summary="Convert a specific eval template to a task")
async def convert_eval_to_task(eval_name: str, experiment_id: int):
    experiment_evaluations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["evaluations"])
    for eval in experiment_evaluations:
        if eval["name"] == eval_name:
            await db.add_task(eval["name"], "EVAL", "{}", json.dumps(eval), eval["plugin"], "{}", experiment_id)
    return {"message":"OK"}

@router.get("/convert_generate_to_task", summary="Convert a specific generation template to a task")
async def convert_generate_to_task(generate_name: str, experiment_id: int):
    experiment_generations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["generations"])
    for generation in experiment_generations:
        if generation["name"] == generate_name:
            await db.add_task(generation["name"], "GENERATE", "{}", json.dumps(generation), generation["plugin"], "{}", experiment_id)
    return {"message":"OK"}

#this function is the "convert all" function so that its just 1 api call
@router.get("/{experiment_id}/convert_all_to_tasks", summary="Convert all templates to tasks")
async def convert_all_to_tasks(experiment_id):
    #train templates
    train_templates = await db.get_training_templates()
    for template in train_templates:
        await convert_training_template_to_task(template[0], experiment_id)
    #evals
    experiment_evaluations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["evaluations"])
    for eval in experiment_evaluations:
        await convert_eval_to_task(eval["name"], experiment_id)
    #generations
    experiment_generations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["generations"])
    for generation in experiment_generations:
        await convert_generate_to_task(generation["name"], experiment_id)
    return {"message":"OK"}



@router.get("/{task_id}/queue", summary="Queue a task to run")
async def queue_task(task_id: int, inputs: str = "{}", outputs:str = "{}"):
    task_to_queue = await db.tasks_get_by_id(task_id)
    job_type = task_to_queue["type"]
    job_status = "QUEUED"
    job_data = {}
    #these are the input and output configs from the task
    input_config = json.loads(task_to_queue["input_config"])
    output_config = json.loads(task_to_queue["output_config"])

    #these are the in runtime changes that will override the input and output config from the task
    inputs = json.loads(inputs)
    outputs = json.loads(outputs)
    if job_type == "TRAIN":
        job_data["config"] = json.loads(task_to_queue["config"])
        job_data["model_name"] = input_config["model_name"]
        job_data["dataset"] = input_config["dataset_name"]
        if "type" not in job_data["config"].keys():
            if "trainingType" not in job_data["config"].keys():
                job_data["config"]["type"] = "LoRA"
            else:
                job_data["config"]["type"] = job_data["config"]["trainingType"]
        #sets the inputs and outputs from the task
        for key in input_config.keys():
            job_data["config"][key] = input_config[key]
        for key in output_config.keys():
            job_data["config"][key] = output_config[key]

        #overrides the inputs and outputs based on the runtime changes requested
        for key in inputs.keys():
            if key=="model_name":
                job_data["model_name"] = inputs["model_name"]
            if key=="dataset":
                job_data["dataset"] = inputs["dataset_name"]
            job_data["config"][key] = inputs[key]
        for key in outputs.keys():
            job_data["config"][key] = outputs[key]


        job_data["template_id"] = task_to_queue["id"]
        job_data["template_name"] = task_to_queue["name"]
    elif job_type == "EVAL":
        job_data["evaluator"] = task_to_queue["name"]
        job_data["config"] = json.loads(task_to_queue["config"])
        for key in input_config.keys():
            job_data["config"][key] = input_config[key]
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]

        job_data["plugin"] = task_to_queue["plugin"]
    elif job_type == "GENERATE":
        job_data["generator"] = task_to_queue["name"]
        job_data["config"] = json.loads(task_to_queue["config"])
        for key in input_config.keys():
            job_data["config"][key] = input_config[key]
        for key in inputs.keys():
            job_data["config"][key] = inputs[key]
        job_data["plugin"] = task_to_queue["plugin"]
    job_id = await db.job_create(job_type, job_status, json.dumps(job_data), task_to_queue["experiment_id"])
    return {"id": job_id}
