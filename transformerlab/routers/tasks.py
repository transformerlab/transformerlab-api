import json
from fastapi import APIRouter

import transformerlab.db as db

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/list")
async def tasks_get_all():
    tasks = await db.tasks_get_all()
    return tasks

@router.get("/list_by_type")
async def tasks_get_by_type(type: str):
    tasks = await db.tasks_get_by_type(type)
    return tasks

@router.get("/delete_all")
async def tasks_delete_all():
    await db.tasks_delete_all()
    return {"message":"OK"}

@router.get("/convert_training_template_to_task")
async def convert_training_template_to_task(template_id: int, experiment_id: int):
    template = await db.get_training_template(template_id)
    template_config = json.loads(template["config"])
    input_config = {"model_name":template_config["model_name"],"model_architecture":template_config["model_architecture"],"dataset_name":template_config["dataset_name"]} 
    output_config = {"adaptor_name":template_config["adaptor_name"]}
    await db.add_task(template["name"], "TRAIN", json.dumps(input_config), template["config"], template_config["plugin_name"], json.dumps(output_config), experiment_id)
    return {"message":"OK"}


@router.get("/convert_eval_to_task")
async def convert_eval_to_task(eval_name: str, experiment_id: int):
    experiment_evaluations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["evaluations"])
    for eval in experiment_evaluations:
        if eval["name"] == eval_name:
            await db.add_task(eval["name"], "EVAL", "{}", json.dumps(eval), eval["plugin"], "{}", experiment_id)
    return {"message":"OK"}

@router.get("/convert_generate_to_task")
async def convert_generate_to_task(generate_name: str, experiment_id: int):
    experiment_generations = json.loads(json.loads((await db.experiment_get(experiment_id))["config"])["generations"])
    for generation in experiment_generations:
        if generation["name"] == generate_name:
            await db.add_task(generation["name"], "GENERATE", "{}", json.dumps(generation), generation["plugin"], "{}", experiment_id)
    return {"message":"OK"}

@router.get("/{experiment_id}/convert_all_to_tasks")
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



@router.get("/{task_id}/queue")
async def queue_task(task_id: int, inputs: str = "{}", outputs:str = "{}"):
    task_to_queue = await db.tasks_get_by_id(task_id)
    job_type = task_to_queue["type"]
    job_status = "QUEUED"
    job_data = {}
    input_config = json.loads(task_to_queue["input_config"])
    output_config = json.loads(task_to_queue["output_config"])
    inputs = json.loads(inputs)
    outputs = json.loads(outputs)
    if job_type == "TRAIN":
        job_data["config"] = json.loads(task_to_queue["config"])
        job_data["model_name"] = input_config["model_name"]
        job_data["dataset"] = input_config["dataset_name"]
        for key in input_config.keys():
            job_data["config"][key] = input_config[key]
        for key in output_config.keys():
            job_data["config"][key] = output_config[key]
        for key in inputs.keys():
            if key=="model_name":
                job_data["model_name"] = inputs["model_name"]
            if key=="datset":
                job_data["dataset"] = inputs["dataset_name"]
            job_data["config"][key] = inputs[key]
        for key in outputs.keys():
            job_data["config"][key] = outputs[key]
        job_data["template_id"] = task_to_queue["id"]
        job_data["template_name"] = task_to_queue["name"]
    elif job_type == "EVAL":
        job_data["evaluator"] = task_to_queue["name"]
        job_data["plugin"] = task_to_queue["plugin"]
    elif job_type == "GENERATE":
        job_data["generator"] = task_to_queue["name"]
        job_data["plugin"] = task_to_queue["plugin"]
    await db.job_create(job_type, job_status, json.dumps(job_data), task_to_queue["experiment_id"])
    return {"message":"OK"}
