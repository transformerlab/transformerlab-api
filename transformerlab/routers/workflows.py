import json
import yaml
import uuid
from fastapi import APIRouter, UploadFile, Body
from fastapi.responses import FileResponse

import transformerlab.db as db
import transformerlab.routers.model as model

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.get("/list")
async def workflows_get_all():
    workflows = await db.workflows_get_all()
    return workflows


@router.get("/delete/{workflow_id}")
async def workflow_delete(workflow_id: str):
    await db.workflow_delete_by_id(workflow_id)
    return {"message": "OK"}


@router.get("/delete_all")
async def workflow_delete_all():
    await db.workflow_delete_all()
    return {"message": "OK"}


@router.get("/create")
async def workflow_create(name: str, config: str = '{"nodes":[]}', experiment_id="1"):
    config = json.loads(config)
    if len(config["nodes"] > 0):
        config["nodes"] = [
            {"type": "START", "id": str(uuid.uuid4()), "name": "START", "out": [config["nodes"][0]["id"]]}
        ] + config["nodes"]
    else:
        config["nodes"] = [{"type": "START", "id": str(uuid.uuid4()), "name": "START", "out": []}]
    workflow_id = await db.workflow_create(name, json.dumps(config), experiment_id)
    return workflow_id


@router.get("/create_empty")
async def workflow_create_empty(name: str, experiment_id="1"):
    config = {"nodes": [{"type": "START", "id": str(uuid.uuid4()), "name": "START", "out": []}]}
    workflow_id = await db.workflow_create(name, json.dumps(config), experiment_id)
    return workflow_id


@router.get("/{workflow_id}/{node_id}/edit_node_metadata")
async def workflow_edit_node_metadata(workflow_id: str, node_id: str, metadata: str):
    workflow = await db.workflows_get_by_id(workflow_id)
    config = json.loads(workflow["config"])

    for node in config["nodes"]:
        if node["id"] == node_id:
            node["metadata"] = json.loads(metadata)

    await db.workflow_update_config(workflow_id, json.dumps(config))
    return {"message": "OK"}


@router.get("/{workflow_id}/add_node")
async def workflow_add_node(workflow_id: str, node: str):
    new_node_json = json.loads(node)
    workflow = await db.workflows_get_by_id(workflow_id)
    config = json.loads(workflow["config"])

    new_node_json["id"] = str(uuid.uuid4())
    new_node_json["out"] = []
    new_node_json["metadata"] = {}

    for node in config["nodes"]:
        if node["out"] == []:
            node["out"].append(new_node_json["id"])

    config["nodes"].append(new_node_json)

    await db.workflow_update_config(workflow_id, json.dumps(config))
    return {"message": "OK"}


@router.post("/{workflow_id}/{node_id}/update_node")
async def workflow_update_node(workflow_id: str, node_id: str, new_node: dict = Body()):
    workflow = await db.workflows_get_by_id(workflow_id)
    config = json.loads(workflow["config"])

    newNodes = []

    for node in config["nodes"]:
        if node["id"] == node_id:
            newNodes.append(new_node)
        else:
            newNodes.append(node)

    config["nodes"] = newNodes

    await db.workflow_update_config(workflow_id, json.dumps(config))
    return {"message": "OK"}


@router.post("/{workflow_id}/{start_node_id}/add_edge")
async def workflow_add_edge(workflow_id: str, start_node_id: str, end_node_id: str):
    workflow = await db.workflows_get_by_id(workflow_id)
    config = json.loads(workflow["config"])

    newNodes = []

    for node in config["nodes"]:
        if node["id"] == start_node_id:
            node["out"].append(end_node_id)

    config["nodes"] = newNodes

    await db.workflow_update_config(workflow_id, json.dumps(config))
    return {"message": "OK"}


@router.get("/{workflow_id}/{node_id}/delete_node")
async def workflow_delete_node(workflow_id: str, node_id: str):
    workflow = await db.workflows_get_by_id(workflow_id)
    config = json.loads(workflow["config"])

    newNodes = []
    removedNode = {}
    for node in config["nodes"]:
        if node["id"] != node_id:
            newNodes.append(node)
        else:
            removedNode = node

    if removedNode["type"] == "START":
        return {"message": "Cannot delete start node"}
    for node in newNodes:
        if node_id in node["out"]:
            node["out"].remove(node_id)
            node["out"] += removedNode["out"]

    config["nodes"] = newNodes

    await db.workflow_update_config(workflow_id, json.dumps(config))
    return {"message": "OK"}


@router.get("/{workflow_id}/export_to_yaml")
async def workflow_export_to_yaml(workflow_id: str):
    workflow = await db.workflows_get_by_id(workflow_id)

    del workflow["current_job_id"]
    del workflow["current_task"]
    del workflow["experiment_id"]
    del workflow["created_at"]
    del workflow["updated_at"]
    del workflow["status"]
    del workflow["id"]

    workflow["config"] = json.loads(workflow["config"])

    filename = f"{workflow['name']}.yaml"
    with open(filename, "w") as yaml_file:
        yaml.dump(workflow, yaml_file)
    return FileResponse(filename, filename=filename)


@router.post("/import_from_yaml")
async def workflow_import_from_yaml(file: UploadFile, experiment_id="1"):
    with open(file.filename, "r") as fileStream:
        workflow = yaml.load(fileStream, Loader=yaml.BaseLoader)
    await db.workflow_create(workflow["name"], json.dumps(workflow["config"]), experiment_id)
    return {"message": "OK"}


@router.get("/{workflow_id}/start")
async def start_workflow(workflow_id):
    await db.workflow_update_status(workflow_id, "RUNNING")
    return {"message": "OK"}


@router.get("/start_next_step")
async def start_next_step_in_workflow():
    num_running_workflows = await db.workflow_count_running()
    if num_running_workflows == 0:
        return {"message": "A workflow is not running"}
    currently_running_workflow = await db.workflow_get_running()

    workflow_id = currently_running_workflow["id"]
    workflow_config = json.loads(currently_running_workflow["config"])
    workflow_current_task = json.loads(currently_running_workflow["current_task"])
    workflow_current_job_id = json.loads(currently_running_workflow["current_job_id"])
    workflow_experiment_id = currently_running_workflow["experiment_id"]

    current_jobs = []

    if workflow_current_job_id != []:
        for job_id in workflow_current_job_id:
            current_job = await db.job_get(job_id)
            current_jobs.append(current_job)

            if current_job["status"] == "FAILED":
                await db.workflow_update_status(workflow_id, "FAILED")
                return {"message": "the current job failed"}

            if current_job["status"] == "CANCELLED" or current_job["status"] == "DELETED":
                await db.workflow_update_with_new_job(workflow_id, "[]", -1)
                await db.workflow_update_status(workflow_id, "CANCELLED")
                return {"message": "the current job was cancelled"}

            if current_job["status"] != "COMPLETE":
                return {"message": "the current job is running"}

    if workflow_current_task != []:
        for node in workflow_config["nodes"]:
            if node["id"] in workflow_current_task:
                workflow_current_task = node["out"]
    else:
        workflow_current_task = workflow_config["nodes"][0]["id"]

    if workflow_current_task == []:
        await db.workflow_update_status(workflow_id, "COMPLETE")
        await db.workflow_update_with_new_job(workflow_id, "[]", -1)
        return {"message": "Workflow Complete!"}

    next_node = {}
    for node in workflow_config["nodes"]:
        if node["id"] in workflow_current_task:
            next_node = node
    next_job_type = next_node["type"]
    if next_job_type == "START":
        workflow_current_task = next_node["out"][0]
        for node in workflow_config["nodes"]:
            if node["id"] in workflow_current_task:
                next_node = node
        next_job_type = next_node["type"]
    next_job_status = "QUEUED"

    if next_job_type == "DOWNLOAD_MODEL":
        output = await model.download_huggingface_model(next_node["model"])
        if output["status"] != "success":
            await db.workflow_update_status(workflow_id, "FAILED")
            return {"message": "failed to download model"}
        else:
            await db.workflow_update_with_new_job(
                workflow_id, json.dumps(workflow_current_task), json.dumps([output["job_id"]])
            )
            return {"message": "model downloaded"}

    elif "template" in next_node.keys():
        template_name = next_node["template"]
        if next_job_type == "TRAIN":
            next_job_data = await db.get_training_template_by_name(template_name)
            next_job_data["config"] = json.loads(next_job_data["config"])
            next_job_data["template_id"] = 1
            next_job_data["template_name"] = template_name
        elif next_job_type == "EVAL":
            experiment_evaluations = json.loads(
                json.loads((await db.experiment_get(workflow_experiment_id))["config"])["evaluations"]
            )
            evaluation_to_run = None
            for evaluation in experiment_evaluations:
                if evaluation["name"] == template_name:
                    evaluation_to_run = evaluation
            if evaluation_to_run is None:
                await db.workflow_update_status(workflow_id, "FAILED")
            next_job_data = {"plugin": evaluation_to_run["plugin"], "evaluator": template_name}

    else:
        next_job_data = next_node["data"]

    next_job_id = await db.job_create(next_job_type, next_job_status, json.dumps(next_job_data), workflow_experiment_id)
    await db.workflow_update_with_new_job(workflow_id, json.dumps(workflow_current_task), json.dumps([next_job_id]))

    return {"message": "Next job created"}
