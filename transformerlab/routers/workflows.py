import json
import yaml
import uuid
from fastapi import APIRouter, UploadFile, Body
from fastapi.responses import FileResponse

import transformerlab.db as db
import transformerlab.routers.tasks as tsks

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
    if len(config["nodes"])>0:
        config["nodes"] = [{"type":"START", "id":str(uuid.uuid4()), "name":"START", "out":[config["nodes"][0]["id"]]}] + config["nodes"]
    else:
        config["nodes"] = [{"type":"START", "id":str(uuid.uuid4()), "name":"START", "out":[]}]
    workflow_id = await db.workflow_create(name, json.dumps(config), experiment_id)
    return workflow_id


@router.get("/create_empty")
async def workflow_create_empty(name: str, experiment_id="1"):
    config = {"nodes":[{"type":"START", "id":str(uuid.uuid4()), "name":"START", "out":[]}]}
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


@router.post("/{workflow_id}/{start_node_id}/remove_edge")
async def workflow_remove_edge(workflow_id: str, start_node_id: str, end_node_id: str):
    workflow = await db.workflows_get_by_id(workflow_id)
    config = json.loads(workflow["config"])

    newNodes = []

    for node in config["nodes"]:
        if node["id"] == start_node_id:
            if end_node_id in node["out"]:
                node["out"].remove(end_node_id)
            newNodes.append(node)
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
            node["out"].append(end_node_id) # Corrected:  Modify node directly
            newNodes.append(node)  # Add the modified node
        else:
            newNodes.append(node)

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

    current_jobs = []

    if workflow_current_job_id != []:
        for job_id in workflow_current_job_id:
            current_job = await db.job_get(job_id)
            current_jobs.append(current_job)

            if current_job["status"] == "FAILED":
                await db.workflow_update_status(workflow_id, "FAILED")
                return {"message": "the current job failed"}

            if current_job["status"] == "CANCELLED" or current_job["status"] == "DELETED":
                await db.workflow_update_with_new_job(workflow_id, "[]", "[]")
                await db.workflow_update_status(workflow_id, "CANCELLED")
                return {"message": "the current job was cancelled"}

            if current_job["status"] != "COMPLETE":
                return {"message": "the current job is running"}

    workflow_next_tasks = []
    # Determine the next task/node.
    if workflow_current_task != []:
        # We had a current task; find its outputs.
        for node in workflow_config["nodes"]:
            if node["id"] in workflow_current_task:
                workflow_next_tasks += node["out"]
                break  # Important: Exit the loop once found.
        else:  # This 'else' belongs to the 'for', executed if no 'break' occurred.
             await db.workflow_update_status(workflow_id, "FAILED")
             return {"message": "Could not find the current task in the workflow."}

    else:
        # No current task; start from the beginning.
        workflow_next_tasks = [workflow_config["nodes"][0]["id"]]  # Now a list

    workflow_current_task = workflow_next_tasks

    if not workflow_current_task:  # Check if the list is empty (end of workflow).
        await db.workflow_update_status(workflow_id, "COMPLETE")
        await db.workflow_update_with_new_job(workflow_id, "[]", "[]")  # Reset current job.
        return {"message": "Workflow Complete!"}

    next_nodes = []
    for node in workflow_config["nodes"]:
        if node["id"] in workflow_current_task:
            next_nodes.append(node)
    if len(next_nodes) == 0:
        await db.workflow_update_status(workflow_id, "FAILED")
        return {"message": "Could not find the next task in the workflow."}
    

    # --- Task Lookup and Job Creation ---
    if next_nodes[0]["type"] == "START":
        workflow_current_task = next_nodes[0]["out"]  # Skip the START node.
        if not workflow_current_task: #if the next node does not exist
            await db.workflow_update_status(workflow_id, "COMPLETE")
            await db.workflow_update_with_new_job(workflow_id, "[]", "[]")  # Reset current job.
            return {"message": "Workflow Complete!"}
        next_nodes = []
        for node in workflow_config["nodes"]:
            if node["id"] in workflow_current_task:
                next_nodes.append(node)
        if len(next_nodes) == 0:
            await db.workflow_update_status(workflow_id, "FAILED")
            return {"message": "Could not find the next task in the workflow."}

    print(next_nodes)

    next_job_ids = []
    for next_node in next_nodes:
        # Get the task definition.  Prioritize metadata.task_name, then node.type
        task_name = next_node["task"]

        next_task = None
        tasks = await db.tasks_get_all()
        for task in tasks:
            if(task["name"] == task_name):
                next_task = task
                break

        if not next_task:
            await db.workflow_update_status(workflow_id, "FAILED")
            return {"message": f"Could not find task '{task_name}' for workflow node."}

        if next_task["type"] == "TRAIN":
            next_task["outputs"] = json.loads(next_task["outputs"])
            next_task["outputs"]["adaptor_name"] = str(uuid.uuid4()).replace("-","")
            next_task["outputs"] = json.dumps(next_task["outputs"])
        if next_task["type"] == "EVAL":
            if current_job is not None:
                if current_job["type"] == "TRAIN":
                    next_task["inputs"] = json.loads(next_task["inputs"])
                    next_task["inputs"]["model_name"] = current_job["job_data"]["config"]["model_name"]
                    next_task["inputs"]["model_architecture"] = current_job["job_data"]["config"]["model_architecture"]
                    next_task["inputs"]["adaptor_name"] = current_job["job_data"]["config"]["adaptor_name"]
                    next_task["inputs"] = json.dumps(next_task["inputs"])

    
        next_job_info = await tsks.queue_task(next_task["id"], next_task["inputs"], next_task["outputs"])
        next_job_ids.append(next_job_info["id"])
    await db.workflow_update_with_new_job(workflow_id, json.dumps(workflow_current_task), json.dumps(next_job_ids))

    return {"message": "Next job created"}
