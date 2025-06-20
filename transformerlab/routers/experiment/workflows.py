from fastapi import APIRouter, UploadFile, Body
from fastapi.responses import FileResponse
import transformerlab.db as db
import transformerlab.routers.tasks as tsks
import json
import yaml
import uuid

from transformerlab.shared import dirs
from transformerlab.shared.shared import slugify

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.get("/list", summary="get all workflows in the experiment")
async def workflows_get_in_experiment(experimentId: int):
    workflows = await db.workflows_get_from_experiment(experimentId)
    return workflows


@router.get("/runs", summary="get all workflow runs in the experiment")
async def workflow_runs_get_in_experiment(experimentId: int):
    workflow_runs = await db.workflow_runs_get_from_experiment(experimentId)
    return workflow_runs


@router.get("/delete/{workflow_id}", summary="delete a workflow")
async def workflow_delete(workflow_id: str, experimentId: int):
    # Delete workflow with experiment enforcement at database level
    success = await db.workflow_delete_by_id(workflow_id, experimentId)
    if not success:
        return {"error": "Workflow not found or does not belong to this experiment"}
    return {"message": "OK"}


@router.get("/create", summary="Create a workflow from config")
async def workflow_create(name: str, config: str = '{"nodes":[]}', experimentId: int = 99):
    config = json.loads(config)
    if len(config["nodes"]) > 0:
        config["nodes"] = [
            {"type": "START", "id": str(uuid.uuid4()), "name": "START", "out": [config["nodes"][0]["id"]]}
        ] + config["nodes"]
    else:
        config["nodes"] = [{"type": "START", "id": str(uuid.uuid4()), "name": "START", "out": []}]
    workflow_id = await db.workflow_create(name, json.dumps(config), experimentId)
    return workflow_id


@router.get("/create_empty", summary="Create an empty workflow")
async def workflow_create_empty(name: str, experimentId: int = 99):
    name = slugify(name)
    config = {"nodes": [{"type": "START", "id": str(uuid.uuid4()), "name": "START", "out": []}]}
    workflow_id = await db.workflow_create(name, json.dumps(config), experimentId)
    return workflow_id


@router.get("/{workflow_id}/{node_id}/edit_node_metadata", summary="Edit metadata of a node in a workflow")
async def workflow_edit_node_metadata(workflow_id: str, node_id: str, metadata: str, experimentId: int):
    # Get workflow with experiment enforcement at database level
    workflow = await db.workflows_get_by_id(workflow_id, experimentId)
    if not workflow:
        return {"error": "Workflow not found or does not belong to this experiment"}

    config = json.loads(workflow["config"])
    for node in config["nodes"]:
        if node["id"] == node_id:
            node["metadata"] = json.loads(metadata)

    success = await db.workflow_update_config(workflow_id, json.dumps(config), experimentId)
    if not success:
        return {"error": "Failed to update workflow"}
    return {"message": "OK"}


@router.get("/{workflow_id}/update_name", summary="Update the name of a workflow")
async def workflow_update_name(workflow_id: str, new_name: str, experimentId: int):
    new_name = slugify(new_name)
    # Update workflow name with experiment enforcement at database level
    success = await db.workflow_update_name(workflow_id, new_name, experimentId)
    if not success:
        return {"error": "Workflow not found or does not belong to this experiment"}
    return {"message": "OK"}


@router.get("/{workflow_id}/add_node", summary="Add a node to a workflow")
async def workflow_add_node(workflow_id: str, node: str, experimentId: int):
    # Get workflow with experiment enforcement at database level
    workflow = await db.workflows_get_by_id(workflow_id, experimentId)
    if not workflow:
        return {"error": "Workflow not found or does not belong to this experiment"}

    new_node_json = json.loads(node)
    config = json.loads(workflow["config"])

    new_node_json["id"] = str(uuid.uuid4())
    new_node_json["out"] = []
    new_node_json["metadata"] = {}

    for node in config["nodes"]:
        if node["out"] == []:
            node["out"].append(new_node_json["id"])

    config["nodes"].append(new_node_json)

    success = await db.workflow_update_config(workflow_id, json.dumps(config), experimentId)
    if not success:
        return {"error": "Failed to update workflow"}
    return {"message": "OK"}


@router.post("/{workflow_id}/{node_id}/update_node", summary="Update a specific node in a workflow")
async def workflow_update_node(workflow_id: str, node_id: str, experimentId: int, new_node: dict = Body()):
    # Get workflow with experiment enforcement at database level
    workflow = await db.workflows_get_by_id(workflow_id, experimentId)
    if not workflow:
        return {"error": "Workflow not found or does not belong to this experiment"}

    config = json.loads(workflow["config"])
    newNodes = []

    for node in config["nodes"]:
        if node["id"] == node_id:
            newNodes.append(new_node)
        else:
            newNodes.append(node)

    config["nodes"] = newNodes
    success = await db.workflow_update_config(workflow_id, json.dumps(config), experimentId)
    if not success:
        return {"error": "Failed to update workflow"}
    return {"message": "OK"}


@router.post("/{workflow_id}/{start_node_id}/remove_edge", summary="Remove an edge between two nodes in a workflow")
async def workflow_remove_edge(workflow_id: str, start_node_id: str, end_node_id: str, experimentId: int):
    # Get workflow with experiment enforcement at database level
    workflow = await db.workflows_get_by_id(workflow_id, experimentId)
    if not workflow:
        return {"error": "Workflow not found or does not belong to this experiment"}

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
    success = await db.workflow_update_config(workflow_id, json.dumps(config), experimentId)
    if not success:
        return {"error": "Failed to update workflow"}
    return {"message": "OK"}


@router.post("/{workflow_id}/{start_node_id}/add_edge", summary="Add an edge between two nodes in a workflow")
async def workflow_add_edge(workflow_id: str, start_node_id: str, end_node_id: str, experimentId: int):
    # Get workflow with experiment enforcement at database level
    workflow = await db.workflows_get_by_id(workflow_id, experimentId)
    if not workflow:
        return {"error": "Workflow not found or does not belong to this experiment"}

    config = json.loads(workflow["config"])
    newNodes = []

    for node in config["nodes"]:
        if node["id"] == start_node_id:
            node["out"].append(end_node_id)
            newNodes.append(node)
        else:
            newNodes.append(node)

    config["nodes"] = newNodes
    success = await db.workflow_update_config(workflow_id, json.dumps(config), experimentId)
    if not success:
        return {"error": "Failed to update workflow"}
    return {"message": "OK"}


@router.get("/{workflow_id}/{node_id}/delete_node", summary="Delete a node from a workflow")
async def workflow_delete_node(workflow_id: str, node_id: str, experimentId: int):
    # Get workflow with experiment enforcement at database level
    workflow = await db.workflows_get_by_id(workflow_id, experimentId)
    if not workflow:
        return {"error": "Workflow not found or does not belong to this experiment"}

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
    success = await db.workflow_update_config(workflow_id, json.dumps(config), experimentId)
    if not success:
        return {"error": "Failed to update workflow"}
    return {"message": "OK"}


@router.get("/{workflow_id}/export_to_yaml", summary="Export a workflow definition to YAML")
async def workflow_export_to_yaml(workflow_id: str, experimentId: int):
    # Get workflow with experiment enforcement at database level
    workflow = await db.workflows_get_by_id(workflow_id, experimentId)
    if not workflow:
        return {"error": "Workflow not found or does not belong to this experiment"}

    for field in (
        "current_job_id",
        "current_task",
        "experiment_id",
        "created_at",
        "updated_at",
        "status",
        "id",
    ):
        workflow.pop(field, None)

    workflow["config"] = json.loads(workflow["config"])

    filename = f"{workflow['name']}.yaml"
    with open(filename, "w") as yaml_file:
        yaml.dump(workflow, yaml_file)
    return FileResponse(filename, filename=filename)


@router.post("/import_from_yaml", summary="Import a workflow definition from YAML")
async def workflow_import_from_yaml(file: UploadFile, experimentId: int = 99):
    with open(file.filename, "r") as fileStream:
        workflow = yaml.load(fileStream, Loader=yaml.BaseLoader)
    await db.workflow_create(workflow["name"], json.dumps(workflow["config"]), experimentId)
    return {"message": "OK"}


@router.get("/{workflow_id}/start", summary="Queue a workflow to start execution")
async def start_workflow(workflow_id: str, experimentId: int):
    # Verify workflow exists and belongs to experiment
    workflow = await db.workflows_get_by_id(workflow_id, experimentId)
    if not workflow:
        return {"error": "Workflow not found or does not belong to this experiment"}

    await db.workflow_queue(workflow_id)
    return {"message": "OK"}


async def start_next_step_in_workflow():
    # 1. Find the active workflow run (Running or next Queued)
    active_run = await get_active_workflow_run()
    if not active_run:
        return {"message": "No workflow is running or queued."}

    # 2. Load context for the active run
    workflow_run_id, workflow_id, workflow_config, current_tasks, current_job_ids = await load_workflow_context(
        active_run
    )

    # 3. Check status of jobs from the *previous* step (if any)
    job_status_message = await check_current_jobs_status(workflow_run_id, current_job_ids)
    if job_status_message:
        return {"message": job_status_message}

    # --- Jobs from previous step are complete, determine next step ---

    # 4. Determine the next task(s) based on completed ones
    next_task_ids = await determine_next_tasks(current_tasks, workflow_config, workflow_run_id)

    if next_task_ids is None:
        return {"message": "Failed to determine next task due to an error."}

    if not next_task_ids:
        # Workflow is complete
        await db.workflow_run_update_status(workflow_run_id, "COMPLETE")
        await db.workflow_run_update_with_new_job(workflow_run_id, "[]", "[]")  # Clear current tasks/jobs
        return {"message": "Workflow Complete!"}

    # 5. Handle START node and get the actual next nodes
    actual_next_task_ids, next_nodes = await handle_start_node_skip(next_task_ids, workflow_config, workflow_run_id)

    if actual_next_task_ids is None:  # Indicates an error occurred in _handle_start_node_skip
        return {"message": "Failed processing potential START node."}

    if not actual_next_task_ids:  # Can happen if START node had no output
        await db.workflow_run_update_status(workflow_run_id, "COMPLETE")
        await db.workflow_run_update_with_new_job(workflow_run_id, "[]", "[]")
        return {"message": "Workflow Complete (ended after START node)!"}

    # 6. Queue jobs for the next node(s)
    next_job_ids = []
    for node in next_nodes:
        # Pass necessary context to the queuing function
        new_job_info = await queue_job_for_node(node, active_run, workflow_config)
        if new_job_info and "id" in new_job_info:
            next_job_ids.append(new_job_info["id"])
        else:
            return {"message": f"Failed to queue job for task '{node.get('task', 'UNKNOWN')}'."}

    # 7. Update workflow run state with new current tasks and job IDs
    await db.workflow_run_update_with_new_job(
        workflow_run_id, json.dumps(actual_next_task_ids), json.dumps(next_job_ids)
    )

    return {"message": f"Started next step with job(s): {next_job_ids}"}


@router.get("/runs/{workflow_run_id}", summary="get a specific workflow run by id")
async def workflow_runs_get_by_id(workflow_run_id: str, experimentId: int):
    workflow_run = await db.workflow_run_get_by_id(workflow_run_id)
    if not workflow_run:
        return {"error": "Workflow run not found"}

    # Verify workflow belongs to experiment
    workflow = await db.workflows_get_by_id(workflow_run.get("workflow_id"), experimentId)
    if not workflow:
        return {"error": "Associated workflow not found or does not belong to this experiment"}

    returnInfo = {"run": workflow_run, "workflow": workflow, "jobs": []}

    try:
        job_ids = json.loads(workflow_run.get("job_ids", "[]"))
        for job_id in job_ids:
            job = await db.job_get(job_id)
            if not job:
                continue

            job_info = {"jobID": job_id, "status": job.get("status", "UNKNOWN")}

            # Safely get job data
            job_data = job.get("job_data", {})
            if not isinstance(job_data, dict):
                job_data = {}

            # Safely extract fields
            if job_data.get("template_name"):
                job_info["taskName"] = job_data["template_name"]
            if job_data.get("start_time"):
                job_info["jobStartTime"] = job_data["start_time"]
            if job_data.get("end_time"):
                job_info["jobEndTime"] = job_data["end_time"]

            returnInfo["jobs"].append(job_info)

    except json.JSONDecodeError:
        # If job_ids JSON is invalid, return empty jobs list
        pass

    return returnInfo


# Helper functions moved from old workflows.py
async def get_active_workflow_run():
    num_running_workflows = await db.workflow_count_running()
    num_queued_workflows = await db.workflow_count_queued()

    if num_running_workflows + num_queued_workflows == 0:
        return None  # No workflows to process

    active_run = await db.workflow_run_get_running()
    # if we have no currenlty active workflow run, then run a queued one
    if active_run is None:
        active_run = await db.workflow_run_get_queued()
        if active_run:
            await db.workflow_run_update_status(active_run["id"], "RUNNING")

    return active_run


async def load_workflow_context(active_run):
    workflow_run_id = active_run["id"]
    workflow_id = active_run["workflow_id"]
    experiment_id = active_run["experiment_id"]  # Get experiment_id from active_run
    workflow = await db.workflows_get_by_id(workflow_id, experiment_id)
    workflow_config = json.loads(workflow["config"])
    current_tasks = json.loads(active_run["current_tasks"])  # List of task IDs (node IDs)
    current_job_ids = json.loads(active_run["current_job_ids"])  # List of job IDs

    return workflow_run_id, workflow_id, workflow_config, current_tasks, current_job_ids


async def check_current_jobs_status(workflow_run_id, current_job_ids):
    """
    Checks the status of currently running jobs for the workflow step.
    Updates workflow status if a job failed or was cancelled.
    Returns None if all jobs are complete, or a message string if waiting, failed, or cancelled.
    """
    if not current_job_ids:
        return None

    for job_id in current_job_ids:
        current_job = await db.job_get(job_id)
        if not current_job:
            await db.workflow_run_update_status(workflow_run_id, "FAILED")
            return f"Could not find job with ID {job_id}"

        status = current_job["status"]

        if status == "FAILED":
            await db.workflow_run_update_status(workflow_run_id, "FAILED")
            return "The current job failed"

        if status in ["CANCELLED", "DELETED", "STOPPED"]:
            await db.workflow_run_update_with_new_job(workflow_run_id, "[]", "[]")
            await db.workflow_run_update_status(workflow_run_id, "CANCELLED")
            return "The current job was cancelled/stopped/deleted"

        if status != "COMPLETE":
            return "The current job is running"

    return None


def find_nodes_by_ids(node_ids, all_nodes):
    """Finds node configuration dictionaries based on a list of node IDs."""
    return [node for node in all_nodes if node.get("id") in node_ids]


async def determine_next_tasks(current_tasks, workflow_config, workflow_run_id):
    """
    Determines the IDs of the next tasks based on the current tasks and workflow config.
    Returns a list of next task IDs, an empty list if workflow is complete, or None on error.
    """
    all_nodes = workflow_config.get("nodes", [])
    next_task_ids = []

    if current_tasks:
        current_nodes = find_nodes_by_ids(current_tasks, all_nodes)
        if not current_nodes:
            await db.workflow_run_update_status(workflow_run_id, "FAILED")
            return None

        for node in current_nodes:
            next_task_ids += node["out"]

    else:
        if all_nodes:
            next_task_ids = [all_nodes[0]["id"]]

    return next_task_ids


async def handle_start_node_skip(next_task_ids, workflow_config, workflow_run_id):
    """
    Checks if the next task is a START node. If so, finds the subsequent tasks.
    Returns the actual next task IDs and their corresponding node dicts.
    Returns (None, None) on error.
    """
    all_nodes = workflow_config.get("nodes", [])
    next_nodes = find_nodes_by_ids(next_task_ids, all_nodes)

    if not next_nodes and next_task_ids:
        await db.workflow_run_update_status(workflow_run_id, "FAILED")
        return None, None

    # Handle potential multiple start nodes or parallel paths from start
    final_next_task_ids = []
    processed_start_nodes = False

    temp_next_nodes = []
    for node in next_nodes:
        if node.get("type") == "START":
            processed_start_nodes = True
            final_next_task_ids.extend(node.get("out", []))
        else:
            temp_next_nodes.append(node)
            final_next_task_ids.append(node["id"])

    if processed_start_nodes:
        if not final_next_task_ids:
            return [], []

        next_nodes = find_nodes_by_ids(final_next_task_ids, all_nodes)
        if not next_nodes:
            await db.workflow_run_update_status(workflow_run_id, "FAILED")
            return None, None
        return final_next_task_ids, next_nodes
    else:
        return next_task_ids, next_nodes


async def find_task_definition(task_name: str, workflow_run_id: int):
    """Finds the task definition from the database by name."""
    tasks = await db.tasks_get_all()
    for task in tasks:
        if task.get("name") == task_name:
            return task

    # if the task isnt found
    await db.workflow_run_update_status(workflow_run_id, "FAILED")
    return None


async def find_previous_node_and_job(current_node, workflow_run, workflow_config):
    """Finds the job corresponding to the node that preceded the current_node."""

    all_nodes = workflow_config.get("nodes", [])
    previous_job = None

    # Find nodes that have current_node["id"] in their "out" list
    potential_previous_nodes = [node for node in all_nodes if current_node.get("id") in node.get("out", [])]

    if not potential_previous_nodes:
        return None

    if potential_previous_nodes:
        # Taking the first predecessor found. Needs improvement for multiple inputs.
        previous_node = potential_previous_nodes[0]
        if previous_node.get("type") != "START":
            ran_nodes_str = workflow_run.get("node_ids", "[]")
            ran_jobs_str = workflow_run.get("job_ids", "[]")
            ran_nodes = json.loads(ran_nodes_str)
            ran_jobs = json.loads(ran_jobs_str)
            if previous_node["id"] in ran_nodes:
                previous_job_ID = ran_jobs[ran_nodes.index(previous_node["id"])]
                previous_job = await db.job_get(previous_job_ID)

    return previous_job


def extract_previous_job_outputs(previous_job):
    """Extracts relevant output information from a completed job."""
    outputs = {}
    if previous_job is None or "job_data" not in previous_job or not previous_job["job_data"]:
        return outputs

    job_data = previous_job["job_data"]  # Assuming job_data is already a dict
    job_type = previous_job.get("type")
    job_config = job_data.get("config", {})  # Handle missing config safely

    fuse_pretext = dirs.MODELS_DIR + "/" if hasattr(dirs, "MODELS_DIR") else ""

    if job_type == "GENERATE":
        # Prefer dataset_id from top-level job_data if present
        dataset_id = job_data.get("dataset_id") or job_config.get("dataset_id")
        if dataset_id:
            outputs["dataset_name"] = str(dataset_id).lower().replace(" ", "-")

    elif job_type == "TRAIN":
        model_name = job_config.get("model_name")
        adaptor_name = job_config.get("adaptor_name")
        model_architecture = job_config.get("model_architecture")

        if job_config.get("fuse_model") and model_name and adaptor_name:
            model_base_name = model_name.split("/")[-1]
            outputs["model_name"] = f"{fuse_pretext}{model_base_name}_{adaptor_name}"
        elif model_name:
            outputs["model_name"] = model_name

        if model_architecture:
            outputs["model_architecture"] = model_architecture
        if adaptor_name and not job_config.get("fuse_model"):
            outputs["adaptor_name"] = adaptor_name

    return outputs


def prepare_next_task_io(task_def: dict, previous_outputs: dict):
    """Prepares the JSON input and output strings for the next task."""
    inputs = json.loads(task_def.get("inputs", "{}"))
    outputs = json.loads(task_def.get("outputs", "{}"))
    task_type = task_def.get("type")

    # Map previous outputs to next inputs
    if task_type == "EVAL":
        # Map relevant outputs to specific input fields
        for key in ["model_name", "model_architecture", "adaptor_name", "dataset_name"]:
            if key in previous_outputs:
                inputs[key] = previous_outputs[key]

    elif task_type == "TRAIN":
        # Map relevant outputs to specific input fields
        for key in ["model_name", "model_architecture", "dataset_name"]:
            if key in previous_outputs:
                inputs[key] = previous_outputs[key]
        # Generate dynamic output fields
        outputs["adaptor_name"] = str(uuid.uuid4()).replace("-", "")

    elif task_type == "GENERATE":
        # Generate dynamic output fields
        outputs["dataset_id"] = str(uuid.uuid4()).replace("-", "")

    return json.dumps(inputs), json.dumps(outputs)


async def queue_job_for_node(node: dict, workflow_run: dict, workflow_config: dict):
    """Finds task def, prepares IO, and queues a job for a given workflow node."""
    task_name = node.get("task")
    if not task_name:
        return None

    task_def = await find_task_definition(task_name, workflow_run["id"])
    if not task_def:
        return None

    # Find the job that ran *before* this node to get its outputs
    previous_job = await find_previous_node_and_job(node, workflow_run, workflow_config)

    # Extract outputs from the previous job
    previous_outputs = extract_previous_job_outputs(previous_job)

    # Prepare inputs and outputs for the new job
    inputs_json, outputs_json = prepare_next_task_io(task_def, previous_outputs)

    # Queue the task
    try:
        task_id_to_queue = task_def.get("id")
        if task_id_to_queue is None:
            await db.workflow_run_update_status(workflow_run["id"], "FAILED")
            return None

        queued_job_info = await tsks.queue_task(task_id_to_queue, inputs_json, outputs_json)
        if not queued_job_info or "id" not in queued_job_info:
            return None
        return queued_job_info
    except Exception as e:
        print(f"Error queueing task {task_name}: {e}")
        await db.workflow_run_update_status(workflow_run["id"], "FAILED")
        return None
