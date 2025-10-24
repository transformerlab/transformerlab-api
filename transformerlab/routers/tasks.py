import json
import time

from fastapi import APIRouter, Body, Request, Form
from typing import Optional
import os
import tempfile
import shutil
import json as json_lib
import subprocess
import httpx
from werkzeug.utils import secure_filename

from lab import Dataset
from transformerlab.services.job_service import job_create
from transformerlab.models import model_helper
from transformerlab.services.tasks_service import tasks_service
from transformerlab.shared import galleries
from transformerlab.shared.shared import slugify
from lab.dirs import get_workspace_dir

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/list", summary="Returns all the tasks")
async def tasks_get_all():
    tasks = tasks_service.tasks_get_all()
    return tasks


@router.get("/{task_id}/get", summary="Gets all the data for a single task")
async def tasks_get_by_id(task_id: str):
    task = tasks_service.tasks_get_by_id(task_id)
    if task is None:
        return {"message": "NOT FOUND"}
    return task


@router.get("/list_by_type", summary="Returns all the tasks of a certain type, e.g TRAIN")
async def tasks_get_by_type(type: str):
    tasks = tasks_service.tasks_get_by_type(type)
    return tasks


@router.get(
    "/list_by_type_in_experiment", summary="Returns all the tasks of a certain type in a certain experiment, e.g TRAIN"
)
async def tasks_get_by_type_in_experiment(type: str, experiment_id: str):
    tasks = tasks_service.tasks_get_by_type_in_experiment(type, experiment_id)
    return tasks


@router.put("/{task_id}/update", summary="Updates a task with new information")
async def update_task(task_id: str, new_task: dict = Body()):
    # Perform secure_filename before updating the task
    if "name" in new_task:
        new_task["name"] = secure_filename(new_task["name"])
    success = tasks_service.update_task(task_id, new_task)
    if success:
        return {"message": "OK"}
    else:
        return {"message": "NOT FOUND"}


@router.get("/{task_id}/delete", summary="Deletes a task")
async def delete_task(task_id: str):
    success = tasks_service.delete_task(task_id)
    if success:
        return {"message": "OK"}
    else:
        return {"message": "NOT FOUND"}


@router.put("/new_task", summary="Create a new task")
async def add_task(new_task: dict = Body()):
    # Perform secure_filename before adding the task
    new_task["name"] = secure_filename(new_task["name"])
    # Support optional remote_task flag to mark remote task templates
    remote_task_flag = False
    try:
        remote_task_flag = bool(new_task.get("remote_task", False))
    except Exception:
        remote_task_flag = False

    tasks_service.add_task(
        new_task["name"],
        new_task["type"],
        new_task["inputs"],
        new_task["config"],
        new_task["plugin"],
        new_task["outputs"],
        new_task["experiment_id"],
        remote_task=remote_task_flag,
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
        local_datasets = Dataset.list_all()
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
    tasks_service.tasks_delete_all()
    return {"message": "OK"}


@router.get("/{task_id}/queue", summary="Queue a task to run")
async def queue_task(task_id: str, input_override: str = "{}", output_override: str = "{}"):
    task_to_queue = await tasks_service.tasks_get_by_id(task_id)
    if task_to_queue is None:
        return {"message": "TASK NOT FOUND"}
    
    # Skip remote tasks - they are handled by the launch_remote route, not the job queue
    if task_to_queue.get("remote_task", False):
        return {"message": "REMOTE TASK - Cannot queue remote tasks, use launch_remote endpoint instead"}
    
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
    job_id = job_create(
        type=("EXPORT" if job_type == "EXPORT" else job_type),
        status=job_status,
        experiment_id=task_to_queue["experiment_id"],
        job_data=json.dumps(job_data),
    )
    return {"id": job_id}


@router.get("/gallery", summary="Returns task gallery entries from remote cache")
async def tasks_gallery_list():
    """List tasks available in the remote galleries index."""
    try:
        gallery = galleries.get_tasks_gallery()
        return {"status": "success", "data": gallery}
    except Exception as e:
        print(f"Error fetching task gallery: {e}")
        return {"status": "error", "message": "An error occurred while fetching the task gallery"}


@router.get("/local_gallery", summary="Returns local tasks-gallery entries from workspace")
async def tasks_local_gallery_list():
    """List tasks available in the local workspace tasks-gallery directory."""
    try:
        workspace_dir = get_workspace_dir()
        local_gallery_dir = os.path.join(workspace_dir, "tasks-gallery")
        
        if not os.path.exists(local_gallery_dir):
            return {"status": "success", "data": []}
        
        local_tasks = []
        for item in os.listdir(local_gallery_dir):
            task_dir = os.path.join(local_gallery_dir, item)
            if os.path.isdir(task_dir):
                task_json_path = os.path.join(task_dir, "task.json")
                if os.path.isfile(task_json_path):
                    try:
                        with open(task_json_path) as f:
                            task_data = json_lib.load(f)
                        local_tasks.append({
                            "name": task_data.get("name", item),
                            "description": task_data.get("description", ""),
                            "task_dir": item,
                            "source": "local",
                            "tag": task_data.get("tag", "OTHER")
                        })
                    except Exception as e:
                        print(f"Error reading {task_json_path}: {e}")
                        continue
        
        return {"status": "success", "data": local_tasks}
    except Exception as e:
        print(f"Error fetching local task gallery: {e}")
        return {"status": "error", "message": "An error occurred while fetching the local task gallery"}


@router.delete("/local_gallery/{task_dir}", summary="Delete a task from local tasks-gallery")
async def delete_task_from_local_gallery(task_dir: str):
    """
    Delete a task from the local tasks-gallery directory.
    """
    try:
        workspace_dir = get_workspace_dir()
        local_gallery_dir = os.path.join(workspace_dir, "tasks-gallery")
        task_path = os.path.normpath(os.path.join(local_gallery_dir, task_dir))
        
        # Security check: ensure the task path is within the local gallery directory
        local_gallery_dir_real = os.path.realpath(local_gallery_dir)
        task_path_real = os.path.realpath(task_path)
        common_path = os.path.commonpath([local_gallery_dir_real, task_path_real])
        
        if common_path != local_gallery_dir_real:
            return {"status": "error", "message": "Invalid task directory"}
        
        if not os.path.exists(task_path):
            return {"status": "error", "message": "Task directory not found"}
        
        # Remove the task directory
        shutil.rmtree(task_path)
        
        return {"status": "success", "message": f"Task '{task_dir}' deleted successfully"}
        
    except Exception as e:
        print(f"Error deleting task from local gallery: {e}")
        return {"status": "error", "message": "An error occurred while deleting the task"}


@router.get("/local_gallery/{task_dir}/files", summary="Get files for a task in local tasks-gallery")
async def get_task_files(task_dir: str):
    """
    Get the list of files in the src/ directory of a task in the local tasks-gallery.
    """
    try:
        workspace_dir = get_workspace_dir()
        local_gallery_dir = os.path.join(workspace_dir, "tasks-gallery")
        task_path = os.path.normpath(os.path.join(local_gallery_dir, task_dir))
        
        # Security check: ensure the task path is within the local gallery directory
        local_gallery_dir_real = os.path.realpath(local_gallery_dir)
        task_path_real = os.path.realpath(task_path)
        common_path = os.path.commonpath([local_gallery_dir_real, task_path_real])
        
        if common_path != local_gallery_dir_real:
            return {"status": "error", "message": "Invalid task directory"}
        
        if not os.path.exists(task_path):
            return {"status": "error", "message": "Task directory not found"}
        
        # Check for src directory
        src_dir = os.path.join(task_path, "src")
        if not os.path.exists(src_dir):
            return {"status": "success", "data": {"files": [], "count": 0}}
        
        # Get all files in src directory recursively
        files = []
        for root, dirs, filenames in os.walk(src_dir):
            for filename in filenames:
                # Get relative path from src directory
                rel_path = os.path.relpath(os.path.join(root, filename), src_dir)
                files.append(rel_path)
        
        return {"status": "success", "data": {"files": files, "count": len(files)}}
        
    except Exception as e:
        print(f"Error getting task files: {e}")
        return {"status": "error", "message": "An error occurred while getting task files"}


@router.post("/import_from_gallery", summary="Import a task from transformerlab/galleries tasks subdirectory")
async def import_task_from_gallery(
    request: Request,
    id: str = Form(...),
    experiment_id: Optional[str] = Form(None),
    upload: Optional[bool] = Form(True),
    repo_url: Optional[str] = Form(None),
):
    """
    Clone the specific tasks/<id> from transformerlab/galleries and create a REMOTE task using task.json.
    This function now installs the task locally first, then creates a task in the experiment.
    """
    # First, install the task locally
    install_result = await install_task_from_gallery(id, repo_url)
    if install_result.get("status") != "success":
        return install_result
    
    task_dir_name = install_result.get("task_dir")
    if not task_dir_name:
        return {"status": "error", "message": "Failed to get task directory name from installation"}
    
    # Now create the task in the experiment using the locally installed task
    try:
        workspace_dir = get_workspace_dir()
        local_gallery_dir = os.path.join(workspace_dir, "tasks-gallery")
        local_task_dir = os.path.join(local_gallery_dir, task_dir_name)
        task_json_path = os.path.join(local_task_dir, "task.json")
        
        if not os.path.isfile(task_json_path):
            return {"status": "error", "message": f"task.json not found in local task directory: {task_dir_name}"}

        with open(task_json_path) as f:
            task_def = json_lib.load(f)

        # Build task fields, marking as remote
        task_name = slugify(task_def.get("name", task_dir_name))
        task_type = task_def.get("type", "REMOTE")
        inputs = task_def.get("inputs", {})
        config = task_def.get("config", {})
        plugin = task_def.get("plugin", "remote_task")
        outputs = task_def.get("outputs", {})

        # Check if task already exists and update instead of creating duplicate
        existing_tasks = tasks_service.tasks_get_all()
        existing_task = None
        for task in existing_tasks:
            if task.get("name") == task_name and task.get("remote_task", False):
                existing_task = task
                break

        if existing_task:
            # Update existing task
            task_id = existing_task["id"]
            tasks_service.update_task(task_id, {
                "name": task_name,
                "inputs": inputs,
                "config": config,
                "outputs": outputs,
                "plugin": plugin
            })
        else:
            # Create new task
            task_id = tasks_service.add_task(
                name=task_name,
                task_type=task_type,
                inputs=inputs,
                config=config,
                plugin=plugin,
                outputs=outputs,
                experiment_id=experiment_id,
                remote_task=True,
            )

        # Optional: Upload files to GPU orchestrator if requested
        if upload:
            try:
                # Get the src directory from local installation
                src_dir = os.path.join(local_task_dir, "src")
                if os.path.exists(src_dir):
                    # Validate that src_dir is within the trusted tasks-gallery location
                    canonical_src_dir = os.path.normpath(os.path.realpath(src_dir))
                    trusted_gallery_root = os.path.normpath(os.path.realpath(os.path.join(get_workspace_dir(), "tasks-gallery")))
                     # Ensure src_dir is strictly contained within trusted gallery root directory using commonpath
                    if os.path.commonpath([trusted_gallery_root, canonical_src_dir]) != trusted_gallery_root:
                        return {"status": "error", "message": "Attempted upload from outside trusted gallery directory"}
                    # Post to GPU orchestrator upload endpoint
                    gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATION_SERVER")
                    gpu_orchestrator_port = os.getenv("GPU_ORCHESTRATION_SERVER_PORT")
                    
                    if not gpu_orchestrator_url or not gpu_orchestrator_port:
                        # If orchestrator not configured, just attach local path hint
                        try:
                            task_obj = tasks_service.tasks_get_by_id(task_id)
                            if isinstance(task_obj.get("config"), str):
                                task_obj["config"] = json_lib.loads(task_obj["config"]) if task_obj["config"] else {}
                            task_obj["config"]["local_upload_staged_dir"] = src_dir
                            tasks_service.update_task(task_id, {"config": task_obj["config"]})
                        except Exception:
                            pass
                    else:
                        # Build multipart form to mirror frontend DirectoryUpload
                        dest = f"{gpu_orchestrator_url}:{gpu_orchestrator_port}/api/v1/instances/upload"
                        files_form = []
                        # Walk src_dir and add each file, preserving relative path inside src/
                        for root, _, filenames in os.walk(src_dir):
                            for filename in filenames:
                                full_path = os.path.join(root, filename)
                                rel_path = os.path.relpath(full_path, src_dir)
                                # Prefix with src/ like the packed structure
                                upload_name = f"src/{rel_path}"
                                with open(full_path, "rb") as f:
                                    files_form.append(("dir_files", (upload_name, f.read(), "application/octet-stream")))

                        form_data = {"dir_name": slugify(task_name)}

                        async with httpx.AsyncClient(timeout=120.0) as client:
                            headers = {}
                            incoming_auth = request.headers.get("AUTHORIZATION")
                            if incoming_auth:
                                headers["AUTHORIZATION"] = incoming_auth

                            resp = await client.post(
                                dest,
                                headers=headers,
                                files=files_form,
                                data=form_data,
                                cookies=request.cookies,
                            )
                            if resp.status_code == 200:
                                remote_info = resp.json()
                                # Try to read path similar to frontend handler
                                remote_path = (
                                    remote_info.get("data", {})
                                    .get("uploaded_files", {})
                                    .get("dir_files", {})
                                    .get("uploaded_dir")
                                )
                                if not remote_path:
                                    remote_path = remote_info.get("path") or remote_info.get("remote_path") or remote_info
                                try:
                                    task_obj = tasks_service.tasks_get_by_id(task_id)
                                    if isinstance(task_obj.get("config"), str):
                                        task_obj["config"] = json_lib.loads(task_obj["config"]) if task_obj["config"] else {}
                                    task_obj["config"]["remote_upload_path"] = remote_path
                                    tasks_service.update_task(task_id, {"config": task_obj["config"]})
                                except Exception:
                                    pass
                            else:
                                return {"status": "error", "message": f"Upload failed: {resp.status_code} {resp.text}"}

            except Exception as e:
                print(f"Upload exception: {e}")
                return {"status": "error", "message": "An error occurred while uploading the task"}

        return {"status": "success", "task_id": task_id}
    except Exception as e:
        print(f"Error importing task from gallery: {e}")
        return {"status": "error", "message": "An error occurred while importing the task from the gallery"}


@router.post("/install_from_gallery", summary="Install a task from transformerlab/galleries to local tasks-gallery")
async def install_task_from_gallery(
    id: str = Form(...),
    repo_url: Optional[str] = Form(None),
):
    """
    Clone the specific tasks/<id> from transformerlab/galleries and store it in workspace/tasks-gallery/.
    This installs the task locally without creating a task in any experiment.
    """
    # Prepare temp directory for shallow clone of specific path
    remote_repo_url = "https://github.com/transformerlab/galleries.git"
    tmp_dir = tempfile.mkdtemp(prefix="tlab_tasks_gallery_")
    try:
        # Sparse checkout only the requested task
        subprocess.check_call(["git", "init"], cwd=tmp_dir)
        subprocess.check_call(["git", "remote", "add", "origin", remote_repo_url], cwd=tmp_dir)
        subprocess.check_call(["git", "config", "core.sparseCheckout", "true"], cwd=tmp_dir)
        sparse_info_dir = os.path.join(tmp_dir, ".git", "info")
        os.makedirs(sparse_info_dir, exist_ok=True)
        with open(os.path.join(sparse_info_dir, "sparse-checkout"), "w") as f:
            f.write(f"tasks/{id}\n")
        subprocess.check_call(["git", "pull", "--depth", "1", "origin", "main"], cwd=tmp_dir)

        # Validate id: reject traversal and normalize path
        if os.path.isabs(id) or ".." in id or "/" in id or "\\" in id or not id.strip():
            return {"status": "error", "message": "Invalid task id"}
        base_tasks_dir = os.path.join(tmp_dir, "tasks")
        task_dir = os.path.normpath(os.path.join(base_tasks_dir, id))
        # Make sure the resolved path is within the expected tasks dir
        if not task_dir.startswith(base_tasks_dir + os.sep):
            return {"status": "error", "message": "Invalid task directory"}
        task_json_path = os.path.join(task_dir, "task.json")
        if not os.path.isfile(task_json_path):
            return {"status": "error", "message": f"task.json not found in tasks/{id}"}

        with open(task_json_path) as f:
            task_def = json_lib.load(f)

        # Create local tasks-gallery directory structure
        workspace_dir = get_workspace_dir()
        local_gallery_dir = os.path.join(workspace_dir, "tasks-gallery")
        os.makedirs(local_gallery_dir, exist_ok=True)
        
        # Create task directory in local gallery
        task_name = slugify(task_def.get("name", id))
        task_dir_name = slugify(task_name)
        local_task_dir = os.path.join(local_gallery_dir, task_dir_name)
        
        # Check if task already exists locally
        if os.path.exists(local_task_dir):
            return {"status": "error", "message": f"Task '{task_name}' is already installed locally"}
        
        os.makedirs(local_task_dir, exist_ok=True)
        
        # Copy task.json to local gallery
        local_task_json_path = os.path.join(local_task_dir, "task.json")
        shutil.copy2(task_json_path, local_task_json_path)
        
        # Copy all other files to local gallery (excluding task.json)
        src_dir = os.path.join(local_task_dir, "src")
        os.makedirs(src_dir, exist_ok=True)
        
        files_to_copy = [f for f in os.listdir(task_dir) if f != "task.json"]
        for name in files_to_copy:
            src_path = os.path.join(task_dir, name)
            dest_path = os.path.join(src_dir, name)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)
        
        # Create metadata file for installation info
        metadata = {
            "installed_from": "gallery",
            "gallery_id": id,
            "install_date": json_lib.dumps({"$date": {"$numberLong": str(int(time.time() * 1000))}}),
            "version": task_def.get("version", "1.0.0")
        }
        metadata_path = os.path.join(local_task_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json_lib.dump(metadata, f, indent=2)
        
        return {"status": "success", "task_dir": task_dir_name, "message": f"Task '{task_name}' installed successfully"}
        
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")
        return {"status": "error", "message": "An error occurred while installing the task from the gallery"}
    except Exception as e:
        print(f"Error installing task from gallery: {e}")
        return {"status": "error", "message": "An error occurred while installing the task from the gallery"}
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


@router.post("/export_to_local_gallery", summary="Import a REMOTE task to local tasks-gallery")
async def export_task_to_local_gallery(
    request: Request,
    task_name: str = Form(...),
    description: str = Form(...),
    source_task_id: str = Form(...),
    tag: str = Form("OTHER"),
    experiment_id: Optional[str] = Form(None),
):
    """
    Import an existing REMOTE task to the local tasks-gallery directory.
    Creates <task_name>/task.json in WORKSPACE_DIR/tasks-gallery/.
    """
    try:
        # Get the source task
        source_task = tasks_service.tasks_get_by_id(source_task_id)
        if not source_task:
            return {"status": "error", "message": f"Source task {source_task_id} not found"}
        
        if not source_task.get("remote_task", False):
            return {"status": "error", "message": "Source task must be a REMOTE task"}
        
        # Create local gallery directory structure
        workspace_dir = get_workspace_dir()
        local_gallery_dir = os.path.join(workspace_dir, "tasks-gallery")
        os.makedirs(local_gallery_dir, exist_ok=True)
        
        # Create task directory
        task_dir_name = slugify(task_name)
        task_dir = os.path.join(local_gallery_dir, task_dir_name)
        os.makedirs(task_dir, exist_ok=True)
        
        # Create task.json for local gallery
        local_task_data = {
            "name": task_name,
            "description": description,
            "type": "REMOTE",
            "plugin": source_task.get("plugin", "remote_task"),
            "inputs": source_task.get("inputs", {}),
            "outputs": source_task.get("outputs", {}),
            "config": source_task.get("config", {}),
            "source": "local_gallery",
            "imported_from": source_task_id,
            "tag": tag
        }
        
        task_json_path = os.path.join(task_dir, "task.json")
        with open(task_json_path, "w") as f:
            json_lib.dump(local_task_data, f, indent=2)
        
        # Copy files from local storage if they exist
        src_dir = os.path.join(task_dir, "src")
        os.makedirs(src_dir, exist_ok=True)
        
        # Check if source task has local files stored
        source_config = source_task.get("config", {})
        if isinstance(source_config, str):
            try:
                source_config = json_lib.loads(source_config)
            except Exception as e:
                print(f"Error loading source config: {e}")
                source_config = {}
        
        # Check for local upload files in both possible config fields
        local_upload_staged_dir = source_config.get("local_upload_staged_dir")
        local_upload_copy = source_config.get("local_upload_copy")
        
        # local_upload_copy is just a folder name, we need to construct the full path
        if local_upload_copy:
            workspace_dir = get_workspace_dir()
            local_upload_copy = os.path.join(workspace_dir, "uploads", local_upload_copy)
        
        local_files_dir = local_upload_staged_dir or local_upload_copy
        if local_files_dir and os.path.exists(local_files_dir):
            # Copy files from local storage
            for root, _, filenames in os.walk(local_files_dir):
                for filename in filenames:
                    src_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(src_path, local_files_dir)
                    dest_path = os.path.join(src_dir, rel_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(src_path, dest_path)
        
        # Create metadata file for export info
        metadata = {
            "exported_from": "experiment",
            "source_task_id": source_task_id,
            "export_date": json_lib.dumps({"$date": {"$numberLong": str(int(time.time() * 1000))}}),
            "has_files": os.path.exists(src_dir) and len(os.listdir(src_dir)) > 0
        }
        metadata_path = os.path.join(task_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json_lib.dump(metadata, f, indent=2)
        
        return {"status": "success", "task_dir": task_dir_name}
    except Exception as e:
        print(f"Error exporting task to local gallery: {e}")
        return {"status": "error", "message": "An error occurred while exporting the task to the local gallery"}


@router.post("/import_from_local_gallery", summary="Import a task from local tasks-gallery")
async def import_task_from_local_gallery(
    request: Request,
    task_dir: str = Form(...),
    experiment_id: Optional[str] = Form(None),
):
    """
    Import a task from the local tasks-gallery directory.
    Creates a REMOTE task from the local gallery task.json.
    """
    try:
        workspace_dir = get_workspace_dir()
        local_gallery_dir = os.path.join(workspace_dir, "tasks-gallery")
        task_path = os.path.normpath(os.path.join(local_gallery_dir, task_dir))
        local_gallery_dir_real = os.path.realpath(local_gallery_dir)
        task_path_real = os.path.realpath(task_path)
        common_path = os.path.commonpath([local_gallery_dir_real, task_path_real])
        if common_path != local_gallery_dir_real:
            print(f"Invalid task directory: {task_path_real} not in {local_gallery_dir_real}")
            return {"status": "error", "message": "Invalid task directory"}
        task_json_path = os.path.join(task_path, "task.json")
        
        task_json_real = os.path.realpath(task_json_path)
        if not task_json_real.startswith(local_gallery_dir_real + os.sep):
            print(f"Invalid task.json path: {task_json_real} not in {local_gallery_dir_real}")
            return {"status": "error", "message": "Invalid task.json path"}
        if not os.path.isfile(task_json_real):
            return {"status": "error", "message": f"task.json not found in local gallery: {task_dir}"}

        with open(task_json_real) as f:
            task_def = json_lib.load(f)

        # Build task fields, marking as remote
        task_name = slugify(task_def.get("name", task_dir))
        task_type = task_def.get("type", "REMOTE")
        inputs = task_def.get("inputs", {})
        config = task_def.get("config", {})
        plugin = task_def.get("plugin", "remote_task")
        outputs = task_def.get("outputs", {})

        # Check if task already exists and update instead of creating duplicate
        existing_tasks = tasks_service.tasks_get_all()
        existing_task = None
        for task in existing_tasks:
            if task.get("name") == task_name and task.get("remote_task", False):
                existing_task = task
                break

        if existing_task:
            # Update existing task
            task_id = existing_task["id"]
            tasks_service.update_task(task_id, {
                "name": task_name,
                "inputs": inputs,
                "config": config,
                "outputs": outputs,
                "plugin": plugin
            })
            return {"status": "success", "task_id": task_id, "action": "updated"}
        else:
            # Create new task
            task_id = tasks_service.add_task(
                name=task_name,
                task_type=task_type,
                inputs=inputs,
                config=config,
                plugin=plugin,
                outputs=outputs,
                experiment_id=experiment_id,
                remote_task=True,
            )
            return {"status": "success", "task_id": task_id, "action": "created"}

    except Exception as e:
        print(f"Error importing task from local gallery: {e}")
        return {"status": "error", "message": "An error occurred while creating the task"}
