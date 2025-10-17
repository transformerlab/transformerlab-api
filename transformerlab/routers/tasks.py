import json

from fastapi import APIRouter, Body, Request
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
        return {"status": "error", "message": str(e)}


@router.post("/import_from_gallery", summary="Import a task from transformerlab/galleries tasks subdirectory")
async def import_task_from_gallery(
    request: Request,
    subdir: str,
    experiment_id: str | None = None,
    upload: bool = True,
    repo_url: str | None = None,
):
    """
    Clone the specific tasks/<subdir> from transformerlab/galleries and create a REMOTE task using task.json.
    If task.json includes an "upload_dir" key, run a test upload using transformerlab-sdk/test_script.py facilities later.
    """
    # Prepare temp directory for shallow clone of specific path
    repo_url = "https://github.com/transformerlab/galleries.git"
    tmp_dir = tempfile.mkdtemp(prefix="tlab_tasks_gallery_")
    try:
        # Sparse checkout only the requested subdir
        subprocess.check_call(["git", "init"], cwd=tmp_dir)
        subprocess.check_call(["git", "remote", "add", "origin", repo_url], cwd=tmp_dir)
        subprocess.check_call(["git", "config", "core.sparseCheckout", "true"], cwd=tmp_dir)
        sparse_info_dir = os.path.join(tmp_dir, ".git", "info")
        os.makedirs(sparse_info_dir, exist_ok=True)
        with open(os.path.join(sparse_info_dir, "sparse-checkout"), "w") as f:
            f.write(f"tasks/{subdir}\n")
        # For testing: pull from specific branch containing tasks library scaffolding
        subprocess.check_call(["git", "pull", "--depth", "1", "origin", "add/tasks-library-v1"], cwd=tmp_dir)

        task_dir = os.path.join(tmp_dir, "tasks", subdir)
        task_json_path = os.path.join(task_dir, "task.json")
        if not os.path.isfile(task_json_path):
            return {"status": "error", "message": f"task.json not found in tasks/{subdir}"}

        with open(task_json_path) as f:
            task_def = json_lib.load(f)

        # Build task fields, marking as remote
        task_name = slugify(task_def.get("name", subdir))
        task_type = task_def.get("type", "REMOTE")
        inputs = task_def.get("inputs", {})
        config = task_def.get("config", {})
        plugin = task_def.get("plugin", "remote_task")
        outputs = task_def.get("outputs", {})

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

        # Optional: if task.json suggests a folder to upload, we will just record it in config
        # Optional remote upload flow to GPU orchestrator
        if upload:
            try:
                # Determine directory to upload: base task directory by default
                dir_to_upload = task_dir
                if repo_url:
                    # If explicit repo_url provided, do a separate sparse clone of that URL root
                    repo_tmp = tempfile.mkdtemp(prefix="tlab_repo_upload_")
                    try:
                        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, repo_tmp])
                        dir_to_upload = repo_tmp
                    except Exception as e:
                        return {"status": "error", "message": f"Failed to clone repo for upload: {e}"}

                if not dir_to_upload or not os.path.isdir(dir_to_upload):
                    return {"status": "error", "message": f"Upload directory not found: {dir_to_upload}"}

                # Stage files under a temp 'src' directory and exclude task.json
                src_stage = os.path.join(tmp_dir, "src")
                os.makedirs(src_stage, exist_ok=True)
                files_to_copy = [f for f in os.listdir(dir_to_upload) if f != "task.json"]
                # If no files besides task.json, skip upload
                has_files_to_upload = len(files_to_copy) > 0
                if has_files_to_upload:
                    for name in files_to_copy:
                        src_path = os.path.join(dir_to_upload, name)
                        dest_path = os.path.join(src_stage, name)
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_path)
                        else:
                            shutil.copy2(src_path, dest_path)

                # Post to GPU orchestrator upload endpoint
                gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATION_SERVER")
                gpu_orchestrator_port = os.getenv("GPU_ORCHESTRATION_SERVER_PORT")
                # Legacy upload endpoint var no longer used when mirroring frontend
                if not gpu_orchestrator_url or not gpu_orchestrator_port:
                    # If orchestrator not configured, just attach local path hint
                    try:
                        task_obj = tasks_service.tasks_get_by_id(task_id)
                        if isinstance(task_obj.get("config"), str):
                            task_obj["config"] = json_lib.loads(task_obj["config"]) if task_obj["config"] else {}
                        if has_files_to_upload:
                            task_obj["config"]["local_upload_staged_dir"] = src_stage
                        tasks_service.update_task(task_id, {"config": task_obj["config"]})
                    except Exception:
                        pass
                else:
                    # Only attempt upload if we actually staged files
                    if has_files_to_upload:
                        # Build multipart form to mirror frontend DirectoryUpload
                        dest = f"{gpu_orchestrator_url}:{gpu_orchestrator_port}/api/v1/instances/upload"
                        files_form = []
                        # Walk src_stage and add each file, preserving relative path inside src/
                        for root, _, filenames in os.walk(src_stage):
                            for filename in filenames:
                                full_path = os.path.join(root, filename)
                                rel_path = os.path.relpath(full_path, src_stage)
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
                return {"status": "error", "message": f"Upload exception: {e}"}

        return {"status": "success", "task_id": task_id}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Git error: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
