import os
import httpx
import json
import re
from fastapi import APIRouter, Form, Request, File, UploadFile, HTTPException
from typing import Optional, List
from transformerlab.services import job_service
from transformerlab.services.job_service import job_update_status
from lab.dirs import get_workspace_dir, get_job_checkpoints_dir


router = APIRouter(prefix="/remote", tags=["remote"])


def validate_gpu_orchestrator_env_vars():
    """
    Validate that required GPU orchestrator environment variables are set.
    Returns a tuple of (url, port) if valid, or (None, error_response) if invalid.
    """
    gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATION_SERVER")
    gpu_orchestrator_port = os.getenv("GPU_ORCHESTRATION_SERVER_PORT")

    if not gpu_orchestrator_url:
        return None, {"status": "error", "message": "GPU_ORCHESTRATION_SERVER environment variable not set"}

    if not gpu_orchestrator_port:
        return None, {"status": "error", "message": "GPU_ORCHESTRATION_SERVER_PORT environment variable not set"}

    return gpu_orchestrator_url, gpu_orchestrator_port


@router.post("/create-job")
async def create_remote_job(
    request: Request,
    experimentId: str,
    cluster_name: str = Form(...),
    command: str = Form("echo 'Hello World'"),
    task_name: Optional[str] = Form(None),
    cpus: Optional[str] = Form(None),
    memory: Optional[str] = Form(None),
    disk_space: Optional[str] = Form(None),
    accelerators: Optional[str] = Form(None),
    num_nodes: Optional[int] = Form(None),
    setup: Optional[str] = Form(None),
    uploaded_dir_path: Optional[str] = Form(None),
):
    """
    Create a remote job without launching it. Returns job info for frontend to show placeholder.
    """
    # First, create a REMOTE job
    job_data = {"task_name": task_name, "command": command, "cluster_name": cluster_name}

    # Add optional parameters if provided
    if cpus:
        job_data["cpus"] = cpus
    if memory:
        job_data["memory"] = memory
    if disk_space:
        job_data["disk_space"] = disk_space
    if accelerators:
        job_data["accelerators"] = accelerators
    if num_nodes:
        job_data["num_nodes"] = num_nodes
    if setup:
        job_data["setup"] = setup
    if uploaded_dir_path:
        job_data["uploaded_dir_path"] = uploaded_dir_path

    try:
        job_id = job_service.job_create(
            type="REMOTE",
            status="LAUNCHING",
            experiment_id=experimentId,
        )
        # Update the job data to add fields from job_data (this ensures default fields stay in the job)
        for key, value in job_data.items():
            job_service.job_update_job_data_insert_key_value(job_id, key, value, experimentId)

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Remote job created successfully",
        }
    except Exception as e:
        print(f"Failed to create job: {str(e)}")
        return {"status": "error", "message": "Failed to create job"}


@router.post("/launch")
async def launch_remote(
    request: Request,
    experimentId: str,
    job_id: Optional[str] = Form(None),
    cluster_name: str = Form(...),
    command: str = Form("echo 'Hello World'"),
    task_name: Optional[str] = Form(None),
    cpus: Optional[str] = Form(None),
    memory: Optional[str] = Form(None),
    disk_space: Optional[str] = Form(None),
    accelerators: Optional[str] = Form(None),
    num_nodes: Optional[int] = Form(None),
    setup: Optional[str] = Form(None),
    uploaded_dir_path: Optional[str] = Form(None),
):
    """
    Launch a remote instance via Lattice orchestrator. If job_id is provided, use existing job, otherwise create new one.
    """
    # If job_id is provided, use existing job, otherwise create a new one
    if job_id:
        # Use existing job
        pass
    else:
        # Create a new REMOTE job
        job_data = {"task_name": task_name, "command": command, "cluster_name": cluster_name}

        try:
            job_id = job_service.job_create(
                type="REMOTE",
                status="LAUNCHING",
                experiment_id=experimentId,
            )
            # Update the job data to add fields from job_data (this ensures default fields stay in the job)
            for key, value in job_data.items():
                job_service.job_update_job_data_insert_key_value(job_id, key, value, experimentId)
        except Exception as e:
            print(f"Failed to create job: {str(e)}")
            return {"status": "error", "message": "Failed to create job"}

    # Validate environment variables
    result = validate_gpu_orchestrator_env_vars()
    gpu_orchestrator_url, gpu_orchestrator_port = result
    if isinstance(gpu_orchestrator_url, dict):
        return gpu_orchestrator_url  # Error response
    elif isinstance(gpu_orchestrator_port, dict):
        return gpu_orchestrator_port  # Error response

    # Prepare the request data for Lattice orchestrator
    request_data = {
        "cluster_name": cluster_name,
        "command": command,
        "tlab_job_id": job_id,  # Pass the job_id to the orchestrator
    }

    # Use task_name as job_name if provided, otherwise fall back to cluster_name
    job_name = task_name if task_name else cluster_name
    request_data["job_name"] = job_name

    # Add optional parameters if provided
    if cpus:
        request_data["cpus"] = cpus
    if memory:
        request_data["memory"] = memory
    if disk_space:
        request_data["disk_space"] = disk_space
    if accelerators:
        request_data["accelerators"] = accelerators
    if num_nodes:
        request_data["num_nodes"] = num_nodes
    if setup:
        request_data["setup"] = setup
    if uploaded_dir_path:
        request_data["uploaded_dir_path"] = uploaded_dir_path

    gpu_orchestrator_url = f"{gpu_orchestrator_url}:{gpu_orchestrator_port}/api/v1/instances/launch"

    try:
        # Make the request to the Lattice orchestrator
        async with httpx.AsyncClient() as client:
            # Build headers: prefer configured API key, otherwise forward incoming Authorization header
            outbound_headers = {"Content-Type": "application/x-www-form-urlencoded"}
            incoming_auth = request.headers.get("AUTHORIZATION")
            if incoming_auth:
                outbound_headers["AUTHORIZATION"] = incoming_auth

            response = await client.post(
                f"{gpu_orchestrator_url}",
                headers=outbound_headers,
                data=request_data,
                cookies=request.cookies,
                timeout=30.0,
            )

            if response.status_code == 200:
                response_data = response.json()
                # Store the request_id in job data for later use
                if "request_id" in response_data:
                    job_service.job_update_job_data_insert_key_value(
                        job_id, "orchestrator_request_id", response_data["request_id"], experimentId
                    )
                # Store the cluster_name in job data for later use
                if "cluster_name" in response_data:
                    job_service.job_update_job_data_insert_key_value(
                        job_id, "cluster_name", response_data["cluster_name"], experimentId
                    )

                # Save the launch_config for potential resume
                launch_config = {
                    "command": command,
                    "setup": setup,
                    "cluster_name": cluster_name,
                    "cpus": cpus,
                    "memory": memory,
                    "disk_space": disk_space,
                    "accelerators": accelerators,
                    "num_nodes": num_nodes,
                    "uploaded_dir_path": uploaded_dir_path,
                }
                job_service.job_update_job_data_insert_key_value(
                    job_id, "launch_config", json.dumps(launch_config), experimentId
                )

                return {
                    "status": "success",
                    "data": response_data,
                    "job_id": job_id,
                    "message": "Remote instance launched successfully",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Lattice orchestrator returned status {response.status_code}: {response.text}",
                }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request to Lattice orchestrator timed out"}
    except httpx.RequestError:
        return {"status": "error", "message": "Request error occurred"}
    except Exception:
        return {"status": "error", "message": "Unexpected error occurred"}


@router.post("/stop")
async def stop_remote(
    request: Request,
    job_id: str = Form(...),
    cluster_name: str = Form(...),
):
    """
    Stop a remote instance via Lattice orchestrator by calling instances/down endpoint
    """
    # Validate environment variables
    result = validate_gpu_orchestrator_env_vars()
    gpu_orchestrator_url, gpu_orchestrator_port = result
    if isinstance(gpu_orchestrator_url, dict):
        return gpu_orchestrator_url  # Error response
    elif isinstance(gpu_orchestrator_port, dict):
        return gpu_orchestrator_port  # Error response

    # First, cancel the job on the cluster
    down_url = f"{gpu_orchestrator_url}:{gpu_orchestrator_port}/api/v1/instances/down"

    try:
        # Make the request to the Lattice orchestrator
        async with httpx.AsyncClient() as client:
            # Build headers: prefer configured API key, otherwise forward incoming Authorization header
            outbound_headers = {"Content-Type": "application/json"}
            incoming_auth = request.headers.get("AUTHORIZATION")
            if incoming_auth:
                outbound_headers["AUTHORIZATION"] = incoming_auth

            # Bring down the cluster
            print(f"Bringing down cluster {cluster_name}")
            # Prepare JSON payload for the orchestrator
            payload = {
                "cluster_name": cluster_name,
                "tlab_job_id": job_id,  # Pass the job_id to the orchestrator
            }

            response = await client.post(
                down_url, headers=outbound_headers, json=payload, cookies=request.cookies, timeout=30.0
            )

            if response.status_code == 200:
                # Update job status to STOPPED on successful down request
                await job_update_status(job_id, "STOPPED")

                return {
                    "status": "success",
                    "data": response.json(),
                    "message": "Remote instance stopped successfully",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Lattice orchestrator returned status {response.status_code}: {response.text}",
                }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request to Lattice orchestrator timed out"}
    except httpx.RequestError:
        return {"status": "error", "message": "Request error occurred"}
    except Exception:
        return {"status": "error", "message": "Unexpected error occurred"}


@router.post("/upload")
async def upload_directory(
    request: Request,
    dir_files: List[UploadFile] = File(...),
    dir_name: Optional[str] = Form(None),
):
    """
    Upload a directory to the remote Lattice orchestrator for later use in cluster launches.
    Files are stored locally first, then sent to orchestrator.
    """

    # Validate environment variables
    result = validate_gpu_orchestrator_env_vars()
    gpu_orchestrator_url, gpu_orchestrator_port = result
    if isinstance(gpu_orchestrator_url, dict):
        return gpu_orchestrator_url  # Error response
    elif isinstance(gpu_orchestrator_port, dict):
        return gpu_orchestrator_port  # Error response
    gpu_orchestrator_url = f"{gpu_orchestrator_url}:{gpu_orchestrator_port}/api/v1/instances/upload"

    # Store files locally first
    local_storage_dir = None
    try:
        # Create local storage directory
        workspace_dir = get_workspace_dir()
        local_uploads_dir = os.path.join(workspace_dir, "uploads")
        os.makedirs(local_uploads_dir, exist_ok=True)

        # Create unique directory for this upload
        import uuid

        upload_id = str(uuid.uuid4())
        base_upload_dir = f"upload_{upload_id}"
        local_storage_dir = os.path.join(local_uploads_dir, base_upload_dir)
        os.makedirs(local_storage_dir, exist_ok=True)

        # Store files locally
        for file in dir_files:
            # Reset file pointer to beginning
            await file.seek(0)
            content = await file.read()

            # Create directory structure if filename contains path separators
            file_path = os.path.join(local_storage_dir, file.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as f:
                f.write(content)

        # Prepare the request data for Lattice orchestrator
        files_data = []
        form_data = {}

        # Add dir_name if provided
        if dir_name:
            form_data["dir_name"] = dir_name

        # Prepare files for upload (reset file pointers)
        for file in dir_files:
            await file.seek(0)
            files_data.append(("dir_files", (file.filename, await file.read(), file.content_type)))

        # Make the request to the Lattice orchestrator
        async with httpx.AsyncClient() as client:
            # Build headers: prefer configured API key, otherwise forward incoming Authorization header
            outbound_headers = {}
            incoming_auth = request.headers.get("AUTHORIZATION")
            if incoming_auth:
                outbound_headers["AUTHORIZATION"] = incoming_auth

            response = await client.post(
                f"{gpu_orchestrator_url}",
                headers=outbound_headers,
                files=files_data,
                data=form_data,
                cookies=request.cookies,
                timeout=60.0,  # Longer timeout for file uploads
            )

            if response.status_code == 200:
                response_data = response.json()
                # Add local storage path to response (just the folder name)
                response_data["local_storage_path"] = base_upload_dir
                return {
                    "status": "success",
                    "data": response_data,
                    "message": "Directory uploaded successfully",
                    "local_storage_path": base_upload_dir,
                }
            else:
                return {
                    "status": "error",
                    "message": f"Lattice orchestrator returned status {response.status_code}: {response.text}",
                }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request to Lattice orchestrator timed out"}
    except httpx.RequestError:
        return {"status": "error", "message": "Request error occurred"}
    except Exception as e:
        print(f"Upload error: {e}")
        return {"status": "error", "message": "Unexpected error occurred"}


async def check_remote_job_status(request: Request, cluster_name: str):
    """
    Check the status of jobs running on a remote cluster via the orchestrator.
    Returns the status of all jobs on the cluster.
    """
    # Validate environment variables
    result = validate_gpu_orchestrator_env_vars()
    gpu_orchestrator_url, gpu_orchestrator_port = result
    if isinstance(gpu_orchestrator_url, dict):
        return gpu_orchestrator_url  # Error response
    elif isinstance(gpu_orchestrator_port, dict):
        return gpu_orchestrator_port  # Error response

    # Build the jobs endpoint URL
    jobs_url = f"{gpu_orchestrator_url}:{gpu_orchestrator_port}/api/v1/jobs/{cluster_name}"

    try:
        async with httpx.AsyncClient() as client:
            # Build headers: prefer configured API key, otherwise forward incoming Authorization header
            outbound_headers = {"Content-Type": "application/json"}
            incoming_auth = request.headers.get("AUTHORIZATION")
            if incoming_auth:
                outbound_headers["AUTHORIZATION"] = incoming_auth

            response = await client.get(jobs_url, headers=outbound_headers, cookies=request.cookies, timeout=30.0)

            if response.status_code == 200:
                return {
                    "status": "success",
                    "data": response.json(),
                    "message": "Remote job status retrieved successfully",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Orchestrator returned status {response.status_code}: {response.text}",
                }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request to orchestrator timed out"}
    except httpx.RequestError:
        return {"status": "error", "message": "Request error occurred"}
    except Exception:
        return {"status": "error", "message": "Unexpected error occurred"}


@router.get("/logs/{request_id}")
async def get_orchestrator_logs(request: Request, request_id: str):
    """
    Get streaming logs from the orchestrator for a specific request_id.
    This endpoint forwards authentication to the orchestrator.
    """
    # Validate environment variables
    result = validate_gpu_orchestrator_env_vars()
    gpu_orchestrator_url, gpu_orchestrator_port = result
    if isinstance(gpu_orchestrator_url, dict):
        return gpu_orchestrator_url  # Error response
    elif isinstance(gpu_orchestrator_port, dict):
        return gpu_orchestrator_port  # Error response

    # Build the logs endpoint URL
    logs_url = f"{gpu_orchestrator_url}:{gpu_orchestrator_port}/api/v1/instances/requests/{request_id}/logs"

    try:
        async with httpx.AsyncClient() as client:
            # Build headers: prefer configured API key, otherwise forward incoming Authorization header
            outbound_headers = {"Content-Type": "application/json"}
            incoming_auth = request.headers.get("AUTHORIZATION")
            if incoming_auth:
                outbound_headers["AUTHORIZATION"] = incoming_auth

            response = await client.get(logs_url, headers=outbound_headers, cookies=request.cookies, timeout=30.0)

            if response.status_code == 200:
                return {
                    "status": "success",
                    "data": response.text,  # Return raw text for streaming logs
                    "message": "Orchestrator logs retrieved successfully",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Orchestrator returned status {response.status_code}: {response.text}",
                }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request to orchestrator timed out"}
    except httpx.RequestError:
        return {"status": "error", "message": "Request error occurred"}
    except Exception:
        return {"status": "error", "message": "Unexpected error occurred"}


@router.get("/check-status")
async def check_remote_jobs_status(request: Request):
    """
    Simple endpoint to check and update status of REMOTE jobs in LAUNCHING state.
    This endpoint can be called by the frontend and forwards authentication.
    """
    try:
        # Get all REMOTE jobs in LAUNCHING state across all experiments
        import transformerlab.services.experiment_service as experiment_service

        launching_remote_jobs = []

        # Get all experiments and check for REMOTE jobs in LAUNCHING state
        experiments = experiment_service.experiment_get_all()
        for exp in experiments:
            # Avoid errors of broken migrations in experiments
            if "id" not in exp:
                continue
            exp_jobs = job_service.jobs_get_all(exp["id"], type="REMOTE", status="LAUNCHING")
            launching_remote_jobs.extend(exp_jobs)

        if not launching_remote_jobs:
            return {"message": "No REMOTE jobs in LAUNCHING state", "updated_jobs": []}

        updated_jobs = []
        print(f"Checking {len(launching_remote_jobs)} REMOTE jobs in LAUNCHING state")

        for job in launching_remote_jobs:
            job_id = job["id"]
            job_data = job.get("job_data", {})
            cluster_name = job_data.get("cluster_name")

            if not cluster_name:
                print(f"Warning: Job {job_id} has no cluster_name in job_data")
                continue

            # Check the status of jobs on this cluster using the actual request
            status_response = await check_remote_job_status(request, cluster_name)

            if status_response["status"] == "success":
                orchestrator_data = status_response["data"]
                jobs_on_cluster = orchestrator_data.get("jobs", [])

                # Check if all jobs on the cluster are in a terminal state (SUCCEEDED or FAILED)
                all_jobs_finished = True
                for cluster_job in jobs_on_cluster:
                    job_status = cluster_job.get("status", "")
                    # Check for both the enum format and plain string format
                    if job_status not in ["JobStatus.SUCCEEDED", "JobStatus.FAILED", "SUCCEEDED", "FAILED"]:
                        all_jobs_finished = False
                        break

                if all_jobs_finished and jobs_on_cluster:
                    # All jobs on the cluster are finished, mark our LAUNCHING job as COMPLETE
                    await job_update_status(job_id, "COMPLETE", experiment_id=job["experiment_id"])
                    updated_jobs.append(
                        {
                            "job_id": job_id,
                            "cluster_name": cluster_name,
                            "status": "COMPLETE",
                            "message": "All jobs on cluster completed",
                        }
                    )
                else:
                    # Jobs are still running on the cluster
                    updated_jobs.append(
                        {
                            "job_id": job_id,
                            "cluster_name": cluster_name,
                            "status": "LAUNCHING",
                            "message": "Jobs still running on cluster",
                        }
                    )
            else:
                print(
                    f"Error checking status for job {job_id} on cluster {cluster_name}: {status_response.get('message', 'Unknown error')}"
                )

        return {
            "status": "success",
            "updated_jobs": updated_jobs,
            "message": f"Checked {len(launching_remote_jobs)} REMOTE jobs in LAUNCHING state",
        }

    except Exception as e:
        print(f"Error checking remote job status: {str(e)}")
        return {"status": "error", "message": "Error checking remote job status"}


@router.post("/{job_id}/resume_from_checkpoint")
async def resume_from_checkpoint(
    job_id: str,
    request: Request,
):
    """Resume training from a checkpoint by creating a new job with the checkpoint parameter"""
    try:
        body = await request.json()
        checkpoint_name = body.get("checkpoint_name")
        experimentId = body.get("experimentId")
        
        if not checkpoint_name:
            raise HTTPException(status_code=400, detail="checkpoint_name is required")
        if not experimentId:
            raise HTTPException(status_code=400, detail="experimentId is required")
        
        # Get the original job
        original_job = job_service.job_get(job_id)
        if not original_job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        original_job_data = original_job.get("job_data", {})
        
        # Get the launch_config from the original job
        launch_config_str = original_job_data.get("launch_config")
        if not launch_config_str:
            raise HTTPException(status_code=400, detail="launch_config not found in original job")
        
        # Parse the launch_config JSON string
        try:
            launch_config = json.loads(launch_config_str)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid launch_config format")
        
        # Get the checkpoint directory path
        checkpoints_dir = get_job_checkpoints_dir(job_id)
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_name} not found at {checkpoint_path}")
        
        # Modify the command to include the checkpoint parameter
        original_command = launch_config.get("command", "")
        if not original_command:
            raise HTTPException(status_code=400, detail="Original command not found in launch_config")
        
        # Parse the command to find the python execution part and add --resume_from_checkpoint
        command_parts = original_command.split(' && ')
        python_part_index = None
        for i, part in enumerate(command_parts):
            if part.strip().startswith('python '):
                python_part_index = i
                break
        
        if python_part_index is None:
            raise HTTPException(status_code=400, detail="Could not find python execution command in original command")
        
        # Modify the python part to include --resume_from_checkpoint
        python_command = command_parts[python_part_index].strip()
        
        # Check if --resume_from_checkpoint already exists
        if "--resume_from_checkpoint" in python_command:
            # Replace existing checkpoint parameter
            resume_python_command = re.sub(
                r'--resume_from_checkpoint\s+\S+',
                f'--resume_from_checkpoint {checkpoint_path}',
                python_command
            )
        else:
            # Add the checkpoint parameter
            resume_python_command = f"{python_command} --resume_from_checkpoint {checkpoint_path}"
        
        # Replace the python part in the command_parts
        command_parts[python_part_index] = resume_python_command
        
        # Rejoin the command parts
        resume_command = ' && '.join(command_parts)        
        # Call launch_remote to create and launch the new job
        launch_result = await launch_remote(
            request=request,
            experimentId=experimentId,
            cluster_name=launch_config.get("cluster_name"),
            command=resume_command,
            task_name=f"Resume from {checkpoint_name}",
            cpus=launch_config.get("cpus"),
            memory=launch_config.get("memory"),
            disk_space=launch_config.get("disk_space"),
            accelerators=launch_config.get("accelerators"),
            num_nodes=launch_config.get("num_nodes"),
            setup=launch_config.get("setup"),
            uploaded_dir_path=launch_config.get("uploaded_dir_path"),
        )
        
        if launch_result.get("status") == "success":
            new_job_id = launch_result.get("job_id")
            # Update the new job's job_data to mark it as resumed
            job_service.job_update_job_data_insert_key_value(new_job_id, "resumed_from_checkpoint", checkpoint_name, experimentId)
            job_service.job_update_job_data_insert_key_value(new_job_id, "parent_job_id", job_id, experimentId)
            return {
                "status": "success",
                "job_id": new_job_id,
                "message": f"Training resumed from checkpoint {checkpoint_name}",
                "data": launch_result
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to launch remote job: {launch_result.get('message')}")

    except Exception as e:
        print(f"Error checking remote job status: {str(e)}")
        return {"status": "error", "message": "Error checking remote job status"}
