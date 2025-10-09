import json
import os
import httpx
from fastapi import APIRouter, Form
from typing import Optional

import transformerlab.services.job_service as job_service
from transformerlab.services.tasks_service import tasks_service

router = APIRouter(prefix="/remote", tags=["remote"])


@router.post("/launch")
async def launch_remote(
    experimentId: str,
    cluster_name: str = Form(...),
    command: str = Form("echo 'Hello World'"),
    cpus: Optional[str] = Form(None),
    memory: Optional[str] = Form(None),
    disk_space: Optional[str] = Form(None),
    accelerators: Optional[str] = Form(None),
    num_nodes: Optional[int] = Form(None),
    setup: Optional[str] = Form(None),
):
    """
    Launch a remote instance via Lattice orchestrator and create a job with remote_task=True
    """
    # Get environment variables
    gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATION_SERVER")
    gpu_orchestrator_port = os.getenv("GPU_ORCHESTRATION_SERVER_PORT")
    gpu_orchestrator_api_key = os.getenv("GPU_ORCHESTRATION_SERVER_API_KEY")
    
    if not gpu_orchestrator_url:
        return {"status": "error", "message": "GPU_ORCHESTRATION_SERVER environment variable not set"}
    
    if not gpu_orchestrator_port:
        return {"status": "error", "message": "GPU_ORCHESTRATION_SERVER_PORT environment variable not set"}
    
    if not gpu_orchestrator_api_key:
        return {"status": "error", "message": "GPU_ORCHESTRATION_SERVER_API_KEY environment variable not set"}
    
    # Prepare the request data for Lattice orchestrator
    request_data = {
        "cluster_name": cluster_name,
        "command": command,
    }
    
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
    
    gpu_orchestrator_url = f"{gpu_orchestrator_url}:{gpu_orchestrator_port}/api/v1/instances/launch"
    
    try:
        # Make the request to the Lattice orchestrator
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{gpu_orchestrator_url}",
                headers={
                    "Authorization": f"Bearer {gpu_orchestrator_api_key}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data=request_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                # Create a job with remote_task=True to track this remote execution
                job_data = {
                    "remote_task": True,
                    "cluster_name": cluster_name,
                    "command": command,
                    "lattice_response": response.json(),
                    "gpu_orchestrator_url": gpu_orchestrator_url,
                }
                
                # Add optional parameters to job data
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
                
                # Create a task with remote_task=True so it appears in the filesystem
                task_config = {
                    "cluster_name": cluster_name,
                    "command": command,
                    "gpu_orchestrator_url": gpu_orchestrator_url,
                    "gpu_orchestrator_port": gpu_orchestrator_port,
                    "gpu_orchestrator_api_key": gpu_orchestrator_api_key,
                    "lattice_response": response.json(),
                }
                
                # Add optional parameters to task config
                if cpus:
                    task_config["cpus"] = cpus
                if memory:
                    task_config["memory"] = memory
                if disk_space:
                    task_config["disk_space"] = disk_space
                if accelerators:
                    task_config["accelerators"] = accelerators
                if num_nodes:
                    task_config["num_nodes"] = num_nodes
                if setup:
                    task_config["setup"] = setup
                
                # Create the task with remote_task=True
                task_id = tasks_service.add_task(
                    name=cluster_name,
                    task_type="REMOTE",
                    inputs={},
                    config=task_config,
                    plugin="remote_orchestrator",
                    outputs={},
                    experiment_id=experimentId,
                    remote_task=True
                )
                
                # Also create a job to track this remote task
                job_id = job_service.job_create(
                    type="REMOTE",
                    status="COMPLETE",  # Mark as complete since it's handled by Lattice
                    experiment_id=experimentId,
                    job_data=json.dumps(job_data)
                )
                
                return {
                    "status": "success", 
                    "data": response.json(),
                    "task_id": task_id,
                    "job_id": job_id,
                    "message": "Remote instance launched successfully"
                }
            else:
                return {
                    "status": "error", 
                    "message": f"Lattice orchestrator returned status {response.status_code}: {response.text}"
                }
                
    except httpx.TimeoutException:
        return {"status": "error", "message": "Request to Lattice orchestrator timed out"}
    except httpx.RequestError:
        return {"status": "error", "message": "Request error occurred"}
    except Exception:
        return {"status": "error", "message": "Unexpected error occurred"}
