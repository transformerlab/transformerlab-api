import os
import httpx
from fastapi import APIRouter, Form, Request
from typing import Optional

from transformerlab.services.tasks_service import tasks_service

router = APIRouter(prefix="/remote", tags=["remote"])


@router.post("/launch")
async def launch_remote(
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
):
    """
    Launch a remote instance via Lattice orchestrator and create a task with remote_task attribute as True
    """
    # Get environment variables
    gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATION_SERVER")
    gpu_orchestrator_port = os.getenv("GPU_ORCHESTRATION_SERVER_PORT")
    
    if not gpu_orchestrator_url:
        return {"status": "error", "message": "GPU_ORCHESTRATION_SERVER environment variable not set"}
    
    if not gpu_orchestrator_port:
        return {"status": "error", "message": "GPU_ORCHESTRATION_SERVER_PORT environment variable not set"}
    # Prepare the request data for Lattice orchestrator
    request_data = {
        "cluster_name": cluster_name,
        "command": command,
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
    
    gpu_orchestrator_url = f"{gpu_orchestrator_url}:{gpu_orchestrator_port}/api/v1/instances/launch"
    
    try:
        # Make the request to the Lattice orchestrator
        async with httpx.AsyncClient() as client:
            # Build headers: prefer configured API key, otherwise forward incoming Authorization header
            outbound_headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            incoming_auth = request.headers.get("AUTHORIZATION")
            if incoming_auth:
                outbound_headers["AUTHORIZATION"] = incoming_auth

            response = await client.post(
                f"{gpu_orchestrator_url}",
                headers=outbound_headers,
                data=request_data,
                cookies=request.cookies,
                timeout=30.0
            )
            
            if response.status_code == 200:
                # Create a task with remote_task=True so it appears in the filesystem
                task_config = {
                    "cluster_name": cluster_name,
                    "command": command,
                    "gpu_orchestrator_url": gpu_orchestrator_url,
                    "gpu_orchestrator_port": gpu_orchestrator_port,
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
                    name=task_name if task_name else cluster_name,
                    task_type="REMOTE",
                    inputs={},
                    config=task_config,
                    plugin="remote_orchestrator",
                    outputs={},
                    experiment_id=experimentId,
                    remote_task=True
                )
                
                return {
                    "status": "success", 
                    "data": response.json(),
                    "task_id": task_id,
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
