import asyncio
import httpx
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json

import transformerlab.db as db

router = APIRouter(prefix="/network", tags=["network"])


# Pydantic models for request/response
class NetworkMachineCreate(BaseModel):
    name: str
    host: str
    port: int = 8338
    api_token: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NetworkMachineResponse(BaseModel):
    id: int
    name: str
    host: str
    port: int
    status: str
    last_seen: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


# Remote job execution models
class RemoteJobDispatch(BaseModel):
    job_id: str
    job_data: Dict[str, Any]
    target_machine_id: int
    plugins_required: Optional[list[str]] = []
    models_required: Optional[list[str]] = []
    datasets_required: Optional[list[str]] = []


class RemoteJobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    machine_id: int
    error_message: Optional[str] = None


@router.get("/info", summary="Get network configuration info")
async def get_network_info():
    """Get information about this machine's network role."""
    try:
        is_host = await db.config_get("IS_HOST_MACHINE")
        is_host_machine = is_host == "True" if is_host else False

        return {
            "status": "success",
            "data": {"is_host_machine": is_host_machine, "role": "host" if is_host_machine else "network_machine"},
        }
    except Exception as e:
        print(f"ERROR: Error getting network info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/machines", summary="List all network machines")
async def list_machines():
    """Get all registered network machines."""
    try:
        machines = await db.network_machine_get_all()
        return {"status": "success", "data": machines}
    except Exception as e:
        print(f"ERROR: Error listing network machines: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/machines", summary="Add a new network machine")
async def add_machine(machine: NetworkMachineCreate):
    """Register a new network machine."""
    try:
        # Check if machine with same name already exists
        existing = await db.network_machine_get_by_name(machine.name)
        if existing:
            raise HTTPException(status_code=400, detail=f"Machine with name '{machine.name}' already exists")

        # Create the machine
        machine_id = await db.network_machine_create(
            name=machine.name,
            host=machine.host,
            port=machine.port,
            api_token=machine.api_token,
            metadata=machine.metadata or {},
        )

        # Try to ping the machine to check if it's reachable
        await ping_machine(machine_id)

        return {
            "status": "success",
            "machine_id": machine_id,
            "message": f"Machine '{machine.name}' added successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error adding network machine: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/machines/{machine_id}", summary="Get specific machine details")
async def get_machine(machine_id: int):
    """Get details of a specific network machine."""
    try:
        machine = await db.network_machine_get(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")
        return {"status": "success", "data": machine}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error getting network machine {machine_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/machines/{machine_id}", summary="Remove a network machine")
async def remove_machine(machine_id: int):
    """Remove a network machine from the registry."""
    try:
        success = await db.network_machine_delete(machine_id)
        if not success:
            raise HTTPException(status_code=404, detail="Machine not found")
        return {"status": "success", "message": "Machine removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error removing network machine {machine_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/machines/by-name/{machine_name}", summary="Remove a network machine by name")
async def remove_machine_by_name(machine_name: str):
    """Remove a network machine by name."""
    try:
        success = await db.network_machine_delete_by_name(machine_name)
        if not success:
            raise HTTPException(status_code=404, detail="Machine not found")
        return {"status": "success", "message": f"Machine '{machine_name}' removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error removing network machine '{machine_name}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status", summary="Get aggregated status of all machines")
async def get_network_status():
    """Get aggregated status information for all network machines."""
    try:
        machines = await db.network_machine_get_all()

        total_machines = len(machines)
        online_machines = len([m for m in machines if m.get("status") == "online"])
        offline_machines = len([m for m in machines if m.get("status") == "offline"])
        error_machines = len([m for m in machines if m.get("status") == "error"])

        return {
            "status": "success",
            "data": {
                "total_machines": total_machines,
                "online": online_machines,
                "offline": offline_machines,
                "error": error_machines,
                "machines": machines,
            },
        }
    except Exception as e:
        print(f"ERROR: Error getting network status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/machines/{machine_id}/ping", summary="Ping a specific machine")
async def ping_machine_endpoint(machine_id: int):
    """Ping a specific network machine to check its status."""
    try:
        result = await ping_machine(machine_id)
        return {"status": "success", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error pinging machine {machine_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/health-check", summary="Run health check on all machines")
async def health_check_all():
    """Run health check on all registered network machines."""
    try:
        machines = await db.network_machine_get_all()
        results = []

        # Create tasks for concurrent pinging
        tasks = []
        for machine in machines:
            task = ping_machine(machine["id"])
            tasks.append(task)

        # Wait for all pings to complete
        ping_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(ping_results):
            machine = machines[i]
            if isinstance(result, Exception):
                results.append(
                    {"machine_id": machine["id"], "name": machine["name"], "status": "error", "error": str(result)}
                )
            else:
                results.append(
                    {
                        "machine_id": machine["id"],
                        "name": machine["name"],
                        "status": result["status"],
                        "response_time": result.get("response_time"),
                        "server_info": result.get("server_info"),
                    }
                )

        return {"status": "success", "data": results}
    except Exception as e:
        print(f"ERROR: Error during health check: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Helper function to ping a machine
async def ping_machine(machine_id: int) -> Dict[str, Any]:
    """
    Ping a network machine by calling its /server/info endpoint.
    Updates the machine's status and last_seen timestamp.
    """
    machine = await db.network_machine_get(machine_id)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    start_time = datetime.now()

    try:
        # Construct the URL for the machine's server info endpoint
        url = f"http://{machine['host']}:{machine['port']}/server/info"

        # Set up headers for authentication if token is provided
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        # Make the HTTP request with timeout
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()

            # Parse server info
            server_info = response.json()

            # Update machine status to online
            await db.network_machine_update_status(machine_id, "online")

            # Update metadata with latest server info
            await db.network_machine_update_metadata(
                machine_id, {"last_server_info": server_info, "last_response_time": response_time}
            )

            return {"status": "online", "response_time": response_time, "server_info": server_info}

    except httpx.TimeoutException:
        await db.network_machine_update_status(machine_id, "offline")
        return {"status": "offline", "error": "Request timeout"}
    except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError, OSError):
        # Server is off/unreachable - mark as offline, not error
        await db.network_machine_update_status(machine_id, "offline")
        return {"status": "offline", "error": "Server unreachable"}
    except httpx.HTTPStatusError:
        await db.network_machine_update_status(machine_id, "error")
        return {"status": "error", "error": "An internal error occurred"}
    except Exception:
        # Only genuine errors should be marked as error status
        await db.network_machine_update_status(machine_id, "error")
        return {"status": "error", "error": "An internal error occurred"}


# =============================================================================
# REMOTE JOB EXECUTION ENDPOINTS
# =============================================================================


@router.post("/machines/{machine_id}/dispatch_job", summary="Dispatch a job to a remote machine")
async def dispatch_job_to_machine(machine_id: int, job_dispatch: RemoteJobDispatch):
    """
    Dispatch a job to a specific remote machine.
    This will first ensure dependencies are installed, then start the job.
    """
    try:
        # Get machine details
        machine = await db.network_machine_get(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")

        if machine["status"] != "online":
            raise HTTPException(status_code=400, detail="Machine is not online")

        # Build URL for remote machine
        base_url = f"http://{machine['host']}:{machine['port']}"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # First, check dependencies and install if needed
            if job_dispatch.plugins_required or job_dispatch.models_required or job_dispatch.datasets_required:
                deps_response = await client.post(
                    f"{base_url}/network/prepare_dependencies",
                    json={
                        "plugins": job_dispatch.plugins_required or [],
                        "models": job_dispatch.models_required or [],
                        "datasets": job_dispatch.datasets_required or [],
                    },
                    headers=headers,
                )
                if deps_response.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to prepare dependencies on remote machine")

            # Dispatch the actual job
            job_response = await client.post(
                f"{base_url}/network/execute_job",
                json={
                    "job_id": job_dispatch.job_id,
                    "job_data": job_dispatch.job_data,
                    "origin_machine": await _get_this_machine_info(),
                },
                headers=headers,
            )

            if job_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to start job on remote machine")

            # Update local job record with target machine info
            await db.job_update_job_data_insert_key_value(job_dispatch.job_id, "target_machine_id", machine_id)
            await db.job_update_job_data_insert_key_value(
                job_dispatch.job_id, "execution_host", f"{machine['host']}:{machine['port']}"
            )

            return {
                "status": "success",
                "message": "Job dispatched successfully",
                "remote_response": job_response.json(),
            }

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Failed to communicate with remote machine: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/prepare_dependencies", summary="Prepare dependencies for incoming remote job")
async def prepare_dependencies(dependencies: Dict[str, list[str]]):
    """
    Endpoint called by other machines to prepare dependencies before job execution.
    Similar to recipe dependency installation but for remote job requirements.
    """
    try:
        plugins = dependencies.get("plugins", [])
        models = dependencies.get("models", [])
        datasets = dependencies.get("datasets", [])

        if not (plugins or models or datasets):
            return {"status": "success", "message": "No dependencies to install"}

        # Create a job to track dependency installation
        job_data = {"dependencies": dependencies, "results": [], "progress": 0}

        job_id = await db.job_create(
            type="INSTALL_REMOTE_DEPS", status="QUEUED", job_data=json.dumps(job_data), experiment_id=""
        )

        # Execute dependency installation synchronously
        await _install_remote_dependencies_job(job_id, dependencies)

        # Wait for job to complete and get final status
        job = await db.job_get(job_id)

        if job["status"] == "COMPLETE":
            return {
                "status": "success",
                "job_id": job_id,
                "message": "Dependency installation completed successfully",
                "results": job["job_data"].get("results", []),
            }
        elif job["status"] == "FAILED":
            error_msg = job["job_data"].get("error_msg", "Unknown error")
            return {
                "status": "failed",
                "job_id": job_id,
                "message": f"Dependency installation failed: {error_msg}",
                "results": job["job_data"].get("results", []),
            }
        else:
            return {
                "status": "error",
                "job_id": job_id,
                "message": f"Unexpected job status: {job['status']}",
                "results": job["job_data"].get("results", []),
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare dependencies: {str(e)}")


@router.post("/execute_job", summary="Execute a job on this machine (called by remote machines)")
async def execute_remote_job(job_data: Dict[str, Any]):
    """
    Endpoint called by other machines to execute a job on this machine.
    This creates a local job and executes it using the standard job execution flow.
    """
    try:
        job_id = job_data["job_id"]
        job_config = job_data["job_data"]
        job_type = job_data.get("job_type", "UNDEFINED")
        experiment_id = job_data.get("experiment_id", "1")
        origin_machine = job_data.get("origin_machine", {})

        # Create local job record
        local_job_data = {
            **job_config,
            "remote_execution": True,
            "origin_machine": origin_machine,
            "original_job_id": job_id,
        }

        # Create job with remote execution flag
        local_job_id = await db.job_create(
            type=job_type,
            status="QUEUED",
            job_data=json.dumps(local_job_data),
            experiment_id=experiment_id,
        )

        # Get the job details for execution
        job_details = await db.job_get(local_job_id)

        # Start job execution using existing job runner
        from transformerlab.shared import shared

        # For now, start immediately. Later we can integrate with the job queue
        experiment_name = job_config.get("experiment_name")
        if not experiment_name:
            data = await db.experiment_get(experiment_id)
            if data:
                experiment_name = data.get("name")
        if not experiment_name:
            raise HTTPException(
                status_code=400, detail="Experiment name or valid experiment id is required for remote job execution"
            )

        # Execute in background
        import asyncio

        asyncio.create_task(
            shared.run_job(
                job_id=local_job_id, job_config=local_job_data, experiment_name=experiment_name, job_details=job_details
            )
        )

        return {
            "status": "started", 
            "local_job_id": local_job_id, 
            "original_job_id": job_id,
            "message": "Job execution started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute remote job: {str(e)}")


@router.get("/machines/{machine_id}/job_status/{job_id}", summary="Get status of a job running on remote machine")
async def get_remote_job_status(machine_id: int, job_id: str):
    """Get the status of a job running on a remote machine."""
    try:
        machine = await db.network_machine_get(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")

        base_url = f"http://{machine['host']}:{machine['port']}"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/jobs/{job_id}", headers=headers)

            if response.status_code == 200:
                return {"status": "success", "job_status": response.json()}
            else:
                return {"status": "error", "message": "Failed to get job status from remote machine"}

    except httpx.RequestError as e:
        return {"status": "error", "message": f"Failed to communicate with remote machine: {str(e)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/machines/{machine_id}/stop_job/{job_id}", summary="Stop a job running on remote machine")
async def stop_remote_job(machine_id: int, job_id: str):
    """Stop a job running on a remote machine."""
    try:
        machine = await db.network_machine_get(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")

        base_url = f"http://{machine['host']}:{machine['port']}"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/jobs/{job_id}/stop", headers=headers)

            if response.status_code == 200:
                return {"status": "success", "message": "Job stop signal sent"}
            else:
                return {"status": "error", "message": "Failed to stop job on remote machine"}

    except httpx.RequestError as e:
        return {"status": "error", "message": f"Failed to communicate with remote machine: {str(e)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/capabilities", summary="Get capabilities of this machine")
async def get_machine_capabilities():
    """
    Return capabilities of this machine for job scheduling decisions.
    This endpoint can be called by other machines to determine if this machine
    is suitable for a particular job.
    """
    try:
        # Get system information
        import platform
        import psutil
        import torch

        capabilities = {
            "system": {
                "platform": platform.system(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
            },
            "resources": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            },
            "ml_frameworks": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            },
        }

        # Add GPU information if available
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append(
                    {
                        "name": props.name,
                        "memory_total_mb": props.total_memory // (1024**2),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
            capabilities["gpu"] = gpu_info

        # Get current load
        current_jobs = await db.job_count_running()
        capabilities["current_load"] = {
            "running_jobs": current_jobs,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
        }

        # Get installed plugins
        from transformerlab.routers import plugins as plugins_router

        installed_plugins = await plugins_router.list_plugins()
        capabilities["installed_plugins"] = [p["uniqueId"] for p in installed_plugins]

        return {"status": "success", "capabilities": capabilities}

    except Exception as e:
        return {"status": "error", "message": f"Failed to get capabilities: {str(e)}"}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _install_remote_dependencies_job(job_id: str, dependencies: Dict[str, list[str]]):
    """
    Background job to install dependencies for remote job execution.
    Similar to _install_recipe_dependencies_job but for remote job deps.
    """
    try:
        await db.job_update_status(job_id, "RUNNING")

        plugins = dependencies.get("plugins", [])
        models = dependencies.get("models", [])
        datasets = dependencies.get("datasets", [])

        # total_deps = len(plugins) +
        total_deps = len(plugins) + len(models) + len(datasets)
        if total_deps == 0:
            await db.job_update_status(job_id, "COMPLETE")
            return

        progress = 0
        results = []

        # Install plugins
        for plugin_name in plugins:
            from transformerlab.routers import plugins as plugins_router

            print(f"Installing plugin: {plugin_name}")
            result = {"type": "plugin", "name": plugin_name, "action": None, "status": None}
            try:
                install_result = await plugins_router.install_plugin(plugin_id=plugin_name)
                print(f"Install result: {install_result}")
                result["action"] = "install_plugin"
                result["status"] = install_result.get("status", "unknown")
            except Exception as e:
                result["action"] = "error"
                result["status"] = str(e)

            results.append(result)
            progress += 1
            await db.job_update_progress(job_id, int(progress * 100 / total_deps))
            await db.job_update_job_data_insert_key_value(job_id, "results", results)

        # Install models
        for model_name in models:
            from transformerlab.routers import model as model_router

            result = {"type": "model", "name": model_name, "action": None, "status": None}
            try:
                download_result = await model_router.download_model_by_huggingface_id(model=model_name)
                result["action"] = "download_model"
                result["status"] = download_result.get("status", "unknown")
            except Exception as e:
                result["action"] = "error"
                result["status"] = str(e)

            results.append(result)
            progress += 1
            await db.job_update_progress(job_id, int(progress * 100 / total_deps))
            await db.job_update_job_data_insert_key_value(job_id, "results", results)

        # Install datasets
        for dataset_name in datasets:
            from transformerlab.routers import data as data_router

            result = {"type": "dataset", "name": dataset_name, "action": None, "status": None}
            try:
                download_result = await data_router.dataset_download(dataset_id=dataset_name)
                result["action"] = "download_dataset"
                result["status"] = download_result.get("status", "unknown")
            except Exception as e:
                result["action"] = "error"
                result["status"] = str(e)

            results.append(result)
            progress += 1
            await db.job_update_progress(job_id, int(progress * 100 / total_deps))
            await db.job_update_job_data_insert_key_value(job_id, "results", results)

        await db.job_update_status(job_id, "COMPLETE")

    except Exception as e:
        await db.job_update_status(job_id, "FAILED", error_msg=str(e))


# async def _sync_remote_job_progress(local_job_id: str, original_job_id: str, origin_machine: dict):
#     """
#     Background task to sync progress from local remote job execution back to origin machine.
#     Polls local job status and sends updates to the origin machine.
#     """
#     import asyncio
#     import httpx

#     try:
#         # Build origin machine URL (assuming same port for simplicity)
#         origin_host = origin_machine.get("ip", origin_machine.get("hostname"))
#         origin_url = f"http://{origin_host}:8338"  # Default port

#         while True:
#             try:
#                 # Get current job status and progress
#                 job = await db.job_get(local_job_id)
#                 if not job:
#                     break

#                 status = job.get("status")
#                 progress = job.get("progress", 0)

#                 # Send progress update to origin machine
#                 async with httpx.AsyncClient(timeout=10.0) as client:
#                     await client.post(
#                         f"{origin_url}/network/progress_update",
#                         json={"job_id": original_job_id, "progress": progress, "status": status},
#                     )

#                 # Stop syncing if job is complete or failed
#                 if status in ["COMPLETE", "FAILED", "CANCELLED"]:
#                     break

#                 # Wait before next sync
#                 await asyncio.sleep(5)  # Sync every 5 seconds

#             except Exception as e:
#                 print(f"Error syncing progress for job {local_job_id}: {str(e)}")
#                 await asyncio.sleep(10)  # Wait longer on error

#     except Exception as e:
#         print(f"Failed to start progress sync for job {local_job_id}: {str(e)}")


async def _get_this_machine_info():
    """Get information about this machine for identification purposes."""
    import socket
    import platform

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        return {"hostname": hostname, "ip": local_ip, "platform": platform.system(), "architecture": platform.machine()}
    except Exception:
        return {"hostname": "unknown", "ip": "unknown"}


@router.post("/progress_update", summary="Receive progress update from remote job")
async def receive_progress_update(progress_data: Dict[str, Any]):
    """
    Endpoint for remote machines to send progress updates back to the origin machine.
    Called by remote machines to update progress of jobs running remotely.
    """
    try:
        job_id = progress_data["job_id"]
        progress = progress_data["progress"]
        status = progress_data.get("status")

        # Update local job progress
        await db.job_update_progress(job_id, progress)

        # Update status if provided
        if status:
            await db.job_update_status(job_id, status)

        return {"status": "success", "message": "Progress updated"}

    except Exception as e:
        print(f"Error updating remote job progress: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.get("/local_job_status/{job_id}", summary="Get status of a local job (called by remote machines)")
async def get_local_job_status(job_id: str):
    """
    Endpoint for other machines to query the status of a job running on this machine.
    Returns job status, progress, and other details for remote monitoring.
    """
    try:
        job = await db.job_get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "status": "success",
            "job": {
                "id": job["id"],
                "status": job["status"],
                "progress": job.get("progress", 0),
                "type": job.get("type"),
                "created_at": job.get("created_at"),
                "updated_at": job.get("updated_at")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")
