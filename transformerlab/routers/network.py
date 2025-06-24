import asyncio
import os
import httpx
import zipfile
import tempfile
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
from enum import Enum

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


class DistributedTrainingRole(str, Enum):
    MASTER = "master"
    WORKER = "worker"


class DistributedJobCreate(BaseModel):
    job_id: str
    job_data: Dict[str, Any]
    target_machine_ids: list[int]  # Multiple machines
    master_machine_id: int  # Which machine acts as coordinator
    distributed_config: Dict[str, Any]  # Contains world_size, backend, etc.
    plugins_required: Optional[list[str]] = []
    models_required: Optional[list[str]] = []
    datasets_required: Optional[list[str]] = []


class DistributedJobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    master_machine_id: int
    worker_statuses: Dict[int, Dict[str, Any]]  # machine_id -> status info
    error_message: Optional[str] = None
    distributed_info: Optional[Dict[str, Any]] = None


class MachineCapabilities(BaseModel):
    machine_id: int
    gpu_count: int
    gpu_memory_total: int
    cpu_count: int
    memory_total: int
    network_bandwidth: Optional[float] = None
    current_load: Dict[str, Any]


class DistributedPlanRequest(BaseModel):
    required_gpus: Optional[int] = None
    model_size_gb: Optional[float] = None
    dataset_size_gb: Optional[float] = None
    preferred_machines: Optional[list[int]] = None
    exclude_host: bool = True


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
            "message": "Job execution started",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute remote job: {str(e)}")


@router.post("/execute_distributed_job", summary="Execute a distributed training job on this machine")
async def execute_distributed_job(job_request: Dict[str, Any]):
    """
    Execute a distributed training job on this machine as part of a distributed setup.
    This endpoint is called by the network orchestrator to start training on each machine.
    """
    try:
        job_id = job_request.get("job_id")
        job_data = job_request.get("job_data", {})
        origin_machine = job_request.get("origin_machine", {})

        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")

        # Create a local job record for tracking
        local_job_id = await db.job_create(
            type="DISTRIBUTED_TRAIN_LOCAL",
            status="QUEUED",
            job_data=json.dumps(job_data),
            experiment_id=job_data.get("experiment_id", ""),
        )

        # Store reference to original job and origin machine
        await db.job_update_job_data_insert_key_value(local_job_id, "original_job_id", job_id)
        await db.job_update_job_data_insert_key_value(local_job_id, "origin_machine", json.dumps(origin_machine))

        # Check if this is a distributed training job
        if job_data.get("distributed_training") or job_data.get("distributed_config"):
            # Use the distributed training runner
            from transformerlab.shared.shared import run_distributed_job

            # Start the job asynchronously
            asyncio.create_task(
                run_distributed_job(
                    job_id=local_job_id,
                    job_config=job_data,
                    experiment_name="default",  # Will be determined from job_data
                    job_details={"id": local_job_id, "type": "DISTRIBUTED_TRAIN_LOCAL"},
                )
            )
        else:
            # Fallback to regular job execution
            from transformerlab.shared.shared import run_job

            asyncio.create_task(
                run_job(
                    job_id=local_job_id,
                    job_config=job_data,
                    experiment_name="default",
                    job_details={"id": local_job_id, "type": "TRAIN"},
                )
            )

        return {
            "status": "success",
            "message": "Distributed training job started",
            "local_job_id": local_job_id,
            "original_job_id": job_id,
        }

    except Exception as e:
        print(f"ERROR: Failed to execute distributed job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute distributed job: {str(e)}")


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
        print("RETURNING CAPABILITIES", capabilities)

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
                "updated_at": job.get("updated_at"),
                "job_data": str(job.get("job_data", "{}")),
                "workspace_dir": os.getenv("_TFL_WORKSPACE_DIR", ""),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.get("/local_job_file/{job_id}/{key}", summary="Get a file from job data by key")
async def get_local_job_file(job_id: str, key: str):
    """
    Endpoint to serve files from job_data using a specified key.
    Returns the file specified by the key in the job's job_data as a FileResponse.
    """
    try:
        job = await db.job_get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Parse job_data if it's a string
        job_data = job.get("job_data", {})
        if isinstance(job_data, str):
            try:
                job_data = json.loads(job_data)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid job_data format")

        # Check if the key exists in job_data
        if key not in job_data:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found in job data")

        file_path = job_data[key]

        # Ensure the file path is a string
        if not isinstance(file_path, str):
            raise HTTPException(status_code=400, detail=f"Value for key '{key}' is not a valid file path")

        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # Handle directories by creating a zip file
        if os.path.isdir(file_path):
            # Create a temporary zip file
            temp_dir = tempfile.gettempdir()
            dir_name = os.path.basename(file_path.rstrip("/"))
            zip_filename = f"{dir_name}.zip"
            zip_path = os.path.join(temp_dir, f"job_{job_id}_{key}_{zip_filename}")

            try:
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(file_path):
                        for file in files:
                            file_path_in_zip = os.path.join(root, file)
                            arcname = os.path.relpath(file_path_in_zip, file_path)
                            zipf.write(file_path_in_zip, arcname)

                return FileResponse(
                    path=zip_path,
                    filename=zip_filename,
                    media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
                )
            except Exception as e:
                # Clean up the temp file if something went wrong
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                raise HTTPException(status_code=500, detail=f"Failed to create zip file: {str(e)}")

        # Get the filename for the response (for regular files)
        filename = os.path.basename(file_path)

        return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")


@router.get("/local_job_output/{job_id}", summary="Get the output file from job data by key")
async def get_local_job_output(job_id: str):
    """
    Endpoint to serve the output.txt file for given job.
    Returns the output.txt file stored in workspace_dir/jobs/<job_id>/output_<job_id>.txt
    """
    try:
        job = await db.job_get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get the workspace directory from environment variable
        workspace_dir = os.getenv("_TFL_WORKSPACE_DIR")
        if not workspace_dir:
            raise HTTPException(status_code=500, detail="Workspace directory is not configured")

        # Construct the output file path
        output_file_path = os.path.join(workspace_dir, "jobs", str(job_id), f"output_{job_id}.txt")

        # Check if the output file exists
        if not os.path.exists(output_file_path):
            raise HTTPException(status_code=404, detail=f"Output file not found: {output_file_path}")

        return FileResponse(path=output_file_path, filename=f"output_{job_id}.txt", media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Failed to serve output file for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve output file")


# =============================================================================
# DISTRIBUTED TRAINING ENDPOINTS
# =============================================================================


@router.post("/distributed/plan", summary="Plan a distributed training job")
async def plan_distributed_training(request: DistributedPlanRequest):
    """
    Analyze available machines and create an optimal plan for distributed training.
    Returns suggested machine allocation and configuration.

    Args:
        request: DistributedPlanRequest containing planning parameters
    """
    print("=== DISTRIBUTED TRAINING PLAN REQUEST RECEIVED ===")
    print(f"Request data: {request}")

    try:
        # Set default values for null parameters
        required_gpus = request.required_gpus if request.required_gpus is not None else 2
        model_size_gb = request.model_size_gb if request.model_size_gb is not None else 1.0
        dataset_size_gb = request.dataset_size_gb if request.dataset_size_gb is not None else 0.1
        preferred_machines = request.preferred_machines
        exclude_host = request.exclude_host

        # Get all online machines
        print("GETTING REQUEST TO PLAN DISTRIBUTED TRAINING")
        print(
            f"Parameters: required_gpus={required_gpus}, model_size_gb={model_size_gb}, dataset_size_gb={dataset_size_gb}"
        )

        all_machines = await db.network_machine_get_all()
        print(f"Total machines in database: {len(all_machines)}")

        online_machines = [m for m in all_machines if m.get("status") == "online"]
        print(f"Online machines: {len(online_machines)}")

        # Filter out host machine if requested (useful for Apple MLX hosts)
        if exclude_host:
            import socket

            host_ip = socket.gethostbyname(socket.gethostname())
            online_machines = [
                m
                for m in online_machines
                if m.get("host") != "localhost" and m.get("host") != "127.0.0.1" and m.get("host") != host_ip
            ]
            print(f"Excluding host machine (IP: {host_ip}). Remaining machines: {len(online_machines)}")

        if len(online_machines) < 1:
            error_msg = "At least 1 online machine required for distributed training"
            print(f"ERROR: {error_msg}")
            print("Available machines:")
            for m in all_machines:
                print(
                    f"  - {m.get('name', 'Unknown')} ({m.get('host', 'Unknown')}:{m.get('port', 'Unknown')}) - Status: {m.get('status', 'Unknown')}"
                )
            raise HTTPException(status_code=500, detail=error_msg)

        # For single machine testing, allow distributed training across multiple GPUs on that machine
        if len(online_machines) == 1:
            print("Single machine detected - will use multi-GPU distributed training on remote machine")

        # Get capabilities for each machine
        machine_capabilities = []
        for machine in online_machines:
            try:
                base_url = f"http://{machine['host']}:{machine['port']}"
                headers = {}
                if machine.get("api_token"):
                    headers["Authorization"] = f"Bearer {machine['api_token']}"

                print(f"Getting capabilities from machine {machine['id']} at {base_url}")
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{base_url}/network/capabilities", headers=headers)
                    if response.status_code == 200:
                        caps = response.json()
                        machine_capabilities.append(
                            {"machine_id": machine["id"], "machine_info": machine, "capabilities": caps}
                        )
                        print(f"Successfully got capabilities from machine {machine['id']}")
                    else:
                        print(f"Failed to get capabilities from machine {machine['id']}: HTTP {response.status_code}")
            except Exception as e:
                print(f"Failed to get capabilities for machine {machine['id']}: {e}")
                continue

        if len(machine_capabilities) == 0:
            error_msg = "No machines available with accessible capabilities"
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        print(f"Successfully got capabilities from {len(machine_capabilities)} machines")

        print("Machine capabilities:", machine_capabilities)

        # Plan the distribution
        try:
            plan = _create_distribution_plan(
                machine_capabilities, required_gpus, model_size_gb, dataset_size_gb, preferred_machines
            )
            print("Distribution plan created successfully")
        except Exception as e:
            print(f"ERROR in _create_distribution_plan: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create distribution plan: {str(e)}")

        return {
            "status": "success",
            "plan": plan,
            "total_machines": len(plan["machines"]),
            "total_gpus": sum(m["allocated_gpus"] for m in plan["machines"]),
            "estimated_memory_usage": plan["distributed_config"],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error planning distributed training: {e}")
        raise HTTPException(status_code=500, detail="Failed to plan distributed training")


@router.post("/distributed/dispatch", summary="Dispatch a distributed training job")
async def dispatch_distributed_training(distributed_job: DistributedJobCreate):
    """
    Dispatch a distributed training job across multiple machines.
    This coordinates the setup and execution across all target machines.
    """
    try:
        # Validate all target machines are online
        for machine_id in distributed_job.target_machine_ids:
            machine = await db.network_machine_get(machine_id)
            if not machine or machine["status"] != "online":
                raise HTTPException(status_code=400, detail=f"Machine {machine_id} is not available")

        master_machine = await db.network_machine_get(distributed_job.master_machine_id)
        if not master_machine:
            raise HTTPException(status_code=400, detail="Master machine not found")

        # Store distributed job info in database
        distributed_info = {
            "type": "DISTRIBUTED_TRAIN",
            "master_machine_id": distributed_job.master_machine_id,
            "worker_machine_ids": [
                mid for mid in distributed_job.target_machine_ids if mid != distributed_job.master_machine_id
            ],
            "distributed_config": distributed_job.distributed_config,
            "status": "INITIALIZING",
        }

        await db.job_update_job_data_insert_key_value(
            distributed_job.job_id, "distributed_info", json.dumps(distributed_info)
        )

        # Step 1: Prepare dependencies on all machines
        dependency_tasks = []
        for machine_id in distributed_job.target_machine_ids:
            dependency_tasks.append(
                _prepare_machine_dependencies(
                    machine_id,
                    distributed_job.plugins_required,
                    distributed_job.models_required,
                    distributed_job.datasets_required,
                )
            )

        # Wait for all dependency installations
        dependency_results = await asyncio.gather(*dependency_tasks, return_exceptions=True)
        failed_machines = []
        for i, result in enumerate(dependency_results):
            if isinstance(result, Exception):
                failed_machines.append(distributed_job.target_machine_ids[i])

        if failed_machines:
            raise HTTPException(
                status_code=500, detail=f"Failed to prepare dependencies on machines: {failed_machines}"
            )

        # Step 2: Set up distributed training environment
        world_size = len(distributed_job.target_machine_ids)
        distributed_config = distributed_job.distributed_config
        distributed_config.update(
            {
                "world_size": world_size,
                "master_addr": master_machine["host"],
                "master_port": distributed_config.get("master_port", 29500),
                "backend": distributed_config.get("backend", "nccl"),
            }
        )

        # Step 3: Start training on all machines
        training_tasks = []

        # Start master first
        master_job_data = distributed_job.job_data.copy()
        master_job_data.update(
            {"distributed_role": "master", "rank": 0, "local_rank": 0, "distributed_config": distributed_config}
        )

        training_tasks.append(
            _start_distributed_training_on_machine(
                distributed_job.master_machine_id, distributed_job.job_id, master_job_data
            )
        )

        # Start workers
        worker_rank = 1
        for machine_id in distributed_job.target_machine_ids:
            if machine_id != distributed_job.master_machine_id:
                worker_job_data = distributed_job.job_data.copy()
                worker_job_data.update(
                    {
                        "distributed_role": "worker",
                        "rank": worker_rank,
                        "local_rank": 0,  # Assuming single GPU per machine for now
                        "distributed_config": distributed_config,
                    }
                )

                training_tasks.append(
                    _start_distributed_training_on_machine(
                        machine_id, f"{distributed_job.job_id}_worker_{worker_rank}", worker_job_data
                    )
                )
                worker_rank += 1

        # Start all training processes
        await asyncio.gather(*training_tasks, return_exceptions=True)

        # Update job status
        await db.job_update_status(distributed_job.job_id, "RUNNING")
        await db.job_update_job_data_insert_key_value(distributed_job.job_id, "distributed_status", "RUNNING")

        return {
            "status": "success",
            "message": "Distributed training job started successfully",
            "job_id": distributed_job.job_id,
            "master_machine_id": distributed_job.master_machine_id,
            "worker_machines": [
                mid for mid in distributed_job.target_machine_ids if mid != distributed_job.master_machine_id
            ],
            "world_size": world_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Failed to dispatch distributed training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to dispatch distributed training: {str(e)}")


@router.get("/distributed/status/{job_id}", summary="Get distributed training job status")
async def get_distributed_training_status(job_id: str):
    """Get comprehensive status of a distributed training job across all machines."""
    try:
        # Get main job info
        job = await db.job_get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        job_data = json.loads(job.get("job_data", "{}"))
        distributed_info = json.loads(job_data.get("distributed_info", "{}"))

        if distributed_info.get("type") != "DISTRIBUTED_TRAIN":
            raise HTTPException(status_code=400, detail="Job is not a distributed training job")

        master_machine_id = distributed_info["master_machine_id"]
        worker_machine_ids = distributed_info["worker_machine_ids"]

        # Get status from master machine
        master_status = await _get_machine_job_status(master_machine_id, job_id)

        # Get status from all worker machines
        worker_statuses = {}
        for i, machine_id in enumerate(worker_machine_ids):
            worker_job_id = f"{job_id}_worker_{i + 1}"
            worker_statuses[machine_id] = await _get_machine_job_status(machine_id, worker_job_id)

        # Aggregate status
        all_statuses = [master_status["status"]] + [ws["status"] for ws in worker_statuses.values()]

        if all(s == "COMPLETE" for s in all_statuses):
            overall_status = "COMPLETE"
        elif any(s == "FAILED" for s in all_statuses):
            overall_status = "FAILED"
        elif any(s == "RUNNING" for s in all_statuses):
            overall_status = "RUNNING"
        else:
            overall_status = "INITIALIZING"

        return {
            "status": "success",
            "job_id": job_id,
            "overall_status": overall_status,
            "master_status": master_status,
            "worker_statuses": worker_statuses,
            "distributed_info": distributed_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Failed to get distributed training status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get distributed training status")


@router.post("/distributed/stop/{job_id}", summary="Stop distributed training job")
async def stop_distributed_training(job_id: str):
    """Stop a distributed training job on all machines."""
    try:
        # Get job info
        job = await db.job_get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        job_data = json.loads(job.get("job_data", "{}"))
        distributed_info = json.loads(job_data.get("distributed_info", "{}"))

        master_machine_id = distributed_info["master_machine_id"]
        worker_machine_ids = distributed_info["worker_machine_ids"]

        # Stop master
        stop_tasks = [_stop_machine_job(master_machine_id, job_id)]

        # Stop all workers
        for i, machine_id in enumerate(worker_machine_ids):
            worker_job_id = f"{job_id}_worker_{i + 1}"
            stop_tasks.append(_stop_machine_job(machine_id, worker_job_id))

        # Execute all stop commands
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Update local job status
        await db.job_update_status(job_id, "STOPPED")

        return {"status": "success", "message": "Distributed training job stop signals sent to all machines"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Failed to stop distributed training: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop distributed training")


# =============================================================================
# DISTRIBUTED TRAINING HELPER FUNCTIONS
# =============================================================================


def _create_distribution_plan(
    machine_capabilities, required_gpus, model_size_gb, dataset_size_gb, preferred_machines=None
):
    """
    Create an optimal distribution plan for training across machines.
    """
    print("=== CREATING DISTRIBUTION PLAN ===")
    print(f"Required GPUs: {required_gpus}")
    print(f"Model size: {model_size_gb}GB")
    print(f"Dataset size: {dataset_size_gb}GB")
    print(f"Preferred machines: {preferred_machines}")
    print(f"Number of machine capabilities: {len(machine_capabilities)}")

    available_machines = []

    for machine_cap in machine_capabilities:
        caps_wrapper = machine_cap["capabilities"]
        # The actual capabilities are nested under capabilities.capabilities
        caps = caps_wrapper.get("capabilities", caps_wrapper)
        print(f"Processing machine {machine_cap['machine_id']}")

        # Extract GPU and memory info
        gpu_count = caps.get("ml_frameworks", {}).get("cuda_device_count", 0)
        gpu_info = caps.get("gpu", [])
        total_gpu_memory = sum(gpu.get("memory_total_mb", 0) for gpu in gpu_info) if gpu_info else 0

        memory_available = caps.get("resources", {}).get("memory_available_gb", 0)
        current_load = caps.get("current_load", {})

        print(f"  GPU count: {gpu_count}")
        print(f"  Total GPU memory: {total_gpu_memory}MB")
        print(f"  Available memory: {memory_available}GB")
        print(f"  Current load: {current_load}")

        # Calculate suitability score
        load_factor = 1.0 - (current_load.get("cpu_percent", 0) / 100.0)
        memory_factor = 1.0 - (current_load.get("memory_percent", 0) / 100.0)

        suitability_score = (gpu_count * 10) + (total_gpu_memory / 1000) + (load_factor * 5) + (memory_factor * 5)

        print(f"  Suitability score: {suitability_score}")

        available_machines.append(
            {
                "machine_id": machine_cap["machine_id"],
                "machine_info": machine_cap["machine_info"],
                "gpu_count": gpu_count,
                "gpu_memory_mb": total_gpu_memory,
                "memory_available_gb": memory_available,
                "suitability_score": suitability_score,
                "current_load": current_load,
            }
        )

    # Sort by suitability score (highest first)
    available_machines.sort(key=lambda x: x["suitability_score"], reverse=True)

    print(f"Available machines after sorting: {len(available_machines)}")
    for i, machine in enumerate(available_machines):
        print(
            f"  {i + 1}. Machine {machine['machine_id']}: {machine['gpu_count']} GPUs, score: {machine['suitability_score']}"
        )

    # Select machines for the plan
    selected_machines = []
    total_allocated_gpus = 0

    # Prefer specific machines if provided
    if preferred_machines:
        print(f"Using preferred machines: {preferred_machines}")
        for machine_id in preferred_machines:
            machine = next((m for m in available_machines if m["machine_id"] == machine_id), None)
            if machine and total_allocated_gpus < required_gpus:
                allocated_gpus = min(machine["gpu_count"], required_gpus - total_allocated_gpus)
                if allocated_gpus > 0:
                    print(f"  Selected preferred machine {machine_id}: {allocated_gpus} GPUs")
                    selected_machines.append(
                        {
                            **machine,
                            "allocated_gpus": allocated_gpus,
                            "role": "master" if len(selected_machines) == 0 else "worker",
                        }
                    )
                    total_allocated_gpus += allocated_gpus

    print(f"After preferred machines: {total_allocated_gpus}/{required_gpus} GPUs allocated")

    # Fill remaining GPU requirements with best available machines
    for machine in available_machines:
        if total_allocated_gpus >= required_gpus:
            print(f"GPU requirement met: {total_allocated_gpus}/{required_gpus}")
            break

        # Skip if already selected
        if any(sm["machine_id"] == machine["machine_id"] for sm in selected_machines):
            print(f"  Skipping already selected machine {machine['machine_id']}")
            continue

        allocated_gpus = min(machine["gpu_count"], required_gpus - total_allocated_gpus)
        if allocated_gpus > 0:
            print(f"  Selecting machine {machine['machine_id']}: {allocated_gpus} GPUs")
            selected_machines.append(
                {
                    **machine,
                    "allocated_gpus": allocated_gpus,
                    "role": "master" if len(selected_machines) == 0 else "worker",
                }
            )
            total_allocated_gpus += allocated_gpus
        else:
            print(f"  Machine {machine['machine_id']} has no available GPUs")

    print(f"Final selection: {len(selected_machines)} machines, {total_allocated_gpus} total GPUs")
    for machine in selected_machines:
        print(f"  - Machine {machine['machine_id']}: {machine['allocated_gpus']} GPUs, role: {machine['role']}")

    # Calculate estimated memory usage
    estimated_memory_per_gpu = (model_size_gb * 1.5) + (dataset_size_gb * 0.1)  # Rough estimate
    estimated_total_memory = estimated_memory_per_gpu * total_allocated_gpus

    # Recommended distributed configuration
    distributed_config = {
        "backend": "nccl" if any(m["gpu_count"] > 0 for m in selected_machines) else "gloo",
        "init_method": "tcp",
        "master_port": 29500,
        "world_size": len(selected_machines),
        "gradient_compression": total_allocated_gpus > 4,  # Enable compression for larger setups
        "bucket_size_mb": 25 if total_allocated_gpus <= 4 else 50,
    }

    return {
        "machines": selected_machines,
        "total_gpus_allocated": total_allocated_gpus,
        "estimated_memory_usage": estimated_total_memory,
        "distributed_config": distributed_config,
        "master_machine_id": selected_machines[0]["machine_id"] if selected_machines else None,
    }


async def _prepare_machine_dependencies(machine_id, plugins_required, models_required, datasets_required):
    """Prepare dependencies on a specific machine."""
    try:
        machine = await db.network_machine_get(machine_id)
        if not machine:
            raise Exception(f"Machine {machine_id} not found")

        base_url = f"http://{machine['host']}:{machine['port']}"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/network/prepare_dependencies",
                json={
                    "plugins": plugins_required or [],
                    "models": models_required or [],
                    "datasets": datasets_required or [],
                },
                headers=headers,
            )

            if response.status_code != 200:
                raise Exception(f"Failed to prepare dependencies on machine {machine_id}")

            return {"machine_id": machine_id, "status": "success"}

    except Exception as e:
        raise Exception(f"Dependency preparation failed for machine {machine_id}: {str(e)}")


async def _start_distributed_training_on_machine(machine_id, job_id, job_data):
    """Start distributed training on a specific machine."""
    try:
        machine = await db.network_machine_get(machine_id)
        if not machine:
            raise Exception(f"Machine {machine_id} not found")

        base_url = f"http://{machine['host']}:{machine['port']}"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/network/execute_distributed_job",
                json={
                    "job_id": job_id,
                    "job_data": job_data,
                    "origin_machine": await _get_this_machine_info(),
                },
                headers=headers,
            )

            if response.status_code != 200:
                raise Exception(f"Failed to start training on machine {machine_id}")

            return {"machine_id": machine_id, "job_id": job_id, "status": "started"}

    except Exception as e:
        raise Exception(f"Training start failed for machine {machine_id}: {str(e)}")


async def _get_machine_job_status(machine_id, job_id):
    """Get job status from a specific machine."""
    try:
        machine = await db.network_machine_get(machine_id)
        if not machine:
            return {"status": "ERROR", "message": "Machine not found"}

        base_url = f"http://{machine['host']}:{machine['port']}"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/jobs/{job_id}", headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "ERROR", "message": "Failed to get status"}

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


async def _stop_machine_job(machine_id, job_id):
    """Stop a job on a specific machine."""
    try:
        machine = await db.network_machine_get(machine_id)
        if not machine:
            return {"status": "ERROR", "message": "Machine not found"}

        base_url = f"http://{machine['host']}:{machine['port']}"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/jobs/{job_id}/stop", headers=headers)

            if response.status_code == 200:
                return {"status": "success"}
            else:
                return {"status": "error", "message": "Failed to stop job"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
