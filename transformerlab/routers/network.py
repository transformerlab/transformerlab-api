import asyncio
import os
import httpx
import zipfile
import tempfile
import socket
import platform
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
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


class NetworkMachineReservationRequest(BaseModel):
    duration_minutes: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class NetworkMachineReservationResponse(BaseModel):
    id: int
    name: str
    host: str
    port: int
    is_reserved: bool
    reserved_by_host: Optional[str] = None
    reserved_at: Optional[datetime] = None
    reservation_duration_minutes: Optional[int] = None
    reservation_metadata: Optional[Dict[str, Any]] = None


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


# Quota system models
class QuotaConfig(BaseModel):
    host_identifier: str
    time_period: str  # 'daily', 'weekly', 'monthly', 'yearly'
    minutes_limit: int
    warning_threshold_percent: int = 80
    is_active: bool = True


class QuotaUsage(BaseModel):
    host_identifier: str
    time_period: str
    period_start_date: str
    minutes_used: int
    minutes_limit: int
    usage_percent: float
    remaining_minutes: int
    is_warning: bool
    is_exceeded: bool


class QuotaCheck(BaseModel):
    can_reserve: bool
    requested_minutes: int
    quota_status: Dict[str, QuotaUsage]
    warnings: list[str] = []
    errors: list[str] = []


class QuotaReservationRequest(BaseModel):
    duration_minutes: int
    machine_id: Optional[int] = None


# Dashboard analytics models
class DashboardAnalyticsRequest(BaseModel):
    time_range: int = 30  # days


class ReservationByHost(BaseModel):
    host: str
    reservations: int
    totalMinutes: int
    activeReservations: int


class ReservationByMachine(BaseModel):
    machine: str
    reservations: int
    totalMinutes: int
    avgDuration: float


class UsageOverTime(BaseModel):
    date: str
    reservations: int
    minutes: int


class QuotaUtilization(BaseModel):
    host: str
    daily: float
    weekly: float
    monthly: float


class DashboardAnalyticsResponse(BaseModel):
    data: Dict[str, Any]


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
    """Get all registered network machines with reservation status."""
    try:
        machines = await db.network_machine_get_all()

        # Clean up expired reservations before returning the list
        await db.network_machine_cleanup_expired_reservations()

        # Get updated machines after cleanup
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
    """Get aggregated status information for all network machines including reservations."""
    try:
        # Clean up expired reservations first
        await db.network_machine_cleanup_expired_reservations()

        machines = await db.network_machine_get_all()

        total_machines = len(machines)
        online_machines = len([m for m in machines if m.get("status") == "online"])
        offline_machines = len([m for m in machines if m.get("status") == "offline"])
        error_machines = len([m for m in machines if m.get("status") == "error"])

        # Add reservation statistics
        reserved_machines = len([m for m in machines if m.get("is_reserved")])
        available_machines = total_machines - reserved_machines

        # Get reservation details by host
        reservations = await db.network_machine_get_all_reservations()
        reservation_by_host = {}
        for machine in reservations:
            host = machine.get("reserved_by_host")
            if host:
                if host not in reservation_by_host:
                    reservation_by_host[host] = []
                reservation_by_host[host].append(
                    {
                        "machine_id": machine["id"],
                        "machine_name": machine["name"],
                        "reserved_at": machine.get("reserved_at"),
                        "duration_minutes": machine.get("reservation_duration_minutes"),
                    }
                )

        return {
            "status": "success",
            "data": {
                "total_machines": total_machines,
                "online": online_machines,
                "offline": offline_machines,
                "error": error_machines,
                "reserved": reserved_machines,
                "available": available_machines,
                "reservation_by_host": reservation_by_host,
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

        # Check if machine is available for this host
        host_identifier = await _get_host_identifier()
        is_available = await _is_machine_available_for_reservation(machine_id, host_identifier)
        if not is_available:
            reserved_by = machine.get("reserved_by_host", "unknown")
            raise HTTPException(status_code=403, detail=f"Machine is reserved by another host: {reserved_by}")

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


async def _get_this_machine_info():
    """Get information about this machine for identification purposes."""

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        return {"hostname": hostname, "ip": local_ip, "platform": platform.system(), "architecture": platform.machine()}
    except Exception:
        return {"hostname": "unknown", "ip": "unknown"}


async def _get_host_identifier():
    """
    Get a unique identifier for this host machine.
    This combines hostname, IP, and optionally a custom identifier from config.
    """

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        # Check if there's a custom host identifier in config
        custom_id = await db.config_get("HOST_IDENTIFIER")
        if custom_id:
            return f"{custom_id}:{hostname}:{local_ip}"

        return f"{hostname}:{local_ip}:{platform.machine()}"
    except Exception:
        return "unknown-host"


async def _check_reservation_allowed(target_machine: Dict[str, Any]) -> bool:
    """
    Check if this machine can be reserved by checking the _TLAB_RESERVED config.
    Returns True if the machine allows reservations, False otherwise.
    """
    try:
        # Make request to target machine to check its reservation config
        url = f"http://{target_machine['host']}:{target_machine['port']}/network/reservation_config"
        headers = {}
        if target_machine.get("api_token"):
            headers["Authorization"] = f"Bearer {target_machine['api_token']}"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                config_data = response.json()
                return config_data.get("allows_reservations", True)
            else:
                # If endpoint doesn't exist, assume reservations are allowed for backward compatibility
                return True
    except Exception:
        # If we can't check, assume reservations are allowed
        return True


async def _is_machine_available_for_reservation(machine_id: int, requesting_host: str) -> bool:
    """
    Check if a machine is available for reservation by the requesting host.
    Returns False if machine is reserved by a different host.
    """
    machine = await db.network_machine_get(machine_id)
    if not machine:
        return False

    # Check if machine is reserved
    if machine.get("is_reserved"):
        # If reserved by the same host, it's available
        return machine.get("reserved_by_host") == requesting_host

    # Not reserved, so it's available
    return True


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


# Default quota configurations
DEFAULT_QUOTAS = {
    "daily": 480,  # 8 hours per day
    "weekly": 2400,  # 40 hours per week
    "monthly": 9600,  # 160 hours per month
    "yearly": 115200,  # 1920 hours per year
}

WARNING_THRESHOLDS = {
    "daily": 80,  # Warn at 80%
    "weekly": 80,  # Warn at 80%
    "monthly": 85,  # Warn at 85%
    "yearly": 90,  # Warn at 90%
}


@router.get("/quota/config", summary="Get quota configuration for current host")
async def get_quota_config():
    """Get quota limits and configuration for the current host."""
    try:
        host_identifier = await _get_host_identifier()
        quota_configs = await db.network_quota_get_config(host_identifier)

        # If no config exists, return defaults
        if not quota_configs:
            await _create_default_quota_config(host_identifier)
            quota_configs = await db.network_quota_get_config(host_identifier)

        return {"status": "success", "data": quota_configs, "host": host_identifier}
    except Exception as e:
        print(f"ERROR: Error getting quota config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/quota/config", summary="Set quota configuration (admin)")
async def set_quota_config(configs: list[QuotaConfig]):
    """Set quota limits for hosts. Admin endpoint."""
    try:
        results = []
        for config in configs:
            success = await db.network_quota_set_config(
                host_identifier=config.host_identifier,
                time_period=config.time_period,
                minutes_limit=config.minutes_limit,
                warning_threshold_percent=config.warning_threshold_percent,
                is_active=config.is_active,
            )
            results.append({"host": config.host_identifier, "period": config.time_period, "success": success})

        return {"status": "success", "results": results}
    except Exception as e:
        print(f"ERROR: Error setting quota config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/quota/usage", summary="Get current quota usage for this host")
async def get_quota_usage():
    """Get current quota usage across all time periods for this host."""
    try:
        host_identifier = await _get_host_identifier()
        usage_data = await _get_host_quota_usage(host_identifier)

        return {"status": "success", "data": usage_data, "host": host_identifier}
    except Exception as e:
        print(f"ERROR: Error getting quota usage: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/quota/usage/{time_period}", summary="Get quota usage for specific time period")
async def get_quota_usage_period(time_period: str):
    """Get quota usage for a specific time period (daily, weekly, monthly, yearly)."""
    try:
        host_identifier = await _get_host_identifier()
        usage = await _get_period_quota_usage(host_identifier, time_period)

        if not usage:
            raise HTTPException(status_code=404, detail=f"No quota configured for period: {time_period}")

        return {"status": "success", "data": usage, "host": host_identifier}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error getting quota usage for {time_period}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/quota/remaining", summary="Get remaining quota time for this host")
async def get_quota_remaining():
    """Get remaining quota time across all periods for this host."""
    try:
        host_identifier = await _get_host_identifier()
        usage_data = await _get_host_quota_usage(host_identifier)

        remaining_data = {}
        for period, usage in usage_data.items():
            remaining_data[period] = {
                "remaining_minutes": usage.remaining_minutes,
                "remaining_hours": round(usage.remaining_minutes / 60, 2),
                "usage_percent": usage.usage_percent,
                "is_warning": usage.is_warning,
                "is_exceeded": usage.is_exceeded,
            }

        return {"status": "success", "data": remaining_data, "host": host_identifier}
    except Exception as e:
        print(f"ERROR: Error getting remaining quota: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/quota/check", summary="Check if quota allows a reservation")
async def check_quota_availability(request: QuotaReservationRequest):
    """
    Check if current quota allows for a reservation of specified duration.
    Returns warnings and errors if quota would be exceeded.
    """
    try:
        host_identifier = await _get_host_identifier()
        quota_check = await _check_quota_for_reservation(host_identifier, request.duration_minutes)

        return {"status": "success", "data": quota_check}
    except Exception as e:
        print(f"ERROR: Error checking quota: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/quota/status/{machine_id}", summary="Check quota status for specific machine reservation")
async def check_machine_quota_status(machine_id: int, duration_minutes: int = 60):
    """Check if current quota allows reserving a specific machine for given duration."""
    try:
        machine = await db.network_machine_get(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")

        host_identifier = await _get_host_identifier()
        quota_check = await _check_quota_for_reservation(host_identifier, duration_minutes)

        # Add machine-specific information
        quota_check_data = quota_check.model_dump()
        quota_check_data["target_machine"] = {
            "id": machine["id"],
            "name": machine["name"],
            "host": machine["host"],
            "port": machine["port"],
        }

        return {"status": "success", "data": quota_check_data}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error checking machine quota status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Admin endpoints for quota management
@router.get("/quota/admin/hosts", summary="Get quota info for all hosts (admin)")
async def get_all_hosts_quota():
    """Get quota configuration and usage for all hosts. Admin endpoint."""
    try:
        all_quota_data = await db.network_quota_get_all_hosts_usage()
        return {"status": "success", "data": all_quota_data}
    except Exception as e:
        print(f"ERROR: Error getting all hosts quota: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/quota/admin/reset/{host_identifier}", summary="Reset quota usage for host (admin)")
async def reset_host_quota(host_identifier: str, time_periods: Optional[list[str]] = None):
    """Reset quota usage for a specific host. Admin endpoint."""
    try:
        periods_to_reset = time_periods or ["daily", "weekly", "monthly", "yearly"]
        reset_results = []

        for period in periods_to_reset:
            success = await db.network_quota_reset_usage(host_identifier, period)
            reset_results.append({"period": period, "success": success})

        return {"status": "success", "host": host_identifier, "results": reset_results}
    except Exception as e:
        print(f"ERROR: Error resetting quota for {host_identifier}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# QUOTA HELPER FUNCTIONS
# =============================================================================


async def _create_default_quota_config(host_identifier: str):
    """Create default quota configuration for a new host."""
    for period, limit in DEFAULT_QUOTAS.items():
        warning_threshold = WARNING_THRESHOLDS.get(period, 80)
        await db.network_quota_set_config(
            host_identifier=host_identifier,
            time_period=period,
            minutes_limit=limit,
            warning_threshold_percent=warning_threshold,
            is_active=True,
        )


async def _get_host_quota_usage(host_identifier: str) -> Dict[str, QuotaUsage]:
    """Get quota usage for all time periods for a host."""
    usage_data = {}

    # Get all configured periods for this host
    configs = await db.network_quota_get_config(host_identifier)

    for config in configs:
        if not config.get("is_active", True):
            continue

        period = config["time_period"]
        usage = await _get_period_quota_usage(host_identifier, period)
        if usage:
            usage_data[period] = usage

    return usage_data


async def _get_period_quota_usage(host_identifier: str, time_period: str) -> Optional[QuotaUsage]:
    """Get quota usage for a specific time period."""
    try:
        # Get quota config
        config = await db.network_quota_get_period_config(host_identifier, time_period)
        if not config:
            return None

        # Get current usage
        period_start = _get_period_start_date(time_period)
        usage_record = await db.network_quota_get_usage(host_identifier, time_period, period_start)

        minutes_used = usage_record.get("minutes_used", 0) if usage_record else 0
        minutes_limit = config["minutes_limit"]
        usage_percent = (minutes_used / minutes_limit * 100) if minutes_limit > 0 else 0
        remaining_minutes = max(0, minutes_limit - minutes_used)
        warning_threshold = config.get("warning_threshold_percent", 80)

        return QuotaUsage(
            host_identifier=host_identifier,
            time_period=time_period,
            period_start_date=period_start,
            minutes_used=minutes_used,
            minutes_limit=minutes_limit,
            usage_percent=round(usage_percent, 2),
            remaining_minutes=remaining_minutes,
            is_warning=usage_percent >= warning_threshold,
            is_exceeded=usage_percent >= 100,
        )
    except Exception as e:
        print(f"ERROR: Error getting period quota usage: {e}")
        return None


async def _check_quota_for_reservation(host_identifier: str, duration_minutes: int) -> QuotaCheck:
    """Check if a reservation would exceed quota limits."""
    usage_data = await _get_host_quota_usage(host_identifier)

    can_reserve = True
    warnings = []
    errors = []

    for period, usage in usage_data.items():
        # Fetch config for this period to get warning threshold
        config = await db.network_quota_get_period_config(host_identifier, period)
        warning_threshold_percent = config.get("warning_threshold_percent", 80) if config else 80

        # Check if adding duration would exceed limit
        new_usage_minutes = usage.minutes_used + duration_minutes
        new_usage_percent = (new_usage_minutes / usage.minutes_limit * 100) if usage.minutes_limit > 0 else 0

        if new_usage_percent >= 100:
            can_reserve = False
            over_limit = new_usage_minutes - usage.minutes_limit
            errors.append(f"{period.title()} quota exceeded: would use {over_limit} minutes over limit")
        elif new_usage_percent >= warning_threshold_percent:
            warnings.append(f"{period.title()} quota warning: would reach {new_usage_percent:.1f}% usage")

    return QuotaCheck(
        can_reserve=can_reserve,
        requested_minutes=duration_minutes,
        quota_status=usage_data,
        warnings=warnings,
        errors=errors,
    )


def _get_period_start_date(time_period: str) -> str:
    """Get the start date for a given time period."""
    now = datetime.now()

    if time_period == "daily":
        return now.strftime("%Y-%m-%d")
    elif time_period == "weekly":
        # Start of week (Monday)
        days_since_monday = now.weekday()
        monday = now - timedelta(days=days_since_monday)
        return monday.strftime("%Y-%m-%d")
    elif time_period == "monthly":
        # Start of month
        return now.strftime("%Y-%m-01")
    elif time_period == "yearly":
        # Start of year
        return now.strftime("%Y-01-01")
    else:
        # Default to daily
        return now.strftime("%Y-%m-%d")


async def _record_quota_usage(host_identifier: str, duration_minutes: int, machine_id: int):
    """Record quota usage when a reservation is completed."""
    try:
        # Record in history
        await db.network_quota_record_history(
            host_identifier=host_identifier, machine_id=machine_id, minutes_used=duration_minutes
        )

        # Update usage for all active periods
        configs = await db.network_quota_get_config(host_identifier)

        for config in configs:
            if not config.get("is_active", True):
                continue

            period = config["time_period"]
            period_start = _get_period_start_date(period)

            await db.network_quota_add_usage(host_identifier, period, period_start, duration_minutes)

    except Exception as e:
        print(f"ERROR: Error recording quota usage: {e}")


# =============================================================================
# MACHINE RESERVATION ENDPOINTS WITH QUOTA CHECKING
# =============================================================================


@router.get("/reservation_config", summary="Get reservation configuration for this machine")
async def get_reservation_config():
    """
    Return whether this machine allows reservations.
    This endpoint is called by other hosts to check if they can reserve this machine.
    """
    try:
        # Check if reservations are disabled via config
        tlab_reserved = await db.config_get("_TLAB_RESERVED")
        allows_reservations = tlab_reserved != "disabled"

        # Get current reservation status
        # Note: This assumes the machine can determine its own ID,
        # but we'll return general reservation config for now

        return {
            "status": "success",
            "allows_reservations": allows_reservations,
            "config": {"_TLAB_RESERVED": tlab_reserved or "enabled"},
        }
    except Exception as e:
        print(f"ERROR: Error getting reservation config: {e}")
        return {
            "status": "success",
            "allows_reservations": True,  # Default to allow for backward compatibility
            "config": {},
        }


@router.post("/machines/{machine_id}/reserve", summary="Reserve a network machine")
async def reserve_machine(machine_id: int, reservation: NetworkMachineReservationRequest):
    """Reserve a network machine for this host with quota checking."""
    try:
        # Get the machine to check if it exists and is reservable
        machine = await db.network_machine_get(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")

        # Check if the target machine allows reservations
        allows_reservations = await _check_reservation_allowed(machine)
        if not allows_reservations:
            raise HTTPException(status_code=403, detail="Target machine has reservations disabled")

        # Get our host identifier
        host_identifier = await _get_host_identifier()

        # Check quota before making reservation
        duration_minutes = reservation.duration_minutes or 60  # Default to 1 hour if not specified
        quota_check = await _check_quota_for_reservation(host_identifier, duration_minutes)

        if not quota_check.can_reserve:
            error_msg = f"Quota exceeded: {'; '.join(quota_check.errors)}"
            raise HTTPException(status_code=400, detail=error_msg)

        # Show warnings if any (but still allow reservation)
        warnings = quota_check.warnings
        if warnings:
            print(f"QUOTA WARNING for {host_identifier}: {'; '.join(warnings)}")

        # Check for expired reservations first
        await db.network_machine_check_reservation_expired(machine_id)

        # Attempt to reserve the machine
        success, message = await db.network_machine_reserve(
            machine_id=machine_id,
            host_identifier=host_identifier,
            duration_minutes=duration_minutes,
            metadata=reservation.metadata or {},
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        # Get updated machine info
        updated_machine = await db.network_machine_get(machine_id)

        # Record quota usage for this reservation
        await _record_quota_usage(host_identifier, duration_minutes, machine_id)

        response_data = {
            "status": "success",
            "message": message,
            "machine": updated_machine,
            "reserved_by": host_identifier,
            "quota_info": {
                "duration_minutes": duration_minutes,
                "warnings": warnings,
                "quota_status": {k: v.model_dump() for k, v in quota_check.quota_status.items()},
            },
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error reserving machine {machine_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/machines/{machine_id}/release", summary="Release a reserved network machine")
async def release_machine(machine_id: int):
    """Release a network machine reservation and update quota usage if released early."""
    try:
        # Get our host identifier
        host_identifier = await _get_host_identifier()

        # Get the machine info before releasing
        machine = await db.network_machine_get(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")
        if not machine.get("is_reserved"):
            raise HTTPException(status_code=400, detail="Machine is not reserved")
        if machine.get("reserved_by_host") != host_identifier:
            raise HTTPException(status_code=403, detail="Machine is reserved by a different host")

        reserved_at = machine.get("reserved_at")
        reserved_minutes = machine.get("reservation_duration_minutes")
        if reserved_at and reserved_minutes:
            # Calculate actual usage in minutes
            from datetime import datetime

            reserved_at_dt = reserved_at if isinstance(reserved_at, datetime) else datetime.fromisoformat(reserved_at)
            now = datetime.now()
            actual_minutes = int((now - reserved_at_dt).total_seconds() // 60)
            if actual_minutes < 1:
                actual_minutes = 1  # Minimum 1 minute
            # Only adjust if released early
            if actual_minutes < reserved_minutes:
                await _adjust_quota_usage_for_actual_time(host_identifier, machine_id, reserved_minutes, actual_minutes)

        # Attempt to release the machine
        success, message = await db.network_machine_release(machine_id=machine_id, host_identifier=host_identifier)

        if not success:
            raise HTTPException(status_code=400, detail=message)

        return {"status": "success", "message": message}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error releasing machine {machine_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _adjust_quota_usage_for_actual_time(
    host_identifier: str, machine_id: int, reserved_minutes: int, actual_minutes: int
):
    """Adjust quota usage for all periods to reflect actual usage instead of reserved duration."""
    try:
        configs = await db.network_quota_get_config(host_identifier)
        for config in configs:
            if not config.get("is_active", True):
                continue
            period = config["time_period"]
            period_start = _get_period_start_date(period)
            # Subtract reserved, add actual
            # Subtract reserved
            await db.network_quota_add_usage(host_identifier, period, period_start, -reserved_minutes)
            # Add actual
            await db.network_quota_add_usage(host_identifier, period, period_start, actual_minutes)
        # Optionally, update quota history (not implemented here)
    except Exception as e:
        print(f"ERROR: Error adjusting quota usage for early release: {e}")


@router.post("/machines/by-name/{machine_name}/reserve", summary="Reserve a network machine by name")
async def reserve_machine_by_name(machine_name: str, reservation: NetworkMachineReservationRequest):
    """Reserve a network machine by name for this host."""
    try:
        # Get machine by name first
        machine = await db.network_machine_get_by_name(machine_name)
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")

        # Use the reserve by ID endpoint
        return await reserve_machine(machine["id"], reservation)

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error reserving machine '{machine_name}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/machines/by-name/{machine_name}/release", summary="Release a reserved network machine by name")
async def release_machine_by_name(machine_name: str):
    """Release a network machine reservation by name."""
    try:
        # Get machine by name first
        machine = await db.network_machine_get_by_name(machine_name)
        if not machine:
            raise HTTPException(status_code=404, detail="Machine not found")

        # Use the release by ID endpoint
        return await release_machine(machine["id"])

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error releasing machine '{machine_name}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reservations", summary="Get all machine reservations")
async def get_all_reservations():
    """Get all machine reservations across the network."""
    try:
        reservations = await db.network_machine_get_all_reservations()
        return {"status": "success", "data": reservations}
    except Exception as e:
        print(f"ERROR: Error getting reservations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reservations/by-host/{host_identifier}", summary="Get reservations by specific host")
async def get_reservations_by_host(host_identifier: str):
    """Get all machines reserved by a specific host."""
    try:
        reservations = await db.network_machine_get_reservations_by_host(host_identifier)
        return {"status": "success", "data": reservations, "host": host_identifier}
    except Exception as e:
        print(f"ERROR: Error getting reservations for host {host_identifier}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reservations/my", summary="Get reservations made by this host")
async def get_my_reservations():
    """Get all machines reserved by this host."""
    try:
        host_identifier = await _get_host_identifier()
        reservations = await db.network_machine_get_reservations_by_host(host_identifier)
        return {"status": "success", "data": reservations, "host": host_identifier}
    except Exception as e:
        print(f"ERROR: Error getting my reservations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reservations/cleanup", summary="Clean up expired reservations")
async def cleanup_expired_reservations():
    """Clean up all expired reservations across all machines."""
    try:
        expired_count = await db.network_machine_cleanup_expired_reservations()
        return {
            "status": "success",
            "message": f"Cleaned up {expired_count} expired reservations",
            "expired_count": expired_count,
        }
    except Exception as e:
        print(f"ERROR: Error cleaning up expired reservations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reservations/network-wide", summary="Get comprehensive network-wide reservation information")
async def get_network_wide_reservations():
    """
    Get comprehensive reservation information across the entire network.
    This includes local reservations and remote machine reservations.
    Perfect for Host A to see what Host B has reserved.
    """
    try:
        # Get local reservations (machines managed by this host)
        local_reservations = await db.network_machine_get_all_reservations()

        # Get information about this host's identity
        host_identifier = await _get_host_identifier()
        this_machine_info = await _get_this_machine_info()

        # Get all registered machines to query their reservations
        machines = await db.network_machine_get_all()

        network_reservations = {
            "local_host": {
                "identifier": host_identifier,
                "machine_info": this_machine_info,
                "managed_machines": local_reservations,
                "total_managed": len(local_reservations),
                "reserved_machines": len([m for m in local_reservations if m.get("is_reserved")]),
            },
            "remote_hosts": [],
        }

        # Query each remote machine for its reservation information
        for machine in machines:
            if machine.get("status") == "online":
                try:
                    # Query remote machine for its reservations
                    remote_reservations = await _get_remote_machine_reservations(machine)
                    if remote_reservations:
                        network_reservations["remote_hosts"].append(
                            {
                                "machine_info": {
                                    "id": machine["id"],
                                    "name": machine["name"],
                                    "host": machine["host"],
                                    "port": machine["port"],
                                },
                                "reservations": remote_reservations.get("data", []),
                                "total_managed": len(remote_reservations.get("data", [])),
                                "reserved_machines": len(
                                    [m for m in remote_reservations.get("data", []) if m.get("is_reserved")]
                                ),
                            }
                        )
                except Exception as e:
                    print(f"Failed to get reservations from {machine['name']}: {e}")
                    # Add machine info even if we couldn't get reservations
                    network_reservations["remote_hosts"].append(
                        {
                            "machine_info": {
                                "id": machine["id"],
                                "name": machine["name"],
                                "host": machine["host"],
                                "port": machine["port"],
                            },
                            "error": f"Could not retrieve reservations: {str(e)}",
                            "reservations": [],
                            "total_managed": 0,
                            "reserved_machines": 0,
                        }
                    )

        return {"status": "success", "data": network_reservations}
    except Exception as e:
        print(f"ERROR: Error getting network-wide reservations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reservations/cascade/{target_host_id}", summary="Get cascaded reservation info from a specific host")
async def get_cascaded_reservations(target_host_id: int):
    """
    Get reservation information from a specific remote host.
    This allows Host A to specifically query Host B for its reservation details.
    """
    try:
        # Get the target machine info
        target_machine = await db.network_machine_get(target_host_id)
        if not target_machine:
            raise HTTPException(status_code=404, detail="Target host machine not found")

        if target_machine.get("status") != "online":
            raise HTTPException(status_code=503, detail="Target host machine is not online")

        # Get reservations from the target host
        remote_reservations = await _get_remote_machine_reservations(target_machine)

        # Also get the network-wide view from that host
        remote_network_view = await _get_remote_network_wide_reservations(target_machine)

        return {
            "status": "success",
            "target_host": {
                "id": target_machine["id"],
                "name": target_machine["name"],
                "host": target_machine["host"],
                "port": target_machine["port"],
            },
            "data": {
                "direct_reservations": remote_reservations.get("data", []),
                "network_wide_view": remote_network_view.get("data", {}),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Error getting cascaded reservations from host {target_host_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reservations/hierarchy", summary="Get hierarchical view of all network reservations")
async def get_reservation_hierarchy():
    """
    Get a hierarchical view of all reservations in the network.
    Shows the tree structure: Host A -> Host B -> Workers managed by Host B
    """
    try:
        # Start with this host as the root
        host_identifier = await _get_host_identifier()
        this_machine_info = await _get_this_machine_info()

        hierarchy = await _build_reservation_hierarchy(host_identifier, this_machine_info, set())

        return {"status": "success", "data": hierarchy}
    except Exception as e:
        print(f"ERROR: Error building reservation hierarchy: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Helper functions for network-wide reservation queries


async def _get_remote_machine_reservations(machine: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get reservations from a remote machine."""
    try:
        url = f"http://{machine['host']}:{machine['port']}/network/reservations"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return None
    except Exception:
        return None


async def _get_remote_network_wide_reservations(machine: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get network-wide reservation view from a remote machine."""
    try:
        url = f"http://{machine['host']}:{machine['port']}/network/reservations/network-wide"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return None
    except Exception:
        return None


async def _build_reservation_hierarchy(
    host_id: str, machine_info: Dict[str, Any], visited: set, depth: int = 0
) -> Dict[str, Any]:
    """Recursively build a hierarchical view of network reservations."""
    if depth > 5 or host_id in visited:  # Prevent infinite loops
        return {"error": "Max depth reached or circular reference detected"}

    visited.add(host_id)

    # Get local machines and reservations
    local_reservations = await db.network_machine_get_all_reservations()
    local_machines = await db.network_machine_get_all()

    hierarchy_node = {
        "host_id": host_id,
        "machine_info": machine_info,
        "depth": depth,
        "local_reservations": local_reservations,
        "managed_machines": local_machines,
        "sub_networks": [],
    }

    # For each online machine, try to get its hierarchy
    for machine in local_machines:
        if machine.get("status") == "online" and depth < 3:  # Limit depth for performance
            try:
                remote_hierarchy = await _get_remote_machine_reservations(machine)
                if remote_hierarchy:
                    # Try to get the machine info from the remote machine
                    remote_info = await _get_remote_machine_info(machine)
                    remote_host_id = f"{machine['host']}:{machine['port']}"

                    if remote_host_id not in visited:
                        sub_hierarchy = await _build_reservation_hierarchy(
                            remote_host_id,
                            remote_info or {"host": machine["host"], "port": machine["port"]},
                            visited.copy(),
                            depth + 1,
                        )
                        hierarchy_node["sub_networks"].append(sub_hierarchy)
            except Exception as e:
                hierarchy_node["sub_networks"].append(
                    {"error": f"Could not retrieve hierarchy from {machine['name']}: {str(e)}", "machine": machine}
                )

    return hierarchy_node


async def _get_remote_machine_info(machine: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get machine info from a remote host."""
    try:
        url = f"http://{machine['host']}:{machine['port']}/server/info"
        headers = {}
        if machine.get("api_token"):
            headers["Authorization"] = f"Bearer {machine['api_token']}"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return None
    except Exception:
        return None


# =============================================================================
# DASHBOARD ANALYTICS ENDPOINTS
# =============================================================================


@router.post("/analytics/reservations/by-host", summary="Get reservation statistics by host")
async def analytics_reservations_by_host(request: DashboardAnalyticsRequest):
    """Get reservation count and total minutes reserved, grouped by host."""
    try:
        time_range_days = request.time_range

        # Get the start date for the query
        start_date = (datetime.now() - timedelta(days=time_range_days)).strftime("%Y-%m-%d")

        # Query the database for reservation data
        reservations = await db.network_quota_get_reservations_by_host(start_date)

        # Calculate totals for each host
        host_totals = {}
        for reservation in reservations:
            host = reservation["host_identifier"]
            if host not in host_totals:
                host_totals[host] = {"reservations": 0, "totalMinutes": 0, "activeReservations": 0}

            host_totals[host]["reservations"] += 1
            host_totals[host]["totalMinutes"] += reservation["minutes_used"]

            # Check if the reservation is active (not expired)
            if reservation.get("is_active"):
                host_totals[host]["activeReservations"] += 1

        # Prepare the response data
        response_data = {"data": []}
        for host, totals in host_totals.items():
            response_data["data"].append({"host": host, **totals})  # Merge host identifier with the totals

        return {"status": "success", "data": response_data["data"]}

    except Exception as e:
        print(f"ERROR: Error getting reservation analytics by host: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analytics/reservations/by-machine", summary="Get reservation statistics by machine")
async def analytics_reservations_by_machine(request: DashboardAnalyticsRequest):
    """Get reservation count and total minutes reserved, grouped by machine."""
    try:
        time_range_days = request.time_range

        # Get the start date for the query
        start_date = (datetime.now() - timedelta(days=time_range_days)).strftime("%Y-%m-%d")

        # Query the database for reservation data
        reservations = await db.network_quota_get_reservations_by_machine(start_date)

        # Calculate totals for each machine
        machine_totals = {}
        for reservation in reservations:
            machine_name = reservation["machine_name"]
            if machine_name not in machine_totals:
                machine_totals[machine_name] = {"reservations": 0, "totalMinutes": 0}

            machine_totals[machine_name]["reservations"] += 1
            machine_totals[machine_name]["totalMinutes"] += reservation["minutes_used"]

        # Calculate average duration for each machine
        for machine_name, totals in machine_totals.items():
            totals["avgDuration"] = totals["totalMinutes"] / totals["reservations"] if totals["reservations"] > 0 else 0

        # Prepare the response data
        response_data = {"data": []}
        for machine, totals in machine_totals.items():
            response_data["data"].append({"machine": machine, **totals})  # Merge machine name with the totals

        return {"status": "success", "data": response_data["data"]}

    except Exception as e:
        print(f"ERROR: Error getting reservation analytics by machine: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analytics/usage/over-time", summary="Get usage statistics over time")
async def analytics_usage_over_time(request: DashboardAnalyticsRequest):
    """Get usage statistics (reservations and minutes) aggregated by day over the past X days."""
    try:
        time_range_days = request.time_range

        # Get the start date for the query
        start_date = (datetime.now() - timedelta(days=time_range_days)).strftime("%Y-%m-%d")

        # Query the database for daily usage data
        daily_usage = await db.network_quota_get_usage_over_time(start_date)

        # Prepare the response data
        response_data = {"data": []}
        for record in daily_usage:
            response_data["data"].append(
                {"date": record["date"], "reservations": record["reservations"], "minutes": record["minutes_used"]}
            )

        return {"status": "success", "data": response_data["data"]}

    except Exception as e:
        print(f"ERROR: Error getting usage analytics over time: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analytics/quota/utilization", summary="Get quota utilization statistics")
async def analytics_quota_utilization(request: DashboardAnalyticsRequest):
    """Get quota utilization (percentage used) for each host over the past X days."""
    try:
        time_range_days = request.time_range

        # Get the start date for the query
        start_date = (datetime.now() - timedelta(days=time_range_days)).strftime("%Y-%m-%d")

        # Query the database for quota utilization data
        quota_utilization = await db.network_quota_get_utilization(start_date)

        # Prepare the response data
        response_data = {"data": []}
        for record in quota_utilization:
            response_data["data"].append(
                {
                    "host": record["host_identifier"],
                    "daily": record["daily_usage_percent"],
                    "weekly": record["weekly_usage_percent"],
                    "monthly": record["monthly_usage_percent"],
                }
            )

        return {"status": "success", "data": response_data["data"]}

    except Exception as e:
        print(f"ERROR: Error getting quota utilization analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/dashboardAnalytics", summary="Get dashboard analytics data")
async def get_dashboard_analytics(request: DashboardAnalyticsRequest):
    """
    Get comprehensive dashboard analytics data including reservations by host/machine,
    usage over time, and quota utilization.
    """
    try:
        time_range_days = request.time_range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_range_days)

        # Get all relevant data
        reservations_by_host = await _get_reservations_by_host(start_date, end_date)
        reservations_by_machine = await _get_reservations_by_machine(start_date, end_date)
        usage_over_time = await _get_usage_over_time(start_date, end_date)
        quota_utilization = await _get_quota_utilization()

        return {
            "status": "success",
            "data": {
                "reservationsByHost": reservations_by_host,
                "reservationsByMachine": reservations_by_machine,
                "usageOverTime": usage_over_time,
                "quotaUtilization": quota_utilization,
            },
        }

    except Exception as e:
        print(f"ERROR: Error getting dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# DASHBOARD ANALYTICS HELPER FUNCTIONS
# =============================================================================


async def _get_reservations_by_host(start_date: datetime, end_date: datetime) -> list[Dict[str, Any]]:
    """Get reservation statistics grouped by host."""
    try:
        # Get reservation history data from database
        reservation_history = await db.network_quota_get_reservation_history(start_date, end_date)

        # Get currently active reservations
        current_reservations = await db.network_machine_get_all_reservations()

        # Group by host
        host_stats = {}

        # Process historical reservations
        for reservation in reservation_history:
            host = reservation.get("host_identifier", "unknown")
            if host not in host_stats:
                host_stats[host] = {"host": host, "reservations": 0, "totalMinutes": 0, "activeReservations": 0}

            host_stats[host]["reservations"] += 1
            host_stats[host]["totalMinutes"] += reservation.get("minutes_used", 0)

        # Count active reservations
        for machine in current_reservations:
            host = machine.get("reserved_by_host")
            if host:
                if host not in host_stats:
                    host_stats[host] = {"host": host, "reservations": 0, "totalMinutes": 0, "activeReservations": 0}
                host_stats[host]["activeReservations"] += 1

        return list(host_stats.values())

    except Exception as e:
        print(f"ERROR: Error getting reservations by host: {e}")
        return []


async def _get_reservations_by_machine(start_date: datetime, end_date: datetime) -> list[Dict[str, Any]]:
    """Get reservation statistics grouped by machine."""
    try:
        # Get reservation history data from database
        reservation_history = await db.network_quota_get_reservation_history(start_date, end_date)

        # Get machine names mapping
        machines = await db.network_machine_get_all()
        machine_name_map = {m["id"]: m["name"] for m in machines}

        # Group by machine
        machine_stats = {}

        for reservation in reservation_history:
            machine_id = reservation.get("machine_id")
            machine_name = machine_name_map.get(machine_id, f"Machine-{machine_id}")
            minutes_used = reservation.get("minutes_used", 0)

            if machine_name not in machine_stats:
                machine_stats[machine_name] = {
                    "machine": machine_name,
                    "reservations": 0,
                    "totalMinutes": 0,
                    "total_duration_for_avg": 0,
                }

            machine_stats[machine_name]["reservations"] += 1
            machine_stats[machine_name]["totalMinutes"] += minutes_used
            machine_stats[machine_name]["total_duration_for_avg"] += minutes_used

        # Calculate averages
        result = []
        for stats in machine_stats.values():
            avg_duration = (stats["total_duration_for_avg"] / stats["reservations"]) if stats["reservations"] > 0 else 0
            result.append(
                {
                    "machine": stats["machine"],
                    "reservations": stats["reservations"],
                    "totalMinutes": stats["totalMinutes"],
                    "avgDuration": round(avg_duration, 1),
                }
            )

        return result

    except Exception as e:
        print(f"ERROR: Error getting reservations by machine: {e}")
        return []


async def _get_usage_over_time(start_date: datetime, end_date: datetime) -> list[Dict[str, Any]]:
    """Get daily usage statistics over the specified time range."""
    try:
        # Get reservation history data
        reservation_history = await db.network_quota_get_reservation_history(start_date, end_date)

        # Create daily buckets
        usage_by_date = {}
        current_date = start_date.date()
        end_date_only = end_date.date()

        # Initialize all dates with zero values
        while current_date <= end_date_only:
            date_str = current_date.strftime("%Y-%m-%d")
            usage_by_date[date_str] = {"date": date_str, "reservations": 0, "minutes": 0}
            current_date += timedelta(days=1)

        # Process reservations and group by date
        for reservation in reservation_history:
            # Use created_at or reservation date from the history
            reservation_date = reservation.get("created_at")
            if reservation_date:
                if isinstance(reservation_date, str):
                    # Parse string date
                    try:
                        reservation_datetime = datetime.fromisoformat(reservation_date.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        reservation_datetime = datetime.strptime(reservation_date, "%Y-%m-%d %H:%M:%S")
                else:
                    reservation_datetime = reservation_date

                date_str = reservation_datetime.strftime("%Y-%m-%d")

                if date_str in usage_by_date:
                    usage_by_date[date_str]["reservations"] += 1
                    usage_by_date[date_str]["minutes"] += reservation.get("minutes_used", 0)

        # Return sorted by date
        return sorted(usage_by_date.values(), key=lambda x: x["date"])

    except Exception as e:
        print(f"ERROR: Error getting usage over time: {e}")
        return []


async def _get_quota_utilization() -> list[Dict[str, Any]]:
    """Get quota utilization percentages for all hosts."""
    try:
        # Get all hosts with quota configurations
        all_quota_data = await db.network_quota_get_all_hosts_usage()

        utilization_data = []

        # The data structure is {host_identifier: {configs: [...], usage: [...]}}
        for host_identifier, host_data in all_quota_data.items():
            # Extract usage data for each period
            usage_by_period = {}
            for usage_record in host_data.get("usage", []):
                period = usage_record.get("time_period")
                usage_by_period[period] = usage_record

            # Extract config data for each period
            config_by_period = {}
            for config_record in host_data.get("configs", []):
                period = config_record.get("time_period")
                config_by_period[period] = config_record

            # Calculate utilization percentages
            daily_percent = 0.0
            weekly_percent = 0.0
            monthly_percent = 0.0

            if "daily" in usage_by_period and "daily" in config_by_period:
                daily_usage = usage_by_period["daily"].get("minutes_used", 0)
                daily_limit = config_by_period["daily"].get("minutes_limit", 1)
                daily_percent = (daily_usage / daily_limit * 100) if daily_limit > 0 else 0.0

            if "weekly" in usage_by_period and "weekly" in config_by_period:
                weekly_usage = usage_by_period["weekly"].get("minutes_used", 0)
                weekly_limit = config_by_period["weekly"].get("minutes_limit", 1)
                weekly_percent = (weekly_usage / weekly_limit * 100) if weekly_limit > 0 else 0.0

            if "monthly" in usage_by_period and "monthly" in config_by_period:
                monthly_usage = usage_by_period["monthly"].get("minutes_used", 0)
                monthly_limit = config_by_period["monthly"].get("minutes_limit", 1)
                monthly_percent = (monthly_usage / monthly_limit * 100) if monthly_limit > 0 else 0.0

            utilization_data.append(
                {
                    "host": host_identifier,
                    "daily": round(daily_percent, 1),
                    "weekly": round(weekly_percent, 1),
                    "monthly": round(monthly_percent, 1),
                }
            )

        return utilization_data

    except Exception as e:
        print(f"ERROR: Error getting quota utilization: {e}")
        return []
