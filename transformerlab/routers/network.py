import asyncio
import httpx
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

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
