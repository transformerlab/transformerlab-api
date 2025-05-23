from typing import Optional, List
from fastapi import APIRouter, Body
from pydantic import BaseModel
import transformerlab.db as db

router = APIRouter(prefix="/workflow_triggers", tags=["workflow_triggers"])


class TriggerUpdate(BaseModel):
    config: Optional[dict] = None
    is_enabled: Optional[bool] = None


@router.get("/list/{experiment_id}", summary="List all workflow triggers for an experiment")
async def list_workflow_triggers(experiment_id: int):
    """List all workflow triggers for a given experiment."""
    triggers = await db.workflow_trigger_get_by_experiment_id(experiment_id)
    return triggers


@router.get("/{trigger_id}", summary="Get a specific workflow trigger by its ID")
async def get_workflow_trigger(trigger_id: int):
    """Get a specific workflow trigger by its ID."""
    trigger = await db.workflow_trigger_get_by_id(trigger_id)
    return trigger


@router.put("/{trigger_id}", summary="Update a workflow trigger configuration or enabled status")
async def update_workflow_trigger(trigger_id: int, update_data: TriggerUpdate):
    """Update the configuration (assigned workflows) and/or enabled status of a workflow trigger."""
    updated_trigger = await db.workflow_trigger_update(
        trigger_id, 
        config=update_data.config, 
        is_enabled=update_data.is_enabled
    )
    return updated_trigger 