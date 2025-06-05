from fastapi import APIRouter
from typing import Optional
import transformerlab.db as db

router = APIRouter(prefix="/workflows", tags=["workflows"])

@router.get("/list", summary="get all workflows in the experiment")
async def workflows_get_in_experiment(experimentId: int):
    workflows = await db.workflows_get_from_experiment(experimentId)
    print(experimentId)
    print(workflows)
    return workflows

@router.get("/runs", summary="get all workflow runs in the experiment")
async def workflow_runs_get_in_experiment(experimentId: int):
    workflow_runs = await db.workflow_runs_get_from_experiment(experimentId)
    print(experimentId)
    print(workflow_runs)
    return workflow_runs 