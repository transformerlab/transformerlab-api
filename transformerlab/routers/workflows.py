import json
import os
import csv
import pandas as pd
from fastapi import APIRouter, Body, Response
from fastapi.responses import StreamingResponse, FileResponse

import transformerlab.db as db
from transformerlab.shared import shared
from transformerlab.shared import dirs
from typing import Annotated
from json import JSONDecodeError

from transformerlab.routers.serverinfo import watch_file


router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.get("/list")
async def workflows_get_all():
    workflows = await db.workflows_get_all()
    return workflows


@router.get("/delete/{workflow_id}")
async def job_delete(workflow_id: str):
    await db.workflow_delete_by_id(workflow_id)
    return {"message": "OK"}


@router.get("/create")
async def workflow_create(name: str, config: str = "{}", experiment_id="1"):
    workflow_id = await db.workflow_create(name, config, experiment_id) 
    return workflow_id


@router.get("/start/{workflow_id}")
async def start_workflow(workflow_id):
    await db.workflow_update_status(workflow_id, "RUNNING")
    return {"message": "OK"}


@router.get("/start_next_step")
async def start_next_step_in_workflow():
    num_running_workflows = await db.workflow_count_running()
    if num_running_workflows == 0:
        return {"message": "A workflow is not running"}
    currently_running_workflow = await db.workflow_get_running()

    workflow_id = currently_running_workflow["id"]
    workflow_config = json.loads(currently_running_workflow["config"])
    workflow_current_task = int(currently_running_workflow["current_task"])
    workflow_current_job_id = int(currently_running_workflow["current_job_id"])
    workflow_experiment_id = currently_running_workflow["experiment_id"]

    if workflow_current_job_id != -1:
        current_job = await db.job_get(workflow_current_job_id)
    
        if current_job["status"] == "FAILED":
            await db.workflow_update_status(workflow_id, "FAILED")

        if current_job["status"] == "CANCELLED" or current_job["status"] == "DELETED":
            await db.workflow_update_with_new_job(workflow_id, -1, -1)
            await db.workflow_update_status(workflow_id, "CANCELLED")

        if current_job["status"] != "COMPLETE":
            return {"message": "the current job is running"}

    workflow_current_task = workflow_config["nodes"]["out"]

    if workflow_current_task == len(workflow_config["nodes"]):
        await db.workflow_update_status(workflow_id, "COMPLETE")
        return {"message": "Workflow Complete!"}

    next_job_type = workflow_config["nodes"][workflow_current_task]["type"]
    next_job_status = "QUEUED" 
    next_job_data = workflow_config["nodes"][workflow_current_task]["data"]

    next_job_id = await db.job_create(next_job_type, next_job_status, json.dumps(next_job_data), workflow_experiment_id)
    await db.workflow_update_with_new_job(workflow_id, workflow_current_task, next_job_id)

    return {"message": "Next job created"}

