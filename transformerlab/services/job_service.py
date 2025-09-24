import os
import json
from typing import Optional

import transformerlab.db.jobs as db_jobs
from transformerlab.db.sync import (
    job_update_status_sync as db_job_update_status_sync,
    job_update_sync as db_job_update_sync,
    get_sync_session,
    job_mark_as_complete_if_running as db_job_mark_as_complete_if_running,
)
from transformerlab.shared.models import models
from transformerlab.db.db import experiment_get
from sqlalchemy import select
from lab import Experiment


# Centralized set of job types that can trigger workflows on completion
SUPPORTED_WORKFLOW_TRIGGERS = ["TRAIN", "LOAD_MODEL", "EXPORT", "EVAL", "GENERATE", "DOWNLOAD_MODEL"]

# For now several service calls will use the SDK for MULTITENANT environments
MULTITENANT = os.getenv("TFL_MULTITENANT", "")


async def _get_experiment_by_id(experimentId: int):
    """
    Helper function to get SDK Experiment from an ID.
    """
    # first get the experiment name:
    data = await experiment_get(experimentId)

    # if the experiment does not exist, return none
    if data is None:
        return None

    experiment_name = data["name"]
    experiment = Experiment.get(experiment_name)
    return experiment


async def list_jobs_by_experiment(experimentId: int, type: str = "", status: str = ""):
    """
    Returns a list of jobs in an experiment.
    Optionally, filter on type or status
    """
    if MULTITENANT:
        experiment = await _get_experiment_by_id(experimentId)
        if not experiment:
            return []
        jobs = experiment.get_jobs(type, status)
    else:
        jobs = await db_jobs.jobs_get_all(type=type, status=status, experiment_id=experimentId)
    return jobs


async def _trigger_workflows_on_job_completion(job_id: str):
    """
    Trigger workflows when a job completes if the job type is in supported triggers.
    """
    try:
        # Get the job details
        job = await db_jobs.job_get(job_id)
        if not job:
            return

        job_type = job.get("type")
        experiment_id = job.get("experiment_id")

        # Define supported triggers based on centralized configuration
        supported_triggers = SUPPORTED_WORKFLOW_TRIGGERS

        # Check if job type is in supported triggers
        if job_type not in supported_triggers:
            return

        # Import here to avoid circular imports
        from transformerlab.routers.experiment.workflows import workflows_get_by_trigger_type

        # Get workflows that should be triggered
        triggered_workflow_ids = await workflows_get_by_trigger_type(experiment_id, job_type)

        # Start each workflow
        if triggered_workflow_ids:
            from transformerlab.db.workflows import workflow_queue

            for workflow_id in triggered_workflow_ids:
                await workflow_queue(workflow_id)
    except Exception as e:
        print(f"Error triggering workflows for job {job_id}: {e}")


async def job_update_status(
    job_id: str, status: str, experiment_id: Optional[str] = None, error_msg: Optional[str] = None
):
    """
    Update job status and trigger workflows if job is completed.

    Args:
        job_id: The ID of the job to update
        status: The new status to set
        experiment_id: The experiment ID (required for most operations, optional for backward compatibility)
        error_msg: Optional error message to add to job data
    """
    # Update the job status in the database
    await db_jobs.job_update_status(job_id, status, experiment_id, error_msg)

    # Trigger workflows if job status is COMPLETE
    if status == "COMPLETE":
        await _trigger_workflows_on_job_completion(job_id)


async def job_update(job_id: str, type: str, status: str, experiment_id: Optional[str] = None):
    """
    Update job type and status and trigger workflows if job is completed.

    Args:
        job_id: The ID of the job to update
        type: The new type to set
        status: The new status to set
        experiment_id: The experiment ID (required for most operations, optional for backward compatibility)
    """
    # Update the job in the database
    await db_jobs.job_update(job_id, type, status, experiment_id)

    # Trigger workflows if job status is COMPLETE
    if status == "COMPLETE":
        await _trigger_workflows_on_job_completion(job_id)


def job_update_status_sync(
    job_id: str, status: str, experiment_id: Optional[str] = None, error_msg: Optional[str] = None
):
    """
    Synchronous version of job status update.

    Args:
        job_id: The ID of the job to update
        status: The new status to set
        experiment_id: The experiment ID (required for most operations, optional for backward compatibility)
        error_msg: Optional error message to add to job data
    """
    # Update the job status in the database
    db_job_update_status_sync(job_id, status, experiment_id, error_msg)

    # Trigger workflows if job status is COMPLETE
    if status == "COMPLETE":
        _trigger_workflows_on_job_completion_sync(job_id)


def job_update_sync(job_id: str, status: str, experiment_id: Optional[str] = None):
    """
    Synchronous version of job update.

    Args:
        job_id: The ID of the job to update
        status: The new status to set
        experiment_id: The experiment ID (required for most operations, optional for backward compatibility)
    """
    # Update the job in the database
    db_job_update_sync(job_id, status, experiment_id)

    # Trigger workflows if job status is COMPLETE
    if status == "COMPLETE":
        _trigger_workflows_on_job_completion_sync(job_id)


def _trigger_workflows_on_job_completion_sync(job_id: str):
    """
    Sync version of workflow triggering for use in sync contexts
    """
    try:
        with get_sync_session() as session:
            # 1. Get job details (sync)
            job_result = session.execute(
                select(models.Job.type, models.Job.experiment_id).where(models.Job.id == job_id)
            )
            job_row = job_result.fetchone()
            if not job_row:
                return

            job_type = job_row[0]
            experiment_id = job_row[1]

            # 2. Check if job type is supported
            supported_triggers = SUPPORTED_WORKFLOW_TRIGGERS
            if job_type not in supported_triggers:
                return

            # 4. Get workflows with matching trigger (sync)
            workflows_result = session.execute(
                select(models.Workflow.id, models.Workflow.config).where(models.Workflow.experiment_id == experiment_id)
            )

            triggered_workflow_ids = []

            for workflow_row in workflows_result:
                workflow_id = workflow_row[0]
                config = workflow_row[1]

                # Parse config and check triggers
                try:
                    if isinstance(config, str):
                        config = json.loads(config)
                    elif not isinstance(config, dict):
                        continue

                    triggers = config.get("triggers", [])

                    if job_type in triggers:
                        triggered_workflow_ids.append(workflow_id)

                except (json.JSONDecodeError, TypeError):
                    continue

            # 5. Queue workflows (sync)
            for workflow_id in triggered_workflow_ids:
                # Get workflow name
                workflow_result = session.execute(select(models.Workflow.name).where(models.Workflow.id == workflow_id))
                workflow_name = workflow_result.scalar_one_or_none()

                # Create workflow run using model object (same as async version)
                workflow_run = models.WorkflowRun(
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    job_ids="[]",
                    node_ids="[]",
                    status="QUEUED",
                    current_tasks="[]",
                    current_job_ids="[]",
                    experiment_id=experiment_id,
                )
                session.add(workflow_run)

            session.commit()
    except Exception as e:
        print(f"Error triggering workflows for job {job_id}: {e}")


def job_mark_as_complete_if_running(job_id: int, experiment_id: int) -> None:
    """Service wrapper: call db.sync.job_mark_as_complete_if_running and then trigger workflows."""
    # We cannot know from the db function whether an update occurred,
    # but it's safe to attempt the trigger; it will read the type and queue accordingly.
    db_job_mark_as_complete_if_running(job_id, experiment_id)
    _trigger_workflows_on_job_completion_sync(job_id)
