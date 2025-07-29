import json
import logging
from typing import Optional

import transformerlab.db.jobs as db_jobs
from transformerlab.db.sync import (
    job_update_status_sync as db_job_update_status_sync,
    job_update_sync as db_job_update_sync,
    get_sync_session,
)
from transformerlab.shared.models import models
from sqlalchemy import select

logger = logging.getLogger(__name__)


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

        # Define supported triggers based on existing ALLOWED_JOB_TYPES
        supported_triggers = ["TRAIN", "LOAD_MODEL", "EXPORT", "EVAL", "GENERATE"]

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
                logger.info(f"Triggered workflow {workflow_id} due to job {job_id} completion, job type: {job_type}.")

    except Exception as e:
        logger.error(f"Error triggering workflows for job {job_id}: {e}")


async def job_update_status(job_id: str, status: str, error_msg: Optional[str] = None):
    """
    Update job status and trigger workflows if job is completed.

    Args:
        job_id: The ID of the job to update
        status: The new status to set
        error_msg: Optional error message to add to job data
    """
    # Update the job status in the database
    await db_jobs.job_update_status(job_id, status, error_msg)

    # Trigger workflows if job status is COMPLETE
    if status == "COMPLETE":
        await _trigger_workflows_on_job_completion(job_id)


async def job_update(job_id: str, type: str, status: str):
    """
    Update job type and status and trigger workflows if job is completed.

    Args:
        job_id: The ID of the job to update
        type: The new type to set
        status: The new status to set
    """
    # Update the job in the database
    await db_jobs.job_update(job_id, type, status)

    # Trigger workflows if job status is COMPLETE
    if status == "COMPLETE":
        await _trigger_workflows_on_job_completion(job_id)


def job_update_status_sync(job_id: str, status: str, error_msg: Optional[str] = None):
    """
    Synchronous version of job status update.

    Args:
        job_id: The ID of the job to update
        status: The new status to set
        error_msg: Optional error message to add to job data
    """
    # Update the job status in the database
    db_job_update_status_sync(job_id, status, error_msg)


def job_update_sync(job_id: str, status: str):
    """
    Synchronous version of job update.

    Args:
        job_id: The ID of the job to update
        status: The new status to set
    """
    # Update the job in the database
    db_job_update_sync(job_id, status)


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
            supported_triggers = ["TRAIN", "LOAD_MODEL", "EXPORT", "EVAL", "GENERATE"]
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
                print(f"Triggered workflow {workflow_id} due to job {job_id} completion, job type: {job_type}")

            session.commit()

    except Exception as e:
        print(f"Error triggering workflows for job {job_id}: {e}")
