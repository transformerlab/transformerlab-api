"""
Synchronous database operations using SQLAlchemy.

This module contains synchronous database functions that were previously
using raw SQLite connections. They have been converted to use SQLAlchemy
with synchronous sessions for consistency.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker

from transformerlab.db.db import models


def get_sync_session():
    """
    Get a synchronous database session for use in sync functions
    """
    database_path = os.path.join(os.environ.get("TFL_HOME_DIR", "."), "llmlab.sqlite3")
    # Create directory if it doesn't exist
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    sync_engine = create_engine(f"sqlite:///{database_path}")
    SessionLocal = sessionmaker(bind=sync_engine)
    return SessionLocal()


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
            supported_triggers = ["TRAIN", "DOWNLOAD_MODEL", "LOAD_MODEL", "EXPORT", "EVAL", "GENERATE"]
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


def job_create_sync(type, status, job_data="{}", experiment_id=None):
    """
    Synchronous version of job creation for use with XML-RPC.
    """
    try:
        with get_sync_session() as session:
            # Check if the job type is in the allowed list
            allowed_types = ["TRAIN", "EVAL", "GENERATE", "EXPORT", "DOWNLOAD_MODEL", "LOAD_MODEL", "TASK", "UNDEFINED"]
            if type not in allowed_types:
                raise ValueError(f"Invalid job type: {type}")

            job_id = str(uuid.uuid4())

            # Create job object
            job = models.Job(
                id=job_id,
                type=type,
                status=status,
                job_data=job_data,
                experiment_id=experiment_id,
            )
            session.add(job)
            session.commit()

            return job_id

    except Exception as e:
        print("Error creating job: " + str(e))
        return None


def job_update_status_sync(job_id, status, error_msg=None):
    """
    Synchronous version of job status update for use with XML-RPC.
    """
    try:
        with get_sync_session() as session:
            stmt = update(models.Job).where(models.Job.id == job_id).values(status=status)
            session.execute(stmt)
            session.commit()

        # Trigger workflows if job status is COMPLETE
        if status == "COMPLETE":
            _trigger_workflows_on_job_completion_sync(job_id)

    except Exception as e:
        print("Error updating job status: " + str(e))


def job_update_sync(job_id, status):
    """
    Synchronous version of job_update.
    This is used by popen_and_call function which can only support synchronous functions.
    This is a hack to get around that limitation.
    """
    try:
        with get_sync_session() as session:
            stmt = update(models.Job).where(models.Job.id == job_id).values(status=status)
            session.execute(stmt)
            session.commit()

        # Trigger workflows if job status is COMPLETE
        if status == "COMPLETE":
            _trigger_workflows_on_job_completion_sync(job_id)

    except Exception as e:
        print("Error updating job status: " + str(e))


def job_mark_as_complete_if_running(job_id):
    """
    Synchronous update to jobs that only marks a job as "COMPLETE" if it is currently "RUNNING".
    This avoids updating "stopped" jobs and marking them as complete.
    """
    try:
        with get_sync_session() as session:
            stmt = (
                update(models.Job)
                .where(models.Job.id == job_id, models.Job.status == "RUNNING")
                .values(status="COMPLETE")
            )
            result = session.execute(stmt)
            session.commit()

            # If a job was actually updated (was running and is now complete), trigger workflows
            if result.rowcount > 0:
                _trigger_workflows_on_job_completion_sync(job_id)

    except Exception as e:
        print("Error updating job status: " + str(e))
