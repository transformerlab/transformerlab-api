"""
Synchronous database operations using SQLAlchemy.

This module contains synchronous database functions that were previously
using raw SQLite connections. They have been converted to use SQLAlchemy
with synchronous sessions for consistency.
"""

import json
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.sqlite import insert

from transformerlab.db.constants import DATABASE_FILE_NAME
from transformerlab.db.jobs import ALLOWED_JOB_TYPES
from transformerlab.shared.models import models


# Create synchronous engine and session factory
sync_engine = create_engine(f"sqlite:///{DATABASE_FILE_NAME}", echo=False)
sync_session_factory = sessionmaker(bind=sync_engine, expire_on_commit=False)


def get_sync_session() -> Session:
    """Get a synchronous SQLAlchemy session."""
    return sync_session_factory()


def job_create_sync(type, status, job_data="{}", experiment_id=""):
    """
    Synchronous version of job_create function for use with XML-RPC.
    """
    try:
        # Check if type is allowed
        if type not in ALLOWED_JOB_TYPES:
            raise ValueError(f"Job type {type} is not allowed")

        # Ensure job_data is a dict for SQLAlchemy JSON field
        if isinstance(job_data, str):
            try:
                job_data_dict = json.loads(job_data)
            except Exception:
                job_data_dict = {}
        else:
            job_data_dict = job_data

        with get_sync_session() as session:
            stmt = insert(models.Job).values(
                type=type,
                status=status,
                experiment_id=experiment_id,
                job_data=job_data_dict,
            )
            result = session.execute(stmt)
            session.commit()
            return result.inserted_primary_key[0]
            
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

    except Exception as e:
        print("Error updating job status: " + str(e))
