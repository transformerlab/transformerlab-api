"""
Asynchronous database operations using SQLAlchemy.

This module provides async equivalents of the synchronous DB operations
defined in sync.py.
"""

import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import update
from sqlalchemy.dialects.sqlite import insert
from collections.abc import AsyncGenerator

from transformerlab.db.constants import DATABASE_FILE_NAME
from transformerlab.db.jobs import ALLOWED_JOB_TYPES
from transformerlab.shared.models import models

# Create async engine and session factory
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_FILE_NAME}"
async_engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionFactory = async_sessionmaker(bind=async_engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an asynchronous SQLAlchemy session."""
    async with AsyncSessionFactory() as session:
        yield session


async def job_create_async(type, status, job_data="{}", experiment_id=""):
    try:
        if type not in ALLOWED_JOB_TYPES:
            raise ValueError(f"Job type {type} is not allowed")

        if isinstance(job_data, str):
            try:
                job_data_dict = json.loads(job_data)
            except Exception:
                job_data_dict = {}
        else:
            job_data_dict = job_data

        async with AsyncSessionFactory() as session:
            stmt = insert(models.Job).values(
                type=type,
                status=status,
                experiment_id=experiment_id,
                job_data=job_data_dict,
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.inserted_primary_key[0]

    except Exception as e:
        print("Error creating job (async):", str(e))
        return None


async def job_update_status_async(job_id, status, error_msg=None):
    try:
        async with AsyncSessionFactory() as session:
            stmt = update(models.Job).where(models.Job.id == job_id).values(status=status)
            await session.execute(stmt)
            await session.commit()

    except Exception as e:
        print("Error updating job status (async):", str(e))


async def job_update_async(job_id, status):
    try:
        async with AsyncSessionFactory() as session:
            stmt = update(models.Job).where(models.Job.id == job_id).values(status=status)
            await session.execute(stmt)
            await session.commit()

    except Exception as e:
        print("Error updating job (async):", str(e))


async def job_mark_as_complete_if_running_async(job_id):
    try:
        async with AsyncSessionFactory() as session:
            stmt = (
                update(models.Job)
                .where(models.Job.id == job_id, models.Job.status == "RUNNING")
                .values(status="COMPLETE")
            )
            await session.execute(stmt)
            await session.commit()

    except Exception as e:
        print("Error marking job as complete (async):", str(e))
