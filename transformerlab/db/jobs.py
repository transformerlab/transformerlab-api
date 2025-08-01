###############
# GENERIC JOBS MODEL
###############
import json
from sqlalchemy import insert, select, text, update
from transformerlab.db.session import async_session
from transformerlab.shared.models import models
from transformerlab.db.utils import sqlalchemy_to_dict

# Allowed job types:
ALLOWED_JOB_TYPES = [
    "TRAIN",
    "DOWNLOAD_MODEL",
    "LOAD_MODEL",
    "TASK",
    "EVAL",
    "EXPORT",
    "UNDEFINED",
    "GENERATE",
    "INSTALL_RECIPE_DEPS",
]


async def job_create(type, status, job_data="{}", experiment_id=""):
    # check if type is allowed
    if type not in ALLOWED_JOB_TYPES:
        raise ValueError(f"Job type {type} is not allowed")
    # Ensure job_data is a dict for SQLAlchemy JSON field
    if isinstance(job_data, str):
        try:
            job_data = json.loads(job_data)
        except Exception:
            job_data = {}
    async with async_session() as session:
        stmt = insert(models.Job).values(
            type=type,
            status=status,
            experiment_id=experiment_id,
            job_data=job_data,
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.inserted_primary_key[0]


async def jobs_get_all(type="", status=""):
    async with async_session() as session:
        stmt = select(models.Job).where(models.Job.status != "DELETED")
        if type != "":
            stmt = stmt.where(models.Job.type == type)
        if status != "":
            stmt = stmt.where(models.Job.status == status)
        stmt = stmt.order_by(models.Job.created_at.desc())
        result = await session.execute(stmt)
        jobs = result.scalars().all()
        data = []
        for job in jobs:
            row = sqlalchemy_to_dict(job)
            # Convert job_data from JSON string to dict if needed
            if "job_data" in row and row["job_data"]:
                if isinstance(row["job_data"], str):
                    try:
                        row["job_data"] = json.loads(row["job_data"])
                    except Exception:
                        pass
            data.append(row)
        return data


async def jobs_get_all_by_experiment_and_type(experiment_id, job_type):
    async with async_session() as session:
        stmt = (
            select(models.Job)
            .where(
                models.Job.experiment_id == experiment_id,
                models.Job.type == job_type,
                models.Job.status != "DELETED",
            )
            .order_by(models.Job.created_at.desc())
        )
        result = await session.execute(stmt)
        jobs = result.scalars().all()
        data = []
        for job in jobs:
            row = sqlalchemy_to_dict(job)
            # Convert job_data from JSON string to dict if needed
            if "job_data" in row and row["job_data"]:
                if isinstance(row["job_data"], str):
                    try:
                        row["job_data"] = json.loads(row["job_data"])
                    except Exception:
                        pass
            data.append(row)
        return data


async def jobs_get_by_experiment(experiment_id):
    """Get all jobs for a specific experiment"""
    async with async_session() as session:
        result = await session.execute(
            select(models.Job)
            .where(models.Job.experiment_id == experiment_id, models.Job.status != "DELETED")
            .order_by(models.Job.created_at.desc())
        )
        jobs = result.scalars().all()
        return [sqlalchemy_to_dict(job) for job in jobs]


async def job_get_status(job_id):
    async with async_session() as session:
        result = await session.execute(select(models.Job.status).where(models.Job.id == job_id))
        status = result.scalar_one_or_none()
        return status


async def job_get_error_msg(job_id):
    async with async_session() as session:
        result = await session.execute(select(models.Job.job_data).where(models.Job.id == job_id))
        job_data_raw = result.scalar_one_or_none()
        # If no job_data, return None
        if not job_data_raw:
            return None
        # Parse JSON string if necessary
        if isinstance(job_data_raw, str):
            try:
                job_data = json.loads(job_data_raw)
            except Exception:
                return None
        else:
            job_data = job_data_raw
        return job_data.get("error_msg", None)


async def job_get(job_id):
    async with async_session() as session:
        result = await session.execute(select(models.Job).where(models.Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            return None
        row = sqlalchemy_to_dict(job)
        # Convert job_data from JSON string to dict if needed
        if "job_data" in row and row["job_data"]:
            if isinstance(row["job_data"], str):
                try:
                    row["job_data"] = json.loads(row["job_data"])
                except Exception:
                    pass
        return row


async def job_count_running():
    async with async_session() as session:
        result = await session.execute(select(models.Job).where(models.Job.status == "RUNNING"))
        count = len(result.scalars().all())
        return count


async def jobs_get_next_queued_job():
    async with async_session() as session:
        result = await session.execute(
            select(models.Job).where(models.Job.status == "QUEUED").order_by(models.Job.created_at.asc()).limit(1)
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None
        row = sqlalchemy_to_dict(job)
        # Convert job_data from JSON string to dict if needed
        if "job_data" in row and row["job_data"]:
            if isinstance(row["job_data"], str):
                try:
                    row["job_data"] = json.loads(row["job_data"])
                except Exception:
                    pass
        return row


async def job_update_status(job_id, status, error_msg=None):
    async with async_session() as session:
        await session.execute(
            update(models.Job)
            .where(models.Job.id == job_id)
            .values(status=status, updated_at=text("CURRENT_TIMESTAMP"))
        )
        if error_msg:
            # Fetch current job_data
            result = await session.execute(select(models.Job.job_data).where(models.Job.id == job_id))
            job_data = result.scalar_one_or_none()
            if isinstance(job_data, str):
                try:
                    job_data = json.loads(job_data)
                except Exception:
                    job_data = {}
            elif not job_data:
                job_data = {}
            job_data["error_msg"] = str(error_msg)
            await session.execute(
                update(models.Job).where(models.Job.id == job_id).values(job_data=json.dumps(job_data))
            )
        await session.commit()

    return


async def job_update(job_id, type, status):
    async with async_session() as session:
        await session.execute(update(models.Job).where(models.Job.id == job_id).values(type=type, status=status))
        await session.commit()

    return


async def job_delete_all():
    async with async_session() as session:
        # Instead of deleting, set status to 'DELETED' for all jobs
        await session.execute(update(models.Job).values(status="DELETED"))
        await session.commit()
    return


async def job_delete(job_id):
    print("Deleting job: " + str(job_id))
    async with async_session() as session:
        # Instead of deleting, set status to 'DELETED' for the job
        await session.execute(update(models.Job).where(models.Job.id == job_id).values(status="DELETED"))
        await session.commit()
    return


async def job_update_job_data_insert_key_value(job_id, key, value):
    async with async_session() as session:
        # Fetch current job_data
        result = await session.execute(select(models.Job.job_data).where(models.Job.id == job_id))
        job_data = result.scalar_one_or_none()
        if isinstance(job_data, str):
            try:
                job_data = json.loads(job_data)
            except Exception:
                job_data = {}
        elif not job_data:
            job_data = {}
        # Update the key
        job_data[key] = value
        # Save back as JSON string
        await session.execute(update(models.Job).where(models.Job.id == job_id).values(job_data=json.dumps(job_data)))
        await session.commit()
    return


async def job_stop(job_id):
    print("Stopping job: " + str(job_id))
    await job_update_job_data_insert_key_value(job_id, "stop", True)
    return


async def job_update_progress(job_id, progress):
    """
    Update the percent complete for this job.

    progress: int representing percent complete
    """
    async with async_session() as session:
        await session.execute(update(models.Job).where(models.Job.id == job_id).values(progress=progress))
        await session.commit()


async def job_update_sweep_progress(job_id, value):
    """
    Update the 'sweep_progress' key in the job_data JSON column for a given job.
    """
    async with async_session() as session:
        # Fetch the job
        result = await session.execute(select(models.Job).where(models.Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            return

        # Parse job_data as dict if needed
        job_data = job.job_data
        if isinstance(job_data, str):
            try:
                job_data = json.loads(job_data)
            except Exception:
                job_data = {}

        # Update the sweep_progress key
        job_data["sweep_progress"] = value

        # Save back as JSON string
        await session.execute(update(models.Job).where(models.Job.id == job_id).values(job_data=json.dumps(job_data)))
        await session.commit()
    return
