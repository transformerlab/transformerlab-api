import itertools
import json
import os
import sqlite3

import aiosqlite
from sqlalchemy import select, delete, text, update
from sqlalchemy.dialects.sqlite import insert  # Correct import for SQLite upsert

# from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


# Make sure SQLAlchemy is installed using pip install sqlalchemy[asyncio] as
# described here https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html

# FastAPI Users
from typing import AsyncGenerator
from fastapi import Depends
from fastapi_users.db import SQLAlchemyUserDatabase

from transformerlab.shared import dirs
from transformerlab.shared.models import models  # noqa: F401
from transformerlab.shared.models.models import Config, Plugin

db = None
DATABASE_FILE_NAME = f"{dirs.WORKSPACE_DIR}/llmlab.sqlite3"
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_FILE_NAME}"

# Create SQLAlchemy engines
# engine = create_engine(DATABASE_URL, echo=True)
async_engine = create_async_engine(DATABASE_URL, echo=False)

# Create a configured "Session" class
async_session = sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)


def unconverted(func):
    """
    Decorator to mark a function as not yet converted to SQLAlchemy.
    """
    func._unconverted = True
    return func


@unconverted
async def migrate_workflows_non_preserving():
    """
    Migration function that renames workflows table as backup and creates new table
    based on current schema definition if experiment_id is not INTEGER type or config is not JSON type
    """

    try:
        # Check if workflows table exists
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workflows'")
        table_exists = await cursor.fetchone()
        await cursor.close()

        if not table_exists:
            print("Workflows table does not exist. Skipping non-preserving migration.")
            return

        # Check column types in the current workflows table
        cursor = await db.execute("PRAGMA table_info(workflows)")
        columns_info = await cursor.fetchall()
        await cursor.close()

        experiment_id_type = None
        config_type = None

        for column in columns_info:
            column_name = column[1]
            column_type = column[2].upper()

            if column_name == "experiment_id":
                experiment_id_type = column_type
            elif column_name == "config":
                config_type = column_type

        # Check if migration is needed based on column types
        needs_migration = False
        migration_reasons = []

        if experiment_id_type and experiment_id_type != "INTEGER":
            needs_migration = True
            migration_reasons.append(f"experiment_id column type is {experiment_id_type}, expected INTEGER")

        # SQLAlchemy JSON type maps to TEXT in SQLite, so we accept both
        if config_type and config_type not in ["JSON", "TEXT"]:
            needs_migration = True
            migration_reasons.append(
                f"config column type is {config_type}, expected JSON/TEXT (SQLAlchemy creates JSON as TEXT in SQLite)"
            )

        if not needs_migration:
            # print("Column types are correct. No migration needed.")
            return

        print("Migration needed due to:")
        for reason in migration_reasons:
            print(f"  - {reason}")

        # Check if backup table already exists and drop it
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workflows_backup'")
        backup_exists = await cursor.fetchone()
        await cursor.close()

        if backup_exists:
            await db.execute("DROP TABLE workflows_backup")

        # Rename current table as backup
        await db.execute("ALTER TABLE workflows RENAME TO workflows_backup")

        # Create new workflows table using SQLAlchemy schema
        async with async_engine.begin() as conn:
            await conn.run_sync(models.Base.metadata.create_all)

        await db.commit()
        print("Successfully created new workflows table with correct schema. Old table saved as workflows_backup.")

    except Exception as e:
        print(f"Failed to perform non-preserving migration: {e}")
        raise e


@unconverted
async def init():
    """
    Create the database, tables, and workspace folder if they don't exist.
    """
    global db
    os.makedirs(os.path.dirname(DATABASE_FILE_NAME), exist_ok=True)
    db = await aiosqlite.connect(DATABASE_FILE_NAME)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=normal")
    await db.execute("PRAGMA busy_timeout = 30000")

    # Create the tables if they don't exist
    async with async_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

    # Check if experiment_id column exists in workflow_runs table
    cursor = await db.execute("PRAGMA table_info(workflow_runs)")
    columns = await cursor.fetchall()
    has_experiment_id = any(column[1] == "experiment_id" for column in columns)

    if not has_experiment_id:
        # Add experiment_id column
        await db.execute("ALTER TABLE workflow_runs ADD COLUMN experiment_id INTEGER")

        # Update existing workflow runs with experiment_id from their workflows
        await db.execute("""
            UPDATE workflow_runs 
            SET experiment_id = (
                SELECT experiment_id 
                FROM workflows 
                WHERE workflows.id = workflow_runs.workflow_id
            )
        """)
        await db.commit()

    print("✅ Database initialized")

    print("✅ SEED DATA")
    async with async_session() as session:
        for name in ["alpha", "beta", "gamma"]:
            # Check if experiment already exists
            exists = await session.execute(select(models.Experiment).where(models.Experiment.name == name))
            if not exists.scalar_one_or_none():
                session.add(models.Experiment(name=name, config={}))
        await session.commit()

    # On startup, look for any jobs that are in the RUNNING state and set them to CANCELLED instead:
    # This is to handle the case where the server is restarted while a job is running.
    await job_cancel_in_progress_jobs()
    # Run migrations
    await migrate_workflows_non_preserving()
    # await init_sql_model()

    return


@unconverted
def get_sync_db_connection():
    global DATABASE_FILE_NAME
    db_sync = sqlite3.connect(DATABASE_FILE_NAME, isolation_level=None)
    db_sync.execute("PRAGMA journal_mode=WAL")
    db_sync.execute("PRAGMA synchronous=normal")
    db_sync.execute("PRAGMA busy_timeout = 30000")
    return db_sync


async def close():
    await db.close()
    await async_engine.dispose()
    print("✅ Database closed")
    return


###############################################
# Dependencies for FastAPI Users
# It wants specific types for session and DB.
###############################################


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, models.User)


# async def init_sql_model():
#     """
#     Initialize the database using SQLModel.
#     """
#     async with async_engine.begin() as conn:
#         await conn.run_sync(SQLModel.metadata.create_all)
#     print("✅ SQLModel Database initialized")


###############
# DATASETS MODEL
###############


async def get_dataset(dataset_id):
    async with async_session() as session:
        result = await session.execute(select(models.Dataset).where(models.Dataset.dataset_id == dataset_id))
        dataset = result.scalar_one_or_none()
        if dataset is None:
            return None
        row = dataset.__dict__.copy()
        if "json_data" in row and row["json_data"]:
            # If json_data is a string, parse it
            if isinstance(row["json_data"], str):
                row["json_data"] = json.loads(row["json_data"])
        return row


async def get_datasets():
    async with async_session() as session:
        result = await session.execute(select(models.Dataset))
        datasets = result.scalars().all()
        data = []
        for dataset in datasets:
            row = dataset.__dict__.copy()
            if "json_data" in row and row["json_data"]:
                if isinstance(row["json_data"], str):
                    row["json_data"] = json.loads(row["json_data"])
            data.append(row)
        return data


async def get_generated_datasets():
    async with async_session() as session:
        # Use SQLAlchemy's JSON path query for SQLite
        stmt = select(models.Dataset).where(text("json_extract(json_data, '$.generated') = 1"))
        result = await session.execute(stmt)
        datasets = result.scalars().all()
        data = []
        for dataset in datasets:
            row = dataset.__dict__.copy()
            if "json_data" in row and row["json_data"]:
                if isinstance(row["json_data"], str):
                    row["json_data"] = json.loads(row["json_data"])
            data.append(row)
        return data


async def create_huggingface_dataset(dataset_id, description, size, json_data):
    async with async_session() as session:
        stmt = insert(models.Dataset).values(
            dataset_id=dataset_id,
            location="huggingfacehub",
            description=description,
            size=size,
            json_data=json_data,
        )
        await session.execute(stmt)
        await session.commit()


async def create_local_dataset(dataset_id, json_data=None):
    async with async_session() as session:
        values = dict(
            dataset_id=dataset_id,
            location="local",
            description="",
            size=-1,
            json_data=json_data if json_data is not None else {},
        )
        stmt = insert(models.Dataset).values(**values)
        await session.execute(stmt)
        await session.commit()


async def delete_dataset(dataset_id):
    async with async_session() as session:
        stmt = delete(models.Dataset).where(models.Dataset.dataset_id == dataset_id)
        await session.execute(stmt)
        await session.commit()


###############
# MODELS MODEL
###############


async def model_local_list():
    async with async_session() as session:
        result = await session.execute(select(models.Model))
        models_list = result.scalars().all()
        data = []
        for model in models_list:
            row = model.__dict__.copy()
            if "json_data" in row and row["json_data"]:
                if isinstance(row["json_data"], str):
                    row["json_data"] = json.loads(row["json_data"])
            data.append(row)
        return data


async def model_local_count():
    async with async_session() as session:
        result = await session.execute(select(models.Model))
        count = len(result.scalars().all())
        return count


async def model_local_create(model_id, name, json_data):
    async with async_session() as session:
        # Upsert using SQLite's ON CONFLICT (model_id) DO UPDATE
        stmt = insert(models.Model).values(
            model_id=model_id,
            name=name,
            json_data=json_data,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["model_id"],
            set_={"name": name, "json_data": json_data},
        )
        await session.execute(stmt)
        await session.commit()


async def model_local_get(model_id):
    async with async_session() as session:
        result = await session.execute(select(models.Model).where(models.Model.model_id == model_id))
        model = result.scalar_one_or_none()
        if model is None:
            return None
        row = model.__dict__.copy()
        if "json_data" in row and row["json_data"]:
            if isinstance(row["json_data"], str):
                row["json_data"] = json.loads(row["json_data"])
        return row


async def model_local_delete(model_id):
    async with async_session() as session:
        result = await session.execute(select(models.Model).where(models.Model.model_id == model_id))
        model = result.scalar_one_or_none()
        if model:
            await session.delete(model)
            await session.commit()


###############
# GENERIC JOBS MODEL
###############

# Allowed job types:
ALLOWED_JOB_TYPES = [
    "TRAIN",
    "EXPORT_MODEL",
    "DOWNLOAD_MODEL",
    "LOAD_MODEL",
    "TASK",
    "EVAL",
    "UNDEFINED",
    "GENERATE",
    "INSTALL_RECIPE_DEPS",
]


@unconverted
async def job_create(type, status, job_data="{}", experiment_id=""):
    # check if type is allowed
    if type not in ALLOWED_JOB_TYPES:
        raise ValueError(f"Job type {type} is not allowed")
    row = await db.execute_insert(
        "INSERT INTO job(type, status, experiment_id, job_data) VALUES (?, ?, ?, json(?))",
        (type, status, experiment_id, job_data),
    )
    await db.commit()  # is this necessary?
    return row[0]


@unconverted
def job_create_sync(type, status, job_data="{}", experiment_id=""):
    """
    Synchronous version of job_create function for use with XML-RPC.
    """
    db_sync = None
    cursor = None
    try:
        global DATABASE_FILE_NAME
        # check if type is allowed
        if type not in ALLOWED_JOB_TYPES:
            raise ValueError(f"Job type {type} is not allowed")

        # Use SQLite directly in synchronous mode
        db_sync = get_sync_db_connection()

        cursor = db_sync.cursor()

        # Execute insert
        cursor.execute(
            "INSERT INTO job(type, status, experiment_id, job_data) VALUES (?, ?, ?, json(?))",
            (type, status, experiment_id, job_data),
        )

        # Get the row ID
        row_id = cursor.lastrowid

        # Commit and close
        db_sync.commit()
        cursor.close()

        return row_id
    except Exception as e:
        print("Error creating job: " + str(e))
        return None
    finally:
        if cursor:
            cursor.close()
        if db_sync:
            db_sync.close()


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
            row = job.__dict__.copy()
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
            row = job.__dict__.copy()
            # Convert job_data from JSON string to dict if needed
            if "job_data" in row and row["job_data"]:
                if isinstance(row["job_data"], str):
                    try:
                        row["job_data"] = json.loads(row["job_data"])
                    except Exception:
                        pass
            data.append(row)
        return data


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
        row = job.__dict__.copy()
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
        row = job.__dict__.copy()
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


@unconverted
def job_update_status_sync(job_id, status, error_msg=None):
    db_sync = None
    cursor = None
    try:
        db_sync = get_sync_db_connection()
        cursor = db_sync.cursor()

        cursor.execute("UPDATE job SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, job_id))
        db_sync.commit()
        return
    except Exception as e:
        print("Error updating job status: " + str(e))
        return
    finally:
        if cursor:
            cursor.close()
        if db_sync:
            db_sync.close()


async def job_update(job_id, type, status):
    async with async_session() as session:
        await session.execute(update(models.Job).where(models.Job.id == job_id).values(type=type, status=status))
        await session.commit()
    return


@unconverted
def job_update_sync(job_id, status):
    # This is a synchronous version of job_update
    # It is used by popen_and_call function
    # which can only support sychronous functions
    # This is a hack to get around that limitation
    db_sync = None
    cursor = None
    try:
        db_sync = get_sync_db_connection()
        cursor = db_sync.cursor()

        cursor.execute("UPDATE job SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, job_id))
        db_sync.commit()
        return
    except Exception as e:
        print("Error updating job status: " + str(e))
        return
    finally:
        if cursor:
            cursor.close()
        if db_sync:
            db_sync.close()


@unconverted
def job_mark_as_complete_if_running(job_id):
    # This synchronous update to jobs
    # only marks a job as "COMPLETE" if it is currenty "RUNNING"
    # This avoids updating "stopped" jobs and marking them as complete
    db_sync = None
    cursor = None
    try:
        db_sync = get_sync_db_connection()
        cursor = db_sync.cursor()
        cursor.execute(
            "UPDATE job SET status = 'COMPLETE', updated_at = CURRENT_TIMESTAMP WHERE id = ? AND status = 'RUNNING'",
            (job_id,),
        )
        db_sync.commit()
        return
    except Exception as e:
        print("Error updating job status: " + str(e))
        return
    finally:
        if cursor:
            cursor.close()
        if db_sync:
            db_sync.close()


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


async def job_cancel_in_progress_jobs():
    async with async_session() as session:
        await session.execute(update(models.Job).where(models.Job.status == "RUNNING").values(status="CANCELLED"))
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


###############
# TASKS MODEL
###############


async def add_task(name, Type, inputs, config, plugin, outputs, experiment_id):
    async with async_session() as session:
        stmt = insert(models.Task).values(
            name=name,
            type=Type,
            inputs=inputs,
            config=config,
            plugin=plugin,
            outputs=outputs,
            experiment_id=experiment_id,
        )
        await session.execute(stmt)
        await session.commit()
    return


async def update_task(task_id, new_task):
    async with async_session() as session:
        values = {}
        if "inputs" in new_task:
            values["inputs"] = new_task["inputs"]
        if "config" in new_task:
            values["config"] = new_task["config"]
        if "outputs" in new_task:
            values["outputs"] = new_task["outputs"]
        if "name" in new_task and new_task["name"] != "":
            values["name"] = new_task["name"]
        if values:
            await session.execute(update(models.Task).where(models.Task.id == task_id).values(**values))
            await session.commit()
    return


async def tasks_get_all():
    async with async_session() as session:
        result = await session.execute(select(models.Task).order_by(models.Task.created_at.desc()))
        tasks = result.scalars().all()
        data = []
        for task in tasks:
            row = task.__dict__.copy()
            data.append(row)
        return data


async def tasks_get_by_type(Type):
    async with async_session() as session:
        result = await session.execute(
            select(models.Task).where(models.Task.type == Type).order_by(models.Task.created_at.desc())
        )
        tasks = result.scalars().all()
        data = []
        for task in tasks:
            row = task.__dict__.copy()
            data.append(row)
        return data


async def tasks_get_by_type_in_experiment(Type, experiment_id):
    async with async_session() as session:
        result = await session.execute(
            select(models.Task)
            .where(models.Task.type == Type, models.Task.experiment_id == experiment_id)
            .order_by(models.Task.created_at.desc())
        )
        tasks = result.scalars().all()
        data = []
        for task in tasks:
            row = task.__dict__.copy()
            data.append(row)
        return data


async def delete_task(task_id):
    async with async_session() as session:
        await session.execute(delete(models.Task).where(models.Task.id == task_id))
        await session.commit()
    return


async def tasks_delete_all():
    async with async_session() as session:
        await session.execute(delete(models.Task))
        await session.commit()
    return


async def tasks_get_by_id(task_id):
    async with async_session() as session:
        result = await session.execute(
            select(models.Task).where(models.Task.id == task_id).order_by(models.Task.created_at.desc()).limit(1)
        )
        task = result.scalar_one_or_none()
        if task is None:
            return None
        return task.__dict__


async def get_training_template(id):
    async with async_session() as session:
        result = await session.execute(select(models.TrainingTemplate).where(models.TrainingTemplate.id == id))
        template = result.scalar_one_or_none()
        if template is None:
            return None
        # Convert ORM object to dict
        return template.__dict__


async def get_training_template_by_name(name):
    async with async_session() as session:
        result = await session.execute(select(models.TrainingTemplate).where(models.TrainingTemplate.name == name))
        template = result.scalar_one_or_none()
        if template is None:
            return None
        # Convert ORM object to dict
        return template.__dict__


async def get_training_templates():
    async with async_session() as session:
        result = await session.execute(
            select(models.TrainingTemplate).order_by(models.TrainingTemplate.created_at.desc())
        )
        templates = result.scalars().all()
        # Convert ORM objects to dicts if needed
        return [t.__dict__ for t in templates]


async def create_training_template(name, description, type, datasets, config):
    async with async_session() as session:
        template = models.TrainingTemplate(
            name=name,
            description=description,
            type=type,
            datasets=datasets,
            config=config,
        )
        session.add(template)
        await session.commit()
    return


async def update_training_template(id, name, description, type, datasets, config):
    async with async_session() as session:
        await session.execute(
            update(models.TrainingTemplate)
            .where(models.TrainingTemplate.id == id)
            .values(
                name=name,
                description=description,
                type=type,
                datasets=datasets,
                config=config,
            )
        )
        await session.commit()
    return


async def delete_training_template(id):
    async with async_session() as session:
        await session.execute(delete(models.TrainingTemplate).where(models.TrainingTemplate.id == id))
        await session.commit()
    return


async def training_jobs_get_all():
    async with async_session() as session:
        # Select jobs of type "TRAIN" and join with TrainingTemplate using the template_id from job_data JSON
        stmt = (
            select(
                models.Job,
                models.TrainingTemplate.id.label("tt_id"),
                models.TrainingTemplate.config,
            )
            .join(
                models.TrainingTemplate,
                text("json_extract(job.job_data, '$.template_id') = training_template.id"),
            )
            .where(models.Job.type == "TRAIN")
        )
        result = await session.execute(stmt)
        rows = result.all()

        data = []
        for job, tt_id, config in rows:
            row = job.__dict__.copy()
            row["tt_id"] = tt_id
            # Convert job_data and config from JSON string to Python object
            if "job_data" in row and row["job_data"]:
                try:
                    row["job_data"] = json.loads(row["job_data"])
                except Exception:
                    pass
            if config:
                try:
                    row["config"] = json.loads(config)
                except Exception:
                    row["config"] = config
            else:
                row["config"] = None
            data.append(row)
        return data


# async def training_job_create(template_id, description, experiment_id):

#     job_data = {
#         "template_id": template_id,
#         "description": description,
#     }

#     job_data = json.dumps(job_data)

#     row = await db.execute_insert(
#         "INSERT INTO job(type, status, experiment_id, job_data) VALUES (?, ?, ?, json(?))",
#         ("TRAIN", "QUEUED", experiment_id, job_data),
#     )
#     await db.commit()  # is this necessary?
#     return row[0]


####################
# EXPEORT JOBS MODEL
# Export jobs use the job_data JSON object to store:
# - exporter_name
# - input_model_id
# - input_model_architecture
# - output_model_id
# - output_model_architecture
# - output_model_name
# - output_model_path
# - params
####################


async def export_job_create(experiment_id, job_data_json):
    job_id = await job_create("EXPORT_MODEL", "Started", job_data_json, experiment_id)
    return job_id


###################
# EXPERIMENTS MODEL
###################


async def experiment_get_all():
    async with async_session() as session:
        result = await session.execute(select(models.Experiment).order_by(models.Experiment.created_at.desc()))
        experiments = result.scalars().all()
        # Convert ORM objects to dicts
        return [e.__dict__ for e in experiments]


async def experiment_create(name: str, config: dict) -> int:
    # test to see if config is a valid dict:
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")
    async with async_session() as session:
        experiment = models.Experiment(name=name, config=config)
        session.add(experiment)
        await session.commit()
        await session.refresh(experiment)
        return experiment.id


async def experiment_get(id):
    if id is None or id == "undefined":
        return None
    async with async_session() as session:
        result = await session.execute(select(models.Experiment).where(models.Experiment.id == id))
        experiment = result.scalar_one_or_none()
        if experiment is None:
            return None
        # Ensure config is always a JSON string
        if isinstance(experiment.config, dict):
            experiment.config = json.dumps(experiment.config)
        return experiment.__dict__


async def experiment_get_by_name(name):
    async with async_session() as session:
        result = await session.execute(select(models.Experiment).where(models.Experiment.name == name))
        experiment = result.scalar_one_or_none()
        if experiment is None:
            return None
        return experiment.__dict__


async def experiment_delete(id):
    async with async_session() as session:
        result = await session.execute(select(models.Experiment).where(models.Experiment.id == id))
        experiment = result.scalar_one_or_none()
        if experiment:
            await session.delete(experiment)
            await session.commit()
    return


async def experiment_update(id, config):
    # Ensure config is JSON string
    if not isinstance(config, str):
        config = json.dumps(config)
    async with async_session() as session:
        result = await session.execute(select(models.Experiment).where(models.Experiment.id == id))
        experiment = result.scalar_one_or_none()
        if experiment:
            experiment.config = config
            await session.commit()
    return


async def experiment_update_config(id, key, value):
    # Fetch the experiment
    async with async_session() as session:
        result = await session.execute(select(models.Experiment).where(models.Experiment.id == id))
        experiment = result.scalar_one_or_none()
        if experiment is None:
            print(f"Experiment with id={id} not found.")
            return

        # Parse config as dict if needed
        config = experiment.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except Exception as e:
                print(f"❌ Could not parse config as JSON: {e}")
                config = {}

        # Update the key
        config[key] = value

        # Use experiment_update to save
        await experiment_update(id, config)
    return


async def experiment_save_prompt_template(id, template):
    # Fetch the experiment config, update prompt_template, and save using experiment_update
    async with async_session() as session:
        result = await session.execute(select(models.Experiment).where(models.Experiment.id == id))
        experiment = result.scalar_one_or_none()
        if experiment is None:
            return
        config = experiment.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except Exception:
                config = {}
        config["prompt_template"] = template
        await experiment_update(id, config)
    return


#################
# WORKFLOWS MODEL
#################


async def workflows_get_all():
    async with async_session() as session:
        result = await session.execute(
            select(models.Workflow)
            .where(models.Workflow.status != "DELETED")
            .order_by(models.Workflow.created_at.desc())
        )
        workflows = result.scalars().all()
        # Convert ORM objects to dicts
        return [w.__dict__ for w in workflows]


async def workflows_get_from_experiment(experiment_id):
    async with async_session() as session:
        result = await session.execute(
            select(models.Workflow)
            .where(
                models.Workflow.experiment_id == experiment_id,
                models.Workflow.status != "DELETED",
            )
            .order_by(models.Workflow.created_at.desc())
        )
        workflows = result.scalars().all()
        return [w.__dict__ for w in workflows]


async def workflow_run_get_all():
    async with async_session() as session:
        result = await session.execute(
            select(models.WorkflowRun)
            .where(models.WorkflowRun.status != "DELETED")
            .order_by(models.WorkflowRun.created_at.desc())
        )
        workflow_runs = result.scalars().all()
        # Convert ORM objects to dicts
        return [wr.__dict__ for wr in workflow_runs]


async def workflows_get_by_id(workflow_id, experiment_id):
    async with async_session() as session:
        result = await session.execute(
            select(models.Workflow)
            .where(
                models.Workflow.id == workflow_id,
                models.Workflow.experiment_id == experiment_id,
                models.Workflow.status != "DELETED",
            )
            .order_by(models.Workflow.created_at.desc())
            .limit(1)
        )
        workflow = result.scalar_one_or_none()
        if workflow is None:
            return None
        return workflow.__dict__


async def workflow_run_get_by_id(workflow_run_id):
    async with async_session() as session:
        result = await session.execute(
            select(models.WorkflowRun)
            .where(models.WorkflowRun.id == workflow_run_id)
            .order_by(models.WorkflowRun.created_at.desc())
            .limit(1)
        )
        workflow_run = result.scalar_one_or_none()
        if workflow_run is None:
            return None
        return workflow_run.__dict__


async def workflow_delete_by_id(workflow_id: str, experiment_id):
    print("Deleting workflow: " + str(workflow_id))
    async with async_session() as session:
        result = await session.execute(
            update(models.Workflow)
            .where(models.Workflow.id == workflow_id, models.Workflow.experiment_id == experiment_id)
            .values(status="DELETED", updated_at=text("CURRENT_TIMESTAMP"))
        )
        await session.commit()
        return result.rowcount > 0


async def workflow_delete_by_name(workflow_name):
    print("Deleting workflow: " + workflow_name)
    async with async_session() as session:
        result = await session.execute(
            update(models.Workflow)
            .where(models.Workflow.name == workflow_name)
            .values(status="DELETED", updated_at=text("CURRENT_TIMESTAMP"))
        )
        await session.commit()
        return result.rowcount > 0


async def workflow_count_running():
    async with async_session() as session:
        result = await session.execute(select(models.WorkflowRun).where(models.WorkflowRun.status == "RUNNING"))
        count = len(result.scalars().all())
        return count


async def workflow_count_queued():
    async with async_session() as session:
        result = await session.execute(select(models.WorkflowRun).where(models.WorkflowRun.status == "QUEUED"))
        count = len(result.scalars().all())
        return count


async def workflow_run_get_running():
    async with async_session() as session:
        result = await session.execute(
            select(models.WorkflowRun)
            .where(models.WorkflowRun.status == "RUNNING")
            .order_by(models.WorkflowRun.created_at.asc())
            .limit(1)
        )
        workflow_run = result.scalar_one_or_none()
        if workflow_run is None:
            return None
        return workflow_run.__dict__


async def workflow_run_get_queued():
    async with async_session() as session:
        result = await session.execute(
            select(models.WorkflowRun)
            .where(models.WorkflowRun.status == "QUEUED")
            .order_by(models.WorkflowRun.created_at.asc())
            .limit(1)
        )
        workflow_run = result.scalar_one_or_none()
        if workflow_run is None:
            return None
        return workflow_run.__dict__


async def workflow_run_update_status(workflow_run_id, status):
    async with async_session() as session:
        await session.execute(
            update(models.WorkflowRun)
            .where(models.WorkflowRun.id == workflow_run_id)
            .values(status=status, updated_at=text("CURRENT_TIMESTAMP"))
        )
        await session.commit()
    return


@unconverted
async def workflow_run_update_with_new_job(workflow_run_id, current_task, current_job_id):
    await db.execute(
        "UPDATE workflow_runs SET current_tasks = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (current_task, workflow_run_id),
    )
    await db.execute(
        "UPDATE workflow_runs SET current_job_ids = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (current_job_id, workflow_run_id),
    )

    current_workflow_run = await workflow_run_get_by_id(workflow_run_id)
    current_workflow_run["job_ids"] = json.loads(current_workflow_run["job_ids"])
    current_workflow_run["job_ids"] += json.loads(current_job_id)
    current_workflow_run["job_ids"] = json.dumps(current_workflow_run["job_ids"])
    current_workflow_run["node_ids"] = json.loads(current_workflow_run["node_ids"])
    current_workflow_run["node_ids"] += json.loads(current_task)
    current_workflow_run["node_ids"] = json.dumps(current_workflow_run["node_ids"])

    await db.execute(
        "UPDATE workflow_runs SET node_ids = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (current_workflow_run["node_ids"], workflow_run_id),
    )
    await db.execute(
        "UPDATE workflow_runs SET job_ids = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (current_workflow_run["job_ids"], workflow_run_id),
    )
    await db.commit()
    return


async def workflow_create(name, config, experiment_id):
    async with async_session() as session:
        workflow = models.Workflow(
            name=name,
            config=config,
            status="CREATED",
            experiment_id=experiment_id,
        )
        session.add(workflow)
        await session.commit()
        await session.refresh(workflow)
        return workflow.id


async def workflow_update_config(workflow_id, config, experiment_id):
    async with async_session() as session:
        result = await session.execute(
            update(models.Workflow)
            .where(models.Workflow.id == workflow_id, models.Workflow.experiment_id == experiment_id)
            .values(config=config, updated_at=text("CURRENT_TIMESTAMP"))
        )
        await session.commit()
        return result.rowcount > 0


async def workflow_update_name(workflow_id, name, experiment_id):
    async with async_session() as session:
        result = await session.execute(
            update(models.Workflow)
            .where(models.Workflow.id == workflow_id, models.Workflow.experiment_id == experiment_id)
            .values(name=name, updated_at=text("CURRENT_TIMESTAMP"))
        )
        await session.commit()
        return result.rowcount > 0


async def workflow_delete_all():
    async with async_session() as session:
        await session.execute(delete(models.Workflow))
        await session.commit()


async def workflow_runs_delete_all():
    async with async_session() as session:
        await session.execute(delete(models.WorkflowRun))
        await session.commit()


@unconverted
async def workflow_queue(workflow_id):
    # Get workflow data directly instead of using workflows_get_by_id which now requires experiment_id
    cursor = await db.execute(
        "SELECT name, experiment_id FROM workflows WHERE id = ? AND status != 'DELETED' LIMIT 1", (workflow_id,)
    )
    row = await cursor.fetchone()
    await cursor.close()

    if row:
        workflow_name = row[0]
        experiment_id = row[1]
        await db.execute(
            "INSERT INTO workflow_runs(workflow_id, workflow_name, job_ids, node_ids, status, current_tasks, current_job_ids, experiment_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (workflow_id, workflow_name, "[]", "[]", "QUEUED", "[]", "[]", experiment_id),
        )
        return True

    return False


async def workflow_runs_get_from_experiment(experiment_id):
    async with async_session() as session:
        result = await session.execute(
            select(models.WorkflowRun)
            .where(
                models.WorkflowRun.experiment_id == experiment_id,
                models.WorkflowRun.status != "DELETED",
            )
            .order_by(models.WorkflowRun.created_at.desc())
        )
        workflow_runs = result.scalars().all()
        # Convert ORM objects to dicts
        return [wr.__dict__ for wr in workflow_runs]


###############
# PLUGINS MODEL
###############


async def get_plugins():
    async with async_session() as session:
        result = await session.execute(select(Plugin))
        plugins = result.scalars().all()
        # Convert ORM objects to dicts
        return [p.__dict__ for p in plugins]


async def get_plugins_of_type(type: str):
    async with async_session() as session:
        result = await session.execute(select(Plugin).where(Plugin.type == type))
        plugins = result.scalars().all()
        return [p.__dict__ for p in plugins]


async def get_plugin(slug: str):
    async with async_session() as session:
        result = await session.execute(select(Plugin).where(Plugin.name == slug))
        plugin = result.scalar_one_or_none()
        return plugin.__dict__ if plugin else None


async def save_plugin(name: str, type: str):
    async with async_session() as session:
        plugin = await session.get(Plugin, name)
        if plugin:
            plugin.type = type
        else:
            plugin = Plugin(name=name, type=type)
            session.add(plugin)
        await session.commit()
    return


async def delete_plugin(name: str):
    async with async_session() as session:
        plugin = await session.get(Plugin, name)
        if plugin:
            await session.delete(plugin)
            await session.commit()
            return True
    return False


###############
# Config MODEL
###############


async def config_get(key: str):
    async with async_session() as session:
        result = await session.execute(select(Config.value).where(Config.key == key))
        row = result.scalar_one_or_none()
        return row


async def config_set(key: str, value: str):
    stmt = insert(Config).values(key=key, value=value)
    stmt = stmt.on_conflict_do_update(index_elements=["key"], set_={"value": value})
    async with async_session() as session:
        await session.execute(stmt)
        await session.commit()
    return
