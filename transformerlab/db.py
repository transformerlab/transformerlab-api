import itertools
import json
import os
import sqlite3

import aiosqlite
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert  # Correct import for SQLite upsert

# from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Make sure SQLAlchemy is installed using pip install sqlalchemy[asyncio] as
# described here https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html

from transformerlab.shared import dirs
from transformerlab.shared.models import models  # noqa: F401
from transformerlab.shared.models.models import Config

db = None
DATABASE_FILE_NAME = f"{dirs.WORKSPACE_DIR}/llmlab.sqlite3"
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_FILE_NAME}"

# Create SQLAlchemy engines
# engine = create_engine(DATABASE_URL, echo=True)
async_engine = create_async_engine(DATABASE_URL, echo=False)

# Create a configured "Session" class
async_session = sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)

# List of triggers to be seeded into the database.
# The `_seed_workflow_triggers` function will process this list during application startup.
TRIGGERS_TO_SEED = [
    {
        "name": "On Training Completed",
        "description": "Workflows will automatically start after every training job is completed.",
        "trigger_type": "TRAIN",
        "experiment_name": "dev", 
        "is_enabled": True,
        "config": {"workflow_ids": []} 
    },
    {
        "name": "On New Model Downloaded",
        "description": "Workflows will automatically start after every new model is downloaded.",
        "trigger_type": "DOWNLOAD_MODEL",
        "experiment_name": "dev",
        "is_enabled": True,
        "config": {"workflow_ids": []}
    },
    {
        "name": "On New Model Loaded",
        "description": "Workflows will automatically start after every new model is loaded for inference.",
        "trigger_type": "LOAD_MODEL",
        "experiment_name": "dev",
        "is_enabled": True,
        "config": {"workflow_ids": []}
    },
    {
        "name": "On Model Export Completed",
        "description": "Workflows will automatically start after every model is exported.",
        "trigger_type": "EXPORT_MODEL",
        "experiment_name": "dev",
        "is_enabled": True,
        "config": {"workflow_ids": []}
    },
    {
        "name": "On Evaluation Completed",
        "description": "Workflows will automatically start after every evaluation job is completed.",
        "trigger_type": "EVAL",
        "experiment_name": "dev",
        "is_enabled": True,
        "config": {"workflow_ids": []}
    },
    {
        "name": "On Generation Completed",
        "description": "Workflows will automatically start after every generation job is completed.",
        "trigger_type": "GENERATE",
        "experiment_name": "dev",
        "is_enabled": True,
        "config": {"workflow_ids": []}
    },
]

async def _seed_workflow_triggers(db_conn):
    """
    Seeds workflow triggers into the database based on the TRIGGERS_TO_SEED list.
    Checks if a trigger for a given experiment_id and trigger_type already exists
    before inserting to prevent duplicates.
    """
    print("🌱 Seeding workflow triggers...")
    for trigger_data in TRIGGERS_TO_SEED:
        experiment_name = trigger_data["experiment_name"]
        trigger_type = trigger_data["trigger_type"]
        name = trigger_data["name"]
        description = trigger_data["description"]

        # Get experiment_id from experiment_name
        # This relies on experiment_get_by_name being defined elsewhere in this file
        experiment = await experiment_get_by_name(experiment_name) 
        if not experiment:
            print(f"  ⚠️  Skipping trigger '{name}': Experiment '{experiment_name}' not found.")
            continue
        
        experiment_id = experiment['id']

        # Check if this specific trigger (by experiment_id and type) already exists
        cursor = await db_conn.execute(
            "SELECT id FROM workflow_triggers WHERE experiment_id = ? AND trigger_type = ?",
            (experiment_id, trigger_type)
        )
        existing_trigger = await cursor.fetchone()
        await cursor.close()

        if not existing_trigger:
            print(f"  Creating trigger: '{name}' for experiment '{experiment_name}' (ID: {experiment_id}), Type: {trigger_type}")
            await db_conn.execute(
                "INSERT OR IGNORE INTO workflow_triggers (name, description, trigger_type, experiment_id, is_enabled, config) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    name,
                    description,
                    trigger_type,
                    experiment_id,
                    1 if trigger_data["is_enabled"] else 0,
                    json.dumps(trigger_data["config"])
                )
            )
        else:
            # Update the description for existing triggers
            print(f"  Updating trigger description for experiment '{experiment_name}' (ID: {experiment_id}), Type: {trigger_type}")
            await db_conn.execute(
                "UPDATE workflow_triggers SET description = ? WHERE id = ?",
                (description, existing_trigger[0])
            )
    
    await db_conn.commit()
    print("✅ Workflow triggers seeding complete.")


async def init():
    """
    Create the database, tables, and workspace folder if they don't exist.
    """
    global db
    os.makedirs(os.path.dirname(DATABASE_FILE_NAME), exist_ok=True)
    db = await aiosqlite.connect(DATABASE_FILE_NAME)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=normal")
    await db.execute("PRAGMA busy_timeout = 5000")

    # Create the tables if they don't exist
    async with async_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

    # Add description column to workflow_triggers if it doesn't exist
    try:
        await db.execute("ALTER TABLE workflow_triggers ADD COLUMN description TEXT")
        print("✅ Added description column to workflow_triggers table")
    except Exception as e:
        if "duplicate column name" not in str(e).lower():
            print(f"⚠️  Note about description column: {str(e)}")

    print("✅ Database initialized")

    print("✅ SEED DATA")
    async with async_session() as session:
        for name in ["alpha", "beta", "gamma"]:
            # Check if experiment already exists
            exists = await session.execute(select(models.Experiment).where(models.Experiment.name == name))
            if not exists.scalar_one_or_none():
                session.add(models.Experiment(name=name, config="{}"))
        await session.commit()

    # On startup, look for any jobs that are in the RUNNING state and set them to CANCELLED instead:
    # This is to handle the case where the server is restarted while a job is running.
    await job_cancel_in_progress_jobs()
    # await init_sql_model()

    return


def get_sync_db_connection():
    global DATABASE_FILE_NAME
    db_sync = sqlite3.connect(DATABASE_FILE_NAME, isolation_level=None)
    db_sync.execute("PRAGMA journal_mode=WAL")
    db_sync.execute("PRAGMA synchronous=normal")
    db_sync.execute("PRAGMA busy_timeout = 5000")
    return db_sync


async def close():
    await db.close()
    await async_engine.dispose()
    print("✅ Database closed")
    return


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
    cursor = await db.execute("SELECT * FROM dataset WHERE dataset_id = ?", (dataset_id,))
    row = await cursor.fetchone()

    # Make sure the dataset exists before formatting repsonse
    if row is None:
        return None

    # convert to json
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))
    if "json_data" in row and row["json_data"]:
        row["json_data"] = json.loads(row["json_data"])

    await cursor.close()
    return row


async def get_datasets():
    cursor = await db.execute("SELECT rowid, * FROM dataset")
    rows = await cursor.fetchall()

    # convert to json:
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()

    return data


async def get_generated_datasets():
    # Get all datasets that have the value `generated` as True in the json_data column
    cursor = await db.execute("SELECT rowid, * FROM dataset WHERE json_extract(json_data, '$.generated') = true")
    rows = await cursor.fetchall()

    # convert to json:
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()

    return data


async def create_huggingface_dataset(dataset_id, description, size, json_data):
    await db.execute(
        """
        INSERT INTO dataset (dataset_id, location, description, size, json_data)
        VALUES (?, ?, ?, ?, json(?))
        """,
        (dataset_id, "huggingfacehub", description, size, json.dumps(json_data)),
    )
    await db.commit()


async def create_local_dataset(dataset_id, json_data=None):
    if json_data is None:
        await db.execute(
            """
            INSERT INTO dataset (dataset_id, location, description, size, json_data)
            VALUES (?, ?, ?, ?, json(?))
            """,
            (dataset_id, "local", "", -1, "{}"),
        )
    else:
        await db.execute(
            """
            INSERT INTO dataset (dataset_id, location, description, size, json_data)
            VALUES (?, ?, ?, ?, json(?))
            """,
            (dataset_id, "local", "", -1, json.dumps(json_data)),
        )
    await db.commit()


async def delete_dataset(dataset_id):
    await db.execute("DELETE FROM dataset WHERE dataset_id = ?", (dataset_id,))
    await db.commit()


###############
# MODELS MODEL
###############


async def model_local_list():
    cursor = await db.execute("SELECT rowid, * FROM model")
    rows = await cursor.fetchall()

    # Convert to JSON
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()

    # convert json_data column from JSON to Python object
    for row in data:
        row["json_data"] = json.loads(row["json_data"])

    return data


async def model_local_count():
    cursor = await db.execute("SELECT COUNT(*) FROM model")
    row = await cursor.fetchone()
    await cursor.close()

    return row[0]


async def model_local_create(model_id, name, json_data):
    json_data = json.dumps(obj=json_data)

    await db.execute(
        "INSERT OR REPLACE INTO model(model_id, name, json_data) VALUES (?, ?,?)", (model_id, name, json_data)
    )

    await db.commit()


async def model_local_get(model_id):
    cursor = await db.execute("SELECT rowid, * FROM model WHERE model_id = ?", (model_id,))
    row = await cursor.fetchone()

    # Returns None if the model_id isn't in the database
    if row is None:
        return None

    # Map column names to row data
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))
    await cursor.close()

    # convert json_data column from JSON to Python object
    row["json_data"] = json.loads(row["json_data"])

    return row


async def model_local_delete(model_id):
    await db.execute("DELETE FROM model WHERE model_id = ?", (model_id,))
    await db.commit()


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
    base_query = "SELECT * FROM job"
    if type != "":
        base_query += " WHERE type = ?"
    else:
        base_query += " WHERE ? != 'x'"

    if status != "":
        base_query += " AND status = ?"
    else:
        base_query += " AND ? != 'x'"

    base_query += " AND status != 'DELETED' ORDER BY created_at DESC"

    cursor = await db.execute(base_query, (type, status))
    rows = await cursor.fetchall()

    # Add column names to output
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()

    # for each row in data, convert the job_data
    # column from JSON to a Python object
    for i in range(len(data)):
        data[i]["job_data"] = json.loads(data[i]["job_data"])

    return data


async def jobs_get_all_by_experiment_and_type(experiment_id, job_type):
    cursor = await db.execute(
        "SELECT * FROM job \
        WHERE experiment_id = ? \
        AND type = ? \
        AND status != 'DELETED' \
        ORDER BY created_at DESC",
        (experiment_id, job_type),
    )
    rows = await cursor.fetchall()

    # Add column names to output
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()

    # for each row in data, convert the job_data
    # column from JSON to a Python object
    for row in data:
        row["job_data"] = json.loads(row["job_data"])

    return data


async def job_get_status(job_id):
    cursor = await db.execute("SELECT status FROM job WHERE id = ?", (job_id,))
    row = await cursor.fetchone()
    await cursor.close()
    return row[0]


async def job_get_error_msg(job_id):
    cursor = await db.execute("SELECT job_data FROM job WHERE id = ?", (job_id,))
    row = await cursor.fetchone()
    await cursor.close()
    job_data = json.loads(row[0])
    return job_data.get("error_msg", None)


async def job_get(job_id):
    cursor = await db.execute("SELECT * FROM job WHERE id = ?", (job_id,))
    row = await cursor.fetchone()

    # if no results, return None
    if row is None:
        return None

    # convert to json:
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))
    await cursor.close()

    row["job_data"] = json.loads(row["job_data"])
    return row


async def job_count_running():
    cursor = await db.execute("SELECT COUNT(*) FROM job WHERE status = 'RUNNING'")
    row = await cursor.fetchone()
    await cursor.close()
    return row[0]


async def jobs_get_next_queued_job():
    cursor = await db.execute("SELECT * FROM job WHERE status = 'QUEUED' ORDER BY created_at ASC LIMIT 1")
    row = await cursor.fetchone()

    # if no results, return None
    if row is None:
        return None

    # convert to json:
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))

    await cursor.close()
    return row


async def job_update_status(job_id, status, error_msg=None):
    print(f"🔄 Updating job {job_id} status to {status}")
    await db.execute("UPDATE job SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, job_id))
    await db.commit()
    if error_msg:
        job_data = json.dumps({"error_msg": str(error_msg)})
        await db.execute("UPDATE job SET job_data = ? WHERE id = ?", (job_data, job_id))
        await db.commit()
    
    # Process triggers if job is completed
    if status == "COMPLETE":
        print(f"🎯 Job {job_id} completed, attempting to process triggers")
        try:
            from transformerlab.jobs_trigger_processing import process_job_completion_triggers
            await process_job_completion_triggers(job_id)
            print(f"✅ Trigger processing completed for job {job_id}")
        except Exception as e:
            print(f"💥 Error processing triggers for job {job_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return


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
    await db.execute(
        "UPDATE job SET type = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (type, status, job_id)
    )
    await db.commit()
    return


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
        
        # Process triggers for completed jobs
        print(f"🎯 Job {job_id} marked as complete, attempting to process triggers")
        try:
            # Since this is a synchronous context, we need to run the async trigger processing
            # using asyncio. We'll do this in a way that doesn't block if there's already an event loop.
            import asyncio
            from transformerlab.jobs_trigger_processing import process_job_completion_triggers
            
            # Try to get the current event loop, if it exists
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an async context, schedule the task
                loop.create_task(process_job_completion_triggers(job_id))
                print(f"✅ Scheduled trigger processing for job {job_id}")
            except RuntimeError:
                # No running event loop, so we can run it directly
                asyncio.run(process_job_completion_triggers(job_id))
                print(f"✅ Trigger processing completed for job {job_id}")
        except Exception as e:
            print(f"💥 Error processing triggers for job {job_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        
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
    # await db.execute("DELETE FROM job")
    await db.execute("UPDATE job SET status = 'DELETED'")
    await db.commit()
    return


async def job_delete(job_id):
    print("Deleting job: " + str(job_id))
    # await db.execute("DELETE FROM job WHERE id = ?", (job_id,))
    # instead of deleting, set status of job to deleted:
    await db.execute("UPDATE job SET status = 'DELETED' WHERE id = ?", (job_id,))
    await db.commit()
    return


async def job_cancel_in_progress_jobs():
    await db.execute("UPDATE job SET status = 'CANCELLED' WHERE status = 'RUNNING'")
    await db.commit()
    return


async def job_update_job_data_insert_key_value(job_id, key, value):
    value = json.dumps(value)

    await db.execute(
        "UPDATE job SET job_data = " + f"json_set(job_data,'$.{key}', json(?))  WHERE id = ?",
        (value, job_id),
    )
    await db.commit()
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
    await db.execute(
        "UPDATE job SET progress = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (progress, job_id),
    )


async def job_update_sweep_progress(job_id, value):
    value = json.dumps(value)

    await db.execute(
        "UPDATE job SET job_data = " + "json_set(job_data,'$.sweep_progress', json(?))  WHERE id = ?",
        (value, job_id),
    )
    await db.commit()
    return


###############
# TASKS MODEL
###############


async def add_task(name, Type, inputs, config, plugin, outputs, experiment_id):
    await db.execute(
        "INSERT INTO tasks(name, type, inputs, config, plugin, outputs, experiment_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (name, Type, inputs, config, plugin, outputs, experiment_id),
    )
    await db.commit()
    return


async def update_task(task_id, new_task):
    await db.execute(
        "UPDATE tasks SET inputs = ? WHERE id = ?",
        (new_task["inputs"], task_id),
    )
    await db.execute(
        "UPDATE tasks SET config = ? WHERE id = ?",
        (new_task["config"], task_id),
    )
    await db.execute(
        "UPDATE tasks SET outputs = ? WHERE id = ?",
        (new_task["outputs"], task_id),
    )
    if "name" in new_task and new_task["name"] != "":
        await db.execute(
            "UPDATE tasks SET name = ? WHERE id = ?",
            (new_task["name"], task_id),
        )
    await db.commit()
    return


async def tasks_get_all():
    cursor = await db.execute("SELECT * FROM tasks ORDER BY created_at desc")
    rows = await cursor.fetchall()
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


async def tasks_get_by_type(Type):
    cursor = await db.execute("SELECT * FROM tasks WHERE type = ? ORDER BY created_at desc", (Type,))
    rows = await cursor.fetchall()
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


async def tasks_get_by_type_in_experiment(Type, experiment_id):
    cursor = await db.execute(
        "SELECT * FROM tasks WHERE type = ? AND experiment_id = ? ORDER BY created_at desc",
        (
            Type,
            experiment_id,
        ),
    )
    rows = await cursor.fetchall()
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


async def delete_task(task_id):
    await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    await db.commit()
    return


async def tasks_delete_all():
    await db.execute("DELETE FROM tasks")
    await db.commit()
    return


async def tasks_get_by_id(task_id):
    cursor = await db.execute("SELECT * FROM tasks WHERE id = ? ORDER BY created_at desc LIMIT 1", (task_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))
    await cursor.close()
    return row


###############
# TRAINING and TRAINING JOBS MODELS
###############


async def get_training_template(id):
    cursor = await db.execute("SELECT * FROM training_template WHERE id = ?", (id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    # convert to json:
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))

    await cursor.close()

    return row


async def get_training_template_by_name(name):
    cursor = await db.execute("SELECT * FROM training_template WHERE name = ?", (name,))
    row = await cursor.fetchone()
    if row is None:
        return None
    # convert to json:
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))

    await cursor.close()

    return row


async def get_training_templates():
    cursor = await db.execute("SELECT * FROM training_template ORDER BY created_at DESC")
    rows = await cursor.fetchall()
    await cursor.close()
    return rows


async def create_training_template(name, description, type, datasets, config):
    await db.execute(
        "INSERT INTO training_template(name, description, type, datasets, config) VALUES (?, ?, ?, ?, ?)",
        (name, description, type, datasets, config),
    )
    await db.commit()
    return


async def update_training_template(id, name, description, type, datasets, config):
    await db.execute(
        "UPDATE training_template SET name = ?, description = ?, type = ?, datasets = ?, config = ? WHERE id = ?",
        (name, description, type, datasets, config, id),
    )
    await db.commit()
    return


async def delete_training_template(id):
    await db.execute("DELETE FROM training_template WHERE id = ?", (id,))
    await db.commit()
    return


# Because this joins on training template it only returns training jobs
async def training_jobs_get_all():
    # Join on the nested JSON value "template_id"
    # #in the job_data column
    cursor = await db.execute(
        "SELECT j.*, tt.id as tt_id, tt.config from job as j \
            JOIN training_template as tt \
            ON  json_extract(j.job_data, '$.template_id') = tt.id \
            "
    )
    rows = await cursor.fetchall()

    # Convert to JSON
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()

    # for each row in data, convert the job_data
    # and config column from JSON to a Python object
    for row in data:
        row["job_data"] = json.loads(row["job_data"])
        row["config"] = json.loads(row["config"])

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
    cursor = await db.execute("SELECT * FROM experiment order by created_at desc")
    rows = await cursor.fetchall()
    # Do the following to convert the return into a JSON object with keys
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


async def experiment_create(name, config):
    # use python insert and commit command
    row = await db.execute_insert("INSERT INTO experiment(name, config) VALUES (?, ?)", (name, config))
    await db.commit()
    return row[0]


async def experiment_get(id):
    if id is None or id == "undefined":
        return None
    cursor = await db.execute("SELECT * FROM experiment WHERE id = ?", (id,))
    row = await cursor.fetchone()

    if row is None:
        return None

    # Convert the SQLite row into a JSON object with keys
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))

    await cursor.close()
    return row


async def experiment_get_by_name(name):
    cursor = await db.execute("SELECT * FROM experiment WHERE name = ?", (name,))
    row = await cursor.fetchone()

    if row is None:
        return None

    # Convert the SQLite row into a JSON object with keys
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))

    await cursor.close()
    return row


async def experiment_delete(id):
    await db.execute("DELETE FROM experiment WHERE id = ?", (id,))
    await db.commit()
    return


async def experiment_update(id, config):
    await db.execute("UPDATE experiment SET config = ? WHERE id = ?", (config, id))
    await db.commit()
    return


async def experiment_update_config(id, key, value):
    value = json.dumps(value)

    await db.execute(
        "UPDATE experiment SET config = " + f"json_set(config,'$.{key}', json(?))  WHERE id = ?",
        (value, id),
    )
    await db.commit()
    return


async def experiment_save_prompt_template(id, template):
    # The following looks the JSON blob called "config" and adds a key called "prompt_template" if it doesn't exist
    # it then sets the value of that key to the value of the template parameter
    # This is the pattern to follow for updating fields in the config JSON blob
    await db.execute(
        "UPDATE experiment SET config = json_set(config,'$.prompt_template', json(?))  WHERE id = ?",
        (template, id),
    )
    await db.commit()
    return


#################
# WORKFLOWS MODEL
#################


async def workflows_get_all():
    cursor = await db.execute("SELECT * FROM workflows WHERE status != 'DELETED' ORDER BY created_at desc")
    rows = await cursor.fetchall()
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


async def workflows_get_from_experiment(experiment_id):
    cursor = await db.execute(
        "SELECT * FROM workflows WHERE experiment_id = ? AND status != 'DELETED' ORDER BY created_at desc",
        (experiment_id,),
    )
    rows = await cursor.fetchall()
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


async def workflow_run_get_all():
    cursor = await db.execute("SELECT * FROM workflow_runs WHERE status != 'DELETED' ORDER BY created_at desc")
    rows = await cursor.fetchall()
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


async def workflows_get_by_id(workflow_id):
    cursor = await db.execute("SELECT * FROM workflows WHERE id = ? ORDER BY created_at desc LIMIT 1", (workflow_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))
    await cursor.close()
    return row


async def workflow_run_get_by_id(workflow_run_id):
    cursor = await db.execute(
        "SELECT * FROM workflow_runs WHERE id = ? ORDER BY created_at desc LIMIT 1", (workflow_run_id,)
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))
    await cursor.close()
    return row


async def workflow_delete_by_id(workflow_id: str):
    print("Deleting workflow: " + str(workflow_id))
    await db.execute(
        "UPDATE workflows SET status = 'DELETED', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (workflow_id,)
    )
    await db.commit()
    return


async def workflow_delete_by_name(workflow_name):
    print("Deleting workflow: " + workflow_name)
    await db.execute(
        "UPDATE workflows SET status = 'DELETED', updated_at = CURRENT_TIMESTAMP WHERE name = ?", (workflow_name,)
    )
    await db.commit()
    return


async def workflow_count_running():
    cursor = await db.execute("SELECT COUNT(*) FROM workflow_runs WHERE status = 'RUNNING'")
    row = await cursor.fetchone()
    await cursor.close()
    return row[0]


async def workflow_count_queued():
    cursor = await db.execute("SELECT COUNT(*) FROM workflow_runs WHERE status = 'QUEUED'")
    row = await cursor.fetchone()
    await cursor.close()
    return row[0]


async def workflow_run_get_running():
    cursor = await db.execute("SELECT * FROM workflow_runs WHERE status = 'RUNNING' LIMIT 1")
    row = await cursor.fetchone()
    if row is None:
        return None
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))
    await cursor.close()
    return row


async def workflow_run_get_queued():
    cursor = await db.execute("SELECT * FROM workflow_runs WHERE status = 'QUEUED' LIMIT 1")
    row = await cursor.fetchone()
    if row is None:
        return None
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))
    await cursor.close()
    return row


async def workflow_run_update_status(workflow_run_id, status):
    await db.execute(
        "UPDATE workflow_runs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, workflow_run_id)
    )
    await db.commit()
    return


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
    # check if type is allowed
    row = await db.execute_insert(
        "INSERT INTO workflows(name, config, status, experiment_id) VALUES (?, json(?), ?, ?)",
        (name, config, "CREATED", experiment_id),
    )
    await db.commit()
    return row[0]


async def workflow_update_config(workflow_id, config):
    await db.execute(
        "UPDATE workflows SET config = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (config, workflow_id)
    )
    await db.commit()


async def workflow_update_name(workflow_id, name):
    await db.execute("UPDATE workflows SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (name, workflow_id))
    await db.commit()


async def workflow_delete_all():
    await db.execute("DELETE FROM workflows")
    await db.commit()


async def workflow_runs_delete_all():
    await db.execute("DELETE FROM workflow_runs")
    await db.commit()


async def workflow_queue(workflow_id):
    workflow_name = (await workflows_get_by_id(workflow_id))["name"]
    await db.execute(
        "INSERT INTO workflow_runs(workflow_id, workflow_name, job_ids, node_ids, status, current_tasks, current_job_ids) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (workflow_id, workflow_name, "[]", "[]", "QUEUED", "[]", "[]"),
    )


###############
# PLUGINS MODEL
###############


async def get_plugins():
    cursor = await db.execute("SELECT id, * FROM plugins")
    rows = await cursor.fetchall()
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


async def get_plugins_of_type(type: str):
    cursor = await db.execute("SELECT id, * FROM plugins WHERE type = ?", (type,))
    rows = await cursor.fetchall()
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


async def get_plugin(slug: str):
    cursor = await db.execute("SELECT id, * FROM plugins WHERE name = ?", (slug,))
    row = await cursor.fetchone()
    if row is None:
        await cursor.close()
        return None
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row = dict(itertools.zip_longest(column_names, row))
    await cursor.close()
    return row


async def save_plugin(name: str, type: str):
    await db.execute("INSERT OR REPLACE INTO plugins (name, type) VALUES (?, ?)", (name, type))
    await db.commit()
    return


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


###############
# WORKFLOW TRIGGERS MODEL
###############


async def workflow_trigger_get_by_experiment_id(experiment_id: int):
    """Fetch all triggers for a given experiment."""
    cursor = await db.execute(
        "SELECT rowid, * FROM workflow_triggers WHERE experiment_id = ?", (experiment_id,)
    )
    rows = await cursor.fetchall()

    # convert to json:
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    
    # Parse JSON config
    for row in data:
        if row["config"]:
            row["config"] = json.loads(row["config"])
    
    await cursor.close()
    return data


async def workflow_trigger_get_by_id(trigger_id: int):
    """Fetch a specific trigger by its ID."""
    cursor = await db.execute("SELECT rowid, * FROM workflow_triggers WHERE id = ?", (trigger_id,))
    row = await cursor.fetchone()

    # Return None if the trigger_id isn't in the database
    if row is None:
        return None

    # Map column names to row data
    desc = cursor.description
    column_names = [col[0] for col in desc]
    row_dict = dict(itertools.zip_longest(column_names, row))
    
    # Parse JSON config
    if row_dict["config"]:
        row_dict["config"] = json.loads(row_dict["config"])
    
    await cursor.close()
    return row_dict


async def workflow_trigger_update(trigger_id: int, config: dict = None, is_enabled: bool = None):
    """Update the config and/or is_enabled status of a specific trigger."""
    update_parts = []
    params = []

    if config is not None:
        update_parts.append("config = json(?)")
        params.append(json.dumps(config))

    if is_enabled is not None:
        update_parts.append("is_enabled = ?")
        params.append(1 if is_enabled else 0)
    
    if not update_parts:
        return None  # Nothing to update

    # Add the current time for updated_at and the trigger_id
    update_parts.append("updated_at = ?")
    params.append(datetime.now().isoformat())
    params.append(trigger_id)

    sql = f"UPDATE workflow_triggers SET {', '.join(update_parts)} WHERE id = ?"
    await db.execute(sql, params)
    await db.commit()
    
    return await workflow_trigger_get_by_id(trigger_id)


async def workflow_trigger_get_enabled_by_experiment_id_and_type(experiment_id: int, trigger_type: str):
    """Fetch all enabled triggers for a specific experiment and type."""
    cursor = await db.execute(
        "SELECT rowid, * FROM workflow_triggers WHERE experiment_id = ? AND trigger_type = ? AND is_enabled = 1",
        (experiment_id, trigger_type)
    )
    rows = await cursor.fetchall()

    # convert to json:
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    
    # Parse JSON config
    for row in data:
        if row["config"]:
            row["config"] = json.loads(row["config"])
    
    await cursor.close()
    return data


async def job_mark_triggers_processed(job_id: int):
    """Mark a job as having had its triggers processed."""
    await db.execute(
        "UPDATE job SET triggers_processed_at = ? WHERE id = ?",
        (datetime.now().isoformat(), job_id)
    )
    await db.commit()


async def experiment_get_directory_by_id(experiment_id: int) -> str:
    """Get experiment directory path by experiment ID"""
    experiment = await experiment_get(experiment_id)
    if not experiment:
        print(f"Error: experiment {experiment_id} not found")
        return os.path.join(dirs.EXPERIMENTS_DIR, "error")
    
    experiment_name = experiment["name"]
    return dirs.experiment_dir_by_name(experiment_name)
