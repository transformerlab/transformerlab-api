import itertools
import json
import os
import sqlite3

import aiosqlite
from sqlalchemy import select, update, text
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


async def migrate_workflows_experiment_id_to_integer():
    """
    Migration function to convert workflows.experiment_id column to integer format
    """

    try:
        # Check if workflows table exists
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workflows'")
        table_exists = await cursor.fetchone()
        await cursor.close()

        if not table_exists:
            print("Workflows table does not exist. Skipping experiment_id migration.")
            return

        # Check current column type
        cursor = await db.execute("PRAGMA table_info(workflows)")
        columns = await cursor.fetchall()
        await cursor.close()

        experiment_id_type = None
        for column in columns:
            if column[1] == "experiment_id":
                experiment_id_type = column[2].upper()
                break

        if experiment_id_type is None:
            print("experiment_id column does not exist in workflows table.")
            return

        if experiment_id_type == "INTEGER":
            # print("experiment_id column is already INTEGER type. No migration needed.")
            return

        print(f"Migrating workflows.experiment_id from {experiment_id_type} to INTEGER...")

        # Create backup and migrate
        await db.execute("CREATE TABLE workflows_backup AS SELECT * FROM workflows")

        # Get original table schema
        cursor = await db.execute("PRAGMA table_info(workflows)")
        columns_info = await cursor.fetchall()
        await cursor.close()

        # Build new schema with INTEGER experiment_id and only expected columns
        expected_columns = {
            "id": "INTEGER PRIMARY KEY",
            "name": "TEXT NOT NULL",
            "description": "TEXT DEFAULT ''",
            "experiment_id": "INTEGER",
            "created_at": "TEXT DEFAULT (datetime('now'))",
            "updated_at": "TEXT DEFAULT (datetime('now'))",
            "config": "TEXT DEFAULT '{}'",
            "status": "TEXT DEFAULT 'pending'"
        }
        
        # Create schema with only expected columns
        column_definitions = []
        existing_columns = {col[1]: col for col in columns_info}
        
        for col_name, default_def in expected_columns.items():
            if col_name in existing_columns:
                # Use existing column definition but update experiment_id type
                col = existing_columns[col_name]
                col_name_db, col_type, not_null, default_val, primary_key = col[1], col[2], col[3], col[4], col[5]
                
                if col_name == "experiment_id":
                    col_type = "INTEGER"
                
                col_def = f"{col_name_db} {col_type}"
                if not_null:
                    col_def += " NOT NULL"
                if default_val is not None:
                    col_def += f" DEFAULT {default_val}"
                if primary_key:
                    col_def += " PRIMARY KEY"
                
                column_definitions.append(col_def)
            else:
                # Use default definition for missing columns
                column_definitions.append(f"{col_name} {default_def}")

        # Recreate table with correct schema
        await db.execute("DROP TABLE workflows")
        await db.execute(f"CREATE TABLE workflows ({', '.join(column_definitions)})")

        # Get column names from backup table to handle missing columns
        cursor = await db.execute("PRAGMA table_info(workflows_backup)")
        backup_columns_info = await cursor.fetchall()
        await cursor.close()
        
        backup_column_names = [col[1] for col in backup_columns_info]
        
        # Copy data with type conversion, handling missing columns
        select_fields = []
        select_fields.append("id")
        select_fields.append("name")
        
        # Handle description column - set to empty string if missing
        if "description" in backup_column_names:
            select_fields.append("description")
        else:
            select_fields.append("'' as description")
        
        # Handle experiment_id with type conversion
        if "experiment_id" in backup_column_names:
            select_fields.append("""
                CASE 
                    WHEN experiment_id IS NULL OR experiment_id = '' THEN NULL
                    WHEN experiment_id GLOB '*[!0-9]*' THEN NULL
                    ELSE CAST(experiment_id AS INTEGER)
                END as experiment_id""")
        else:
            select_fields.append("NULL as experiment_id")
        
        # Handle other columns
        for col_name in ["created_at", "updated_at", "config", "status"]:
            if col_name in backup_column_names:
                select_fields.append(col_name)
            else:
                # Set appropriate defaults for missing columns
                if col_name in ["created_at", "updated_at"]:
                    select_fields.append("datetime('now') as " + col_name)
                elif col_name == "config":
                    select_fields.append("'{}' as " + col_name)
                elif col_name == "status":
                    select_fields.append("'pending' as " + col_name)
        
        select_query = f"SELECT {', '.join(select_fields)} FROM workflows_backup"
        
        await db.execute(f"""
            INSERT INTO workflows 
            {select_query}
        """)

        # Clean up backup
        await db.execute("DROP TABLE workflows_backup")

        await db.commit()
        print("Successfully migrated workflows.experiment_id to INTEGER type")

    except Exception as e:
        print(f"Failed to migrate workflows.experiment_id: {e}")
        # Try to restore from backup
        try:
            await db.execute("DROP TABLE IF EXISTS workflows")
            await db.execute("ALTER TABLE workflows_backup RENAME TO workflows")
            await db.commit()
            print("Restored workflows table from backup")
        except Exception:
            pass
        raise e


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
    await migrate_workflows_experiment_id_to_integer()
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
    await db.execute("UPDATE job SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, job_id))
    await db.commit()
    if error_msg:
        job_data = json.dumps({"error_msg": str(error_msg)})
        await db.execute("UPDATE job SET job_data = ? WHERE id = ?", (job_data, job_id))
        await db.commit()
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
    async with async_session() as session:
        result = await session.execute(select(models.Experiment).where(models.Experiment.name == name))
        experiment = result.scalar_one_or_none()
        if experiment is None:
            return None
        return experiment.__dict__


async def experiment_delete(id):
    await db.execute("DELETE FROM experiment WHERE id = ?", (id,))
    await db.commit()
    return


async def experiment_update(id, config):
    await db.execute("UPDATE experiment SET config = ? WHERE id = ?", (config, id))
    await db.commit()
    return


async def experiment_update_config(id, key, value):
    value_json = json.dumps(value)
    async with async_session() as session:
        stmt = (
            update(models.Experiment)
            .where(models.Experiment.id == id)
            .values(config=text("json_set(config, :json_key, json(:value))"))
        )
        await session.execute(stmt, {"json_key": f"$.{key}", "value": value_json})
        await session.commit()
    return


async def experiment_save_prompt_template(id, template):
    async with async_session() as session:
        stmt = update(models.Experiment).where(models.Experiment.id == id)
        stmt = stmt.values(config=text("json_set(config, '$.prompt_template', json(:template))"))
        await session.execute(stmt, {"template": template})
        await session.commit()
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


async def workflows_get_by_id(workflow_id, experiment_id):
    # Query with both workflow_id and experiment_id to enforce relationship at DB level
    cursor = await db.execute(
        "SELECT * FROM workflows WHERE id = ? AND experiment_id = ? AND status != 'DELETED' ORDER BY created_at desc LIMIT 1",
        (workflow_id, experiment_id),
    )

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


async def workflow_delete_by_id(workflow_id: str, experiment_id):
    print("Deleting workflow: " + str(workflow_id))
    # Delete with experiment_id check to enforce relationship at DB level
    result = await db.execute(
        "UPDATE workflows SET status = 'DELETED', updated_at = CURRENT_TIMESTAMP WHERE id = ? AND experiment_id = ?",
        (workflow_id, experiment_id),
    )
    await db.commit()
    return result.rowcount > 0


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


async def workflow_update_config(workflow_id, config, experiment_id):
    # Update with experiment_id check to enforce relationship at DB level
    result = await db.execute(
        "UPDATE workflows SET config = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND experiment_id = ?",
        (config, workflow_id, experiment_id),
    )
    await db.commit()
    return result.rowcount > 0


async def workflow_update_name(workflow_id, name, experiment_id):
    # Update with experiment_id check to enforce relationship at DB level
    result = await db.execute(
        "UPDATE workflows SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND experiment_id = ?",
        (name, workflow_id, experiment_id),
    )
    await db.commit()
    return result.rowcount > 0


async def workflow_delete_all():
    await db.execute("DELETE FROM workflows")
    await db.commit()


async def workflow_runs_delete_all():
    await db.execute("DELETE FROM workflow_runs")
    await db.commit()


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
    cursor = await db.execute(
        "SELECT * FROM workflow_runs WHERE experiment_id = ? AND status != 'DELETED' ORDER BY created_at desc",
        (experiment_id,),
    )
    rows = await cursor.fetchall()
    desc = cursor.description
    column_names = [col[0] for col in desc]
    data = [dict(itertools.zip_longest(column_names, row)) for row in rows]
    await cursor.close()
    return data


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
