import os
import aiosqlite
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from transformerlab.db.constants import DATABASE_FILE_NAME, DATABASE_URL
from transformerlab.shared.models import models


# --- SQLAlchemy Async Engine ---
# This engine is the core entry point to the database.
# It is created once and can be imported elsewhere.
async_engine = create_async_engine(DATABASE_URL, echo=False)

# --- SQLAlchemy Async Session Factory ---
# This is a factory that creates new AsyncSession objects.
# You will import this into other files to get a session.
async_session = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


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


async def job_cancel_in_progress_jobs():
    async with async_session() as session:
        await session.execute(update(models.Job).where(models.Job.status == "RUNNING").values(status="CANCELLED"))
        await session.commit()
    return


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


async def close():
    await db.close()
    await async_engine.dispose()
    print("✅ Database closed")
    return
