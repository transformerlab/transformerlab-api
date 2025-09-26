import json

from sqlalchemy import select, delete, text, update
from sqlalchemy.dialects.sqlite import insert  # Correct import for SQLite upsert

# from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession

from lab import Experiment

# Make sure SQLAlchemy is installed using pip install sqlalchemy[asyncio] as
# described here https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html

# FastAPI Users
from typing import AsyncGenerator
from fastapi import Depends
from fastapi_users.db import SQLAlchemyUserDatabase

from transformerlab.db.jobs import job_create
from transformerlab.db.workflows import (
    workflow_delete_by_id,
    workflows_get_from_experiment,
    workflow_runs_get_from_experiment,
    workflow_run_delete,
)
from transformerlab.shared.models import models
from transformerlab.shared.models.models import Config, Plugin
from transformerlab.db.utils import sqlalchemy_to_dict, sqlalchemy_list_to_dict

from transformerlab.db.session import async_session


###############################################
# Dependencies for FastAPI Users
# It wants specific types for session and DB.
###############################################


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, models.User)


###############
# MODELS MODEL
###############


async def model_local_list():
    async with async_session() as session:
        result = await session.execute(select(models.Model))
        models_list = result.scalars().all()
        data = []
        for model in models_list:
            row = sqlalchemy_to_dict(model)
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
        row = sqlalchemy_to_dict(model)
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
            row = sqlalchemy_to_dict(task)
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
            row = sqlalchemy_to_dict(task)
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
            row = sqlalchemy_to_dict(task)
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
        return sqlalchemy_to_dict(task)


async def tasks_get_by_experiment(experiment_id):
    """Get all tasks for a specific experiment"""
    async with async_session() as session:
        result = await session.execute(
            select(models.Task)
            .where(models.Task.experiment_id == experiment_id)
            .order_by(models.Task.created_at.desc())
        )
        tasks = result.scalars().all()
        return [sqlalchemy_to_dict(task) for task in tasks]


async def get_training_template(id):
    async with async_session() as session:
        result = await session.execute(select(models.TrainingTemplate).where(models.TrainingTemplate.id == id))
        template = result.scalar_one_or_none()
        if template is None:
            return None
        # Convert ORM object to dict
        return sqlalchemy_to_dict(template)


async def get_training_template_by_name(name):
    async with async_session() as session:
        result = await session.execute(select(models.TrainingTemplate).where(models.TrainingTemplate.name == name))
        template = result.scalar_one_or_none()
        if template is None:
            return None
        # Convert ORM object to dict
        return sqlalchemy_to_dict(template)


async def get_training_templates():
    async with async_session() as session:
        result = await session.execute(
            select(models.TrainingTemplate).order_by(models.TrainingTemplate.created_at.desc())
        )
        templates = result.scalars().all()
        # Convert ORM objects to dicts if needed
        return sqlalchemy_list_to_dict(templates)


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
            row = sqlalchemy_to_dict(job)
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
    job_id = await job_create(type="EXPORT", status="Started", experiment_id=experiment_id, job_data=job_data_json)
    return job_id


###################
# EXPERIMENTS MODEL
###################


async def experiment_create(name: str, config: dict) -> str:
    # test to see if config is a valid dict:
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")
    Experiment.create(name)
    exp = Experiment.get(name)
    exp._update_json_data_field("config", config)
    return name

# TODO: Make sure a name is always passed to this function
# or actually remove this and only keep experiment_get_by_name
async def experiment_get(id):
    if id is None or id == "undefined":
        return None
    return Experiment.get(id)


async def experiment_get_by_name(name):
    return Experiment.get(name)

# TODO: This needs to be expanded to delete all associated tasks, jobs, workflows, and itself
async def experiment_delete(id):
    # Delete associated tasks from DB
    tasks = await tasks_get_by_experiment(id)
    for task in tasks:
        await delete_task(task["id"])

    # Delete associated workflow runs from DB
    workflow_runs = await workflow_runs_get_from_experiment(id)
    for workflow_run in workflow_runs:
        await workflow_run_delete(workflow_run["id"])

    # Delete associated workflows from DB
    workflows = await workflows_get_from_experiment(id)
    for workflow in workflows:
        await workflow_delete_by_id(workflow["id"], id)

    # Delete jobs and experiment from filesystem
    exp = Experiment.get(id)
    if exp:
        exp.delete()
    return


async def experiment_update(id, config):
    # Ensure config is JSON string
    if not isinstance(config, str):
        config = json.dumps(config)
    exp = Experiment.get(id)
    if exp:
        exp._update_json_data_field("config", config)
    return


async def experiment_update_config(id, key, value):
    exp = Experiment.get(id)
    if exp:
        exp.update_config_field(key, value)
    return


async def experiment_save_prompt_template(id, template):
    exp = Experiment.get(id)
    if exp:
        exp.update_config_field("prompt_template", str(template))
    return


async def experiment_update_configs(id, updates: dict):
    exp = Experiment.get(id)
    if exp:
        exp._update_json_data_field("config", updates)
    return


###############
# PLUGINS MODEL
###############


async def get_plugins():
    async with async_session() as session:
        result = await session.execute(select(Plugin))
        plugins = result.scalars().all()
        # Convert ORM objects to dicts
        return sqlalchemy_list_to_dict(plugins)


async def get_plugins_of_type(type: str):
    async with async_session() as session:
        result = await session.execute(select(Plugin).where(Plugin.type == type))
        plugins = result.scalars().all()
        return sqlalchemy_list_to_dict(plugins)


async def get_plugin(slug: str):
    async with async_session() as session:
        result = await session.execute(select(Plugin).where(Plugin.name == slug))
        plugin = result.scalar_one_or_none()
        return sqlalchemy_to_dict(plugin) if plugin else None


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
