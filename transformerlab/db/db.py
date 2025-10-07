import json
import os
import shutil

from sqlalchemy import select, delete, text, update
from sqlalchemy.dialects.sqlite import insert  # Correct import for SQLite upsert

# from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession


# Make sure SQLAlchemy is installed using pip install sqlalchemy[asyncio] as
# described here https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html

from typing import AsyncGenerator

from transformerlab.db.jobs import job_create
from transformerlab.shared.models import models
from transformerlab.shared.models.models import Config, Plugin
from transformerlab.db.utils import sqlalchemy_to_dict, sqlalchemy_list_to_dict

from transformerlab.db.session import async_session
from lab import Experiment, dirs as lab_dirs


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


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
