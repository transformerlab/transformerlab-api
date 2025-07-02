###############
# DATASETS MODEL
###############

import json
from sqlalchemy import select, insert, delete, text
from transformerlab.db.session import async_session
from transformerlab.shared.models import models


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
