import json
import os

from lab import dirs as lab_dirs
from lab.dataset import Dataset as dataset_service

async def migrate_datasets_table_to_filesystem():
    """
    One-time migration: copy rows from the legacy dataset DB table into the filesystem
    registry via transformerlab-sdk, then drop the table.
    Safe to run multiple times; it will no-op if table is missing or empty.
    """
    try:
        # Late import to avoid hard dependency during tests without DB
        from transformerlab.db.session import async_session
        from sqlalchemy import text as sqlalchemy_text

        # Read existing rows
        rows = []
        try:
            # First check if the table exists
            async with async_session() as session:
                result = await session.execute(
                    sqlalchemy_text("SELECT name FROM sqlite_master WHERE type='table' AND name='dataset'")
                )
                exists = result.fetchone() is not None
            if not exists:
                return
            # Migrated db.dataset.get_datasets() to run here as we are deleting that code
            rows = []
            async with async_session() as session:
                result = await session.execute(sqlalchemy_text("SELECT * FROM dataset"))
                datasets = result.mappings().all()
                dict_rows = [dict(dataset) for dataset in datasets]
                for row in dict_rows:
                    if "json_data" in row and row["json_data"]:
                        if isinstance(row["json_data"], str):
                            row["json_data"] = json.loads(row["json_data"])
                    rows.append(row)
        except Exception as e:
            print(f"Failed to read datasets for migration: {e}")
            rows = []

        migrated = 0
        for row in rows:
            dataset_id = str(row.get("dataset_id")) if row.get("dataset_id") is not None else None
            if not dataset_id:
                continue
            location = row.get("location", "local")
            description = row.get("description", "")
            size = int(row.get("size", -1)) if row.get("size") is not None else -1
            json_data = row.get("json_data", {})
            if isinstance(json_data, str):
                try:
                    json_data = json.loads(json_data)
                except Exception:
                    json_data = {}

            try:
                try:
                    ds = dataset_service.get(dataset_id)
                except FileNotFoundError:
                    ds = dataset_service.create(dataset_id)
                ds.set_metadata(
                    location=location,
                    description=description,
                    size=size,
                    json_data=json_data,
                )
                migrated += 1
            except Exception:
                # Best-effort migration; continue
                continue

        # Drop the legacy table if present
        try:
            async with async_session() as session:
                await session.execute(sqlalchemy_text("ALTER TABLE dataset RENAME TO zzz_archived_dataset"))
                await session.commit()
        except Exception:
            pass

        if migrated:
            print(f"Datasets migration completed: {migrated} entries migrated to filesystem store.")
    except Exception as e:
        # Do not block startup on migration issues
        print(f"Datasets migration skipped due to error: {e}")


async def migrate_models_table_to_filesystem():
    """
    One-time migration: copy rows from the legacy model DB table into the filesystem
    registry via transformerlab-sdk, then drop the table.
    Safe to run multiple times; it will no-op if table is missing or empty.
    """
    try:
        # Late import to avoid hard dependency during tests without DB
        from transformerlab.db.session import async_session
        from sqlalchemy import text as sqlalchemy_text
        from lab.model import Model as model_service

        # Read existing rows
        rows = []
        try:
            # First check if the table exists
            async with async_session() as session:
                result = await session.execute(sqlalchemy_text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='model'"
                ))
                exists = result.fetchone() is not None
            if not exists:
                rows = []
            else:
                # Inline the legacy models query here to avoid relying on removed DB helpers
                async with async_session() as session:
                    result = await session.execute(sqlalchemy_text("SELECT * FROM model"))
                    models_rows = result.mappings().all()
                    dict_rows = [dict(model) for model in models_rows]
                    rows = []
                    for row in dict_rows:
                        if "json_data" in row and row["json_data"]:
                            if isinstance(row["json_data"], str):
                                try:
                                    row["json_data"] = json.loads(row["json_data"])
                                except Exception:
                                    # If malformed, keep as original string
                                    pass
                        rows.append(row)
        except Exception as e:
            print(f"Error getting models: {e}")
            rows = []
        migrated = 0
        for row in rows:
            model_id = str(row.get("model_id")) if row.get("model_id") is not None else None
            print(f"Migrating model: {model_id}")
            if not model_id:
                continue
            name = row.get("name", model_id)
            json_data = row.get("json_data", {})
            if isinstance(json_data, str):
                try:
                    json_data = json.loads(json_data)
                except Exception:
                    json_data = {}

            try:
                try:
                    model = model_service.get(model_id)
                except FileNotFoundError:
                    model = model_service.create(model_id)
                model.set_metadata(
                    model_id=model_id,
                    name=name,
                    json_data=json_data,
                )
                migrated += 1
            except Exception as e:
                print(f"Error migrating model: {e}")
                # Best-effort migration; continue
                continue

        # Drop the legacy table if present
        try:
            async with async_session() as session:
                await session.execute(sqlalchemy_text("DROP TABLE IF EXISTS model"))
                await session.commit()
        except Exception as e:
            print(f"Error dropping models table: {e}")
            pass

        # Additionally, scan filesystem models directory for legacy models that
        # have info.json but are missing index.json, and create SDK metadata.
        try:
            models_dir = lab_dirs.MODELS_DIR
            if os.path.isdir(models_dir):
                fs_migrated = 0
                for entry in os.listdir(models_dir):
                    entry_path = os.path.join(models_dir, entry)
                    if not os.path.isdir(entry_path):
                        continue
                    info_path = os.path.join(entry_path, "info.json")
                    index_path = os.path.join(entry_path, "index.json")
                    if os.path.isfile(info_path) and not os.path.isfile(index_path):
                        model_id = entry
                        # Load legacy info.json as best-effort metadata
                        name = model_id
                        json_data = {}
                        try:
                            with open(info_path, "r") as f:
                                info_obj = json.load(f)
                                if isinstance(info_obj, dict):
                                    name = info_obj.get("name", name)
                                    json_data = info_obj
                        except Exception:
                            # Skip malformed info.json but continue migration
                            pass

                        try:
                            try:
                                model = model_service.get(model_id)
                            except FileNotFoundError:
                                model = model_service.create(model_id)
                            model.set_metadata(
                                model_id=model_id,
                                name=name,
                                json_data=json_data,
                            )
                            fs_migrated += 1
                        except Exception as e:
                            print(f"Error migrating local model: {e}")
                            # Best-effort; continue scanning others
                            continue

                if fs_migrated:
                    print(
                        f"Filesystem models migration: {fs_migrated} entries created from info.json (no index.json)."
                    )
        except Exception as e:
            # Do not block startup on filesystem migration issues
            print(f"Error migrating models: {e}")
            pass

        if migrated:
            print(f"Models migration completed: {migrated} entries migrated to filesystem store.")
    except Exception as e:
        # Do not block startup on migration issues
        print(f"Models migration skipped due to error: {e}")