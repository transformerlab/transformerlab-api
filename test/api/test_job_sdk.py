"""
Covers job_sdk.py line 186 – the call to job_create_sync inside start_training().
"""
import json
import os
import unittest.mock as mock
import uuid
import tempfile
import shutil
import atexit

import pytest
import transformerlab.db as db
import transformerlab.routers.job_sdk as job_sdk
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

# Create a unique test directory using absolute paths to prevent contamination
TEST_BASE_DIR = os.path.abspath(os.path.join(tempfile.gettempdir(), f"transformerlab_job_sdk_test_{uuid.uuid4().hex[:8]}"))
os.makedirs(TEST_BASE_DIR, exist_ok=True)

# Set environment variables BEFORE any transformerlab imports
os.environ["TFL_HOME_DIR"] = TEST_BASE_DIR
os.environ["TFL_WORKSPACE_DIR"] = TEST_BASE_DIR
os.environ["_TFL_WORKSPACE_DIR"] = TEST_BASE_DIR

# Patch the database path to ensure complete isolation
TEST_DB_PATH = os.path.join(TEST_BASE_DIR, "test_llmlab.sqlite3")

# Patch database module
db.DATABASE_FILE_NAME = TEST_DB_PATH
db.DATABASE_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH}"

# Recreate the async engine with the new path
db.async_engine = create_async_engine(f"sqlite+aiosqlite:///{TEST_DB_PATH}", echo=False)
db.async_session = sessionmaker(db.async_engine, expire_on_commit=False, class_=AsyncSession)

# Register cleanup function to run at exit
def cleanup_test_dir():
    if os.path.exists(TEST_BASE_DIR):
        shutil.rmtree(TEST_BASE_DIR, ignore_errors=True)

atexit.register(cleanup_test_dir)

# ---------- minimal trainer stub ---------- #
class _DummyJob:
    def add_to_job_data(self, *_):
        pass

    def update_progress(self, *_):
        pass

    def get_job_data(self):
        return {}

    def get_status(self):
        return "RUNNING"

    def get_progress(self):
        return 0


class _DummyTrainer:
    def __init__(self):
        self.params = mock.MagicMock()
        self.params.reported_metrics = []
        self._job = _DummyJob()

    @property
    def job(self):
        return self._job

    def setup_train_logging(self, *_, **__):
        pass

    def log_metric(self, *_, **__):
        pass


def _trainer_factory():
    return _DummyTrainer()


@pytest.mark.asyncio
async def test_start_training_invokes_job_create_sync(monkeypatch):
    # Capture every function XMLRPCRouter registers, so we can call start_training directly
    registered = {}

    def _capture_register(self, func, name=None):
        registered[name or func.__name__] = func
        # We do NOT need to call the original register_function

    monkeypatch.setattr(
        "transformerlab.routers.job_sdk.XMLRPCRouter.register_function",
        _capture_register,
        raising=True,
    )

    # Patch out dependencies used by start_training
    mocked_job_create_sync = mock.MagicMock(return_value=42)
    monkeypatch.setattr("transformerlab.routers.job_sdk.job_create_sync", mocked_job_create_sync)

    class _FakeCore:
        WORKSPACE_DIR = "/tmp"

        @staticmethod
        def get_experiment_id_from_name(_):
            return 7

    monkeypatch.setattr("transformerlab.routers.job_sdk.tlab_core", _FakeCore)

    # Build the router – this populates `registered`
    job_sdk.get_trainer_xmlrpc_router(prefix="/test", trainer_factory=_trainer_factory)

    # Retrieve and call the captured start_training()
    start_training = registered["start_training"]
    result = start_training(json.dumps({"experiment_name": "alpha", "template_name": "demo"}))

    # Assertions
    mocked_job_create_sync.assert_called_once_with(
        "TRAIN", "RUNNING", job_data=mock.ANY, experiment_id=7
    )
    assert result == {"status": "started", "job_id": 42}
