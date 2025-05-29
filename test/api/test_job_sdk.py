"""
Covers job_sdk.py line 186 – the call to job_create_sync inside start_training().
"""
import json
import os
import unittest.mock as mock

import pytest

# Set up environment variables before importing
os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"
os.environ["_TFL_WORKSPACE_DIR"] = "./test/tmp"

import transformerlab.routers.job_sdk as job_sdk


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
