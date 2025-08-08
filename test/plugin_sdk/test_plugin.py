import os
import tempfile
import pytest
import json

from transformerlab.plugin_sdk.transformerlab import plugin


def test_register_process_single_and_multiple_pids():
    original_env = os.environ.get("LLM_LAB_ROOT_PATH")

    with tempfile.TemporaryDirectory() as temp_dir:
        pid_file = os.path.join(temp_dir, "worker.pid")
        try:
            os.environ["LLM_LAB_ROOT_PATH"] = temp_dir

            # Test single PID without job_id
            pids = plugin.register_process(12345)
            assert pids == [12345]
            with open(pid_file) as f:
                data = json.load(f)
            expected_data = {"processes": [{"pid": 12345}]}
            assert data == expected_data

            # Test multiple PIDs without job_id (should overwrite previous)
            pids = plugin.register_process([111, 222, 333])
            assert pids == [111, 222, 333]
            with open(pid_file) as f:
                data = json.load(f)
            expected_data = {
                "processes": [
                    {"pid": 12345},  # Previous entry preserved
                    {"pid": 111},
                    {"pid": 222},
                    {"pid": 333},
                ]
            }
            assert data == expected_data

            # Test single PID with job_id
            pids = plugin.register_process(555, job_id="test-job-123")
            assert pids == [555]
            with open(pid_file) as f:
                data = json.load(f)
            # Should have all previous processes plus the new one with job_id
            assert len(data["processes"]) == 5
            assert data["processes"][-1] == {"pid": 555, "job_id": "test-job-123"}

        finally:
            if original_env is not None:
                os.environ["LLM_LAB_ROOT_PATH"] = original_env
            else:
                os.environ.pop("LLM_LAB_ROOT_PATH", None)

    # Test error when environment variable is not set
    original_env = os.environ.pop("LLM_LAB_ROOT_PATH", None)
    try:
        with pytest.raises(EnvironmentError):
            plugin.register_process(1)
    finally:
        if original_env is not None:
            os.environ["LLM_LAB_ROOT_PATH"] = original_env
