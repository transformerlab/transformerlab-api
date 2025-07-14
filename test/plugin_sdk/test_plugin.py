import os
import tempfile
import pytest
from unittest import mock

from transformerlab.plugin_sdk.transformerlab import plugin

def test_register_process_single_and_multiple_pids():
    with tempfile.TemporaryDirectory() as temp_dir:
        pid_file = os.path.join(temp_dir, "worker.pid")
        with mock.patch.dict(os.environ, {"LLM_LAB_ROOT_PATH": temp_dir}):
            pids = plugin.register_process(12345)
            assert pids == [12345]
            with open(pid_file) as f:
                lines = f.read().splitlines()
            assert lines == ["12345"]

            pids = plugin.register_process([111, 222, 333])
            assert pids == [111, 222, 333]
            with open(pid_file) as f:
                lines = f.read().splitlines()
            assert lines == ["111", "222", "333"]

    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError):
            plugin.register_process(1) 