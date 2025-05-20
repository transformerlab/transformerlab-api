from fastapi.testclient import TestClient
from api import app
import json
from unittest.mock import patch, AsyncMock, MagicMock
import pytest


def test_export_jobs():
    with TestClient(app) as client:
        resp = client.get("/experiment/1/export/jobs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


def test_export_job():
    with TestClient(app) as client:
        resp = client.get("/experiment/1/export/job?jobId=job123")
        assert resp.status_code == 200


@patch('transformerlab.db.experiment_get')
@patch('transformerlab.db.export_job_create')
@patch('asyncio.create_subprocess_exec')
@patch('transformerlab.routers.experiment.export.get_output_file_name')
@patch('transformerlab.db.job_update_status')
@patch('os.makedirs')
@patch('os.path.join')
@patch('json.dump')
@patch('builtins.open')
def test_run_exporter_script_success(
    mock_open, mock_json_dump, mock_path_join, mock_makedirs, 
    mock_job_update, mock_get_output_file, mock_subprocess, 
    mock_job_create, mock_experiment_get
):
    # Setup mocks
    mock_experiment_get.return_value = {
        "config": json.dumps({
            "foundation": "huggingface/model1",
            "foundation_model_architecture": "pytorch"
        })
    }
    mock_job_create.return_value = "job123"
    mock_get_output_file.return_value = "/tmp/output_job123.txt"
    
    # Mock for file opening
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    # Mock subprocess
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (None, b"")
    mock_subprocess.return_value = mock_process
    
    # Mock path join to return predictable paths
    mock_path_join.side_effect = lambda *args: "/".join(args)
    
    with TestClient(app) as client:
        resp = client.get(
            "/experiment/1/export/run_exporter_script?plugin_name=test_plugin&plugin_architecture=GGUF&plugin_params=%7B%22q_bits%22%3A%224%22%7D"
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "success"
        assert result["job_id"] == "job123"
        
        # Verify that status was updated to COMPLETE
        mock_job_update.assert_called_with(job_id="job123", status="COMPLETE")


@patch('transformerlab.db.experiment_get')
def test_run_exporter_script_invalid_experiment(mock_experiment_get):
    # Setup mock to simulate experiment not found
    mock_experiment_get.return_value = None
    
    with TestClient(app) as client:
        resp = client.get(
            "/experiment/999/export/run_exporter_script?plugin_name=test_plugin&plugin_architecture=GGUF"
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["message"] == "Experiment 999 does not exist"


@patch('transformerlab.db.experiment_get')
@patch('transformerlab.db.export_job_create')
@patch('asyncio.create_subprocess_exec')
@patch('transformerlab.routers.experiment.export.get_output_file_name')
@patch('transformerlab.db.job_update_status')
@patch('os.makedirs')
def test_run_exporter_script_process_error(
    mock_makedirs, mock_job_update, mock_get_output_file, 
    mock_subprocess, mock_job_create, mock_experiment_get
):
    # Setup mocks
    mock_experiment_get.return_value = {
        "config": json.dumps({
            "foundation": "huggingface/model1",
            "foundation_model_architecture": "pytorch"
        })
    }
    mock_job_create.return_value = "job123"
    mock_get_output_file.return_value = "/tmp/output_job123.txt"
    
    # Mock subprocess with error
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (None, b"Error")
    mock_subprocess.return_value = mock_process
    
    with TestClient(app) as client:
        resp = client.get(
            "/experiment/1/export/run_exporter_script?plugin_name=test_plugin&plugin_architecture=GGUF"
        )
        assert resp.status_code == 200
        result = resp.json()
        assert "Export failed" in result["message"]
        
        # Verify that status was updated to FAILED
        mock_job_update.assert_called_with(job_id="job123", status="FAILED")


@patch('transformerlab.db.experiment_get')
@patch('transformerlab.db.export_job_create')
@patch('asyncio.create_subprocess_exec')
@patch('transformerlab.routers.experiment.export.get_output_file_name')
@patch('transformerlab.db.job_update_status')
@patch('os.makedirs')
def test_run_exporter_script_stderr_decode_error(
    mock_makedirs, mock_job_update, mock_get_output_file, 
    mock_subprocess, mock_job_create, mock_experiment_get
):
    # Setup mocks
    mock_experiment_get.return_value = {
        "config": json.dumps({
            "foundation": "huggingface/model1",
            "foundation_model_architecture": "pytorch"
        })
    }
    mock_job_create.return_value = "job123"
    mock_get_output_file.return_value = "/tmp/output_job123.txt"
    
    # Mock subprocess with stderr decode error
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (None, b"\xff\xfe")  # Invalid UTF-8 sequence
    mock_subprocess.return_value = mock_process
    
    with TestClient(app) as client:
        resp = client.get(
            "/experiment/1/export/run_exporter_script?plugin_name=test_plugin&plugin_architecture=GGUF"
        )
        assert resp.status_code == 200
        result = resp.json()
        assert "Export failed due to an internal error" in result["message"]

        # Verify that status was updated to FAILED
        mock_job_update.assert_called_with(job_id="job123", status="FAILED")


@patch('transformerlab.db.job_get')
@patch('transformerlab.routers.experiment.export.dirs.plugin_dir_by_name')
@patch('os.path.exists')
def test_get_output_file_name_with_custom_path(
    mock_exists, mock_plugin_dir, mock_job_get
):
    # Setup mocks
    mock_job_get.return_value = {
        "job_data": {
            "output_file_path": "/custom/path/output.txt",
            "plugin": "test_plugin"
        }
    }
    mock_plugin_dir.return_value = "/plugins/test_plugin"
    mock_exists.return_value = True
    
    from transformerlab.routers.experiment.export import get_output_file_name
    import asyncio
    
    result = asyncio.run(get_output_file_name("job123"))
    assert result == "/custom/path/output.txt"


@patch('transformerlab.db.job_get')
@patch('transformerlab.routers.experiment.export.dirs.plugin_dir_by_name')
@patch('os.path.exists')
def test_get_output_file_name_without_plugin(
    mock_exists, mock_plugin_dir, mock_job_get
):
    # Setup mocks
    mock_job_get.return_value = {
        "job_data": {}  # No plugin specified
    }
    
    from transformerlab.routers.experiment.export import get_output_file_name
    import asyncio
    
    with pytest.raises(ValueError, match="Plugin not found in job data"):
        asyncio.run(get_output_file_name("job123"))


@patch('transformerlab.db.job_get')
@patch('transformerlab.routers.experiment.export.dirs.plugin_dir_by_name')
@patch('os.path.exists')
def test_get_output_file_name_with_plugin(
    mock_exists, mock_plugin_dir, mock_job_get
):
    # Setup mocks
    mock_job_get.return_value = {
        "job_data": {
            "plugin": "test_plugin"
        }
    }
    mock_plugin_dir.return_value = "/plugins/test_plugin"
    mock_exists.return_value = True
    
    from transformerlab.routers.experiment.export import get_output_file_name
    import asyncio
    
    result = asyncio.run(get_output_file_name("job123"))
    assert result == "/plugins/test_plugin/output_job123.txt"


@patch('transformerlab.routers.experiment.export.get_output_file_name')
def test_watch_export_log_value_error(mock_get_output_file):
    mock_get_output_file.side_effect = ValueError("File not found for job") 
    
    with TestClient(app) as client:
        resp = client.get("/experiment/1/export/job/job123/stream_output")
        assert resp.status_code == 200
        response_text = resp.text.strip('"')
        assert response_text == "An internal error has occurred!"


@patch('transformerlab.routers.experiment.export.get_output_file_name')
def test_watch_export_log_other_error(mock_get_output_file):
    # Setup mock to raise a different ValueError
    mock_get_output_file.side_effect = ValueError("Some other error")
    
    with TestClient(app) as client:
        resp = client.get("/experiment/1/export/job/job123/stream_output")
        assert resp.status_code == 200
        response_text = resp.text.strip('"')
        assert response_text == "An internal error has occurred!" 