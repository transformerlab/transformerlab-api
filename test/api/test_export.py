from fastapi.testclient import TestClient
from api import app
import json
from unittest.mock import patch, AsyncMock, MagicMock


def test_export_jobs():
    with TestClient(app) as client:
        resp = client.get("/export/jobs?id=1")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


def test_export_job():
    with TestClient(app) as client:
        resp = client.get("/export/job?id=1&jobId=job123")
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
            "/export/run_exporter_script?id=1&plugin_name=test_plugin&plugin_architecture=GGUF&plugin_params=%7B%22q_bits%22%3A%224%22%7D"
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["message"] == "success"
        assert result["job_id"] == "job123"
        
        # Verify that status was updated to COMPLETE
        mock_job_update.assert_called_with(job_id="job123", status="COMPLETE")


@patch('transformerlab.db.experiment_get')
def test_run_exporter_script_invalid_experiment(mock_experiment_get):
    # Setup mock to simulate experiment not found
    mock_experiment_get.return_value = None
    
    with TestClient(app) as client:
        resp = client.get(
            "/export/run_exporter_script?id=999&plugin_name=test_plugin&plugin_architecture=GGUF"
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
            "/export/run_exporter_script?id=1&plugin_name=test_plugin&plugin_architecture=GGUF"
        )
        assert resp.status_code == 200
        result = resp.json()
        assert "Failed to export model" in result["message"]
        
        # Verify that status was updated to FAILED
        mock_job_update.assert_called_with(job_id="job123", status="FAILED") 