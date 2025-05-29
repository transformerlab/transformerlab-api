import os
import pytest
import json
from unittest.mock import AsyncMock, patch

# Set environment variables before importing modules
os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

import transformerlab.db as db
from transformerlab.shared.shared import run_job


pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="module", autouse=True)
async def setup_db():
    await db.init()
    yield
    await db.close()


@pytest.fixture
async def test_experiment():
    """Create a test experiment"""
    experiment_id = await db.experiment_create("test_experiment", "{}")
    yield experiment_id
    await db.experiment_delete(experiment_id)


class TestRunJobShared:
    """Test cases for the run_job function in shared.py"""

    @pytest.mark.asyncio
    async def test_eval_job_completes_when_not_stopped(self, test_experiment):
        """Test that EVAL job is marked as COMPLETE when stop flag is not set"""
        # Create a job
        job_id = await db.job_create("EVAL", "QUEUED", experiment_id=test_experiment)
        
        # Mock the evaluation script function
        with patch('transformerlab.shared.shared.run_evaluation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        mock_plugin_dir.return_value = "/fake/plugin/dir"
                        
                        job_config = {
                            "plugin": "test_plugin",
                            "evaluator": "test_evaluator"
                        }
                        
                        # Run the job
                        await run_job(str(job_id), job_config, "test_experiment", {"type": "EVAL"})
                        
                        # Verify the job was marked as COMPLETE
                        job_status = await db.job_get_status(job_id)
                        assert job_status == "COMPLETE"

    @pytest.mark.asyncio
    async def test_eval_job_not_completed_when_stopped(self, test_experiment):
        """Test that EVAL job is marked as STOPPED when stop flag is set"""
        # Create a job with stop flag set
        job_data = {"stop": True}
        job_id = await db.job_create("EVAL", "QUEUED", json.dumps(job_data), test_experiment)
        
        # Mock the evaluation script function
        with patch('transformerlab.shared.shared.run_evaluation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        mock_plugin_dir.return_value = "/fake/plugin/dir"
                        
                        job_config = {
                            "plugin": "test_plugin", 
                            "evaluator": "test_evaluator"
                        }
                        
                        # Set job to RUNNING first (as run_job does)
                        await db.job_update_status(job_id, "RUNNING")
                        
                        # Run the job
                        await run_job(str(job_id), job_config, "test_experiment", {"type": "EVAL"})
                        
                        # Verify the job was marked as STOPPED
                        job_status = await db.job_get_status(job_id)
                        assert job_status == "STOPPED"

    @pytest.mark.asyncio
    async def test_generate_job_completes_when_not_stopped(self, test_experiment):
        """Test that GENERATE job is marked as COMPLETE when stop flag is not set"""
        # Create a job
        job_id = await db.job_create("GENERATE", "QUEUED", experiment_id=test_experiment)
        
        # Mock the generation script function
        with patch('transformerlab.shared.shared.run_generation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        mock_plugin_dir.return_value = "/fake/plugin/dir"
                        
                        job_config = {
                            "plugin": "test_plugin",
                            "generator": "test_generator"
                        }
                        
                        # Run the job
                        await run_job(str(job_id), job_config, "test_experiment", {"type": "GENERATE"})
                        
                        # Verify the job was marked as COMPLETE
                        job_status = await db.job_get_status(job_id)
                        assert job_status == "COMPLETE"

    @pytest.mark.asyncio
    async def test_generate_job_not_completed_when_stopped(self, test_experiment):
        """Test that GENERATE job is marked as STOPPED when stop flag is set"""
        # Create a job with stop flag set
        job_data = {"stop": True}
        job_id = await db.job_create("GENERATE", "QUEUED", json.dumps(job_data), test_experiment)
        
        # Mock the generation script function
        with patch('transformerlab.shared.shared.run_generation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        mock_plugin_dir.return_value = "/fake/plugin/dir"
                        
                        job_config = {
                            "plugin": "test_plugin",
                            "generator": "test_generator"
                        }
                        
                        # Set job to RUNNING first (as run_job does)
                        await db.job_update_status(job_id, "RUNNING")
                        
                        # Run the job
                        await run_job(str(job_id), job_config, "test_experiment", {"type": "GENERATE"})
                        
                        # Verify the job was marked as STOPPED
                        job_status = await db.job_get_status(job_id)
                        assert job_status == "STOPPED"

    @pytest.mark.asyncio
    async def test_eval_job_handles_missing_job_data(self, test_experiment):
        """Test that EVAL job handles case where job_data is None or missing stop key"""
        # Create a job with empty job_data (no stop key)
        job_id = await db.job_create("EVAL", "QUEUED", "{}", test_experiment)
        
        # Mock the evaluation script function
        with patch('transformerlab.shared.shared.run_evaluation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        mock_plugin_dir.return_value = "/fake/plugin/dir"
                        
                        job_config = {
                            "plugin": "test_plugin",
                            "evaluator": "test_evaluator"
                        }
                        
                        # Run the job
                        await run_job(str(job_id), job_config, "test_experiment", {"type": "EVAL"})
                        
                        # Verify the job was marked as COMPLETE (since stop is not True)
                        job_status = await db.job_get_status(job_id)
                        assert job_status == "COMPLETE"

    @pytest.mark.asyncio
    async def test_generate_job_handles_missing_job_data(self, test_experiment):
        """Test that GENERATE job handles case where job_data is None or missing stop key"""
        # Create a job with empty job_data (no stop key)
        job_id = await db.job_create("GENERATE", "QUEUED", "{}", test_experiment)
        
        # Mock the generation script function
        with patch('transformerlab.shared.shared.run_generation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        mock_plugin_dir.return_value = "/fake/plugin/dir"
                        
                        job_config = {
                            "plugin": "test_plugin",
                            "generator": "test_generator"
                        }
                        
                        # Run the job
                        await run_job(str(job_id), job_config, "test_experiment", {"type": "GENERATE"})
                        
                        # Verify the job was marked as COMPLETE (since stop is not True)
                        job_status = await db.job_get_status(job_id)
                        assert job_status == "COMPLETE"

    @pytest.mark.asyncio
    async def test_eval_job_fails_when_job_data_is_none(self, test_experiment):
        """Test that EVAL job is marked as FAILED when job_data is None"""
        # Create a job with None job_data
        job_id = await db.job_create("EVAL", "QUEUED", experiment_id=test_experiment)
        
        # Mock the evaluation script function
        with patch('transformerlab.shared.shared.run_evaluation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        # Mock db.job_get to return a job with None job_data
                        with patch('transformerlab.shared.shared.db.job_get') as mock_job_get:
                            mock_job_get.return_value = {"job_data": None}
                            mock_plugin_dir.return_value = "/fake/plugin/dir"
                            
                            job_config = {
                                "plugin": "test_plugin",
                                "evaluator": "test_evaluator"
                            }
                            
                            # Run the job
                            result = await run_job(str(job_id), job_config, "test_experiment", {"type": "EVAL"})
                            
                            # Verify the result indicates error
                            assert result["status"] == "error"
                            assert "No job data found" in result["message"]

    @pytest.mark.asyncio
    async def test_generate_job_fails_when_job_data_is_none(self, test_experiment):
        """Test that GENERATE job is marked as FAILED when job_data is None"""
        # Create a job with None job_data
        job_id = await db.job_create("GENERATE", "QUEUED", experiment_id=test_experiment)
        
        # Mock the generation script function
        with patch('transformerlab.shared.shared.run_generation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        # Mock db.job_get to return a job with None job_data
                        with patch('transformerlab.shared.shared.db.job_get') as mock_job_get:
                            mock_job_get.return_value = {"job_data": None}
                            mock_plugin_dir.return_value = "/fake/plugin/dir"
                            
                            job_config = {
                                "plugin": "test_plugin",
                                "generator": "test_generator"
                            }
                            
                            # Run the job
                            result = await run_job(str(job_id), job_config, "test_experiment", {"type": "GENERATE"})
                            
                            # Verify the result indicates error
                            assert result["status"] == "error"
                            assert "No job data found" in result["message"]

    @pytest.mark.asyncio
    async def test_eval_job_completes_when_stop_key_missing(self, test_experiment):
        """Test that EVAL job is marked as COMPLETE when stop key is missing entirely"""
        # Create a job with job_data that has no "stop" key
        job_data = {"other_key": "value"}
        job_id = await db.job_create("EVAL", "QUEUED", json.dumps(job_data), test_experiment)
        
        # Mock the evaluation script function
        with patch('transformerlab.shared.shared.run_evaluation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        mock_plugin_dir.return_value = "/fake/plugin/dir"
                        
                        job_config = {
                            "plugin": "test_plugin",
                            "evaluator": "test_evaluator"
                        }
                        
                        # Run the job
                        await run_job(str(job_id), job_config, "test_experiment", {"type": "EVAL"})
                        
                        # Verify the job was marked as COMPLETE
                        job_status = await db.job_get_status(job_id)
                        assert job_status == "COMPLETE"

    @pytest.mark.asyncio
    async def test_generate_job_completes_when_stop_key_missing(self, test_experiment):
        """Test that GENERATE job is marked as COMPLETE when stop key is missing entirely"""
        # Create a job with job_data that has no "stop" key
        job_data = {"other_key": "value"}
        job_id = await db.job_create("GENERATE", "QUEUED", json.dumps(job_data), test_experiment)
        
        # Mock the generation script function
        with patch('transformerlab.shared.shared.run_generation_script', new_callable=AsyncMock):
            with patch('transformerlab.shared.shared.dirs.plugin_dir_by_name') as mock_plugin_dir:
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', create=True):
                        mock_plugin_dir.return_value = "/fake/plugin/dir"
                        
                        job_config = {
                            "plugin": "test_plugin",
                            "generator": "test_generator"
                        }
                        
                        # Run the job
                        await run_job(str(job_id), job_config, "test_experiment", {"type": "GENERATE"})
                        
                        # Verify the job was marked as COMPLETE
                        job_status = await db.job_get_status(job_id)
                        assert job_status == "COMPLETE"

    @pytest.mark.asyncio
    async def test_task_job_completes_successfully(self, test_experiment):
        """Test that TASK job is marked as COMPLETE and returns correct response"""
        # Create a job
        job_id = await db.job_create("TASK", "QUEUED", experiment_id=test_experiment)
        
        job_config = {
            "plugin": "test_plugin"
        }
        
        # Run the job
        result = await run_job(str(job_id), job_config, "test_experiment", {"type": "TASK"})
        
        # Verify the job was marked as COMPLETE
        job_status = await db.job_get_status(job_id)
        assert job_status == "COMPLETE"
        
        # Verify the return value matches expected format
        assert result["status"] == "complete"
        assert result["job_id"] == str(job_id)
        assert result["message"] == "Task job completed successfully"
