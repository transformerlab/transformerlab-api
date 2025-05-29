import json
import os
import pytest
import uuid
import unittest.mock

os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from transformerlab import db
from transformerlab.jobs_trigger_processing import process_job_completion_triggers
from transformerlab.db import (
    experiment_create,
    workflow_create,
    workflow_update_trigger_configs,
    workflow_delete_all,
    job_create,
    PREDEFINED_TRIGGER_BLUEPRINTS
)


@pytest.fixture(scope="module", autouse=True)
async def setup_db():
    """Initialize database for testing."""
    await db.init()
    try:
        yield
    finally:
        await db.close()


@pytest.fixture
async def test_experiment():
    """Create a test experiment with unique name."""
    unique_name = f"test_trigger_exp_{uuid.uuid4().hex[:8]}"
    exp_id = await experiment_create(unique_name, "{}")
    yield exp_id


@pytest.fixture
async def clean_workflows():
    """Clean up workflows before and after tests."""
    await workflow_delete_all()
    yield
    await workflow_delete_all()


class TestProcessJobCompletionTriggersErrorHandling:
    """Test error handling in process_job_completion_triggers function."""

    @pytest.mark.asyncio
    async def test_process_job_completion_triggers_job_not_found(self):
        """Test trigger processing when job doesn't exist."""
        # Process triggers for non-existent job
        await process_job_completion_triggers(999999)
        # Should complete without error (logged as warning)

    @pytest.mark.asyncio
    async def test_process_job_completion_triggers_job_not_complete(self, test_experiment):
        """Test trigger processing when job is not complete."""
        # Create a job that's not complete
        job_id = await job_create("TRAIN", "RUNNING", "{}", test_experiment)
        
        # Process triggers
        await process_job_completion_triggers(job_id)
        
        # Should complete without processing triggers

    @pytest.mark.asyncio
    async def test_process_job_completion_triggers_already_processed(self, test_experiment):
        """Test trigger processing when triggers already processed."""
        # Create and complete a job
        job_data = {"triggers_processed": True}
        job_id = await job_create("TRAIN", "COMPLETE", json.dumps(job_data), test_experiment)
        
        # Process triggers
        await process_job_completion_triggers(job_id)
        
        # Should complete without processing triggers again

    @pytest.mark.asyncio
    async def test_process_job_completion_triggers_no_experiment_id(self):
        """Test trigger processing when job has no experiment_id."""
        # Create job without experiment_id
        job_id = await job_create("TRAIN", "COMPLETE", "{}", None)
        
        # Process triggers
        await process_job_completion_triggers(job_id)
        
        # Should mark as processed but not queue workflows

    @pytest.mark.asyncio
    async def test_process_job_completion_triggers_no_matching_workflows(self, test_experiment, clean_workflows):
        """Test trigger processing when no workflows match."""
        # Create job
        job_id = await job_create("TRAIN", "COMPLETE", "{}", test_experiment)
        
        # Process triggers (no workflows exist)
        await process_job_completion_triggers(job_id)
        
        # Should complete without error

    @pytest.mark.asyncio
    async def test_process_job_completion_triggers_workflow_queue_error(self, test_experiment, clean_workflows):
        """Test trigger processing when workflow queueing fails."""
        # Create workflow with TRAIN trigger enabled
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": bp["trigger_type"] == "TRAIN"
            })
        await workflow_update_trigger_configs(workflow_id, new_configs)
        
        # Create job
        job_id = await job_create("TRAIN", "COMPLETE", "{}", test_experiment)
        
        # Mock workflow_queue to raise exception
        with unittest.mock.patch('transformerlab.db.workflow_queue') as mock_queue:
            mock_queue.side_effect = RuntimeError("Database error")
            
            # Process triggers - should handle the error gracefully
            await process_job_completion_triggers(job_id)

    @pytest.mark.asyncio
    async def test_process_job_completion_triggers_general_exception(self, test_experiment):
        """Test general exception handling in trigger processing."""
        # Create job
        job_id = await job_create("TRAIN", "COMPLETE", "{}", test_experiment)
        
        # Mock job_get to raise exception
        with unittest.mock.patch('transformerlab.db.job_get') as mock_get:
            mock_get.side_effect = RuntimeError("Database connection failed")
            
            # Process triggers - should handle the error gracefully
            await process_job_completion_triggers(job_id)

    @pytest.mark.asyncio
    async def test_process_job_completion_triggers_successful_processing(self, test_experiment, clean_workflows):
        """Test successful trigger processing with workflow queueing."""
        # Create workflow with TRAIN trigger enabled
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": bp["trigger_type"] == "TRAIN"
            })
        await workflow_update_trigger_configs(workflow_id, new_configs)
        
        # Create job
        job_id = await job_create("TRAIN", "COMPLETE", "{}", test_experiment)
        
        # Process triggers
        await process_job_completion_triggers(job_id)
        
        # Job should be marked as processed
        job = await db.job_get(job_id)
        job_data = job.get("job_data", {})
        if isinstance(job_data, str):
            job_data = json.loads(job_data)
        
        assert job_data.get("triggers_processed") is True 