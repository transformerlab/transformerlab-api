"""
Test the event-based trigger processing functionality.
"""
import pytest
import json
import transformerlab.db as db
from transformerlab.jobs_trigger_processing import process_job_completion_triggers


@pytest.fixture
async def setup_test_db():
    """Set up a test database with necessary data."""
    await db.init()
    
    # Create a test experiment
    experiment_id = await db.experiment_create("test_experiment", "{}")
    
    # Create a test workflow 
    workflow_id = await db.workflow_create("test_workflow", "{}", experiment_id)
    
    # Update a trigger to use our test workflow
    triggers = await db.workflow_trigger_get_by_experiment_id(experiment_id)
    if triggers:
        trigger = triggers[0]  # Get the first trigger
        await db.workflow_trigger_update(
            trigger["id"], 
            config={"workflow_ids": [workflow_id]}, 
            is_enabled=True
        )
        return experiment_id, workflow_id, trigger["id"]
    
    # If no triggers exist, skip the test
    pytest.skip("No triggers found in test database")


@pytest.mark.asyncio
async def test_event_based_trigger_processing(setup_test_db):
    """Test that completing a job triggers workflows via event-based processing."""
    experiment_id, workflow_id, trigger_id = await setup_test_db
    
    # Create a training job
    job_id = await db.job_create(
        type="TRAIN",
        status="RUNNING", 
        job_data="{}",
        experiment_id=experiment_id
    )
    
    # Verify no workflow runs exist yet
    initial_runs = await db.workflow_run_get_all()
    initial_count = len(initial_runs)
    
    # Mark the job as complete (this should trigger the event-based processing)
    await db.job_update_status(job_id, "COMPLETE")
    
    # Check that a workflow run was created
    final_runs = await db.workflow_run_get_all()
    final_count = len(final_runs)
    
    assert final_count > initial_count, "Event-based trigger should have created a workflow run"
    
    # Verify the job was marked as having triggers processed
    updated_job = await db.job_get(job_id)
    job_data = updated_job.get("job_data", {})
    assert job_data.get("triggers_processed", False), "Job should be marked as having triggers processed"


@pytest.mark.asyncio
async def test_trigger_not_processed_twice(setup_test_db):
    """Test that triggers are not processed multiple times for the same job."""
    experiment_id, workflow_id, trigger_id = await setup_test_db
    
    # Create a training job
    job_id = await db.job_create(
        type="TRAIN",
        status="RUNNING",
        job_data="{}",
        experiment_id=experiment_id
    )
    
    # Get initial workflow run count
    initial_runs = await db.workflow_run_get_all()
    initial_count = len(initial_runs)
    
    # Mark the job as complete (first time)
    await db.job_update_status(job_id, "COMPLETE")
    
    # Check workflow run count after first completion
    after_first_runs = await db.workflow_run_get_all()
    after_first_count = len(after_first_runs)
    
    # Call the trigger processing again (simulating a second completion)
    await process_job_completion_triggers(job_id)
    
    # Check workflow run count after second processing
    after_second_runs = await db.workflow_run_get_all()
    after_second_count = len(after_second_runs)
    
    assert after_first_count == after_second_count, "Triggers should not be processed twice for the same job"
    assert after_first_count > initial_count, "At least one workflow run should have been created"


@pytest.mark.asyncio  
async def test_disabled_trigger_not_processed(setup_test_db):
    """Test that disabled triggers are not processed."""
    experiment_id, workflow_id, trigger_id = await setup_test_db
    
    # Disable the trigger
    await db.workflow_trigger_update(trigger_id, is_enabled=False)
    
    # Create a training job
    job_id = await db.job_create(
        type="TRAIN",
        status="RUNNING",
        job_data="{}",
        experiment_id=experiment_id
    )
    
    # Get initial workflow run count
    initial_runs = await db.workflow_run_get_all()
    initial_count = len(initial_runs)
    
    # Mark the job as complete
    await db.job_update_status(job_id, "COMPLETE")
    
    # Check that no new workflow runs were created
    final_runs = await db.workflow_run_get_all()
    final_count = len(final_runs)
    
    assert final_count == initial_count, "Disabled triggers should not create workflow runs" 