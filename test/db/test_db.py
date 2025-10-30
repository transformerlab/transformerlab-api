import json
import os
import asyncio

# Create test directories before setting environment variables
os.makedirs("./test/tmp/workspace", exist_ok=True)

os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp/workspace"


from transformerlab.db.db import (  # noqa: E402
    delete_plugin,
    get_plugins_of_type,
    get_training_template_by_name,
    get_plugins,
    get_plugin,
    save_plugin,
    config_get,
    config_set,
    get_training_template,
    get_training_templates,
    create_training_template,
    update_training_template,
    delete_training_template,
)

from transformerlab.services import experiment_service  # noqa: E402
from transformerlab.services.job_service import (  # noqa: E402
    job_create,
    job_update_status as service_job_update_status,
    job_update_status_sync as service_job_update_status_sync,
    job_update_sync as service_job_update_sync,
    job_mark_as_complete_if_running as service_job_mark_as_complete_if_running,
)
from transformerlab.db.workflows import (  # noqa: E402
    workflow_count_queued,
    workflow_count_running,
    workflow_create,
    workflow_delete_all,
    workflow_delete_by_id,
    workflow_delete_by_name,
    workflow_queue,
    workflow_run_get_all,
    workflow_run_get_by_id,
    workflow_run_update_status,
    workflow_runs_delete_all,
    workflow_runs_get_from_experiment,
    workflow_update_config,
    workflow_update_name,
    workflows_get_all,
    workflows_get_from_experiment,
    workflows_get_by_id,
)

import transformerlab.db.session as db  # noqa: E402


import pytest  # noqa: E402


@pytest.mark.asyncio
async def test_get_training_template_and_by_name_returns_none_for_missing():
    tmpl = await get_training_template(999999)
    assert tmpl is None
    tmpl = await get_training_template_by_name("does_not_exist")
    assert tmpl is None


@pytest.mark.asyncio
@pytest.mark.skip("skipping workflow tests")
async def test_workflows_get_by_id_returns_none_for_missing():
    workflow = await workflows_get_by_id(999999, 1)
    assert workflow is None


@pytest.mark.asyncio
@pytest.mark.skip("skipping workflow tests")
async def test_workflow_run_get_by_id_returns_none_for_missing():
    run = await workflow_run_get_by_id(999999)
    assert run is None


@pytest.mark.asyncio
async def test_get_plugin_returns_none_for_missing():
    plugin = await get_plugin("does_not_exist")
    assert plugin is None


@pytest.mark.asyncio
async def test_get_plugins_of_type_returns_list():
    plugins = await get_plugins()
    if plugins:
        plugin_type = plugins[0].get("type", "test_type")
    else:
        plugin_type = "test_type"
        await save_plugin("plugin_type_test", plugin_type)
    plugins_of_type = await get_plugins_of_type(plugin_type)
    assert isinstance(plugins_of_type, list)


@pytest.mark.asyncio
async def test_config_get_returns_none_for_missing():
    value = await config_get("missing_config_key")
    assert value is None


pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_training_template_crud():
    await create_training_template("tmpl", "desc", "type", "[]", "{}")
    templates = await get_training_templates()
    assert any(t.get("name") == "tmpl" for t in templates)
    tmpl_id = templates[0].get("id")
    await update_training_template(tmpl_id, "tmpl2", "desc2", "type2", "[]", "{}")
    tmpl = await get_training_template(tmpl_id)
    assert tmpl["name"] == "tmpl2"
    await delete_training_template(tmpl_id)
    tmpl = await get_training_template(tmpl_id)
    assert tmpl is None


pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session", autouse=True)
def manage_test_tmp_dir():
    yield
    # delete the database:
    db_path = os.path.join("./test/tmp", "llmlab.sqlite3")
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture(scope="module", autouse=True)
async def setup_db():
    await db.init()
    yield
    await db.close()


@pytest.fixture
async def test_experiment():
    # Converted from using DB to use service
    # Setup code to create test_experiment
    existing = experiment_service.experiment_get("test_experiment")
    if existing:
        experiment_service.experiment_delete(existing["id"])
    experiment_id = experiment_service.experiment_create("test_experiment", {})
    yield experiment_id
    # Teardown code to delete test_experiment
    experiment_service.experiment_delete(experiment_id)


# content of test_sample.py


def test_db_exists():
    global db
    assert db is not None


class TestPlugins:
    @pytest.mark.asyncio
    async def test_save_and_get_plugin(self):
        await save_plugin("test_plugin", "test_type")
        plugin = await get_plugin("test_plugin")
        assert plugin is not None
        assert plugin["name"] == "test_plugin"
        # now delete the plugin
        await delete_plugin("test_plugin")

    @pytest.mark.asyncio
    async def test_get_plugins(self):
        plugins = await get_plugins()
        assert isinstance(plugins, list)


class TestConfig:
    @pytest.mark.asyncio
    async def test_config_set_and_get(self):
        await config_set("test_key", "test_value")
        value = await config_get("test_key")
        assert value == "test_value"
        # now try to set the same key with a different value
        await config_set("test_key", "test_value2")
        value = await config_get("test_key")
        assert value == "test_value2"
        # now try to get a key that does not exist
        value = await config_get("test_key2_SHOULD_NOT_EXIST")
        assert value is None
        # now try to set a key with None value
        await config_set("test_key3", None)
        value = await config_get("test_key3")
        assert value is None
        # now try to set a key with empty string value
        await config_set("test_key4", "")
        value = await config_get("test_key4")
        assert value == ""


@pytest.mark.skip("skipping workflow tests")
class TestWorkflows:
    @pytest.mark.asyncio
    async def test_workflows_get_all(self):
        workflows = await workflows_get_all()
        assert isinstance(workflows, list)

    @pytest.mark.asyncio
    async def test_workflows_get_from_experiment(self, test_experiment):
        workflows = await workflows_get_from_experiment(test_experiment)
        assert isinstance(workflows, list)

    @pytest.mark.asyncio
    async def test_workflow_create_and_get_by_id(self, test_experiment):
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        workflow = await workflows_get_by_id(workflow_id, test_experiment)
        assert workflow is not None
        assert workflow["name"] == "test_workflow"

    @pytest.mark.asyncio
    async def test_workflow_update_name(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_update", "{}", test_experiment)
        await workflow_update_name(workflow_id, "updated_workflow", test_experiment)
        workflow = await workflows_get_by_id(workflow_id, test_experiment)
        assert workflow["name"] == "updated_workflow"

    @pytest.mark.asyncio
    async def test_workflow_update_config(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_config", "{}", test_experiment)
        await workflow_update_config(workflow_id, '{"key": "value"}', test_experiment)
        workflow = await workflows_get_by_id(workflow_id, test_experiment)
        workflow_config = json.loads(workflow["config"])  # should be a string containing JSON
        assert workflow_config == {"key": "value"}

    @pytest.mark.asyncio
    async def test_workflow_delete_by_id(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_delete", "{}", test_experiment)
        await workflow_delete_by_id(workflow_id, test_experiment)
        workflow = await workflows_get_by_id(workflow_id, test_experiment)
        assert workflow is None

    @pytest.mark.asyncio
    async def test_workflow_delete_by_name(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_delete_name", "{}", test_experiment)
        await workflow_delete_by_name("test_workflow_delete_name")  # noqa: F821
        workflow = await workflows_get_by_id(workflow_id, test_experiment)
        assert workflow is None  # Should return None since workflow is deleted

    @pytest.mark.skip(reason="Skipping as it causes db lock issues")
    async def test_workflow_queue(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_queue", "{}", test_experiment)
        result = await workflow_queue(workflow_id)
        assert result is True
        # Verify workflow exists
        workflow = await workflows_get_by_id(workflow_id, test_experiment)
        assert workflow is not None

    @pytest.mark.asyncio
    async def test_workflow_queue_nonexistent(self):
        # Test queueing a workflow that doesn't exist
        result = await workflow_queue(999999)  # Using a workflow ID that shouldn't exist
        assert result is False

    @pytest.mark.asyncio
    async def test_workflow_run_get_all(self):
        workflow_runs = await workflow_run_get_all()
        assert isinstance(workflow_runs, list)

    @pytest.mark.skip(reason="Skipping as it causes db lock issues")
    async def test_workflow_run_get_by_id(self):
        # Assuming a workflow run is created during testing
        workflow_run_id = 1  # Replace with actual logic to create a workflow run
        workflow_run = await workflow_run_get_by_id(workflow_run_id)
        assert workflow_run is not None

    @pytest.mark.skip(reason="Skipping as it causes db lock issues")
    async def test_workflow_run_update_status(self):
        # Assuming a workflow run is created during testing
        workflow_run_id = 1  # Replace with actual logic to create a workflow run
        await workflow_run_update_status(workflow_run_id, "COMPLETED")
        workflow_run = await workflow_run_get_by_id(workflow_run_id)
        assert workflow_run["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_workflow_count_running(self):
        count = await workflow_count_running()
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_workflow_count_queued(self):
        count = await workflow_count_queued()
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_workflow_runs_delete_all(self):
        await workflow_runs_delete_all()
        workflow_runs = await workflow_run_get_all()
        assert len(workflow_runs) == 0

    @pytest.mark.asyncio
    async def test_workflow_delete_all(self):
        await workflow_delete_all()
        workflows = await workflows_get_all()
        assert len(workflows) == 0

    @pytest.mark.asyncio
    async def test_workflow_trigger_on_job_completion(self, test_experiment):
        """Test that workflows are triggered when jobs complete with matching trigger types"""
        # Create a workflow with TRAIN trigger
        workflow_config = {
            "nodes": [{"type": "START", "id": "start", "name": "START", "out": []}],
            "triggers": ["TRAIN"],
        }
        workflow_id = await workflow_create("test_trigger_workflow", json.dumps(workflow_config), test_experiment)

        # Create a TRAIN job
        job_id = job_create("TRAIN", "RUNNING", test_experiment, "{}")

        # Complete the job - this should trigger the workflow
        await service_job_update_status(job_id, "COMPLETE", test_experiment)
        await asyncio.sleep(0.1)

        # Check that workflow was queued
        workflow_runs = await workflow_runs_get_from_experiment(test_experiment)
        assert len(workflow_runs) > 0
        assert workflow_runs[0]["workflow_id"] == workflow_id
        assert workflow_runs[0]["status"] in ["QUEUED", "RUNNING"]

    @pytest.mark.asyncio
    async def test_workflow_trigger_error_handling(self, test_experiment):
        """Test that workflows with malformed configs don't cause errors"""
        # Create a workflow with malformed JSON config
        await workflow_create("test_malformed_config", "invalid json", test_experiment)

        # Create a TRAIN job
        job_id = job_create("TRAIN", "RUNNING", test_experiment, "{}")

        # Complete the job - this should not crash even with malformed config
        await service_job_update_status(job_id, "COMPLETE", test_experiment)
        await asyncio.sleep(0.1)

        # Check that no workflow was triggered due to malformed config
        workflow_runs = await workflow_runs_get_from_experiment(test_experiment)
        assert len(workflow_runs) == 0

    @pytest.mark.asyncio
    async def test_sync_job_functions_trigger_workflows(self, test_experiment):
        """Test that sync job functions also trigger workflows when jobs complete"""
        # Create a workflow with TRAIN trigger
        workflow_config = {
            "nodes": [{"type": "START", "id": "start", "name": "START", "out": []}],
            "triggers": ["TRAIN"],
        }
        workflow_id = await workflow_create("test_sync_trigger", json.dumps(workflow_config), test_experiment)

        # Test job_update_status_sync
        job_id1 = job_create("TRAIN", "RUNNING", test_experiment, "{}")
        service_job_update_status_sync(job_id1, "COMPLETE", test_experiment)

        # Wait a moment for async trigger to complete
        import asyncio

        await asyncio.sleep(0.1)

        # Check that workflow was triggered
        workflow_runs = await workflow_runs_get_from_experiment(test_experiment)
        assert len(workflow_runs) > 0
        # Check if our workflow was triggered (it might not be the first one)
        triggered_workflow_ids = [run["workflow_id"] for run in workflow_runs]
        assert workflow_id in triggered_workflow_ids

        # Test job_update_sync
        job_id2 = job_create("TRAIN", "RUNNING", test_experiment, "{}")
        service_job_update_sync(job_id2, "COMPLETE", test_experiment)

        # Wait a moment for async trigger to complete
        await asyncio.sleep(0.1)

        # Check that workflow was triggered again
        workflow_runs = await workflow_runs_get_from_experiment(test_experiment)
        assert len(workflow_runs) >= 2

        # Test job_mark_as_complete_if_running
        job_id3 = job_create("TRAIN", "RUNNING", test_experiment, "{}")
        service_job_mark_as_complete_if_running(job_id3, test_experiment)

        # Wait a moment for async trigger to complete
        await asyncio.sleep(0.1)

        # Check that workflow was triggered again
        workflow_runs = await workflow_runs_get_from_experiment(test_experiment)
        assert len(workflow_runs) >= 3


# @pytest.mark.skip(reason="Skipping  because I can't get it to work")
# @pytest.mark.asyncio
# async def test_workflow_run_get_running(setup_db):
#     """Test the workflow_run_get_running function."""
#     # Create a workflow and workflow_run using db methods
#     workflow_id = await db.workflow_create("test_workflow", "{}", "test_experiment")
#     await db.workflow_queue(workflow_id)

#     # Sleep for 3 seconds, async:
#     await asyncio.sleep(3)

#     # Test the function
#     running_workflow = await db.workflow_run_get_running()

#     # Verify results
#     assert running_workflow is not None
#     assert running_workflow["status"] == "RUNNING"
#     assert running_workflow["workflow_name"] == "test_workflow"


# @pytest.mark.skip(reason="Skipping because I can't get it to work")
# @pytest.mark.asyncio
# async def test_training_jobs_get_all(setup_db):
#     """Test the training_jobs_get_all function."""
#     # Create a training template using db method
#     template_id = await db.create_training_template("test_template", "Test description", "fine-tuning", "[]", "{}")

#     # Create a job that references this training template
#     job_data = {"template_id": template_id, "description": "Test training job"}
#     job_id = await db.job_create("TRAIN", "QUEUED", json.dumps(job_data), "test_experiment")

#     # Test the function
#     training_jobs = await db.training_jobs_get_all()

#     # Verify results
#     assert len(training_jobs) > 0
#     found_job = False
#     for job in training_jobs:
#         if job["id"] == job_id:
#             found_job = True
#             assert job["type"] == "TRAIN"
#             assert job.get("status") == "QUEUED"
#             assert job["job_data"]["template_id"] == template_id
#             assert job["job_data"]["description"] == "Test training job"
#             assert "config" in job

#     assert found_job, "The created training job was not found in the results"


# @pytest.mark.skip(reason="Skipping test_workflow_run_get_running because I can't get it to work")
# @pytest.mark.asyncio
# async def test_workflow_run_get_queued(setup_db):
#     """Test the workflow_run_get_queued function."""
#     # Create a workflow and workflow_run using db methods
#     workflow_id = await db.workflow_create("queued_workflow", "{}", "test_experiment")

#     # Test the function
#     queued_workflow = await db.workflow_run_get_queued()

#     # Verify results
#     assert queued_workflow is not None
#     assert queued_workflow["status"] == "QUEUED"
#     assert queued_workflow["workflow_name"] == "queued_workflow"


# @pytest.mark.skip(reason="Skipping test_workflow_run_get_running because I can't get it to work")
# @pytest.mark.asyncio
# async def test_workflow_run_update_with_new_job(setup_db):
#     """Test the workflow_run_update_with_new_job function."""
#     # Create a workflow and workflow_run using db methods
#     workflow_id = await db.workflow_create("test_workflow", "{}", "test_experiment")  # noqa: F841

#     # New task and job IDs
#     current_task = '["task1"]'
#     current_job_id = "[1]"

#     # Test the function
#     await db.workflow_run_update_with_new_job(workflow_run_id, current_task, current_job_id)

#     # Verify results
#     updated_workflow_run = await db.workflow_run_get_by_id(workflow_run_id)
#     assert updated_workflow_run is not None
#     assert updated_workflow_run["current_tasks"] == current_task
#     assert updated_workflow_run["current_job_ids"] == current_job_id
#     assert json.loads(updated_workflow_run["job_ids"]) == [1]
#     assert json.loads(updated_workflow_run["node_ids"]) == ["task1"]
