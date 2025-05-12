import os


os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from transformerlab.db import (
    get_dataset,
    get_datasets,
    create_local_dataset,
    delete_dataset,
    model_local_list,
    model_local_count,
    model_local_create,
    model_local_get,
    model_local_delete,
    job_create,
    jobs_get_all,
    job_get_status,
    job_get,
    job_update_status,
    experiment_get_all,
    experiment_create,
    experiment_get,
    experiment_delete,
    get_plugins,
    get_plugin,
    save_plugin,
    config_get,
    config_set,
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
    workflow_update_config,
    workflow_update_name,
    workflows_get_all,
    workflows_get_by_id,
    workflows_get_from_experiment,
)
import pytest
import transformerlab.db as db

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
async def test_dataset():
    # Setup code to create test_dataset
    dataset = await db.create_local_dataset("test_dataset")
    yield dataset
    # Teardown code to delete test_dataset
    await db.delete_dataset("test_dataset")


@pytest.fixture
async def test_experiment():
    # Setup code to create test_experiment
    experiment_id = await db.experiment_create("test_experiment", "{}")
    yield experiment_id
    # Teardown code to delete test_experiment
    await db.experiment_delete(experiment_id)


# content of test_sample.py


def test_db_exists():
    global db
    assert db is not None


class TestDatasets:
    @pytest.mark.asyncio
    async def test_create_and_get_dataset(self, test_dataset):
        dataset = await get_dataset("test_dataset")
        assert dataset is not None
        assert dataset["dataset_id"] == "test_dataset"

    @pytest.mark.asyncio
    async def test_get_datasets(self):
        datasets = await get_datasets()
        assert isinstance(datasets, list)

    @pytest.mark.asyncio
    async def test_delete_dataset(self):
        await create_local_dataset("test_dataset_delete")
        await delete_dataset("test_dataset_delete")
        dataset = await get_dataset("test_dataset_delete")
        assert dataset is None


class TestModels:
    @pytest.mark.asyncio
    async def test_create_and_get_model(self):
        await model_local_create("test_model", "Test Model", {})
        model = await model_local_get("test_model")
        assert model is not None
        assert model["model_id"] == "test_model"

    @pytest.mark.asyncio
    async def test_model_local_list(self):
        models = await model_local_list()
        assert isinstance(models, list)

    @pytest.mark.asyncio
    async def test_model_local_count(self):
        count = await model_local_count()
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_model_local_delete(self):
        await model_local_create("test_model_delete", "Test Model Delete", {})
        await model_local_delete("test_model_delete")
        model = await model_local_get("test_model_delete")
        assert model is None


class TestJobs:
    @pytest.mark.asyncio
    async def test_create_and_get_job(self):
        job_id = await job_create("TRAIN", "QUEUED")
        job = await job_get(job_id)
        assert job is not None
        assert job["id"] == job_id

    @pytest.mark.asyncio
    async def test_jobs_get_all(self):
        jobs = await jobs_get_all()
        assert isinstance(jobs, list)

    @pytest.mark.asyncio
    async def test_job_update_status(self):
        job_id = await job_create("TRAIN", "QUEUED")
        print(job_id)
        await job_update_status(job_id, "RUNNING")
        status = await job_get_status(job_id)
        assert status == "RUNNING"


class TestExperiments:
    @pytest.mark.asyncio
    async def test_create_and_get_experiment(self, test_experiment):
        experiment = await experiment_get(test_experiment)
        assert experiment is not None
        assert experiment["name"] == "test_experiment"
        # now try to get an experiment that does not exist
        experiment = await experiment_get(999999)
        assert experiment is None
        # now try to create a second experiment with the same name:
        # experiment_id = await experiment_create("test_experiment", "{}")
        # assert experiment_id is None

    @pytest.mark.asyncio
    async def test_experiment_get_all(self):
        experiments = await experiment_get_all()
        assert isinstance(experiments, list)

    @pytest.mark.asyncio
    async def test_experiment_delete(self):
        experiment_id = await experiment_create("test_experiment_delete", "{}")
        await experiment_delete(experiment_id)
        experiment = await experiment_get(experiment_id)
        assert experiment is None


class TestPlugins:
    @pytest.mark.asyncio
    async def test_save_and_get_plugin(self):
        await save_plugin("test_plugin", "test_type")
        plugin = await get_plugin("test_plugin")
        assert plugin is not None
        assert plugin["name"] == "test_plugin"

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
        value = await config_get("test_key2")
        assert value is None
        # now try to set a key with None value
        await config_set("test_key3", None)
        value = await config_get("test_key3")
        assert value is None
        # now try to set a key with empty string value
        await config_set("test_key4", "")
        value = await config_get("test_key4")
        assert value == ""


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
        workflow = await workflows_get_by_id(workflow_id)
        assert workflow is not None
        assert workflow["name"] == "test_workflow"

    @pytest.mark.asyncio
    async def test_workflow_update_name(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_update", "{}", test_experiment)
        await workflow_update_name(workflow_id, "updated_workflow")
        workflow = await workflows_get_by_id(workflow_id)
        assert workflow["name"] == "updated_workflow"

    @pytest.mark.asyncio
    async def test_workflow_update_config(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_config", "{}", test_experiment)
        await workflow_update_config(workflow_id, '{"key": "value"}')
        workflow = await workflows_get_by_id(workflow_id)
        assert workflow["config"] == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_workflow_delete_by_id(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_delete", "{}", test_experiment)
        await workflow_delete_by_id(workflow_id)
        workflow = await workflows_get_by_id(workflow_id)
        assert workflow["status"] == "DELETED"

    @pytest.mark.asyncio
    async def test_workflow_delete_by_name(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_delete_name", "{}", test_experiment)
        await workflow_delete_by_name("test_workflow_delete_name")  # noqa: F821
        workflow = await workflows_get_by_id(workflow_id)
        assert workflow["status"] == "DELETED"

    @pytest.mark.asyncio
    async def test_workflow_queue(self, test_experiment):
        workflow_id = await workflow_create("test_workflow_queue", "{}", test_experiment)
        await workflow_queue(workflow_id)
        # Assuming queuing updates the status or similar
        workflow = await workflows_get_by_id(workflow_id)
        assert workflow is not None

    @pytest.mark.asyncio
    async def test_workflow_run_get_all(self):
        workflow_runs = await workflow_run_get_all()
        assert isinstance(workflow_runs, list)

    @pytest.mark.asyncio
    async def test_workflow_run_get_by_id(self):
        # Assuming a workflow run is created during testing
        workflow_run_id = 1  # Replace with actual logic to create a workflow run
        workflow_run = await workflow_run_get_by_id(workflow_run_id)
        assert workflow_run is not None

    @pytest.mark.asyncio
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
