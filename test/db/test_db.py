import os


os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from transformerlab.db import (
    create_huggingface_dataset,
    get_dataset,
    get_datasets,
    create_local_dataset,
    delete_dataset,
    get_generated_datasets,
    job_count_running,
    job_delete,
    job_get_error_msg,
    jobs_get_next_queued_job,
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
    class TestJobs:
        @pytest.mark.asyncio
        async def test_job_create_invalid_type(self):
            with pytest.raises(Exception):
                await job_create("INVALID_TYPE", "QUEUED")

        @pytest.mark.asyncio
        async def test_job_get_nonexistent(self):
            job = await job_get(999999)
            assert job is None

        @pytest.mark.asyncio
        async def test_job_delete(self):
            job_id = await job_create("TRAIN", "QUEUED")
            await job_delete(job_id)
            job = await job_get(job_id)
            print(job)
            assert job["status"] == "DELETED"

        @pytest.mark.asyncio
        async def test_job_count_running(self):
            await job_create("TRAIN", "RUNNING")
            count = await job_count_running()
            assert isinstance(count, int)
            assert count > 0

        @pytest.mark.asyncio
        async def test_jobs_get_next_queued_job(self):
            await job_create("TRAIN", "QUEUED")
            job = await jobs_get_next_queued_job()
            assert job is not None
            assert job["status"] == "QUEUED"

        @pytest.mark.asyncio
        async def test_job_update_status_with_error_msg(self):
            job_id = await job_create("TRAIN", "QUEUED")
            await job_update_status(job_id, "FAILED", error_msg="Test error")
            status = await job_get_status(job_id)
            assert status == "FAILED"
            error_msg = await job_get_error_msg(job_id)
            assert error_msg == "Test error"

    class TestDatasets:
        @pytest.mark.asyncio
        async def test_get_generated_datasets(self):
            dataset = await get_dataset("test_generated_dataset")
            if dataset:
                await delete_dataset("test_generated_dataset")
            await create_local_dataset("test_generated_dataset", {"generated": True})
            datasets = await get_generated_datasets()
            assert isinstance(datasets, list)
            assert any(dataset["dataset_id"] == "test_generated_dataset" for dataset in datasets)
            await delete_dataset("test_generated_dataset")

        @pytest.mark.asyncio
        async def test_create_huggingface_dataset(self):
            dataset = await get_dataset("hf_dataset")
            if dataset:
                await delete_dataset("hf_dataset")
            await create_huggingface_dataset("hf_dataset", "HuggingFace Dataset", 100, {"key": "value"})
            dataset = await get_dataset("hf_dataset")
            assert dataset is not None
            assert dataset["location"] == "huggingfacehub"
            assert dataset["description"] == "HuggingFace Dataset"
            assert dataset["size"] == 100
            assert dataset["json_data"]["key"] == "value"

    class TestModels:
        @pytest.mark.asyncio
        async def test_model_local_get_nonexistent(self):
            model = await model_local_get("nonexistent_model")
            assert model is None

        @pytest.mark.asyncio
        async def test_model_local_create_duplicate(self):
            await model_local_create("duplicate_model", "Duplicate Model", {})
            await model_local_create("duplicate_model", "Duplicate Model Updated", {"key": "value"})
            model = await model_local_get("duplicate_model")
            assert model is not None
            assert model["name"] == "Duplicate Model Updated"
            assert model["json_data"]["key"] == "value"


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
