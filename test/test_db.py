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
)
import pytest
import transformerlab.db as db

pytest_plugins = ("pytest_asyncio",)

# FILE: transformerlab/test_db.py


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
