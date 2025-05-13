import os

os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from transformerlab import db
from transformerlab.db import (
    create_huggingface_dataset,
    experiment_get_by_name,
    get_dataset,
    get_datasets,
    create_local_dataset,
    delete_dataset,
    get_generated_datasets,
    get_plugins_of_type,
    get_training_template_by_name,
    job_count_running,
    job_delete,
    job_get_error_msg,
    jobs_get_next_queued_job,
    model_local_create,
    model_local_get,
    model_local_list,
    model_local_count,
    model_local_delete,
    job_create,
    job_get_status,
    job_get,
    job_update_status,
    job_delete_all,
    job_cancel_in_progress_jobs,
    job_update_job_data_insert_key_value,
    job_stop,
    jobs_get_all,
    jobs_get_all_by_experiment_and_type,
    experiment_get_all,
    experiment_create,
    experiment_get,
    experiment_delete,
    experiment_update,
    experiment_update_config,
    experiment_save_prompt_template,
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
    add_task,
    update_task,
    tasks_get_all,
    tasks_get_by_type,
    tasks_get_by_type_in_experiment,
    delete_task,
    tasks_delete_all,
    tasks_get_by_id,
    get_training_template,
    get_training_templates,
    create_training_template,
    update_training_template,
    delete_training_template,
    export_job_create,
)

import pytest


@pytest.mark.asyncio
async def test_get_dataset_returns_none_for_missing():
    dataset = await get_dataset("does_not_exist")
    assert dataset is None


@pytest.mark.asyncio
async def test_model_local_delete_nonexistent():
    # Should not raise
    await model_local_delete("nonexistent_model")


@pytest.mark.asyncio
async def test_job_get_status_and_error_msg_for_nonexistent():
    # Should raise or return None for missing job
    with pytest.raises(Exception):
        await job_get_status(999999)
    with pytest.raises(Exception):
        await job_get_error_msg(999999)


@pytest.mark.asyncio
async def test_job_update_job_data_insert_key_value_overwrite():
    job_id = await job_create("TRAIN", "QUEUED")
    await job_update_job_data_insert_key_value(job_id, "foo", {"bar": 1})
    await job_update_job_data_insert_key_value(job_id, "foo", {"baz": 2})
    job = await job_get(job_id)
    assert job["job_data"]["foo"] == {"baz": 2}


@pytest.mark.asyncio
async def test_job_update_status_without_error_msg():
    job_id = await job_create("TRAIN", "QUEUED")
    await job_update_status(job_id, "RUNNING")
    status = await job_get_status(job_id)
    assert status == "RUNNING"


@pytest.mark.asyncio
async def test_job_delete_marks_deleted():
    job_id = await job_create("TRAIN", "QUEUED")
    await job_delete(job_id)
    job = await job_get(job_id)
    assert job["status"] == "DELETED"


@pytest.mark.asyncio
async def test_job_cancel_in_progress_jobs_sets_cancelled():
    job_id = await job_create("TRAIN", "RUNNING")
    await job_cancel_in_progress_jobs()
    job = await job_get(job_id)
    assert job["status"] == "CANCELLED"


@pytest.mark.asyncio
async def test_tasks_get_by_id_returns_none_for_missing():
    task = await tasks_get_by_id(999999)
    assert task is None


@pytest.mark.asyncio
async def test_get_training_template_and_by_name_returns_none_for_missing():
    tmpl = await get_training_template(999999)
    assert tmpl is None
    tmpl = await get_training_template_by_name("does_not_exist")
    assert tmpl is None


@pytest.mark.asyncio
async def test_experiment_get_by_name_returns_none_for_missing():
    exp = await experiment_get_by_name("does_not_exist")
    assert exp is None


@pytest.mark.asyncio
async def test_workflows_get_by_id_returns_none_for_missing():
    workflow = await workflows_get_by_id(999999)
    assert workflow is None


@pytest.mark.asyncio
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
async def test_model_local_list_and_count():
    await model_local_create("model1", "Model 1", {})
    await model_local_create("model2", "Model 2", {})
    models = await model_local_list()
    count = await model_local_count()
    assert isinstance(models, list)
    assert count == len(models)
    await model_local_delete("model1")
    await model_local_delete("model2")


@pytest.mark.asyncio
async def test_job_delete_all_and_cancel_in_progress_jobs():
    await job_create("TRAIN", "RUNNING")
    await job_create("TRAIN", "QUEUED")
    await job_delete_all()
    jobs = await jobs_get_all()
    assert all(job["status"] == "DELETED" for job in jobs)
    await job_cancel_in_progress_jobs()  # Should not raise


@pytest.mark.asyncio
async def test_job_update_job_data_insert_key_value_and_stop():
    job_id = await job_create("TRAIN", "QUEUED")
    await job_update_job_data_insert_key_value(job_id, "foo", {"bar": 1})
    job = await job_get(job_id)
    assert job["job_data"]["foo"] == {"bar": 1}
    await job_stop(job_id)
    job = await job_get(job_id)
    assert job["job_data"]["stop"] is True


@pytest.mark.asyncio
async def test_jobs_get_all_and_by_experiment_and_type(test_experiment):
    job_id = await job_create("TRAIN", "QUEUED", "{}", test_experiment)
    jobs = await jobs_get_all("TRAIN", "QUEUED")
    assert any(j["id"] == job_id for j in jobs)
    jobs_by_exp = await jobs_get_all_by_experiment_and_type(test_experiment, "TRAIN")
    assert any(j["id"] == job_id for j in jobs_by_exp)


@pytest.mark.asyncio
async def test_experiment_update_and_update_config_and_save_prompt_template(test_experiment):
    await experiment_update(test_experiment, '{"foo": "bar"}')
    exp = await experiment_get(test_experiment)
    assert exp["config"] == '{"foo": "bar"}'
    await experiment_update_config(test_experiment, "baz", 123)
    exp = await experiment_get(test_experiment)
    assert '"baz":123' in exp["config"]
    await experiment_save_prompt_template(test_experiment, '"prompt"')
    exp = await experiment_get(test_experiment)
    assert "prompt_template" in exp["config"]


@pytest.mark.asyncio
async def test_task_crud(test_experiment):
    await add_task("task1", "TYPE", "{}", "{}", "plugin", "{}", test_experiment)
    tasks = await tasks_get_all()
    assert any(t["name"] == "task1" for t in tasks)
    task = tasks[0]
    await update_task(task["id"], {"inputs": "[]", "config": "{}", "outputs": "[]", "name": "task1_updated"})
    updated = await tasks_get_by_id(task["id"])
    assert updated["name"] == "task1_updated"
    await delete_task(task["id"])
    deleted = await tasks_get_by_id(task["id"])
    assert deleted is None


@pytest.mark.asyncio
async def test_tasks_get_by_type_and_in_experiment(test_experiment):
    await add_task("task2", "TYPE2", "{}", "{}", "plugin", "{}", test_experiment)
    by_type = await tasks_get_by_type("TYPE2")
    assert any(t["name"] == "task2" for t in by_type)
    by_type_exp = await tasks_get_by_type_in_experiment("TYPE2", test_experiment)
    assert any(t["name"] == "task2" for t in by_type_exp)
    await tasks_delete_all()
    all_tasks = await tasks_get_all()
    assert len(all_tasks) == 0


@pytest.mark.asyncio
async def test_training_template_crud():
    await create_training_template("tmpl", "desc", "type", "[]", "{}")
    templates = await get_training_templates()
    assert any(t[1] == "tmpl" for t in templates)
    tmpl_id = templates[0][0]
    await update_training_template(tmpl_id, "tmpl2", "desc2", "type2", "[]", "{}")
    tmpl = await get_training_template(tmpl_id)
    assert tmpl["name"] == "tmpl2"
    await delete_training_template(tmpl_id)
    tmpl = await get_training_template(tmpl_id)
    assert tmpl is None


@pytest.mark.asyncio
async def test_export_job_create(test_experiment):
    job_id = await export_job_create(test_experiment, '{"exporter_name": "exp"}')
    job = await job_get(job_id)
    assert job is not None
    assert job["type"] == "EXPORT_MODEL"


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
