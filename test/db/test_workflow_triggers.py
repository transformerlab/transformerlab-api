import json
import os
import pytest
import uuid

os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from transformerlab import db
from transformerlab.db import (
    PREDEFINED_TRIGGER_BLUEPRINTS,
    _normalize_trigger_configs,
    workflow_update_trigger_configs,
    workflow_get_by_job_event,
    workflow_create,
    workflows_get_by_id,
    workflows_get_all,
    workflows_get_from_experiment,
    workflow_delete_all,
    experiment_create,
    experiment_get_directory_by_id
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


class TestNormalizeTriggerConfigs:
    """Test the _normalize_trigger_configs helper function."""

    def test_normalize_none_input(self):
        """Test normalizing None input returns default configs."""
        result = _normalize_trigger_configs(None)
        
        assert isinstance(result, list)
        assert len(result) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
        
        # Check all predefined types are present with default values
        trigger_types = {config["trigger_type"] for config in result}
        expected_types = {bp["trigger_type"] for bp in PREDEFINED_TRIGGER_BLUEPRINTS}
        assert trigger_types == expected_types
        
        # Check default enabled values
        for config in result:
            blueprint = next(bp for bp in PREDEFINED_TRIGGER_BLUEPRINTS if bp["trigger_type"] == config["trigger_type"])
            assert config["is_enabled"] == blueprint["default_is_enabled"]

    def test_normalize_empty_string_input(self):
        """Test normalizing empty string returns default configs."""
        result = _normalize_trigger_configs("")
        
        assert isinstance(result, list)
        assert len(result) == len(PREDEFINED_TRIGGER_BLUEPRINTS)

    def test_normalize_invalid_json_string(self):
        """Test normalizing invalid JSON string returns default configs."""
        result = _normalize_trigger_configs("invalid json")
        
        assert isinstance(result, list)
        assert len(result) == len(PREDEFINED_TRIGGER_BLUEPRINTS)

    def test_normalize_valid_json_string(self):
        """Test normalizing valid JSON string."""
        input_configs = [
            {"trigger_type": "TRAIN", "is_enabled": True},
            {"trigger_type": "DOWNLOAD_MODEL", "is_enabled": False}
        ]
        json_string = json.dumps(input_configs)
        
        result = _normalize_trigger_configs(json_string)
        
        assert isinstance(result, list)
        assert len(result) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
        
        # Check that provided configs are preserved
        train_config = next(config for config in result if config["trigger_type"] == "TRAIN")
        assert train_config["is_enabled"] is True
        
        download_config = next(config for config in result if config["trigger_type"] == "DOWNLOAD_MODEL")
        assert download_config["is_enabled"] is False

    def test_normalize_partial_config_list(self):
        """Test normalizing list with only some trigger types."""
        input_configs = [
            {"trigger_type": "TRAIN", "is_enabled": True},
            {"trigger_type": "EVAL", "is_enabled": True}
        ]
        
        result = _normalize_trigger_configs(input_configs)
        
        assert isinstance(result, list)
        assert len(result) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
        
        # Check provided configs are preserved
        train_config = next(config for config in result if config["trigger_type"] == "TRAIN")
        assert train_config["is_enabled"] is True
        
        eval_config = next(config for config in result if config["trigger_type"] == "EVAL")
        assert eval_config["is_enabled"] is True
        
        # Check missing types get default values
        missing_types = {"DOWNLOAD_MODEL", "LOAD_MODEL", "EXPORT_MODEL", "GENERATE"}
        for config in result:
            if config["trigger_type"] in missing_types:
                blueprint = next(bp for bp in PREDEFINED_TRIGGER_BLUEPRINTS if bp["trigger_type"] == config["trigger_type"])
                assert config["is_enabled"] == blueprint["default_is_enabled"]

    def test_normalize_complete_config_list(self):
        """Test normalizing complete config list."""
        input_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            input_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": not bp["default_is_enabled"]  # Flip defaults
            })
        
        result = _normalize_trigger_configs(input_configs)
        
        assert isinstance(result, list)
        assert len(result) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
        
        # Check all configs are preserved with flipped values
        for config in result:
            blueprint = next(bp for bp in PREDEFINED_TRIGGER_BLUEPRINTS if bp["trigger_type"] == config["trigger_type"])
            assert config["is_enabled"] == (not blueprint["default_is_enabled"])

    def test_normalize_config_with_missing_is_enabled(self):
        """Test normalizing config where is_enabled is missing."""
        input_configs = [
            {"trigger_type": "TRAIN"},  # Missing is_enabled
            {"trigger_type": "DOWNLOAD_MODEL", "is_enabled": True}
        ]
        
        result = _normalize_trigger_configs(input_configs)
        
        assert isinstance(result, list)
        assert len(result) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
        
        # Config with missing is_enabled should get default value
        train_config = next(config for config in result if config["trigger_type"] == "TRAIN")
        train_blueprint = next(bp for bp in PREDEFINED_TRIGGER_BLUEPRINTS if bp["trigger_type"] == "TRAIN")
        assert train_config["is_enabled"] == train_blueprint["default_is_enabled"]


class TestWorkflowTriggerConfigs:
    """Test workflow trigger configuration database functions."""

    @pytest.mark.asyncio
    async def test_workflow_create_with_default_triggers(self, test_experiment, clean_workflows):
        """Test that workflow creation includes default trigger configs."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        workflow = await workflows_get_by_id(workflow_id)
        
        assert workflow is not None
        assert "trigger_configs" in workflow
        assert isinstance(workflow["trigger_configs"], list)
        assert len(workflow["trigger_configs"]) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
        
        # Check all trigger types are present with default values
        trigger_types = {config["trigger_type"] for config in workflow["trigger_configs"]}
        expected_types = {bp["trigger_type"] for bp in PREDEFINED_TRIGGER_BLUEPRINTS}
        assert trigger_types == expected_types
        
        for config in workflow["trigger_configs"]:
            blueprint = next(bp for bp in PREDEFINED_TRIGGER_BLUEPRINTS if bp["trigger_type"] == config["trigger_type"])
            assert config["is_enabled"] == blueprint["default_is_enabled"]

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_success(self, test_experiment, clean_workflows):
        """Test successful trigger config update."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Create new configs with all triggers enabled
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        updated_workflow = await workflow_update_trigger_configs(workflow_id, new_configs)
        
        assert updated_workflow is not None
        assert len(updated_workflow["trigger_configs"]) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
        
        # Check all triggers are enabled
        for config in updated_workflow["trigger_configs"]:
            assert config["is_enabled"] is True

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_invalid_workflow_id(self, clean_workflows):
        """Test updating trigger configs for non-existent workflow."""
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        with pytest.raises(ValueError, match="Workflow with id 999999 not found"):
            await workflow_update_trigger_configs(999999, new_configs)

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_wrong_count(self, test_experiment, clean_workflows):
        """Test updating with wrong number of trigger configs."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Provide too few configs
        new_configs = [
            {"trigger_type": "TRAIN", "is_enabled": True}
        ]
        
        with pytest.raises(ValueError, match="new_configs_list must be a list of exactly 6 trigger configurations"):
            await workflow_update_trigger_configs(workflow_id, new_configs)

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_invalid_trigger_type(self, test_experiment, clean_workflows):
        """Test updating with invalid trigger type."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        # Replace one with invalid type
        new_configs[0]["trigger_type"] = "INVALID_TYPE"
        
        with pytest.raises(ValueError, match="Invalid trigger_type: INVALID_TYPE"):
            await workflow_update_trigger_configs(workflow_id, new_configs)

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_duplicate_trigger_type(self, test_experiment, clean_workflows):
        """Test updating with duplicate trigger types."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        # Make one duplicate
        new_configs[1]["trigger_type"] = new_configs[0]["trigger_type"]
        
        with pytest.raises(ValueError, match="Duplicate trigger_type"):
            await workflow_update_trigger_configs(workflow_id, new_configs)

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_missing_trigger_type(self, test_experiment, clean_workflows):
        """Test updating with missing trigger types."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Only provide 5 out of 6 required configs
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS[:-1]:  # Skip last one
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        # Add another config of existing type to maintain count of 6
        new_configs.append({
            "trigger_type": PREDEFINED_TRIGGER_BLUEPRINTS[0]["trigger_type"],  # Duplicate
            "is_enabled": False
        })
        
        with pytest.raises(ValueError, match="Duplicate trigger_type"):
            await workflow_update_trigger_configs(workflow_id, new_configs)

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_missing_some_trigger_types(self, test_experiment, clean_workflows):
        """Test updating with only some trigger types present."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Only provide 3 out of 6 required configs (missing multiple types)
        new_configs = [
            {"trigger_type": "TRAIN", "is_enabled": True},
            {"trigger_type": "EVAL", "is_enabled": True},
            {"trigger_type": "GENERATE", "is_enabled": False}
        ]
        
        with pytest.raises(ValueError, match="new_configs_list must be a list of exactly 6 trigger configurations"):
            await workflow_update_trigger_configs(workflow_id, new_configs)

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_missing_trigger_types_when_count_is_correct(self, test_experiment, clean_workflows):
        """Test updating where count is 6 but missing some required trigger types."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Provide 6 configs but with missing types (duplicates instead)
        new_configs = [
            {"trigger_type": "TRAIN", "is_enabled": True},
            {"trigger_type": "TRAIN", "is_enabled": False},  # Duplicate
            {"trigger_type": "EVAL", "is_enabled": True},
            {"trigger_type": "EVAL", "is_enabled": False},   # Duplicate
            {"trigger_type": "GENERATE", "is_enabled": True},
            {"trigger_type": "GENERATE", "is_enabled": False} # Duplicate
        ]
        
        with pytest.raises(ValueError, match="Duplicate trigger_type"):
            await workflow_update_trigger_configs(workflow_id, new_configs)

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_invalid_is_enabled_type(self, test_experiment, clean_workflows):
        """Test updating with non-boolean is_enabled value."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        # Make one is_enabled a string instead of boolean
        new_configs[0]["is_enabled"] = "true"
        
        with pytest.raises(ValueError, match="is_enabled must be a boolean"):
            await workflow_update_trigger_configs(workflow_id, new_configs)

    @pytest.mark.asyncio
    async def test_workflow_update_trigger_configs_non_dict_config(self, test_experiment, clean_workflows):
        """Test updating with non-dictionary config."""
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        # Replace one config with a string
        new_configs[0] = "invalid_config"
        
        with pytest.raises(ValueError, match="Each trigger config must be a dictionary"):
            await workflow_update_trigger_configs(workflow_id, new_configs)


class TestWorkflowGetByJobEvent:
    """Test workflow_get_by_job_event function."""

    @pytest.mark.asyncio
    async def test_get_workflows_by_job_event_no_workflows(self, test_experiment, clean_workflows):
        """Test finding workflows when none exist."""
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_workflows_by_job_event_no_matching_triggers(self, test_experiment, clean_workflows):
        """Test finding workflows when none have matching enabled triggers."""
        # Create workflow with all triggers disabled
        await workflow_create("test_workflow", "{}", test_experiment)
        
        # Keep all triggers disabled (default state)
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_workflows_by_job_event_different_experiment(self, clean_workflows):
        """Test that workflows from different experiments don't match."""
        # Create two experiments with unique names
        exp1_name = f"exp1_{uuid.uuid4().hex[:8]}"
        exp2_name = f"exp2_{uuid.uuid4().hex[:8]}"
        exp1_id = await experiment_create(exp1_name, "{}")
        exp2_id = await experiment_create(exp2_name, "{}")
        
        # Create workflow in exp1 with TRAIN trigger enabled
        workflow_id = await workflow_create("test_workflow", "{}", exp1_id)
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": bp["trigger_type"] == "TRAIN"
            })
        await workflow_update_trigger_configs(workflow_id, new_configs)
        
        # Search for workflows in exp2
        result = await workflow_get_by_job_event("TRAIN", exp2_id)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_workflows_by_job_event_matching_single_workflow(self, test_experiment, clean_workflows):
        """Test finding single workflow with matching trigger."""
        # Create workflow with TRAIN trigger enabled
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": bp["trigger_type"] == "TRAIN"
            })
        await workflow_update_trigger_configs(workflow_id, new_configs)
        
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        
        assert len(result) == 1
        assert result[0]["id"] == workflow_id
        assert result[0]["name"] == "test_workflow"

    @pytest.mark.asyncio
    async def test_get_workflows_by_job_event_matching_multiple_workflows(self, test_experiment, clean_workflows):
        """Test finding multiple workflows with matching triggers."""
        # Create two workflows with TRAIN trigger enabled
        workflow1_id = await workflow_create("workflow1", "{}", test_experiment)
        workflow2_id = await workflow_create("workflow2", "{}", test_experiment)
        
        for workflow_id in [workflow1_id, workflow2_id]:
            new_configs = []
            for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
                new_configs.append({
                    "trigger_type": bp["trigger_type"],
                    "is_enabled": bp["trigger_type"] == "TRAIN"
                })
            await workflow_update_trigger_configs(workflow_id, new_configs)
        
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        
        assert len(result) == 2
        result_ids = {w["id"] for w in result}
        assert result_ids == {workflow1_id, workflow2_id}

    @pytest.mark.asyncio
    async def test_get_workflows_by_job_event_different_trigger_types(self, test_experiment, clean_workflows):
        """Test finding workflows for different trigger types."""
        # Create workflow with multiple triggers enabled
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": bp["trigger_type"] in ["TRAIN", "EVAL"]
            })
        await workflow_update_trigger_configs(workflow_id, new_configs)
        
        # Test TRAIN trigger
        result_train = await workflow_get_by_job_event("TRAIN", test_experiment)
        assert len(result_train) == 1
        assert result_train[0]["id"] == workflow_id
        
        # Test EVAL trigger
        result_eval = await workflow_get_by_job_event("EVAL", test_experiment)
        assert len(result_eval) == 1
        assert result_eval[0]["id"] == workflow_id
        
        # Test disabled trigger
        result_download = await workflow_get_by_job_event("DOWNLOAD_MODEL", test_experiment)
        assert result_download == []

    @pytest.mark.asyncio
    async def test_get_workflows_by_job_event_invalid_trigger_configs(self, test_experiment, clean_workflows):
        """Test finding workflows with invalid trigger configs."""
        # Create workflow and manually set invalid trigger_configs
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Manually update with invalid JSON
        await db.db.execute(
            "UPDATE workflows SET trigger_configs = ? WHERE id = ?",
            ("invalid_json", workflow_id)
        )
        await db.db.commit()
        
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_workflows_by_job_event_deleted_workflows(self, test_experiment, clean_workflows):
        """Test that deleted workflows are not returned."""
        # Create workflow with TRAIN trigger enabled
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        new_configs = []
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            new_configs.append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": bp["trigger_type"] == "TRAIN"
            })
        await workflow_update_trigger_configs(workflow_id, new_configs)
        
        # Verify workflow is found
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        assert len(result) == 1
        
        # Mark workflow as deleted
        await db.db.execute(
            "UPDATE workflows SET status = 'DELETED' WHERE id = ?",
            (workflow_id,)
        )
        await db.db.commit()
        
        # Verify workflow is no longer found
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        assert result == []


class TestWorkflowsGetWithTriggerNormalization:
    """Test that workflows_get functions properly normalize trigger configs."""

    @pytest.mark.asyncio
    async def test_workflows_get_all_normalizes_triggers(self, test_experiment, clean_workflows):
        """Test that workflows_get_all normalizes trigger configs."""
        # Create workflow
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Get all workflows
        workflows = await workflows_get_all()
        
        test_workflow = next(w for w in workflows if w["id"] == workflow_id)
        assert "trigger_configs" in test_workflow
        assert isinstance(test_workflow["trigger_configs"], list)
        assert len(test_workflow["trigger_configs"]) == len(PREDEFINED_TRIGGER_BLUEPRINTS)

    @pytest.mark.asyncio
    async def test_workflows_get_from_experiment_normalizes_triggers(self, test_experiment, clean_workflows):
        """Test that workflows_get_from_experiment normalizes trigger configs."""
        # Create workflow
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Get workflows from experiment
        workflows = await workflows_get_from_experiment(test_experiment)
        
        assert len(workflows) == 1
        test_workflow = workflows[0]
        assert test_workflow["id"] == workflow_id
        assert "trigger_configs" in test_workflow
        assert isinstance(test_workflow["trigger_configs"], list)
        assert len(test_workflow["trigger_configs"]) == len(PREDEFINED_TRIGGER_BLUEPRINTS)

    @pytest.mark.asyncio
    async def test_workflows_get_by_id_normalizes_triggers(self, test_experiment, clean_workflows):
        """Test that workflows_get_by_id normalizes trigger configs."""
        # Create workflow
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Get workflow by id
        workflow = await workflows_get_by_id(workflow_id)
        
        assert workflow is not None
        assert workflow["id"] == workflow_id
        assert "trigger_configs" in workflow
        assert isinstance(workflow["trigger_configs"], list)
        assert len(workflow["trigger_configs"]) == len(PREDEFINED_TRIGGER_BLUEPRINTS)

    @pytest.mark.asyncio
    async def test_workflows_get_normalizes_corrupted_trigger_configs(self, test_experiment, clean_workflows):
        """Test that workflows get functions handle corrupted trigger configs."""
        # Create workflow
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Manually corrupt trigger_configs
        await db.db.execute(
            "UPDATE workflows SET trigger_configs = ? WHERE id = ?",
            ("corrupted_json", workflow_id)
        )
        await db.db.commit()
        
        # Get workflow - should normalize corrupted configs
        workflow = await workflows_get_by_id(workflow_id)
        
        assert workflow is not None
        assert "trigger_configs" in workflow
        assert isinstance(workflow["trigger_configs"], list)
        assert len(workflow["trigger_configs"]) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
        
        # All should have default values
        for config in workflow["trigger_configs"]:
            blueprint = next(bp for bp in PREDEFINED_TRIGGER_BLUEPRINTS if bp["trigger_type"] == config["trigger_type"])
            assert config["is_enabled"] == blueprint["default_is_enabled"]


class TestExperimentDirectoryByIdErrorHandling:
    """Test experiment_get_directory_by_id error handling."""

    @pytest.mark.asyncio
    async def test_experiment_get_directory_by_id_not_found(self):
        """Test experiment_get_directory_by_id with non-existent experiment."""
        result = await experiment_get_directory_by_id(999999)
        
        # Should return error directory path when experiment not found
        assert "error" in result.lower()


class TestWorkflowGetByJobEventWithComplexScenarios:
    """Test workflow_get_by_job_event with complex edge cases."""

    @pytest.mark.asyncio
    async def test_workflow_get_by_job_event_with_json_decode_error(self, test_experiment, clean_workflows):
        """Test workflow_get_by_job_event with JSON decode errors in trigger configs."""
        # Create workflow
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Manually set invalid JSON that will cause JSONDecodeError
        await db.db.execute(
            "UPDATE workflows SET trigger_configs = ? WHERE id = ?",
            ("{invalid json", workflow_id)  # Malformed JSON
        )
        await db.db.commit()
        
        # Should handle JSONDecodeError gracefully
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        assert result == []

    @pytest.mark.asyncio
    async def test_workflow_get_by_job_event_with_non_list_trigger_configs(self, test_experiment, clean_workflows):
        """Test workflow_get_by_job_event when trigger_configs is not a list."""
        # Create workflow
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Set trigger_configs to a non-list (dict)
        await db.db.execute(
            "UPDATE workflows SET trigger_configs = ? WHERE id = ?",
            ('{"not": "a list"}', workflow_id)
        )
        await db.db.commit()
        
        # Should handle non-list trigger_configs gracefully
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        assert result == []

    @pytest.mark.asyncio
    async def test_workflow_get_by_job_event_with_non_dict_trigger_config(self, test_experiment, clean_workflows):
        """Test workflow_get_by_job_event when individual trigger config is not a dict."""
        # Create workflow
        workflow_id = await workflow_create("test_workflow", "{}", test_experiment)
        
        # Set trigger_configs to list with non-dict elements
        await db.db.execute(
            "UPDATE workflows SET trigger_configs = ? WHERE id = ?",
            ('["not_a_dict", "also_not_a_dict"]', workflow_id)
        )
        await db.db.commit()
        
        # Should handle non-dict trigger configs gracefully
        result = await workflow_get_by_job_event("TRAIN", test_experiment)
        assert result == [] 