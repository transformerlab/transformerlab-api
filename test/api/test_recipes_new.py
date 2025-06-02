import os
import json
import pytest
from fastapi.testclient import TestClient
from api import app
from transformerlab.shared.galleries import gallery_cache_file_path, EXP_RECIPES_GALLERY_FILE


# Test recipe data focused on new functionality: notes and tasks
TEST_EXP_RECIPES = [
    {
        "id": 1,
        "title": "Test Recipe - With Notes",
        "description": "A test recipe with notes to test notes creation",
        "notes": "# Test Recipe Notes\n\nThis is a test recipe for unit testing.\n\n## Features\n- Notes creation\n- Task generation\n\n## Usage\nThis should create a readme.md file in the experiment.",
        "dependencies": [
            {
                "type": "model",
                "name": "test-model"
            },
            {
                "type": "dataset", 
                "name": "test-dataset"
            }
        ]
    },
    {
        "id": 2,
        "title": "Test Recipe - With Tasks and Notes",
        "description": "A test recipe that includes both notes and tasks",
        "notes": "# Training Recipe\n\nThis recipe includes training tasks.\n\n## Training Configuration\n- Uses LoRA training\n- Batch size: 4\n- Learning rate: 0.0001",
        "dependencies": [
            {
                "type": "model",
                "name": "test-model-2"
            },
            {
                "type": "dataset",
                "name": "test-dataset-for-training"
            }
        ],
        "tasks": [
            {
                "task_type": "TRAIN",
                "type": "LoRA",
                "plugin": "test_trainer",
                "formatting_template": "{{prompt}}\n{{completion}}",
                "config_json": "{\"template_name\":\"TestTemplate\",\"plugin_name\":\"test_trainer\",\"model_name\":\"test-model-2\",\"dataset_name\":\"test-dataset-for-training\",\"batch_size\":\"4\",\"learning_rate\":\"0.0001\"}"
            }
        ]
    },
    {
        "id": 3,
        "title": "Test Recipe - Tasks Only",
        "description": "A test recipe with only tasks, no notes",
        "dependencies": [
            {
                "type": "model",
                "name": "test-model-3"
            },
            {
                "type": "dataset",
                "name": "test-dataset-3"
            }
        ],
        "tasks": [
            {
                "task_type": "TRAIN",
                "type": "LoRA",
                "plugin": "mlx_lora_trainer",
                "formatting_template": "{{text}}",
                "config_json": "{\"template_name\":\"NoNotesTemplate\",\"plugin_name\":\"mlx_lora_trainer\",\"model_name\":\"test-model-3\",\"dataset_name\":\"test-dataset-3\",\"batch_size\":\"8\",\"learning_rate\":\"0.001\"}"
            }
        ]
    },
    {
        "id": 4,
        "title": "Test Recipe - With Adaptor Name",
        "description": "A test recipe that includes adaptor_name in config to test line 281",
        "dependencies": [
            {
                "type": "model",
                "name": "test-model-4"
            },
            {
                "type": "dataset",
                "name": "test-dataset-4"
            }
        ],
        "tasks": [
            {
                "task_type": "TRAIN",
                "type": "LoRA",
                "plugin": "test_trainer",
                "formatting_template": "{{prompt}}\n{{completion}}",
                "config_json": "{\"template_name\":\"AdaptorTest\",\"plugin_name\":\"test_trainer\",\"model_name\":\"test-model-4\",\"dataset_name\":\"test-dataset-4\",\"adaptor_name\":\"test_adaptor\",\"batch_size\":\"4\",\"learning_rate\":\"0.0001\"}"
            }
        ]
    },
    {
        "id": 5,
        "title": "Test Recipe - Invalid JSON Config",
        "description": "A test recipe with invalid JSON to test exception handling",
        "dependencies": [
            {
                "type": "model",
                "name": "test-model-5"
            },
            {
                "type": "dataset",
                "name": "test-dataset-5"
            }
        ],
        "tasks": [
            {
                "task_type": "TRAIN",
                "type": "LoRA",
                "plugin": "test_trainer",
                "formatting_template": "{{prompt}}\n{{completion}}",
                "config_json": "{invalid json syntax to trigger exception"
            }
        ]
    },
    {
        "id": 6,
        "title": "Test Recipe - With Multiple Task Types",
        "description": "A test recipe that includes training, evaluation and generation tasks",
        "dependencies": [
            {
                "type": "model",
                "name": "test-model-6"
            },
            {
                "type": "dataset",
                "name": "test-dataset-6"
            }
        ],
        "tasks": [
            {
                "task_type": "TRAIN",
                "type": "LoRA",
                "plugin": "test_trainer",
                "formatting_template": "{{prompt}}\n{{completion}}",
                "config_json": "{\"template_name\":\"TestTemplate\",\"plugin_name\":\"test_trainer\",\"model_name\":\"test-model-6\",\"dataset_name\":\"test-dataset-6\",\"batch_size\":\"4\",\"learning_rate\":\"0.0001\"}"
            },
            {
                "task_type": "EVAL",
                "plugin": "test_evaluator",
                "config_json": "{\"template_name\":\"TestEval\",\"plugin_name\":\"test_evaluator\",\"model_name\":\"test-model-6\",\"eval_type\":\"basic\",\"metrics\":[\"accuracy\",\"f1\"],\"eval_dataset\":\"test-eval-dataset\"}"
            },
            {
                "task_type": "GENERATE",
                "plugin": "test_generator",
                "config_json": "{\"template_name\":\"TestGen\",\"plugin_name\":\"test_generator\",\"model_name\":\"test-model-6\",\"prompt_template\":\"Generate a response: {{input}}\",\"generation_params\":{\"max_length\":100,\"temperature\":0.7}}"
            }
        ]
    }
]


@pytest.fixture(autouse=True)
def setup_test_recipes():
    """Setup test recipe file by overwriting the cached gallery file"""
    cache_file_path = gallery_cache_file_path(EXP_RECIPES_GALLERY_FILE)
    
    # Store original file if it exists
    original_content = None
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as f:
            original_content = f.read()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
    
    # Write our test recipes to the cache file
    with open(cache_file_path, 'w') as f:
        json.dump(TEST_EXP_RECIPES, f)
    
    yield  # This is where the test runs
    
    # Cleanup: restore original file or remove test file
    if original_content is not None:
        with open(cache_file_path, 'w') as f:
            f.write(original_content)
    elif os.path.exists(cache_file_path):
        os.remove(cache_file_path)


def test_recipes_list():
    with TestClient(app) as client:
        resp = client.get("/recipes/list")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 3  # Should have our 3 test recipes


def test_recipes_get_by_id_with_notes():
    with TestClient(app) as client:
        resp = client.get("/recipes/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 1
        assert data["title"] == "Test Recipe - With Notes"
        assert "notes" in data
        assert "# Test Recipe Notes" in data["notes"]


def test_recipes_get_by_id_with_tasks():
    with TestClient(app) as client:
        resp = client.get("/recipes/2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 2
        assert "tasks" in data
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["task_type"] == "TRAIN"


def test_create_experiment_with_notes():
    with TestClient(app) as client:
        test_experiment_name = f"test_notes_exp_{os.getpid()}"
        resp = client.post(f"/recipes/1/create_experiment?experiment_name={test_experiment_name}")
        assert resp.status_code == 200
        data = resp.json()
        # Should either succeed or have a reasonable response
        assert "status" in data or "message" in data


def test_create_experiment_with_tasks():
    with TestClient(app) as client:
        test_experiment_name = f"test_tasks_exp_{os.getpid()}"
        resp = client.post(f"/recipes/2/create_experiment?experiment_name={test_experiment_name}")
        assert resp.status_code == 200
        data = resp.json()
        # Should either succeed or have a reasonable response
        assert "status" in data or "message" in data
        # If it succeeds, should have task results in the data section
        if data.get("status") == "success":
            assert "data" in data
            assert "task_results" in data["data"]


def test_create_experiment_tasks_only():
    with TestClient(app) as client:
        test_experiment_name = f"test_tasks_only_{os.getpid()}"
        resp = client.post(f"/recipes/3/create_experiment?experiment_name={test_experiment_name}")
        assert resp.status_code == 200
        data = resp.json()
        # Should either succeed or have a reasonable response
        assert "status" in data or "message" in data


def test_create_experiment_duplicate_name():
    with TestClient(app) as client:
        test_experiment_name = f"duplicate_test_{os.getpid()}"
        
        # First creation
        resp1 = client.post(f"/recipes/1/create_experiment?experiment_name={test_experiment_name}")
        assert resp1.status_code == 200
        
        # Second creation with same name should fail
        resp2 = client.post(f"/recipes/1/create_experiment?experiment_name={test_experiment_name}")
        assert resp2.status_code == 200
        data = resp2.json()
        assert data.get("status") == "error"
        assert "already exists" in data.get("message", "")


def test_create_experiment_invalid_recipe_id():
    with TestClient(app) as client:
        test_experiment_name = f"invalid_recipe_test_{os.getpid()}"
        resp = client.post(f"/recipes/999/create_experiment?experiment_name={test_experiment_name}")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "error"
        assert "not found" in data.get("message", "")


def test_create_experiment_with_adaptor_name():
    """Test creating experiment with recipe that has adaptor_name in config (covers line 281)"""
    with TestClient(app) as client:
        test_experiment_name = f"test_adaptor_{os.getpid()}"
        resp = client.post(f"/recipes/4/create_experiment?experiment_name={test_experiment_name}")
        assert resp.status_code == 200
        data = resp.json()
        # Should either succeed or have a reasonable response
        assert "status" in data or "message" in data
        # If it succeeds, should have task results
        if data.get("status") == "success":
            assert "data" in data
            assert "task_results" in data["data"]


def test_create_experiment_with_invalid_json_config():
    """Test creating experiment with invalid JSON config to trigger exception handling (covers lines 306-307)"""
    with TestClient(app) as client:
        test_experiment_name = f"test_invalid_json_{os.getpid()}"
        resp = client.post(f"/recipes/5/create_experiment?experiment_name={test_experiment_name}")
        assert resp.status_code == 200
        data = resp.json()
        # Should succeed but with error in task results due to invalid JSON
        assert "status" in data or "message" in data
        if data.get("status") == "success" and "data" in data and "task_results" in data["data"]:
            task_results = data["data"]["task_results"]
            # Should have at least one task result with an error status
            assert len(task_results) > 0
            # At least one task should have error status due to invalid JSON
            has_error = any("error" in result.get("status", "") for result in task_results)
            assert has_error


def test_recipes_get_by_id_with_multiple_task_types():
    """Test that a recipe with multiple task types (TRAIN, EVAL, GENERATE) is handled correctly"""
    with TestClient(app) as client:
        resp = client.get("/recipes/6")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 6
        assert "tasks" in data
        assert len(data["tasks"]) == 3
        task_types = [task["task_type"] for task in data["tasks"]]
        assert "TRAIN" in task_types
        assert "EVAL" in task_types
        assert "GENERATE" in task_types


def test_create_experiment_with_multiple_task_types():
    """Test creating an experiment with multiple task types"""
    with TestClient(app) as client:
        test_experiment_name = f"test_multi_tasks_{os.getpid()}"
        resp = client.post(f"/recipes/6/create_experiment?experiment_name={test_experiment_name}")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        if data.get("status") == "success":
            assert "data" in data
            assert "task_results" in data["data"]
            task_results = data["data"]["task_results"]
            assert len(task_results) == 3
            task_types = [result.get("task_type") for result in task_results if "task_type" in result]
            assert "TRAIN" in task_types
            assert "EVAL" in task_types
            assert "GENERATE" in task_types 