import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from api import app

RECIPES = [
    {
        "id": 1,
        "title": "Test Recipe",
        "dependencies": [
            {"type": "model", "name": "model-a"},
            {"type": "dataset", "name": "dataset-x"},
            {"type": "plugin", "name": "plugin-1"},
            {"type": "workflow", "name": "wf-should-be-skipped"},
        ],
    },
    {
        "id": 2,
        "title": "No Deps",
        "dependencies": [],
    },
]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_check_dependencies_all_installed(client):
    models = [{"model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit"}]
    datasets = [{"dataset_id": "spencer/samsum_reformat"}]
    plugins = [
        {"uniqueId": "llama-trainer", "installed": True},
        {"uniqueId": "eleuther-ai-lm-evaluation-harness-mlx", "installed": True},
    ]
    with (
        patch(
            "transformerlab.shared.galleries.get_exp_recipe_gallery",
            return_value=[
                {
                    "id": 1,
                    "dependencies": [
                        {"type": "model", "name": "mlx-community/Llama-3.2-1B-Instruct-4bit"},
                        {"type": "plugin", "name": "llama-trainer"},
                        {"type": "dataset", "name": "spencer/samsum_reformat"},
                        {"type": "plugin", "name": "eleuther-ai-lm-evaluation-harness-mlx"},
                        {"type": "workflow", "name": "eval-and-deploy"},
                    ],
                }
            ],
        ),
        patch("transformerlab.models.model_helper.list_installed_models", AsyncMock(return_value=models)),
        patch("transformerlab.db.get_datasets", AsyncMock(return_value=datasets)),
        patch("transformerlab.routers.plugins.plugin_gallery", AsyncMock(return_value=plugins)),
    ):
        resp = client.get("/recipes/1/check_dependencies")
        assert resp.status_code == 200
        data = resp.json()
        deps = data["dependencies"]
        assert {d["name"]: d["installed"] for d in deps} == {
            "mlx-community/Llama-3.2-1B-Instruct-4bit": True,
            "llama-trainer": True,
            "spencer/samsum_reformat": True,
            "eleuther-ai-lm-evaluation-harness-mlx": True,
        }


def test_check_dependencies_some_missing(client):
    models = []
    datasets = []
    plugins = [
        {"uniqueId": "llama-trainer", "installed": False},
        {"uniqueId": "eleuther-ai-lm-evaluation-harness-mlx", "installed": True},
    ]
    with (
        patch(
            "transformerlab.shared.galleries.get_exp_recipe_gallery",
            return_value=[
                {
                    "id": 1,
                    "dependencies": [
                        {"type": "model", "name": "mlx-community/Llama-3.2-1B-Instruct-4bit"},
                        {"type": "plugin", "name": "llama-trainer"},
                        {"type": "dataset", "name": "spencer/samsum_reformat"},
                        {"type": "plugin", "name": "eleuther-ai-lm-evaluation-harness-mlx"},
                        {"type": "workflow", "name": "eval-and-deploy"},
                    ],
                }
            ],
        ),
        patch("transformerlab.models.model_helper.list_installed_models", AsyncMock(return_value=models)),
        patch("transformerlab.db.get_datasets", AsyncMock(return_value=datasets)),
        patch("transformerlab.routers.plugins.plugin_gallery", AsyncMock(return_value=plugins)),
    ):
        resp = client.get("/recipes/1/check_dependencies")
        assert resp.status_code == 200
        data = resp.json()
        deps = data["dependencies"]
        assert {d["name"]: d["installed"] for d in deps} == {
            "mlx-community/Llama-3.2-1B-Instruct-4bit": False,
            "llama-trainer": False,
            "spencer/samsum_reformat": False,
            "eleuther-ai-lm-evaluation-harness-mlx": True,
        }


def test_check_dependencies_no_deps(client):
    with patch("transformerlab.shared.galleries.get_exp_recipe_gallery", return_value=[{"id": 2, "dependencies": []}]):
        resp = client.get("/recipes/2/check_dependencies")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dependencies"] == []


def test_check_dependencies_not_found(client):
    with patch("transformerlab.shared.galleries.get_exp_recipe_gallery", return_value=[]):
        resp = client.get("/recipes/999/check_dependencies")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data and "not found" in data["error"]


def test_create_experiment_with_notes(client):
    """Test that experiments created from recipes with notes field have the notes content saved as readme.md"""
    recipe_with_notes = {
        "id": 1,
        "title": "Test Recipe with Notes",
        "notes": "# Test Notes\n\nThis is test content.",
        "dependencies": []
    }
    
    with (
        patch("transformerlab.shared.galleries.get_exp_recipe_gallery", return_value=[recipe_with_notes]),
        patch("transformerlab.db.experiment_get_by_name", return_value=None),
        patch("transformerlab.db.experiment_create", return_value=123),
        patch("transformerlab.routers.experiment.experiment.experiment_save_file_contents", 
              return_value={"message": "readme.md file contents saved"}),
        patch("transformerlab.models.model_helper.list_installed_models", return_value=[]),
    ):
        resp = client.post("/recipes/1/create_experiment?experiment_name=test_experiment")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"]["experiment_id"] == 123
        assert "notes_result" in data["data"]


def test_create_experiment_without_notes(client):
    """Test that experiments created from recipes without notes field work normally"""
    recipe_without_notes = {
        "id": 2,
        "title": "Test Recipe without Notes",
        "dependencies": []
    }
    
    with (
        patch("transformerlab.shared.galleries.get_exp_recipe_gallery", return_value=[recipe_without_notes]),
        patch("transformerlab.db.experiment_get_by_name", return_value=None),
        patch("transformerlab.db.experiment_create", return_value=124),
        patch("transformerlab.models.model_helper.list_installed_models", return_value=[]),
    ):
        resp = client.post("/recipes/2/create_experiment?experiment_name=test_experiment")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"]["notes_result"] is None


@pytest.mark.asyncio
async def test_create_experiment_notes_error_handling(client):
    """Test error handling when notes file creation fails"""
    
    # Mock recipe with notes field
    test_recipe = {
        "id": 3,
        "title": "Test Recipe with Notes Error",
        "notes": "# Notes that will fail to save",
        "dependencies": []
    }
    
    with (
        patch("transformerlab.shared.galleries.get_exp_recipe_gallery", return_value=[test_recipe]),
        patch("transformerlab.db.experiment_get_by_name", AsyncMock(return_value=None)),
        patch("transformerlab.db.experiment_create", AsyncMock(return_value=125)),
        patch("transformerlab.routers.experiment.experiment.experiment_save_file_contents", 
              AsyncMock(side_effect=Exception("File save error"))),
        patch("transformerlab.models.model_helper.list_installed_models", AsyncMock(return_value=[])),
    ):
        response = client.post("/recipes/3/create_experiment?experiment_name=test_experiment_error")
        
        # Verify response - experiment should still be created but notes_result should contain error
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["experiment_id"] == 125
        assert "notes_result" in data["data"]
        assert "error" in data["data"]["notes_result"]
        assert "Failed to create Notes file" in data["data"]["notes_result"]["error"]
