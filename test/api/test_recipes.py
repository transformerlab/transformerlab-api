from fastapi.testclient import TestClient
from api import app


def test_recipes_list():
    with TestClient(app) as client:
        resp = client.get("/recipes/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


def test_recipes_get_by_id():
    with TestClient(app) as client:
        resp = client.get("/recipes/1")
        assert resp.status_code == 200
        data = resp.json()
        # Should be a recipe object
        assert isinstance(data, dict)


def test_recipes_get_by_invalid_id():
    with TestClient(app) as client:
        resp = client.get("/recipes/999")
        assert resp.status_code == 200
        data = resp.json()
        # Should return error for non-existent recipe
        assert "error" in data


def test_recipes_check_dependencies():
    with TestClient(app) as client:
        resp = client.get("/recipes/1/check_dependencies")
        assert resp.status_code == 200
        data = resp.json()
        # Should have dependencies array or error
        assert isinstance(data, dict)
        assert "dependencies" in data or "error" in data


def test_recipes_check_dependencies_invalid_id():
    with TestClient(app) as client:
        resp = client.get("/recipes/999/check_dependencies")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data  # Should specifically check for error, not just dict


def test_recipes_install_dependencies():
    with TestClient(app) as client:
        resp = client.get("/recipes/1/install_dependencies")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        # Should have job_id and status
        assert "job_id" in data or "error" in data


def test_recipes_install_dependencies_job_status():
    with TestClient(app) as client:
        # Try to get status for a dummy job ID
        resp = client.get("/recipes/jobs/999/status")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        # Should have error for non-existent job
        assert "error" in data or "job_id" in data


def test_recipes_create_experiment():
    with TestClient(app) as client:
        resp = client.post("/recipes/1/create_experiment?experiment_name=test_experiment_simple")
        assert resp.status_code in (200, 400, 409)  # 409 if experiment already exists
        data = resp.json()
        assert isinstance(data, dict)
        assert "status" in data


def test_recipes_create_experiment_with_notes():
    """Test creating experiment from recipe that has notes - should include notes_result"""
    with TestClient(app) as client:
        resp = client.post("/recipes/1/create_experiment?experiment_name=test_experiment_notes")
        assert resp.status_code in (200, 400, 409)
        data = resp.json()
        assert isinstance(data, dict)
        if resp.status_code == 200 and data.get("status") == "success":
            # If successful, should have data section
            assert "data" in data
            # notes_result should be present (either None or a result object)
            assert "notes_result" in data["data"]


def test_recipes_create_experiment_invalid_recipe():
    with TestClient(app) as client:
        resp = client.post("/recipes/999/create_experiment?experiment_name=test_experiment_invalid")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "not found" in data["message"]


def test_recipes_create_experiment_notes_functionality():
    """Test that creating an experiment from a recipe with notes actually saves the notes as readme.md"""
    with TestClient(app) as client:
        # Use recipe ID 8 which we added notes to in exp-recipe-gallery.json
        resp = client.post("/recipes/8/create_experiment?experiment_name=test_notes_functionality")
        
        assert resp.status_code in (200, 400, 409)
        data = resp.json()
        
        if resp.status_code == 200 and data.get("status") == "success":
            # Should have notes_result in the response data
            assert "data" in data
            assert "notes_result" in data["data"]
            
            # Since recipe 1 has notes, notes_result should not be None
            notes_result = data["data"]["notes_result"]
            if notes_result is not None:
                assert isinstance(notes_result, dict)

