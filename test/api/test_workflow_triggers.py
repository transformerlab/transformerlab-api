import json
import os
import uuid
import tempfile
import shutil
import atexit
from fastapi.testclient import TestClient

# Create a unique test directory using absolute paths to prevent contamination
TEST_BASE_DIR = os.path.abspath(os.path.join(tempfile.gettempdir(), f"transformerlab_api_test_{uuid.uuid4().hex[:8]}"))
os.makedirs(TEST_BASE_DIR, exist_ok=True)

# Set environment variables BEFORE any transformerlab imports
os.environ["TFL_HOME_DIR"] = TEST_BASE_DIR
os.environ["TFL_WORKSPACE_DIR"] = TEST_BASE_DIR

# Patch the database path to ensure complete isolation BEFORE importing modules
TEST_DB_PATH = os.path.join(TEST_BASE_DIR, "test_llmlab.sqlite3")

# Import and patch database module
import transformerlab.db as db
db.DATABASE_FILE_NAME = TEST_DB_PATH
db.DATABASE_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH}"

# Recreate the async engine with the new path
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

db.async_engine = create_async_engine(f"sqlite+aiosqlite:///{TEST_DB_PATH}", echo=False)
db.async_session = sessionmaker(db.async_engine, expire_on_commit=False, class_=AsyncSession)

# Now import the rest
from api import app
from transformerlab.db import (
    PREDEFINED_TRIGGER_BLUEPRINTS
)

# Register cleanup function to run at exit
def cleanup_test_dir():
    if os.path.exists(TEST_BASE_DIR):
        shutil.rmtree(TEST_BASE_DIR, ignore_errors=True)

atexit.register(cleanup_test_dir)

class TestPredefinedTriggersEndpoints:
    """Test the predefined triggers API endpoints."""

    def test_get_predefined_triggers(self):
        """Test GET /workflows/predefined_triggers endpoint."""
        with TestClient(app) as client:
            response = client.get("/workflows/predefined_triggers")
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
            
            # Check structure of each trigger blueprint
            for trigger in data:
                assert "trigger_type" in trigger
                assert "name" in trigger
                assert "description" in trigger
                assert "default_is_enabled" in trigger
                assert isinstance(trigger["default_is_enabled"], bool)
            
            # Check that all expected trigger types are present
            trigger_types = {trigger["trigger_type"] for trigger in data}
            expected_types = {bp["trigger_type"] for bp in PREDEFINED_TRIGGER_BLUEPRINTS}
            assert trigger_types == expected_types

    def test_get_trigger_blueprints(self):
        """Test GET /workflows/predefined_triggers endpoint (alias test)."""
        with TestClient(app) as client:
            response = client.get("/workflows/predefined_triggers")
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
            
            # Should be identical to predefined_triggers
            response2 = client.get("/workflows/predefined_triggers")
            assert response.json() == response2.json()


class TestWorkflowTriggerConfigsEndpoint:
    """Test the workflow trigger configs update endpoint."""

    def test_update_trigger_configs_invalid_workflow_id(self):
        """Test updating trigger configs for non-existent workflow."""
        update_data = {
            "configs": []
        }
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            update_data["configs"].append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/999999/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400
            assert "not found" in response.json()["detail"].lower()

    def test_update_trigger_configs_wrong_count(self):
        """Test updating with wrong number of trigger configs."""
        update_data = {
            "configs": [
                {"trigger_type": "TRAIN", "is_enabled": True}
            ]
        }
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400
            assert "exactly 6 trigger configurations" in response.json()["detail"]

    def test_update_trigger_configs_invalid_trigger_type(self):
        """Test updating with invalid trigger type."""
        update_data = {
            "configs": []
        }
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            update_data["configs"].append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        # Replace one with invalid type
        update_data["configs"][0]["trigger_type"] = "INVALID_TYPE"
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400
            assert "Invalid trigger_type" in response.json()["detail"]

    def test_update_trigger_configs_duplicate_trigger_type(self):
        """Test updating with duplicate trigger types."""
        update_data = {
            "configs": []
        }
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            update_data["configs"].append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        # Make one duplicate
        update_data["configs"][1]["trigger_type"] = update_data["configs"][0]["trigger_type"]
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400
            assert "Duplicate trigger_type" in response.json()["detail"]

    def test_update_trigger_configs_missing_trigger_type(self):
        """Test updating with missing trigger_type field."""
        update_data = {
            "configs": []
        }
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            update_data["configs"].append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        # Remove trigger_type from one config
        del update_data["configs"][0]["trigger_type"]
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400  # API converts validation errors to 400

    def test_update_trigger_configs_missing_is_enabled(self):
        """Test updating with missing is_enabled field."""
        update_data = {
            "configs": []
        }
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            update_data["configs"].append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        # Remove is_enabled from one config
        del update_data["configs"][0]["is_enabled"]
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400  # API converts validation errors to 400

    def test_update_trigger_configs_invalid_request_body(self):
        """Test updating with invalid request body structure."""
        update_data = {
            "invalid_field": "invalid_value"
        }
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400  # API converts validation errors to 400

    def test_update_trigger_configs_empty_configs_array(self):
        """Test updating with empty configs array."""
        update_data = {
            "configs": []
        }
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400
            assert "exactly 6 trigger configurations" in response.json()["detail"]

    def test_update_trigger_configs_missing_configs_field(self):
        """Test updating with missing configs field."""
        update_data = {}  # No configs field
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400  # API converts validation errors to 400

    def test_update_trigger_configs_content_type_validation(self):
        """Test updating with wrong content type."""
        update_data = {
            "configs": []
        }
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            update_data["configs"].append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        with TestClient(app) as client:
            # Send as form data instead of JSON
            response = client.put(
                "/workflows/1/trigger_configs",
                data=json.dumps(update_data),  # Send as raw data, not JSON
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            assert response.status_code == 400

    def test_update_trigger_configs_method_not_allowed(self):
        """Test using wrong HTTP method."""
        update_data = {
            "configs": []
        }
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            update_data["configs"].append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        with TestClient(app) as client:
            # Use POST instead of PUT
            response = client.post(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 405  # Method Not Allowed

    def test_update_trigger_configs_general_exception_handling(self):
        """Test general exception handling in the endpoint."""
        import unittest.mock
        
        update_data = {
            "configs": []
        }
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            update_data["configs"].append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        with TestClient(app) as client:
            # Mock the database function to raise a general exception
            with unittest.mock.patch('transformerlab.db.workflow_update_trigger_configs') as mock_update:
                mock_update.side_effect = RuntimeError("Database connection failed")
                
                response = client.put(
                    "/workflows/1/trigger_configs",
                    json=update_data
                )
                
                assert response.status_code == 500
                assert "Internal server error" in response.json()["detail"]


class TestWorkflowGetByIdErrorHandling:
    """Test workflow_get_by_id endpoint error handling."""

    def test_workflow_get_by_id_not_found(self):
        """Test getting workflow by non-existent ID."""
        with TestClient(app) as client:
            response = client.get("/workflows/999999")
            
            assert response.status_code == 200
            data = response.json()
            assert "error" in data
            assert "not found" in data["error"].lower()

    def test_workflow_get_by_id_success(self):
        """Test successfully getting workflow by ID."""
        with TestClient(app) as client:
            # First create a workflow
            create_response = client.get("/workflows/create?name=test_workflow&experiment_id=1")
            assert create_response.status_code == 200
            
            # Get all workflows to find the created one
            list_response = client.get("/workflows/list")
            assert list_response.status_code == 200
            workflows = list_response.json()
            
            if workflows:
                workflow_id = workflows[0]["id"]
                
                # Get specific workflow by ID
                response = client.get(f"/workflows/{workflow_id}")
                
                assert response.status_code == 200
                data = response.json()
                assert "id" in data
                assert data["id"] == workflow_id
                assert "error" not in data


class TestWorkflowTriggerConfigsSuccessPath:
    """Test successful trigger config updates."""

    def test_update_trigger_configs_success(self):
        """Test successful trigger config update."""
        with TestClient(app) as client:
            # First create a workflow
            create_response = client.get("/workflows/create?name=test_success_workflow&experiment_id=1")
            assert create_response.status_code == 200
            
            # Get all workflows to find the created one
            list_response = client.get("/workflows/list")
            assert list_response.status_code == 200
            workflows = list_response.json()
            
            if workflows:
                workflow_id = workflows[0]["id"]
                
                # Prepare valid update data
                update_data = {
                    "configs": []
                }
                for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
                    update_data["configs"].append({
                        "trigger_type": bp["trigger_type"],
                        "is_enabled": True
                    })
                
                # Update trigger configs
                response = client.put(
                    f"/workflows/{workflow_id}/trigger_configs",
                    json=update_data
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "id" in data
                assert "trigger_configs" in data
                assert len(data["trigger_configs"]) == len(PREDEFINED_TRIGGER_BLUEPRINTS)
                
                # Verify all triggers are enabled
                for config in data["trigger_configs"]:
                    assert config["is_enabled"] is True 