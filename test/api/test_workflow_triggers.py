import json
import os
from fastapi.testclient import TestClient

os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from api import app
from transformerlab.db import (
    PREDEFINED_TRIGGER_BLUEPRINTS
)


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
        """Test updating without configs field."""
        update_data = {}
        
        with TestClient(app) as client:
            response = client.put(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            
            assert response.status_code == 400  # API converts validation errors to 400

    def test_update_trigger_configs_content_type_validation(self):
        """Test that content type must be JSON."""
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
                "/workflows/1/trigger_configs",
                data=json.dumps(update_data),  # Send as string instead of JSON
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            assert response.status_code == 400  # API converts validation errors to 400

    def test_update_trigger_configs_method_not_allowed(self):
        """Test that other HTTP methods are not allowed."""
        update_data = {
            "configs": []
        }
        for bp in PREDEFINED_TRIGGER_BLUEPRINTS:
            update_data["configs"].append({
                "trigger_type": bp["trigger_type"],
                "is_enabled": True
            })
        
        with TestClient(app) as client:
            # Test GET method
            response = client.get("/workflows/1/trigger_configs")
            assert response.status_code == 404  # Route doesn't exist
            
            # Test POST method
            response = client.post(
                "/workflows/1/trigger_configs",
                json=update_data
            )
            assert response.status_code == 405  # Method not allowed
            
            # Test DELETE method
            response = client.delete("/workflows/1/trigger_configs")
            assert response.status_code == 405  # Method not allowed 