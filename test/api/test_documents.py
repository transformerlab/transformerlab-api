import os
import pytest
import time
from fastapi.testclient import TestClient

# Set environment variables before importing modules
os.environ["TFL_HOME_DIR"] = "./test/tmp/"
os.environ["TFL_WORKSPACE_DIR"] = "./test/tmp"

from api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_download_zip_endpoint_exists(client):
    """Test that the download_zip endpoint exists"""
    # Test with empty request body - should get validation error, not 404
    response = client.post(
        "/experiment/test_exp_id/documents/download_zip",
        json={}
    )
    
    # Should have some validation error, not a 404
    assert response.status_code != 404
    # Should be a client error (400 range) due to missing required fields
    assert 400 <= response.status_code < 500


def test_download_zip_missing_url(client):
    """Test download_zip without URL returns proper error"""
    test_data = {
        "extract_folder_name": "test_folder"
    }
    
    response = client.post(
        "/experiment/test_exp_id/documents/download_zip",
        json=test_data
    )
    
    assert response.status_code == 400
    response_data = response.json()
    assert "detail" in response_data
    assert "URL is required" in response_data["detail"]


def test_download_zip_invalid_url(client):
    """Test download_zip with invalid URL format"""
    test_data = {
        "url": "invalid-url-format",
        "extract_folder_name": "test_folder"
    }
    
    response = client.post(
        "/experiment/test_exp_id/documents/download_zip",
        json=test_data
    )
    
    assert response.status_code == 400
    response_data = response.json()
    assert "detail" in response_data
    assert "Invalid or unauthorized URL" in response_data["detail"]


def test_download_zip_malformed_request(client):
    """Test download_zip with malformed JSON"""
    response = client.post(
        "/experiment/test_exp_id/documents/download_zip",
        data="invalid json"
    )
    
    # Should return a client error (400 range)
    assert 400 <= response.status_code < 500


def test_download_zip_valid_url_format(client):
    """Test download_zip with valid URL format"""
    # Create a test experiment first
    unique_name = f"test_download_zip_valid_url_format_{int(time.time() * 1000)}"
    exp_response = client.get(f"/experiment/create?name={unique_name}")
    assert exp_response.status_code == 200
    experiment_id = exp_response.json()
    
    test_data = {
        "url": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-zip-file.zip",
        "extract_folder_name": "test_folder"
    }
    
    response = client.post(
        f"/experiment/{experiment_id}/documents/download_zip",
        json=test_data
    )
    
    # Should process the ZIP download (may succeed or fail at download/extract stage)
    assert response.status_code != 404  # Not a routing error
    assert response.status_code != 422  # Not a validation error
    # Should be either success (200) or server error (500) during processing
    assert response.status_code in [200, 500]


def test_download_zip_with_folder_parameter(client):
    """Test download_zip with folder query parameter"""
    # Create a test experiment first
    unique_name = f"test_download_zip_with_folder_parameter_{int(time.time() * 1000)}"
    exp_response = client.get(f"/experiment/create?name={unique_name}")
    assert exp_response.status_code == 200
    experiment_id = exp_response.json()
    
    test_data = {
        "url": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-zip-file.zip", 
        "extract_folder_name": "test_folder"
    }
    
    response = client.post(
        f"/experiment/{experiment_id}/documents/download_zip?folder=custom_folder",
        json=test_data
    )
    
    # Should handle the folder parameter without validation errors
    assert response.status_code != 422  # Not a validation error
    assert response.status_code != 404  # Not a routing error
    # Should be either success (200) or server error (500) during processing
    assert response.status_code in [200, 500]


def test_download_zip_experiment_id_in_path(client):
    """Test that experiment ID is properly handled in the path"""
    # Create test experiments first
    experiment_ids = []
    timestamp = int(time.time() * 1000)
    for i, exp_name in enumerate(["test_exp", "exp123", "my-experiment"]):
        unique_name = f"test_download_zip_experiment_id_in_path_{exp_name}_{timestamp}_{i}"
        exp_response = client.get(f"/experiment/create?name={unique_name}")
        assert exp_response.status_code == 200
        experiment_ids.append(exp_response.json())
    
    test_data = {
        "url": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-zip-file.zip",
        "extract_folder_name": "test_folder"
    }
    
    # Test with different experiment IDs
    for exp_id in experiment_ids:
        response = client.post(
            f"/experiment/{exp_id}/documents/download_zip",
            json=test_data
        )
        
        # Should not return 404 (path not found) - proves routing works
        assert response.status_code != 404
        # Should be either success (200) or server error (500) during processing
        assert response.status_code in [200, 500]


def test_download_zip_optional_fields(client):
    """Test download_zip with minimal required fields"""
    # Create a test experiment first
    unique_name = f"test_download_zip_optional_fields_{int(time.time() * 1000)}"
    exp_response = client.get(f"/experiment/create?name={unique_name}")
    assert exp_response.status_code == 200
    experiment_id = exp_response.json()
    
    test_data = {
        "url": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-zip-file.zip"
        # extract_folder_name is optional
    }
    
    response = client.post(
        f"/experiment/{experiment_id}/documents/download_zip",
        json=test_data
    )
    
    # Should pass validation (extract_folder_name is optional)
    assert response.status_code != 422  # Not a validation error  
    assert response.status_code != 404  # Not a routing error
    # Should be either success (200) or server error (500) during processing
    assert response.status_code in [200, 500] 