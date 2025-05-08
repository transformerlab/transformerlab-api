# Test FastAPI endpoints using TestClient (in-process, for coverage)
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_set_config():
    with TestClient(app) as client:
        response = client.get("/config/set", params={"k": "test_key", "v": "test_value"})
        assert response.status_code == 200
        assert response.json() == {"key": "test_key", "value": "test_value"}


def test_get_config():
    # Ensure the value is set first
    with TestClient(app) as client:
        client.get("/config/set", params={"k": "test_key2", "v": "test_value2"})
        response = client.get("/config/get/test_key2")
        assert response.status_code == 200
        assert response.json() == "test_value2"
