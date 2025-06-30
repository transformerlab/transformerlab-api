import pytest
from fastapi.testclient import TestClient
from api import app


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
