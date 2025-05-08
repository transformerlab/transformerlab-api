import requests
from fastapi.testclient import TestClient
from api import app


def test_server_info():
    with TestClient(app) as client:
        response = client.get("/server/info")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "cpu" in data
        assert "memory" in data and isinstance(data["memory"], dict)
        assert "disk" in data and isinstance(data["disk"], dict)
        assert "gpu" in data
        mem = data["memory"]
        for key in ("total", "available", "percent", "used", "free"):
            assert key in mem
        disk = data["disk"]
        for key in ("total", "used", "free", "percent"):
            assert key in disk


def test_server_python_libraries():
    with TestClient(app) as client:
        response = client.get("/server/python_libraries")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        for package in data:
            assert isinstance(package, dict)
            assert "name" in package and isinstance(package["name"], str) and package["name"]
            assert "version" in package and isinstance(package["version"], str) and package["version"]


def test_server_pytorch_collect_env():
    with TestClient(app) as client:
        response = client.get("/server/pytorch_collect_env")
        assert response.status_code == 200
        data = response.text
        assert "PyTorch" in data
