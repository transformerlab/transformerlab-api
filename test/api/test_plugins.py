from fastapi.testclient import TestClient
from api import app


def test_plugins_gallery():
    with TestClient(app) as client:
        resp = client.get("/plugins/gallery")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            plugin = data[0]
            assert "name" in plugin or "description" in plugin


def test_plugins_list():
    with TestClient(app) as client:
        resp = client.get("/plugins/list")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            plugin = data[0]
            assert "name" in plugin or "description" in plugin


def test_plugins_install():
    with TestClient(app) as client:
        resp = client.get("/plugins/gallery/fastchat_server/install")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert "message" in data or "status" in data
