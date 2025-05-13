from fastapi.testclient import TestClient
from api import app


def test_train_templates():
    with TestClient(app) as client:
        resp = client.get("/train/templates")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_train_recipe_gallery():
    with TestClient(app) as client:
        resp = client.get("/train/template/gallery")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_train_export_recipe():
    with TestClient(app) as client:
        resp = client.get("/train/template/1/export")
        assert resp.status_code in (200, 404)


def test_train_create_template():
    with TestClient(app) as client:
        data = {"name": "test_template", "description": "desc", "type": "test", "config": "{}"}
        resp = client.post("/train/template/create", data=data)
        assert resp.status_code in (200, 422, 400)
