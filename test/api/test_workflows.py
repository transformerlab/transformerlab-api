from fastapi.testclient import TestClient
from api import app


def test_workflows_list():
    with TestClient(app) as client:
        resp = client.get("/workflows/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_workflows_delete():
    with TestClient(app) as client:
        resp = client.get("/workflows/delete/dummy_workflow")
        assert resp.status_code in (200, 404)


def test_workflows_create():
    with TestClient(app) as client:
        resp = client.get("/workflows/create?name=test_workflow")
        assert resp.status_code in (200, 400, 404)
