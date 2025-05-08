from fastapi.testclient import TestClient
from api import app


def test_tasks_list():
    with TestClient(app) as client:
        resp = client.get("/tasks/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_tasks_get_by_id():
    with TestClient(app) as client:
        resp = client.get("/tasks/1/get")
        assert resp.status_code in (200, 404)


def test_tasks_list_by_type():
    with TestClient(app) as client:
        resp = client.get("/tasks/list_by_type?type=TRAIN")
        assert resp.status_code in (200, 404)
