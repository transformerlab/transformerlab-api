from fastapi.testclient import TestClient
from api import app


def test_jobs_list():
    with TestClient(app) as client:
        resp = client.get("/jobs/list")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list) or isinstance(data, dict)
        if isinstance(data, list) and data:
            job = data[0]
            assert "id" in job or "status" in job


def test_jobs_delete_all():
    with TestClient(app) as client:
        resp = client.get("/jobs/delete_all")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data or data == []
        if "message" in data:
            assert isinstance(data["message"], str)


def test_jobs_get_by_id():
    with TestClient(app) as client:
        resp = client.get("/jobs/1")
        assert resp.status_code in (200, 404)


def test_jobs_delete_by_id():
    with TestClient(app) as client:
        resp = client.get("/jobs/delete/1")
        assert resp.status_code in (200, 404)


def test_jobs_get_template():
    with TestClient(app) as client:
        resp = client.get("/jobs/template/1")
        assert resp.status_code in (200, 404)
