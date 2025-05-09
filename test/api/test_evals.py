from fastapi.testclient import TestClient
from api import app


def test_evals_list():
    with TestClient(app) as client:
        resp = client.get("/evals/list")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            eval_item = data[0]
            assert "name" in eval_item or "info" in eval_item


def test_evals_compare():
    with TestClient(app) as client:
        resp = client.get("/evals/compare_evals?job_list=1,2,3")
        assert resp.status_code in (200, 400, 404)
