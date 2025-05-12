from fastapi.testclient import TestClient
from api import app


def test_prompts_list():
    with TestClient(app) as client:
        resp = client.get("/prompts/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_prompts_dummy():
    with TestClient(app) as client:
        resp = client.get("/prompts/list?prompt_id=dummy")
        assert resp.status_code in (200, 400, 404)
