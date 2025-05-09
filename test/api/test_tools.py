from fastapi.testclient import TestClient
from api import app


def test_tools_list():
    with TestClient(app) as client:
        resp = client.get("/tools/list")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_tools_call():
    with TestClient(app) as client:
        resp = client.get("/tools/call/add?params={}")
        assert resp.status_code in (200, 400, 404)
