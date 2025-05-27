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


def test_tools_prompt():
    with TestClient(app) as client:
        resp = client.get("/tools/prompt")
        assert resp.status_code == 200
        assert "<tools>" in resp.text


def test_tools_call_invalid_tool():
    with TestClient(app) as client:
        resp = client.get("/tools/call/invalid_tool?params={}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"


def test_tools_call_invalid_params():
    with TestClient(app) as client:
        resp = client.get("/tools/call/add?params=not_a_json")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"


def test_tools_install_mcp_server_invalid_file():
    with TestClient(app) as client:
        resp = client.get("/tools/install_mcp_server?server_name=/not/a/real/path.py")
        assert resp.status_code == 403
        assert resp.json()["status"] == "error"
