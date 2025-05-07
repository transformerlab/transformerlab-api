import requests


def test_workflows_list(live_server):
    resp = requests.get(f"{live_server}/workflows/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_workflows_delete(live_server):
    resp = requests.get(f"{live_server}/workflows/delete/dummy_workflow")
    assert resp.status_code in (200, 404)


def test_workflows_create(live_server):
    resp = requests.get(f"{live_server}/workflows/create?name=test_workflow")
    assert resp.status_code in (200, 400, 404)
