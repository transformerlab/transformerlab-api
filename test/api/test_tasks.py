import requests


def test_tasks_list(live_server):
    resp = requests.get(f"{live_server}/tasks/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)
