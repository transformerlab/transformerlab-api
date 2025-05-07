import requests


def test_tasks_list(live_server):
    resp = requests.get(f"{live_server}/tasks/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_tasks_get_by_id(live_server):
    resp = requests.get(f"{live_server}/tasks/1/get")
    assert resp.status_code in (200, 404)


def test_tasks_list_by_type(live_server):
    resp = requests.get(f"{live_server}/tasks/list_by_type?type=TRAIN")
    assert resp.status_code in (200, 404)
