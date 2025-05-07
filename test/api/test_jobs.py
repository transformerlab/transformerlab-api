import requests


def test_jobs_list(live_server):
    resp = requests.get(f"{live_server}/jobs/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_jobs_delete_all(live_server):
    resp = requests.get(f"{live_server}/jobs/delete_all")
    assert resp.status_code == 200
    assert "message" in resp.json() or resp.json() == []
