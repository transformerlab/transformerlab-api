import requests


def test_jobs_list(live_server):
    resp = requests.get(f"{live_server}/jobs/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_jobs_delete_all(live_server):
    resp = requests.get(f"{live_server}/jobs/delete_all")
    assert resp.status_code == 200
    assert "message" in resp.json() or resp.json() == []


def test_jobs_get_by_id(live_server):
    resp = requests.get(f"{live_server}/jobs/1")
    assert resp.status_code in (200, 404)


def test_jobs_delete_by_id(live_server):
    resp = requests.get(f"{live_server}/jobs/delete/1")
    assert resp.status_code in (200, 404)


def test_jobs_get_template(live_server):
    resp = requests.get(f"{live_server}/jobs/template/1")
    assert resp.status_code in (200, 404)
