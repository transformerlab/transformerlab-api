import requests


def test_jobs_list(live_server):
    resp = requests.get(f"{live_server}/jobs/list")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) or isinstance(data, dict)
    if isinstance(data, list) and data:
        job = data[0]
        assert "id" in job or "status" in job


def test_jobs_delete_all(live_server):
    resp = requests.get(f"{live_server}/jobs/delete_all")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data or data == []
    if "message" in data:
        assert isinstance(data["message"], str)


def test_jobs_get_by_id(live_server):
    resp = requests.get(f"{live_server}/jobs/1")
    assert resp.status_code in (200, 404)


def test_jobs_delete_by_id(live_server):
    resp = requests.get(f"{live_server}/jobs/delete/1")
    assert resp.status_code in (200, 404)


def test_jobs_get_template(live_server):
    resp = requests.get(f"{live_server}/jobs/template/1")
    assert resp.status_code in (200, 404)
