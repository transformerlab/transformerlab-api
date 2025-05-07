import requests


def test_data_gallery(live_server):
    resp = requests.get(f"{live_server}/data/gallery")
    assert resp.status_code == 200
    assert "data" in resp.json() or "status" in resp.json()


def test_data_list(live_server):
    resp = requests.get(f"{live_server}/data/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_data_info(live_server):
    resp = requests.get(f"{live_server}/data/info?dataset_id=dummy_dataset")
    assert resp.status_code in (200, 400, 404)


def test_data_preview(live_server):
    resp = requests.get(f"{live_server}/data/preview?dataset_id=dummy_dataset")
    assert resp.status_code in (200, 400, 404)
