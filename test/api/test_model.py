import requests


def test_model_gallery(live_server):
    resp = requests.get(f"{live_server}/model/gallery")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


def test_model_list_local_uninstalled(live_server):
    resp = requests.get(f"{live_server}/model/list_local_uninstalled")
    assert resp.status_code == 200
    assert "data" in resp.json() or "status" in resp.json()
